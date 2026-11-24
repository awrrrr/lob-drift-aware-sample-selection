"""Step 5 results: aggregate runs.db across seeds into tables and plots.

Reads the test/updates/latency metrics written by src/evaluate.py, averages
over seeds per policy, and writes:
  results/summary.csv         one row per policy (mean +/- std over seeds)
  results/differences.csv     E minus B/C/D on high-vol macro-F1, in pp
  results/high_vol_macrof1.png bar chart with per-seed spread

Usage:  .venv/bin/python scripts/make_tables.py
        .venv/bin/python scripts/make_tables.py --revision

The default reads only runs tagged 'main' (the original step 5), so it
reproduces results/summary.csv exactly. --revision (post-hoc reviewer
revision) writes results/revision/ablation.csv and overhead.csv instead and
never touches the step-5 files.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sqlite3  # noqa: E402

POLICY_ORDER = ["A", "B", "C", "D", "E", "E_minus_audit"]
POLICY_LABEL = {"A": "A fixed", "B": "B random", "C": "C uncert",
                "D": "D dens×unc", "E": "E proposed", "E_minus_audit": "E−audit"}


def load(db, tags=("main",)):
    """Runs with one of `tags` (finished only) and their metrics."""
    marks = ",".join("?" * len(tags))
    runs = pd.read_sql(f"SELECT run_id, policy, seed, tag FROM runs "
                       f"WHERE tag IN ({marks}) AND end_time IS NOT NULL", db, params=list(tags))
    metrics = pd.read_sql("SELECT * FROM metrics", db)
    m = metrics.merge(runs, on="run_id")
    return runs, m


def pick(m, split, cond, metric):
    """Per-run values for one metric as a Series indexed by (policy, seed)."""
    sub = m[(m.split == split) & (m.condition == cond) & (m.metric == metric)]
    return sub.set_index(["policy", "seed"]).value


def agg(series):
    """mean and std over seeds, per policy."""
    g = series.groupby(level="policy")
    return g.mean(), g.std().fillna(0.0)


ABLATION_ORDER = ["E_drift", "E", "E_minus_audit", "E_minus_cluster",
                  "E_drift_minus_cluster", "A", "B", "C", "D"]
ABLATION_DESC = {
    "E_drift": "full: cluster diversity + audit + drift adaptation",
    "E": "no drift adaptation (fixed 20% audit)",
    "E_minus_audit": "no audit",
    "E_minus_cluster": "fixed audit + top-entropy (no clustering)",
    "E_drift_minus_cluster": "drift-adaptive audit + top-entropy (no clustering)",
    "A": "baseline: fixed model", "B": "baseline: random",
    "C": "baseline: uncertainty", "D": "baseline: density x uncertainty (adapted from Liu et al.)",
}


def revision_tables(db, out_dir):
    """R3 ablation table and R5 overhead table (post-hoc revision)."""
    runs, m = load(db, tags=("main", "revision"))
    present = [p for p in ABLATION_ORDER if p in set(runs.policy)]
    rows = {}
    for key, cond in [("high", "high_vol"), ("low", "low_vol"), ("all", "all")]:
        mean, std = agg(pick(m, "test", cond, "macro_f1"))
        rows[f"macroF1_{key}_mean"], rows[f"macroF1_{key}_std"] = mean, std
    tab = pd.DataFrame(rows).reindex(present)
    tab.insert(0, "n_seeds", runs.groupby("policy").seed.nunique().reindex(present))
    tab.insert(0, "runs_tag", runs.groupby("policy").tag.first().reindex(present))
    tab.insert(0, "description", [ABLATION_DESC[p] for p in present])
    if "E_drift" in present:
        for key in ["high", "low", "all"]:
            ref = tab.loc["E_drift", f"macroF1_{key}_mean"]
            tab[f"pp_vs_E_drift_{key}"] = (tab[f"macroF1_{key}_mean"] - ref) * 100
    tab.to_csv(out_dir / "ablation.csv")
    show = [c for c in tab.columns if c.endswith("_mean") or c.startswith("pp_") or c == "n_seeds"]
    print(tab[show].round(4).to_string())

    # Overhead (R5): per policy, from the `timings` rows of runs that have them.
    tim = pd.read_sql("SELECT t.*, r.policy, r.tag FROM timings t JOIN runs r USING(run_id) "
                      "WHERE r.tag IN ('revision', 'revision_check')", db)
    lat = m[m.split == "latency"].pivot_table(index="policy", columns="metric",
                                               values="value", aggfunc="mean")
    ov = tim.groupby("policy").agg(
        n_windows=("window", "size"),
        select_ms_median=("select_ms", "median"),
        select_ms_p99=("select_ms", lambda s: s.quantile(0.99)),
        retrain_ms_median=("retrain_ms", "median"),
        retrain_ms_p99=("retrain_ms", lambda s: s.quantile(0.99)))
    ov = ov.join(lat.rename(columns={"p50_ms": "predict_p50_ms", "p99_ms": "predict_p99_ms"}),
                 how="left")
    ov.to_csv(out_dir / "overhead.csv")
    print("\nOverhead (timed runs only; ms per window, prediction per snapshot):")
    print(ov.round(2).to_string())
    print(f"\nWrote {out_dir.relative_to(ROOT)}/ablation.csv and overhead.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", action="store_true",
                    help="write the post-hoc revision tables to results/revision/")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    db = sqlite3.connect(ROOT / cfg["harness"]["runs_db"])
    if args.revision:
        out = ROOT / "results" / "revision"
        out.mkdir(parents=True, exist_ok=True)
        revision_tables(db, out)
        return
    runs, m = load(db)
    present = [p for p in POLICY_ORDER if p in set(runs.policy)]

    fields = {
        "macroF1_high": ("test", "high_vol", "macro_f1"),
        "macroF1_all": ("test", "all", "macro_f1"),
        "macroF1_low": ("test", "low_vol", "macro_f1"),
        "acc_all": ("test", "all", "accuracy"),
        "accepted": ("updates", "all", "n_accepted"),
        "rejected": ("updates", "all", "n_rejected"),
        "lat_p50_ms": ("latency", "all", "p50_ms"),
        "lat_p99_ms": ("latency", "all", "p99_ms"),
    }
    table = {}
    for name, (s, c, mt) in fields.items():
        mean, std = agg(pick(m, s, c, mt))
        table[name + "_mean"] = mean
        table[name + "_std"] = std
    summary = pd.DataFrame(table).reindex(present)
    summary.insert(0, "n_seeds", runs.groupby("policy").seed.nunique().reindex(present))
    summary.to_csv(ROOT / "results" / "summary.csv")

    # Primary comparison: E vs B, C, D on high-vol macro-F1, in percentage points.
    hv_mean = summary["macroF1_high_mean"]
    diffs = []
    if "E" in present:
        for base in ["B", "C", "D"]:
            if base in present:
                diffs.append({"comparison": f"E - {base}",
                              "high_vol_pp": (hv_mean["E"] - hv_mean[base]) * 100,
                              "E_high_vol": hv_mean["E"], f"{base}_high_vol": hv_mean[base]})
    if "E" in present and "E_minus_audit" in present:
        diffs.append({"comparison": "E - E_minus_audit (audit effect)",
                      "high_vol_pp": (hv_mean["E"] - hv_mean["E_minus_audit"]) * 100,
                      "E_high_vol": hv_mean["E"], "E_minus_audit_high_vol": hv_mean["E_minus_audit"]})
    pd.DataFrame(diffs).to_csv(ROOT / "results" / "differences.csv", index=False)

    # Plot: high-vol macro-F1 per policy with per-seed spread as error bars.
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [POLICY_LABEL[p] for p in present]
    ax.bar(labels, summary["macroF1_high_mean"], yerr=summary["macroF1_high_std"],
           capsize=4, color="#4C78A8")
    ax.set_ylabel("macro-F1 (high-volatility test windows)")
    ax.set_title("Step 5: high-volatility test macro-F1 by policy (mean ± std over seeds)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "high_vol_macrof1.png", dpi=130)

    # Console view.
    show = summary[["n_seeds", "macroF1_high_mean", "macroF1_high_std",
                    "macroF1_all_mean", "macroF1_low_mean", "acc_all_mean",
                    "accepted_mean", "rejected_mean", "lat_p50_ms_mean", "lat_p99_ms_mean"]]
    print(show.round(4).to_string())
    print("\nE vs baselines on high-vol macro-F1 (percentage points):")
    print(pd.DataFrame(diffs).round(3).to_string(index=False))
    print(f"\nWrote results/summary.csv, results/differences.csv, results/high_vol_macrof1.png")


if __name__ == "__main__":
    main()
