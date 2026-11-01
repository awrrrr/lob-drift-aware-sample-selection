"""Step 5 results: aggregate runs.db across seeds into tables and plots.

Reads the test/updates/latency metrics written by src/evaluate.py, averages
over seeds per policy, and writes:
  results/summary.csv         one row per policy (mean +/- std over seeds)
  results/differences.csv     E minus B/C/D on high-vol macro-F1, in pp
  results/high_vol_macrof1.png bar chart with per-seed spread

Usage:  .venv/bin/python scripts/make_tables.py
"""

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


def load(db):
    runs = pd.read_sql("SELECT run_id, policy, seed FROM runs", db)
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


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    db = sqlite3.connect(ROOT / cfg["harness"]["runs_db"])
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
