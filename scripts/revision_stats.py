"""Revision R4 (statistical variation) and R5 (selection-cost benchmark).

Post-hoc reviewer revision. Reads finished, test-scored runs ('main' step-5 runs
and 'revision' runs), rebuilds each run's final model exactly as
src/evaluate.py does, and predicts the test days once more to get per-window
confusion matrices. No setting is chosen here; this only measures spread.

Paired bootstrap over TEST WINDOWS (the W-snapshot blocks used for the
volatility split), 2,000 resamples, fixed seed. Windows -- not snapshots -- are
resampled to respect time dependence inside a window. In each resample every
run is scored on the same drawn windows (paired); a policy's score is its
per-seed macro-F1 averaged over its seeds ("per-seed then averaged"). The
difference reported is E_drift minus the other policy, in percentage points.

Also: per-seed test macro-F1 for every run (per_seed.csv), and a selection-
time benchmark of every policy on the same 93 stream pools under V1's
probabilities (selection_benchmark.csv) -- no retraining, no runs.db writes.

Usage:  .venv/bin/python scripts/revision_stats.py
"""

import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data, evaluate, harness, models  # noqa: E402
from src.policies import POLICIES  # noqa: E402

OUT = ROOT / "results" / "revision"
REF = "E_drift"
COMPARE = ["E", "E_minus_audit", "E_minus_cluster", "E_drift_minus_cluster", "A", "B", "C", "D"]
DETERMINISTIC = {"A", "C", "D"}   # no randomness: one run, spread 0 by construction
N_BOOT, BOOT_SEED = 2000, 0


def macro_f1_from_conf(conf):
    """Macro-F1 from (..., 3, 3) confusion matrices [true, pred]; 0/0 -> 0."""
    tp = np.diagonal(conf, axis1=-2, axis2=-1)
    fp = conf.sum(axis=-2) - tp
    fn = conf.sum(axis=-1) - tp
    den = 2 * tp + fp + fn
    f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    return f1.mean(axis=-1)


def window_confusions(df, cfg, db, runs):
    """(n_runs, n_windows, 3, 3) confusion counts on the test days, plus window info."""
    feats = models.feature_columns(cfg)
    test = df[df.day.isin(cfg["splits"]["test"]) & (df.target >= 0)]
    vol = data.classify_windows(df, cfg["splits"]["test"], cfg, cfg["volatility"]["threshold"])
    test = test.merge(vol[["sample_id", "window", "high_vol"]], on="sample_id", how="left")
    X, y, w = test[feats].to_numpy(), test.target.to_numpy(), test.window.to_numpy()
    wins = np.unique(w)
    widx = np.searchsorted(wins, w)
    high = test.groupby("window").high_vol.first().reindex(wins).to_numpy()
    conf = np.zeros((len(runs), len(wins), 3, 3))
    for r, rid in enumerate(runs.run_id):
        t = time.perf_counter()
        pred = evaluate.reconstruct_final_model(df, cfg, db, rid).predict(X)
        np.add.at(conf[r], (widx, y, pred), 1)
        print(f"  {rid}: rebuilt + predicted in {time.perf_counter() - t:.1f}s", flush=True)
    return conf, high, wins


def bootstrap(conf, runs, high):
    rng = np.random.default_rng(BOOT_SEED)
    pols = list(dict.fromkeys(runs.policy))
    members = {p: np.where(runs.policy.to_numpy() == p)[0] for p in pols}
    rows = []
    for cond, mask in [("high_vol", high), ("all", np.ones_like(high, bool))]:
        idx = np.where(mask)[0]
        sub = conf[:, idx]                                       # (runs, nw, 3, 3)
        point = macro_f1_from_conf(sub.sum(axis=1))              # per run, all windows
        draws = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=N_BOOT)
        boot = macro_f1_from_conf(np.einsum("bw,rwij->brij", draws, sub))   # (boot, runs)
        pol_point = {p: point[members[p]].mean() for p in pols}
        pol_boot = {p: boot[:, members[p]].mean(axis=1) for p in pols}
        for other in [p for p in COMPARE if p in pols]:
            d = (pol_boot[REF] - pol_boot[other]) * 100
            rows.append({
                "comparison": f"{REF} - {other}", "condition": cond, "n_windows": len(idx),
                "seeds_ref": len(members[REF]), "seeds_other": len(members[other]),
                "diff_pp": (pol_point[REF] - pol_point[other]) * 100,
                "ci95_low_pp": np.percentile(d, 2.5), "ci95_high_pp": np.percentile(d, 97.5),
                "p_ref_better": float((d > 0).mean()),
                "other_spread": "0 by construction (deterministic)" if other in DETERMINISTIC
                                else "seed std in ablation.csv",
            })
    return pd.DataFrame(rows)


def selection_benchmark(df, cfg):
    """Time each policy's selection call on the same 93 pools (V1 probabilities)."""
    feats = models.feature_columns(cfg)
    Xtr, ytr, _ = models.split_xy(df, cfg["splits"]["train"], cfg)
    v1 = models.make_v1(cfg).fit(Xtr, ytr)
    stream = harness.build_stream(df, cfg)
    B = cfg["harness"]["budget"]
    pools = []
    for k in range(int(stream.pool_window.max()) + 1):
        pool = stream[stream.pool_window == k]
        if len(pool):
            pools.append((pool[["sample_id"] + feats + ["snap_vol"]],
                          v1.predict_proba(pool[feats].to_numpy())))
    rows = []
    for name in ["B", "C", "D", "E", "E_minus_audit", "E_drift",
                 "E_minus_cluster", "E_drift_minus_cluster"]:
        rng, ms = np.random.default_rng(0), []
        for pool, proba in pools:
            t = time.perf_counter()
            POLICIES[name](pool, proba, B, rng, set(), cfg)
            ms.append((time.perf_counter() - t) * 1000)
        ms = np.array(ms)
        rows.append({"policy": name, "windows": len(ms), "select_ms_median": np.median(ms),
                     "select_ms_p99": np.percentile(ms, 99), "select_s_total": ms.sum() / 1000})
    return pd.DataFrame(rows)


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    if cfg["revision"]["audit_low"] is None:
        raise SystemExit("revision.audit_low/high not frozen yet (run R2 first)")
    df = pd.read_pickle(ROOT / cfg["data"]["processed_path"])
    db = sqlite3.connect(ROOT / cfg["harness"]["runs_db"], timeout=600)
    runs = pd.read_sql(
        "SELECT run_id, policy, seed, tag FROM runs WHERE tag IN ('main','revision') "
        "AND end_time IS NOT NULL AND run_id IN (SELECT run_id FROM metrics WHERE split='test') "
        "ORDER BY policy, seed", db)
    if REF not in set(runs.policy):
        raise SystemExit(f"no scored {REF} runs yet (run R3 first)")
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Rebuilding {len(runs)} final models and predicting the test days ...")
    conf, high, wins = window_confusions(df, cfg, db, runs)

    # Sanity: window-sum macro-F1 equals the scored metric in runs.db.
    met = pd.read_sql("SELECT run_id, condition, value FROM metrics WHERE split='test' "
                      "AND metric='macro_f1'", db).pivot(index="run_id", columns="condition",
                                                         values="value")
    per_seed = runs.copy()
    per_seed["macroF1_high"] = macro_f1_from_conf(conf[:, high].sum(axis=1))
    per_seed["macroF1_low"] = macro_f1_from_conf(conf[:, ~high].sum(axis=1))
    per_seed["macroF1_all"] = macro_f1_from_conf(conf.sum(axis=1))
    gap = np.abs(per_seed.set_index("run_id").macroF1_high - met.high_vol.reindex(per_seed.run_id))
    print(f"max |recomputed - stored| high-vol macro-F1: {gap.max():.2e}")
    per_seed.to_csv(OUT / "per_seed.csv", index=False)

    res = bootstrap(conf, runs, high)
    res.to_csv(OUT / "bootstrap.csv", index=False)
    print(f"\nTest windows: {len(wins)} ({int(high.sum())} high-volatility). "
          f"Paired window bootstrap, {N_BOOT} resamples, seed {BOOT_SEED}:")
    print(res.drop(columns=["other_spread"]).round(3).to_string(index=False))

    print("\nSelection-time benchmark (same pools, V1 probabilities):")
    bench = selection_benchmark(df, cfg)
    bench.to_csv(OUT / "selection_benchmark.csv", index=False)
    print(bench.round(2).to_string(index=False))
    print("\nWrote results/revision/per_seed.csv, bootstrap.csv, selection_benchmark.csv")


if __name__ == "__main__":
    main()
