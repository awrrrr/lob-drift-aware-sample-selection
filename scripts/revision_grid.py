"""Revision R2: choose E_drift's audit shares (a_low, a_high) on validation.

Post-hoc reviewer revision. Each grid point replays the stream with policy
E_drift (seed 0) and is judged by the FINAL validation (day 6) macro-F1 of the
replayed model. No test data is touched: grid runs are tagged
'revision_grid:a_low=..,a_high=..' and are never scored on days 9-10.

One grid point per process, so the four can run in parallel:
    .venv/bin/python scripts/revision_grid.py --a-low 0.1 --a-high 0.4
Then summarise (and print the winner to freeze in config.yaml):
    .venv/bin/python scripts/revision_grid.py --summary
"""

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import harness  # noqa: E402

TAG = "revision_grid:a_low={},a_high={}"


def run_point(cfg, a_low, a_high):
    df = pd.read_pickle(ROOT / cfg["data"]["processed_path"])
    c = copy.deepcopy(cfg)
    c["revision"]["audit_low"], c["revision"]["audit_high"] = a_low, a_high
    db = harness.connect(ROOT / cfg["harness"]["runs_db"])
    out = harness.replay(df, c, "E_drift", 0, db, verbose=True,
                         tag=TAG.format(a_low, a_high))
    db.close()
    print(out)


def summary(cfg):
    db = harness.connect(ROOT / cfg["harness"]["runs_db"])
    q = ("SELECT r.tag, r.run_id, m.value AS final_val_macro_f1 FROM runs r "
         "JOIN metrics m USING(run_id) WHERE r.tag LIKE 'revision_grid:%' "
         "AND m.split='validation' AND m.condition='all' AND m.metric='macro_f1' "
         "AND r.end_time IS NOT NULL")
    res = pd.read_sql(q, db)
    ups = pd.read_sql("SELECT run_id, SUM(accepted) acc, SUM(1-accepted) rej FROM updates "
                      "GROUP BY run_id", db)
    res = res.merge(ups, on="run_id", how="left")
    res["a_low"] = res.tag.str.extract(r"a_low=([\d.]+)").astype(float)
    res["a_high"] = res.tag.str.extract(r"a_high=([\d.]+)").astype(float)
    # Highest final validation macro-F1; ties broken by grid order in config.yaml.
    order = {tuple(p): i for i, p in enumerate(cfg["revision"]["audit_grid"])}
    res["grid_pos"] = [order.get((lo, hi), 99) for lo, hi in zip(res.a_low, res.a_high)]
    res = res.sort_values(["final_val_macro_f1", "grid_pos"], ascending=[False, True])
    print(res[["a_low", "a_high", "final_val_macro_f1", "acc", "rej", "run_id"]]
          .to_string(index=False))
    best = res.iloc[0]
    print(f"\nwinner: a_low={best.a_low}, a_high={best.a_high} "
          f"(final val macro-F1 {best.final_val_macro_f1:.5f}); "
          f"spread across grid {res.final_val_macro_f1.max() - res.final_val_macro_f1.min():.5f}")
    out = ROOT / "results" / "revision" / "audit_grid.csv"
    res.drop(columns="grid_pos").to_csv(out, index=False)
    print(f"Wrote {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-low", type=float)
    ap.add_argument("--a-high", type=float)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    if args.summary:
        summary(cfg)
    else:
        run_point(cfg, args.a_low, args.a_high)


if __name__ == "__main__":
    main()
