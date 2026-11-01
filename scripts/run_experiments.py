"""Run replay policies and log to runs.db.

Step 3 usage (exercise the harness with the baselines):
    .venv/bin/python scripts/run_experiments.py --policies A B --seeds 0
    .venv/bin/python scripts/run_experiments.py --policies B --seeds 0 --max-windows 5

Step 5 will call this with all policies and 5 seeds.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import harness  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["A", "B"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a policy/seed already completed at the current config hash")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    df = pd.read_pickle(ROOT / cfg["data"]["processed_path"])
    db = harness.connect(ROOT / cfg["harness"]["runs_db"])
    chash = harness.config_hash(ROOT / "config.yaml")

    for policy in args.policies:
        for seed in args.seeds:
            if args.skip_existing:
                done = db.execute(
                    "SELECT 1 FROM runs WHERE policy=? AND seed=? AND config_hash=? "
                    "AND end_time IS NOT NULL", (policy, seed, chash)).fetchone()
                if done:
                    print(f"policy {policy} seed {seed}: already done, skip", flush=True)
                    continue
            print(f"policy {policy} seed {seed} ...", flush=True)
            out = harness.replay(df, cfg, policy, seed, db,
                                 max_windows=args.max_windows, verbose=True)
            print(f"  {out}")
    db.close()


if __name__ == "__main__":
    main()
