"""Score every run in runs.db on the test days (step 5).

This is the missing link between run_experiments.py (which replays the stream
and logs selections/updates) and make_tables.py (which reads the test metrics).
It reconstructs each run's final model from runs.db and config.yaml, scores the
held-out test days once, and writes the results into the metrics table. It
never changes a setting and never touches the test days for any choice
(CLAUDE.md leakage rule 3).

Usage:  .venv/bin/python scripts/evaluate.py
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import evaluate, harness  # noqa: E402


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    df = pd.read_pickle(ROOT / cfg["data"]["processed_path"])
    db = harness.connect(ROOT / cfg["harness"]["runs_db"])
    run_ids = evaluate.evaluate_all(df, cfg, db)
    db.close()
    print(f"\nScored {len(run_ids)} run(s) on the test days; metrics written to "
          f"{cfg['harness']['runs_db']}.")


if __name__ == "__main__":
    main()
