"""Tests for step-5 model reconstruction and the volatility condition."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data, evaluate, harness  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config.yaml"))


def _db_with(updates, selections):
    db = harness.connect(":memory:")
    db.executemany("INSERT INTO updates VALUES (?,?,?,?,?,?,?)",
                   [("r", w, 0, 0, a, 0.0, 0.0) for w, a in updates])
    db.executemany("INSERT INTO selections VALUES (?,?,?,?,?,?,?,?)",
                   [("r", w, s, "x", 0, 0, 0, 0) for w, s in selections])
    db.commit()
    return db


def test_reconstruction_uses_selections_up_to_last_accepted_window():
    sels = [(0, 10), (0, 11), (1, 20), (1, 21), (2, 30), (2, 31)]
    # last accepted window = 2 -> all six ids
    db = _db_with([(0, 1), (1, 0), (2, 1)], sels)
    assert set(evaluate.final_model_training_ids(db, "r")) == {10, 11, 20, 21, 30, 31}
    # last accepted window = 0 -> only window-0 ids (window 1,2 rejected)
    db = _db_with([(0, 1), (1, 0), (2, 0)], sels)
    assert set(evaluate.final_model_training_ids(db, "r")) == {10, 11}
    # nothing accepted -> empty (model stays V1)
    db = _db_with([(0, 0), (1, 0)], sels)
    assert len(evaluate.final_model_training_ids(db, "r")) == 0


def test_volatility_threshold_and_classification():
    # Two stock-days: one with large mid swings (high vol), one flat (low vol).
    cfg = {k: dict(v) for k, v in CFG.items() if isinstance(v, dict)}
    cfg["harness"] = dict(CFG["harness"]); cfg["harness"]["window"] = 100
    cfg["volatility"] = dict(CFG["volatility"]); cfg["volatility"]["vol_window"] = 10
    cfg["volatility"]["percentile"] = 50
    cfg["splits"] = dict(CFG["splits"])

    rng = np.random.default_rng(0)
    rows = []
    for day, scale in [(1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (5, 0.01)]:
        mid = np.cumsum(scale * rng.standard_normal(300))
        cols = {f"f{c}": 0.0 for c in range(144)}
        cols["f0"], cols["f2"] = mid, mid - 0.01
        cols.update(day=day, stock_id=0, t=np.arange(300), target=0)
        rows.append(pd.DataFrame(cols))
    df = pd.concat(rows, ignore_index=True)
    df.insert(0, "sample_id", np.arange(len(df)))

    thr = data.fit_volatility_threshold(df, cfg)
    assert thr > 0
    cls = data.classify_windows(df, [1, 2, 3, 4, 5], cfg, thr)
    # Day 5 (scale 0.01) windows should be low-vol; day 1 (scale 1.0) high-vol.
    by_day = cls.merge(df[["sample_id", "day"]], on="sample_id")
    assert by_day[by_day.day == 1].high_vol.mean() > by_day[by_day.day == 5].high_vol.mean()
    assert by_day[by_day.day == 5].high_vol.mean() == 0.0
