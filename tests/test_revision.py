"""Tests for the post-hoc reviewer revision: the label-free drift score and the
policies built on it. Synthetic data only, so these run in about a second."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data, harness  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
THR = CFG["volatility"]["threshold"]


def vol_pool(level, n=200, seed=0):
    """A pool whose snapshot volatility averages `level` x the frozen threshold."""
    rng = np.random.default_rng(seed)
    pool = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    pool.insert(0, "sample_id", np.arange(n))
    pool["snap_vol"] = level * THR
    return pool


# --- drift_score ---------------------------------------------------------------

def test_drift_score_is_mean_volatility_over_threshold():
    assert data.drift_score(vol_pool(1.5), CFG) == pytest.approx(1.5)
    assert data.drift_score(vol_pool(0.4), CFG) == pytest.approx(0.4)


def test_drift_score_ignores_undefined_volatility():
    pool = vol_pool(2.0)
    pool.loc[:49, "snap_vol"] = np.nan          # first rows of a stock-day: no estimate yet
    assert data.drift_score(pool, CFG) == pytest.approx(2.0)
    pool["snap_vol"] = np.nan
    assert data.drift_score(pool, CFG) == 0.0    # nothing defined -> calm


@pytest.mark.parametrize("col", ["target", "label_k10"])
def test_drift_score_refuses_label_columns(col):
    # Leakage rule 2: the drift score (and so selection) must never be handed labels.
    pool = vol_pool(1.0)
    pool[col] = 0
    with pytest.raises(ValueError):
        data.drift_score(pool, CFG)


# --- harness hands policies features + snap_vol, never the target ---------------

def test_policy_pool_has_snap_vol_and_no_target():
    from tests.test_harness import make_cfg, make_df
    seen = []

    def spy(pool, proba, B, rng, selected, cfg):
        seen.append(list(pool.columns))
        return []

    harness.POLICIES["_spy"] = spy
    try:
        harness.replay(make_df(), make_cfg(), "_spy", 0, harness.connect(":memory:"))
    finally:
        del harness.POLICIES["_spy"]
    assert seen
    for cols in seen:
        assert "snap_vol" in cols and "target" not in cols
