"""Data recovery tests. The raw-file tests skip if data/raw is empty."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data  # noqa: E402

RAW = ROOT / "data" / "raw"
needs_raw = pytest.mark.skipif(not data.train_file(RAW, 1).exists(), reason="raw data not present")


def test_find_stock_boundaries_on_synthetic_day():
    # Five flat price levels with small noise; boundaries at 100, 250, 300, 420.
    rng = np.random.default_rng(0)
    lengths = [100, 150, 50, 120, 80]
    levels = [-1.0, 2.0, 0.5, -0.3, 1.2]
    ask = np.concatenate([lv + 0.001 * rng.standard_normal(n) for lv, n in zip(levels, lengths)])
    x = np.zeros((len(ask), data.N_ROWS))
    x[:, data.ASK1_COL] = ask
    x[:, data.BID1_COL] = ask - 0.01
    starts, _ = data.find_stock_boundaries(x)
    assert list(starts) == [0, 100, 250, 300, 420]


def test_add_target_shifts_within_stock_day():
    # Two stock-days of 12 rows each; labels 1..3 cycling so every row is distinct-ish.
    import pandas as pd
    raw = [1, 2, 3, 1, 2, 3, 3, 3, 1, 2, 2, 1]
    df = pd.DataFrame({
        "day": [1] * 24,
        "stock_id": [0] * 12 + [1] * 12,
        "label_k10": raw + raw[::-1],
    })
    data.add_target(df, "label_k10", shift=9, label_map={1: 0, 2: 1, 3: 2})
    for s, labels in [(0, raw), (1, raw[::-1])]:
        got = df[df.stock_id == s].target.tolist()
        assert got[:3] == [lab - 1 for lab in labels[9:12]]
        assert got[3:] == [-1] * 9  # no target crosses into the next stock


def test_fit_affine_recovers_scale_map_and_keeps_constant_features():
    rng = np.random.default_rng(1)
    src = rng.standard_normal((500, 3))
    src[:, 2] = 0.7  # constant feature
    ref = src.copy()
    ref[:, 0] = 1.05 * src[:, 0] - 0.02
    ref[:, 1] = 0.98 * src[:, 1] + 0.03
    a, b, resid = data.fit_affine(src, ref)
    assert np.allclose(a, [1.05, 0.98, 1.0]) and np.allclose(b, [-0.02, 0.03, 0.0])
    assert resid < 1e-12


@needs_raw
def test_fold_sum_check():
    for k in range(1, 9):
        n_next = data.count_columns(data.train_file(RAW, k + 1))
        n_sum = data.count_columns(data.train_file(RAW, k)) + data.count_columns(data.test_file(RAW, k))
        assert n_next == n_sum, f"fold {k}"
