"""Tests for step-2 model builders."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import models  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config.yaml"))


def test_feature_columns_drops_constant_features():
    cols = models.feature_columns(CFG)
    assert "f0" in cols and "f128" in cols
    for c in CFG["models"]["drop_features"]:
        assert c not in cols
    assert len(cols) == CFG["data"]["n_features"] - len(CFG["models"]["drop_features"])


def test_split_xy_excludes_missing_target_and_other_days():
    cols = [f"f{i}" for i in range(CFG["data"]["n_features"])]
    df = pd.DataFrame(np.zeros((6, len(cols))), columns=cols)
    df["day"] = [1, 1, 2, 6, 6, 6]
    df["target"] = [0, -1, 2, 1, -1, 0]
    df["sample_id"] = range(6)
    X, y, ids = models.split_xy(df, [6], CFG)
    assert list(y) == [1, 0] and list(ids) == [3, 5]  # day 6, target >= 0 only


def _toy(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + 0.5 * rng.standard_normal(n) > 0).astype(int)
    y[rng.random(n) < 0.2] = 2  # make it 3-class
    return X, y


def test_subsampled_svm_uses_cap_and_is_deterministic():
    X, y = _toy(3000)
    a = models.SubsampledSVM(subsample=500, calibration_cv=2, seed=0).fit(X, y)
    b = models.SubsampledSVM(subsample=500, calibration_cv=2, seed=0).fit(X, y)
    Xt, _ = _toy(200, seed=9)
    assert np.array_equal(a.predict(Xt), b.predict(Xt))  # same seed -> identical
    proba = a.predict_proba(Xt)
    assert proba.shape == (200, len(a.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_iter_grid_is_cartesian_product():
    grid = list(models.iter_grid("knn", CFG))
    assert len(grid) == 3 * 2  # n_neighbors x weights
    assert {"n_neighbors": 25, "weights": "uniform"} in grid
