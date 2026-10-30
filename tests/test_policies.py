"""Tests for the step-4 selection policies.

These exercise the policies directly on a synthetic pool, independent of the
harness, so they are fast and precise about each policy's contract.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import policies  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
ALL = ["B", "C", "D", "E", "E_minus_audit"]  # A selects nothing, tested elsewhere


def make_pool(n=400, n_features=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    pool = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    pool.insert(0, "sample_id", np.arange(1000, 1000 + n))
    logits = rng.standard_normal((n, 3))
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return pool, proba


def call(name, pool, proba, B=50, seed=0):
    rng = np.random.default_rng(seed)
    return policies.POLICIES[name](pool, proba, B, rng, set(), CFG)


@pytest.mark.parametrize("name", ALL)
def test_returns_exactly_B_unique_from_pool(name):
    pool, proba = make_pool()
    picks = call(name, pool, proba, B=50)
    ids = [s for s, _ in picks]
    assert len(ids) == 50
    assert len(set(ids)) == 50
    assert set(ids) <= set(pool.sample_id)


@pytest.mark.parametrize("name", ALL)
def test_small_pool_returns_all(name):
    pool, proba = make_pool(n=30)
    picks = call(name, pool, proba, B=50)
    assert {s for s, _ in picks} == set(pool.sample_id)


def test_uncertainty_picks_highest_entropy():
    pool, proba = make_pool(n=200)
    picks = call("C", pool, proba, B=20)
    H = policies.entropy(proba)
    want = set(pool.sample_id.to_numpy()[np.argsort(-H)[:20]])
    assert {s for s, _ in picks} == want


def test_density_differs_from_pure_uncertainty():
    # With structured features, density weighting should reorder some picks.
    pool, proba = make_pool(n=300)
    c = {s for s, _ in call("C", pool, proba, B=30)}
    d = {s for s, _ in call("D", pool, proba, B=30)}
    assert c != d


def test_proposed_reserves_audit_share_and_ablation_drops_it():
    pool, proba = make_pool(n=400)
    e = call("E", pool, proba, B=50)
    reasons = [r for _, r in e]
    assert sum(r == "audit" for r in reasons) == round(CFG["policies"]["audit_share"] * 50)
    assert any(r.startswith("cluster") for r in reasons)
    ablation = call("E_minus_audit", pool, proba, B=50)
    assert all(r != "audit" for _, r in ablation)


@pytest.mark.parametrize("name", ALL)
def test_determinism_and_seed_sensitivity(name):
    pool, proba = make_pool()
    a = [s for s, _ in call(name, pool, proba, seed=0)]
    b = [s for s, _ in call(name, pool, proba, seed=0)]
    assert a == b  # same seed -> identical
    c = [s for s, _ in call(name, pool, proba, seed=1)]
    if name in ("B", "E", "E_minus_audit"):
        assert a != c  # stochastic (random draw, or k-means init) -> varies with seed
    else:
        assert a == c  # C and D are deterministic regardless of seed
