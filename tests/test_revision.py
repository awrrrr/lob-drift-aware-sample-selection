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


# --- E_drift and the cluster ablations ------------------------------------------

from src import policies  # noqa: E402

REV_CFG = {**CFG, "revision": {**CFG["revision"], "audit_low": 0.1, "audit_high": 0.4}}
NEW = ["E_drift", "E_minus_cluster", "E_drift_minus_cluster"]


def pool_with_proba(level, n=400, seed=0):
    pool = vol_pool(level, n=n, seed=seed)
    logits = np.random.default_rng(seed + 1).standard_normal((n, 3))
    return pool, np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)


def call(name, pool, proba, B=50, seed=0):
    return policies.POLICIES[name](pool, proba, B, np.random.default_rng(seed), set(), REV_CFG)


@pytest.mark.parametrize("name", NEW)
@pytest.mark.parametrize("level", [0.5, 1.5])
def test_new_policies_return_exactly_B_unique_from_pool(name, level):
    pool, proba = pool_with_proba(level)
    ids = [s for s, _ in call(name, pool, proba)]
    assert len(ids) == 50 and len(set(ids)) == 50 and set(ids) <= set(pool.sample_id)


@pytest.mark.parametrize("level,share", [(0.5, 0.1), (1.5, 0.4)])
def test_e_drift_audit_count_follows_drift_rule(level, share):
    pool, proba = pool_with_proba(level)
    reasons = [r for _, r in call("E_drift", pool, proba)]
    assert sum(r == "audit" for r in reasons) == round(share * 50)


@pytest.mark.parametrize("name", NEW)
def test_new_policies_same_seed_same_picks(name):
    pool, proba = pool_with_proba(1.5)
    assert call(name, pool, proba, seed=3) == call(name, pool, proba, seed=3)


@pytest.mark.parametrize("name", ["E_drift", "E_drift_minus_cluster"])
def test_drift_policies_never_see_labels(name):
    pool, proba = pool_with_proba(1.5)
    pool["target"] = 0
    with pytest.raises(ValueError):
        call(name, pool, proba)


def test_minus_cluster_is_audit_plus_top_entropy():
    pool, proba = pool_with_proba(0.5)
    picks = call("E_minus_cluster", pool, proba)
    audit = [s for s, r in picks if r == "audit"]
    top = [s for s, r in picks if r == "top_entropy"]
    assert len(audit) == round(CFG["policies"]["audit_share"] * 50) and len(top) == 40
    # the top_entropy picks are exactly the 40 most uncertain non-audit samples
    H = pd.Series(policies.entropy(proba), index=pool.sample_id)
    assert set(top) == set(H.drop(audit).nlargest(40).index)


def test_e_unchanged_by_cluster_flag_default():
    # E itself must behave exactly as in step 5 (same picks with and without snap_vol).
    pool, proba = pool_with_proba(1.5)
    with_vol = call("E", pool, proba)
    without = call("E", pool.drop(columns="snap_vol"), proba)
    assert with_vol == without


# --- parallel-safe logging and scoring filters ----------------------------------

def test_parallel_replays_share_one_db_file(tmp_path):
    # Runs write short transactions (selections committed before labels are read,
    # no lock held while retraining), so concurrent runs on one file all finish.
    import threading
    from tests.test_harness import make_cfg, make_df
    p = tmp_path / "shared.db"
    harness.connect(p).close()
    errors = []

    def work(seed):
        try:
            db = harness.connect(p)
            harness.replay(make_df(), make_cfg(), "B", seed, db, tag="revision")
            db.close()
        except Exception as e:      # noqa: BLE001 -- surfaced by the assert below
            errors.append(repr(e))

    threads = [threading.Thread(target=work, args=(s,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors
    db = harness.connect(p)
    assert db.execute("SELECT COUNT(*) FROM runs WHERE end_time IS NOT NULL").fetchone()[0] == 4


def test_runs_to_score_filters_tag_unfinished_and_scored(tmp_path):
    from src import evaluate
    db = harness.connect(tmp_path / "r.db")
    cols = "(run_id, policy, seed, config_hash, git_commit, start_time, end_time, tag)"
    for rid, end, tag in [("main1", "t", "main"), ("rev1", "t", "revision"),
                          ("rev_unfinished", None, "revision"), ("grid1", "t", "revision_grid:x"),
                          ("rev_scored", "t", "revision")]:
        db.execute(f"INSERT INTO runs {cols} VALUES (?,?,?,?,?,?,?,?)",
                   (rid, "E", 0, "h", "g", "s", end, tag))
    db.execute("INSERT INTO metrics VALUES ('rev_scored','test','all','macro_f1',0.4)")
    assert evaluate.runs_to_score(db, tag="revision", skip_scored=True) == ["rev1"]
    assert "rev_unfinished" not in evaluate.runs_to_score(db)


def test_same_policy_same_seed_runs_get_distinct_ids_and_same_picks():
    # Parallel grid points share policy + seed: ids must differ, picks must not.
    from tests.test_harness import make_cfg, make_df
    import pandas as pd
    db = harness.connect(":memory:")
    a = harness.replay(make_df(), make_cfg(), "B", 0, db)
    b = harness.replay(make_df(), make_cfg(), "B", 0, db)
    assert a["run_id"] != b["run_id"]
    q = "SELECT sample_id FROM selections WHERE run_id=? ORDER BY window, sample_id"
    assert (pd.read_sql(q, db, params=[a["run_id"]]).sample_id.tolist()
            == pd.read_sql(q, db, params=[b["run_id"]]).sample_id.tolist())


def test_volatility_threshold_ignores_non_training_days():
    # Leakage rule 4: the threshold (and so d_k's scale) depends on days 1-5 only.
    rng = np.random.default_rng(0)
    frames = []
    for day in range(1, 11):
        mid = np.cumsum(rng.standard_normal(400))
        cols = {f"f{c}": 0.0 for c in range(144)}
        cols["f0"], cols["f2"] = mid, mid - 0.01
        cols.update(day=day, stock_id=0, t=np.arange(400), target=0)
        frames.append(pd.DataFrame(cols))
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "sample_id", np.arange(len(df)))
    before = data.fit_volatility_threshold(df, CFG)
    later = df.day >= 6
    df.loc[later, "f0"] *= 50.0          # make validation/stream/test days wildly volatile
    df.loc[later, "f2"] *= 50.0
    assert data.fit_volatility_threshold(df, CFG) == before
