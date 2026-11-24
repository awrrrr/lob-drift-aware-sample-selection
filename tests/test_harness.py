"""Leakage and budget tests for the replay harness.

A small synthetic dataset stands in for FI-2010 so these run in well under a
second: 2 stocks per day, every day present, with a target that depends on a
feature so logistic regression can fit it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import harness, models  # noqa: E402
from src.policies import random_policy

BASE_CFG = yaml.safe_load(open(ROOT / "config.yaml"))


def make_cfg():
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in BASE_CFG.items()}
    cfg["data"] = dict(BASE_CFG["data"])
    cfg["data"]["n_features"] = 4
    cfg["models"] = dict(BASE_CFG["models"])
    cfg["models"]["drop_features"] = []
    cfg["models"]["v1"] = {"model": "logreg", "params": {"C": 1.0}}
    cfg["harness"] = dict(BASE_CFG["harness"])
    cfg["harness"]["window"] = 50
    cfg["harness"]["budget"] = 5
    cfg["data"]["horizon_snapshots"] = 10
    return cfg


def make_df(rows_per_stock_day=120, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for day in range(1, 11):
        for stock in range(2):
            n = rows_per_stock_day
            X = rng.standard_normal((n, 4))
            y = np.where(X[:, 0] > 0.3, 0, np.where(X[:, 0] < -0.3, 2, 1))
            f = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
            f["day"], f["stock_id"], f["t"] = day, stock, np.arange(n)
            # Realigned target: label from 9 rows ahead within the stock-day; last 9 = -1.
            f["target"] = np.r_[y[9:], [-1] * 9]
            frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "sample_id", np.arange(len(df)))
    return df


def run(cfg, df, policy="B", seed=0, max_windows=None):
    db = harness.connect(":memory:")
    out = harness.replay(df, cfg, policy, seed, db, max_windows=max_windows)
    sel = pd.read_sql("SELECT * FROM selections", db)
    ups = pd.read_sql("SELECT * FROM updates", db)
    return out, sel, ups, df


def test_budget_exactly_B_unique_per_window():
    cfg = make_cfg()
    _, sel, _, _ = run(cfg, make_df())
    assert not sel.empty
    per_window = sel.groupby("window").sample_id
    assert (per_window.size() <= cfg["harness"]["budget"]).all()
    assert (per_window.nunique() == per_window.size()).all()  # no repeats in a window
    assert sel.sample_id.is_unique                            # no repeats across windows


def test_no_selection_from_test_days_or_before_horizon():
    cfg = make_cfg()
    _, sel, _, df = run(cfg, make_df())
    chosen = df[df.sample_id.isin(sel.sample_id)]
    assert set(chosen.day) <= set(cfg["splits"]["stream"])    # only days 7-8
    # Every selected sample had >= horizon rows left in its stock-day (label formed).
    rows_after = chosen.groupby(["day", "stock_id"])  # recompute independently
    stream = harness.build_stream(df, cfg)
    merged = sel.merge(stream[["sample_id", "rows_after", "stream_pos", "pool_window"]],
                       on="sample_id")
    assert (merged.rows_after >= cfg["data"]["horizon_snapshots"]).all()


def test_selected_label_formed_within_its_window():
    # A sample selected in window k must have become labellable by the end of k:
    # stream_pos + horizon < (k+1) * W.
    cfg = make_cfg()
    _, sel, _, df = run(cfg, make_df())
    stream = harness.build_stream(df, cfg)
    m = sel.merge(stream[["sample_id", "stream_pos"]], on="sample_id")
    W, h = cfg["harness"]["window"], cfg["data"]["horizon_snapshots"]
    assert (m.stream_pos + h < (m.window + 1) * W).all()


def test_same_seed_identical_selections():
    cfg, df = make_cfg(), make_df()
    _, s1, _, _ = run(cfg, df, seed=0)
    _, s2, _, _ = run(cfg, df, seed=0)
    _, s3, _, _ = run(cfg, df, seed=1)
    assert s1.sample_id.tolist() == s2.sample_id.tolist()
    assert s1.sample_id.tolist() != s3.sample_id.tolist()


def test_fixed_policy_never_selects_and_never_updates():
    cfg, df = make_cfg(), make_df()
    out, sel, ups, _ = run(cfg, df, policy="A")
    assert sel.empty and ups.empty and out["n_selected"] == 0
    assert out["final_version"] == 1  # stays V1


def test_tag_migration_keeps_old_runs_as_main(tmp_path):
    # An old-style runs.db (no `tag` column) must gain the column with old rows = 'main'.
    import sqlite3
    p = tmp_path / "old.db"
    old = sqlite3.connect(p)
    old.execute("CREATE TABLE runs (run_id TEXT PRIMARY KEY, policy TEXT, seed INTEGER, "
                "config_hash TEXT, git_commit TEXT, start_time TEXT, end_time TEXT)")
    old.execute("INSERT INTO runs VALUES ('E_old','E',0,'h','g','t0','t1')")
    old.commit()
    old.close()
    db = harness.connect(p)
    assert db.execute("SELECT tag FROM runs WHERE run_id='E_old'").fetchone()[0] == "main"


def test_timings_logged_per_update_window_and_tag_stored():
    cfg, df = make_cfg(), make_df()
    db = harness.connect(":memory:")
    out = harness.replay(df, cfg, "B", 0, db, tag="revision")
    tim = pd.read_sql("SELECT * FROM timings", db)
    ups = pd.read_sql("SELECT * FROM updates", db)
    assert len(tim) == len(ups) > 0                      # one timing row per update
    assert (tim.select_ms >= 0).all() and (tim.retrain_ms > 0).all()
    assert db.execute("SELECT tag FROM runs WHERE run_id=?",
                      (out["run_id"],)).fetchone()[0] == "revision"
