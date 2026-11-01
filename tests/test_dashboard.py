"""Tests for the dashboard's pure data helpers (no Streamlit runtime needed)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import dashboard  # noqa: E402
from src import harness  # noqa: E402


def _seed_db(path):
    db = harness.connect(str(path))
    db.execute("INSERT INTO runs VALUES ('E_s0','E',0,'h','g','t0','t1')")
    db.execute("INSERT INTO runs VALUES ('B_s0','B',0,'h','g','t0','t1')")
    db.executemany("INSERT INTO metrics VALUES (?,?,?,?,?)", [
        ("E_s0", "test", "high_vol", "macro_f1", 0.42),
        ("E_s0", "test", "all", "macro_f1", 0.43),
        ("B_s0", "test", "high_vol", "macro_f1", 0.40),
        ("B_s0", "test", "all", "macro_f1", 0.43),
    ])
    db.commit()
    db.close()


def test_load_tables_and_summarize(tmp_path):
    p = tmp_path / "r.db"
    _seed_db(p)
    tables = dashboard.load_tables(p)
    assert set(tables) == {"runs", "selections", "updates", "metrics"}
    assert len(tables["runs"]) == 2

    summary = dashboard.summarize(tables)
    assert list(summary.index) == ["B", "E"]  # POLICY_ORDER, present only
    assert summary.loc["E", "high_vol_F1_mean"] == 0.42
    assert dashboard.ordered_policies(tables["runs"]) == ["B", "E"]


def test_summarize_empty_is_empty():
    assert dashboard.summarize({"runs": pd.DataFrame(), "metrics": pd.DataFrame()}).empty
