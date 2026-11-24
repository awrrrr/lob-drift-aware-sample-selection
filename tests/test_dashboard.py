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
    cols = "(run_id, policy, seed, config_hash, git_commit, start_time, end_time)"
    db.execute(f"INSERT INTO runs {cols} VALUES ('E_s0','E',0,'h','g','t0','t1')")
    db.execute(f"INSERT INTO runs {cols} VALUES ('B_s0','B',0,'h','g','t0','t1')")
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


def test_fixed_model_vs_chance_and_gain_over_fixed(tmp_path):
    p = tmp_path / "r.db"
    db = harness.connect(str(p))
    cols = "(run_id, policy, seed, config_hash, git_commit, start_time, end_time)"
    for rid, pol, seed in [("A_s0", "A", 0), ("B_s0", "B", 0), ("B_s1", "B", 1)]:
        db.execute(f"INSERT INTO runs {cols} VALUES (?,?,?,'h','g','t0','t1')", (rid, pol, seed))
    db.executemany("INSERT INTO metrics VALUES (?,?,?,?,?)", [
        ("A_s0", "test", "all", "macro_f1", 0.43),
        ("A_s0", "test", "all", "precision_Up", 0.5), ("A_s0", "test", "all", "recall_Up", 0.5),
        ("A_s0", "test", "high_vol", "macro_f1", 0.410),
        ("B_s0", "test", "high_vol", "macro_f1", 0.413),
        ("B_s1", "test", "high_vol", "macro_f1", 0.412),
    ])
    db.commit()
    db.close()
    t = dashboard.load_tables(p)

    d = dashboard.fixed_model_vs_chance(t).set_index("class")
    assert abs(d.loc["Up", "F1"] - 0.5) < 1e-12          # 2PR/(P+R) with P = R = 0.5
    assert abs(d.loc["macro", "F1_minus_chance_pp"] - (0.43 - 1 / 3) * 100) < 1e-9

    g = dashboard.gain_over_fixed(t, "high_vol").set_index("seed")
    assert set(g.policy) == {"B"}                          # A is the reference, left out
    assert abs(g.loc[0, "gain_pp"] - 0.3) < 1e-9
    assert abs(g.loc[1, "gain_pp"] - 0.2) < 1e-9
    assert dashboard.gain_over_fixed(t, "low_vol").empty   # no A metric -> nothing to compare


def test_dashboard_hides_grid_and_check_runs(tmp_path):
    p = tmp_path / "r.db"
    db = harness.connect(str(p))
    cols = "(run_id, policy, seed, config_hash, git_commit, start_time, end_time, tag)"
    for rid, pol, tag in [("E_main", "E", "main"), ("E_check", "E", "revision_check"),
                          ("Ed_grid", "E_drift", "revision_grid:a_low=0.1,a_high=0.4"),
                          ("Ed_rev", "E_drift", "revision")]:
        db.execute(f"INSERT INTO runs {cols} VALUES (?,?,0,'h','g','t0','t1',?)", (rid, pol, tag))
        db.execute("INSERT INTO updates VALUES (?,0,2,1,1,0.4,0.4)", (rid,))
    db.commit()
    db.close()
    t = dashboard.load_tables(p)
    assert set(t["runs"].run_id) == {"E_main", "Ed_rev"}
    assert set(t["updates"].run_id) == {"E_main", "Ed_rev"}
    assert dashboard.ordered_policies(t["runs"]) == ["E", "E_drift"]
