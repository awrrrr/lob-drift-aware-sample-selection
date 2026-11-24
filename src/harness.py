"""Replay harness for step 3: replay days 7-8, let a policy add B samples per
window, retrain V1, gate the update on validation macro-F1, and log everything
to SQLite.

Leakage rules enforced here:
  1. A sample joins window (stream_pos + horizon)//W, so it is selectable only
     after its label has formed. Samples with < horizon rows left in their
     stock-day are never selectable.
  2. The policy is handed features and the current model's probabilities only;
     labels are revealed after selection and logged afterwards.
  3. Nothing here touches days 9-10. Test scoring lives in src/evaluate.py.
"""

import hashlib
import sqlite3
import subprocess
import time
import uuid

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src import data, models
from src.policies import POLICIES

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, policy TEXT, seed INTEGER,
    config_hash TEXT, git_commit TEXT, start_time TEXT, end_time TEXT);
CREATE TABLE IF NOT EXISTS selections (
    run_id TEXT, window INTEGER, sample_id INTEGER, reason TEXT,
    p_up REAL, p_stat REAL, p_down REAL, model_version INTEGER);
CREATE TABLE IF NOT EXISTS updates (
    run_id TEXT, window INTEGER, model_version INTEGER, parent_version INTEGER,
    accepted INTEGER, val_f1_before REAL, val_f1_after REAL);
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT, split TEXT, condition TEXT, metric TEXT, value REAL);
CREATE TABLE IF NOT EXISTS timings (
    run_id TEXT, window INTEGER, select_ms REAL, retrain_ms REAL);
"""


def connect(path):
    """Open runs.db, creating tables if needed.

    Revision migration: older databases have no `tag` column on `runs`. Add it
    with default 'main', so every original step-5 run is tagged 'main' and new
    revision runs can be told apart without touching the old rows.
    """
    # timeout: several replays may write to the same runs.db in parallel; each
    # commits per window, so a writer waits briefly for another's commit.
    db = sqlite3.connect(str(path), timeout=600)
    db.executescript(SCHEMA)
    cols = [r[1] for r in db.execute("PRAGMA table_info(runs)")]
    if "tag" not in cols:
        db.execute("ALTER TABLE runs ADD COLUMN tag TEXT DEFAULT 'main'")
        db.commit()
    return db


def config_hash(cfg_path):
    return hashlib.sha256(open(cfg_path, "rb").read()).hexdigest()[:16]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_stream(df, cfg):
    """Days 7-8 in replay order, with per-sample window assignment.

    Adds: stream_pos (0..N-1), rows_after (rows left in the stock-day),
    selectable (target present and >= horizon rows left), pool_window
    (the window a selectable sample belongs to, else -1), and snap_vol (the
    row's rolling mid-price volatility from current and past prices only;
    label-free, used by the drift score in src/data.py).
    """
    W = cfg["harness"]["window"]
    horizon = cfg["data"]["horizon_snapshots"]
    stream = (df[df.day.isin(cfg["splits"]["stream"])]
              .sort_values(["day", "stock_id", "t"]).reset_index(drop=True).copy())
    stream["snap_vol"] = data.snapshot_volatility(stream, cfg["volatility"]["vol_window"])
    stream["stream_pos"] = np.arange(len(stream))
    stream["rows_after"] = stream.groupby(["day", "stock_id"]).cumcount(ascending=False)
    stream["selectable"] = (stream.target >= 0) & (stream.rows_after >= horizon)
    avail_pos = stream.stream_pos + horizon
    stream["pool_window"] = np.where(stream.selectable, avail_pos // W, -1)
    return stream


def macro_f1(model, X, y):
    return f1_score(y, model.predict(X), average="macro")


def replay(df, cfg, policy_name, seed, db, max_windows=None, verbose=False, tag="main"):
    """Run one policy/seed over the stream, logging to the open db connection.

    `tag` labels the run in `runs` ('main' = original step 5, 'revision' = the
    post-hoc reviewer revision). Per window, the time spent selecting (model
    probabilities + policy) and retraining (fit + validation gate) is logged to
    `timings`; timing does not change what is selected.
    """
    feats = models.feature_columns(cfg)
    Xtr, ytr, _ = models.split_xy(df, cfg["splits"]["train"], cfg)
    Xva, yva, _ = models.split_xy(df, cfg["splits"]["validation"], cfg)
    stream = build_stream(df, cfg)
    B = cfg["harness"]["budget"]
    tol = cfg["harness"]["gate_tolerance"]
    policy = POLICIES[policy_name]
    rng = np.random.default_rng(seed)

    # The rng draw stays (it advances the seeded stream exactly as in step 5, so
    # selections are unchanged); the uuid suffix keeps run_ids unique when runs
    # with the same policy and seed start in the same second in parallel.
    run_id = (f"{policy_name}_seed{seed}_{time.strftime('%Y%m%d_%H%M%S')}_"
              f"{rng.integers(1e6):06d}_{uuid.uuid4().hex[:6]}")
    db.execute("INSERT INTO runs (run_id, policy, seed, config_hash, git_commit, "
               "start_time, end_time, tag) VALUES (?,?,?,?,?,?,?,?)",
               (run_id, policy_name, seed, config_hash("config.yaml"), git_commit(),
                time.strftime("%Y-%m-%d %H:%M:%S"), None, tag))
    db.commit()

    # Current (accepted) model starts as a freshly fit V1.
    current = models.make_v1(cfg).fit(Xtr, ytr)
    current_val = macro_f1(current, Xva, yva)
    version = 1
    sel_X, sel_y, selected_ids = [], [], set()

    n_windows = int(stream.pool_window.max()) + 1 if len(stream) else 0
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    for k in range(n_windows):
        pool = stream[stream.pool_window == k]
        t0 = time.perf_counter()
        if len(pool):
            proba = current.predict_proba(pool[feats].values)
            # Policies get ids, features and snap_vol (label-free) -- never the target.
            picks = policy(pool[["sample_id"] + feats + ["snap_vol"]], proba, B, rng,
                           selected_ids, cfg)
        else:
            picks = []
        select_ms = (time.perf_counter() - t0) * 1000

        if picks:
            pool_idx = {int(s): i for i, s in enumerate(pool.sample_id.to_numpy())}
            for sid, reason in picks:
                row = stream.loc[stream.sample_id == sid].iloc[0]
                p = proba[pool_idx[sid]]
                db.execute("INSERT INTO selections VALUES (?,?,?,?,?,?,?,?)",
                           (run_id, k, sid, reason, float(p[0]), float(p[1]),
                            float(p[2]), version))
                sel_X.append(row[feats].to_numpy(dtype=float))
                sel_y.append(int(row.target))
                selected_ids.add(sid)

            # Candidate retrains on all training data + every sample selected so far.
            t0 = time.perf_counter()
            cand = models.make_v1(cfg).fit(
                np.vstack([Xtr, np.array(sel_X)]), np.concatenate([ytr, sel_y]))
            cand_val = macro_f1(cand, Xva, yva)
            retrain_ms = (time.perf_counter() - t0) * 1000
            db.execute("INSERT INTO timings VALUES (?,?,?,?)",
                       (run_id, k, select_ms, retrain_ms))
            accepted = cand_val >= current_val - tol
            db.execute("INSERT INTO updates VALUES (?,?,?,?,?,?,?)",
                       (run_id, k, version + 1, version, int(accepted),
                        current_val, cand_val))
            if accepted:
                current, current_val, version = cand, cand_val, version + 1
            db.commit()   # short transactions, so parallel runs can share runs.db
            if verbose:
                print(f"  window {k:4d}: pool {len(pool):4d} picked {len(picks):3d} "
                      f"val {current_val:.4f} cand {cand_val:.4f} "
                      f"{'accept' if accepted else 'REJECT'} v{version}")

    # Final-model metrics on validation (test scoring is step 5, in evaluate.py).
    final_pred = current.predict(Xva)
    db.execute("INSERT INTO metrics VALUES (?,?,?,?,?)",
               (run_id, "validation", "all", "macro_f1",
                float(f1_score(yva, final_pred, average="macro"))))
    db.execute("INSERT INTO metrics VALUES (?,?,?,?,?)",
               (run_id, "validation", "all", "accuracy", float(accuracy_score(yva, final_pred))))
    db.execute("UPDATE runs SET end_time=? WHERE run_id=?",
               (time.strftime("%Y-%m-%d %H:%M:%S"), run_id))
    db.commit()
    return {"run_id": run_id, "n_windows": n_windows, "n_selected": len(selected_ids),
            "final_version": version, "final_val_f1": current_val}
