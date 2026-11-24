"""Step 5 test scoring, from runs.db (leakage rule 3: test scored once, here only).

The replay harness never touches days 9-10. This module reconstructs each run's
final model exactly from what runs.db records -- the training data plus the
samples selected up to the last accepted update -- scores it once on the test
days, splits results by the frozen volatility condition, and writes test metrics
back to runs.db. Nothing here feeds back into any setting.
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support)

from src import data, models

CLASS_NAMES = ["Up", "Stat", "Down"]


def final_model_training_ids(db, run_id):
    """sample_ids in the final accepted model: those selected up to the last
    accepted window. If no update was ever accepted, the model is plain V1."""
    ups = pd.read_sql("SELECT window, accepted FROM updates WHERE run_id=?", db,
                      params=[run_id])
    sel = pd.read_sql("SELECT window, sample_id FROM selections WHERE run_id=?", db,
                      params=[run_id])
    accepted = ups[ups.accepted == 1]
    if accepted.empty:
        return np.array([], dtype=int)
    last_acc = int(accepted.window.max())
    return sel[sel.window <= last_acc].sample_id.to_numpy()


def reconstruct_final_model(df, cfg, db, run_id):
    """Fit V1 on training data + the run's final selected samples."""
    Xtr, ytr, _ = models.split_xy(df, cfg["splits"]["train"], cfg)
    ids = final_model_training_ids(db, run_id)
    if len(ids):
        feats = models.feature_columns(cfg)
        extra = df[df.sample_id.isin(ids)]
        Xtr = np.vstack([Xtr, extra[feats].to_numpy()])
        ytr = np.concatenate([ytr, extra.target.to_numpy()])
    return models.make_v1(cfg).fit(Xtr, ytr)


def latency_ms(model, X, n=2000, seed=0):
    """Per-snapshot predict_proba latency (ms), p50/p99. Includes only the model
    call on a pre-extracted feature vector; excludes feature building and I/O."""
    rng = np.random.default_rng(seed)
    rows = X[rng.choice(len(X), size=min(n, len(X)), replace=False)]
    times = np.empty(len(rows))
    for i, r in enumerate(rows):
        t = time.perf_counter()
        model.predict_proba(r.reshape(1, -1))
        times[i] = (time.perf_counter() - t) * 1000
    return float(np.percentile(times, 50)), float(np.percentile(times, 99))


def evaluate_run(df, cfg, db, run_id, threshold):
    """Score one run on the test days, split by volatility, and log metrics."""
    feats = models.feature_columns(cfg)
    model = reconstruct_final_model(df, cfg, db, run_id)

    test = df[df.day.isin(cfg["splits"]["test"]) & (df.target >= 0)].copy()
    vol = data.classify_windows(df, cfg["splits"]["test"], cfg, threshold)
    test = test.merge(vol[["sample_id", "high_vol"]], on="sample_id", how="left")
    X, y = test[feats].to_numpy(), test.target.to_numpy()
    pred = model.predict(X)

    rows = []  # (split, condition, metric, value)
    def add(cond, metric, val):
        rows.append(("test", cond, metric, float(val)))

    for cond, mask in [("all", np.ones(len(test), bool)),
                       ("high_vol", test.high_vol.to_numpy()),
                       ("low_vol", ~test.high_vol.to_numpy())]:
        if mask.sum() == 0:
            continue
        add(cond, "macro_f1", f1_score(y[mask], pred[mask], average="macro"))
        add(cond, "accuracy", accuracy_score(y[mask], pred[mask]))
        add(cond, "n", int(mask.sum()))
    # Per-class precision/recall on all test snapshots.
    p, r, _, _ = precision_recall_fscore_support(
        y, pred, labels=models.CLASSES, zero_division=0)
    for i, name in enumerate(CLASS_NAMES):
        add("all", f"precision_{name}", p[i])
        add("all", f"recall_{name}", r[i])

    # Update counts (about the replay) and latency.
    ups = pd.read_sql("SELECT accepted FROM updates WHERE run_id=?", db, params=[run_id])
    rows.append(("updates", "all", "n_accepted", float((ups.accepted == 1).sum())))
    rows.append(("updates", "all", "n_rejected", float((ups.accepted == 0).sum())))
    p50, p99 = latency_ms(model, X)
    rows.append(("latency", "all", "p50_ms", p50))
    rows.append(("latency", "all", "p99_ms", p99))

    db.execute("DELETE FROM metrics WHERE run_id=? AND split IN ('test','updates','latency')",
               [run_id])
    db.executemany("INSERT INTO metrics VALUES (?,?,?,?,?)",
                   [(run_id, s, c, m, v) for s, c, m, v in rows])
    db.commit()
    return {r[2]: r[3] for r in rows if r[1] in ("all", "high_vol") and r[0] == "test"}


def runs_to_score(db, tag=None, skip_scored=False):
    """run_ids to score: finished runs only (end_time set), optionally only one
    tag, optionally only runs with no test metrics yet (score each run once)."""
    q = "SELECT run_id FROM runs WHERE end_time IS NOT NULL"
    params = []
    if tag is not None:
        q += " AND tag = ?"
        params.append(tag)
    if skip_scored:
        q += " AND run_id NOT IN (SELECT run_id FROM metrics WHERE split = 'test')"
    return pd.read_sql(q + " ORDER BY run_id", db, params=params).run_id.tolist()


def evaluate_all(df, cfg, db, tag=None, skip_scored=False):
    """Score finished runs in runs.db on the test days (see runs_to_score)."""
    threshold = cfg["volatility"]["threshold"]
    if threshold is None:
        raise ValueError("volatility.threshold is not frozen; run freeze_volatility.py")
    run_ids = runs_to_score(db, tag=tag, skip_scored=skip_scored)
    for rid in run_ids:
        out = evaluate_run(df, cfg, db, rid, threshold)
        print(f"  {rid}: test macroF1 {out.get('macro_f1', float('nan')):.4f}")
    return run_ids
