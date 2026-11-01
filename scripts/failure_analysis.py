"""Step 5 failure analysis: why does smart selection not beat random here?

Read-only diagnosis (CLAUDE.md: examine failure cases when E does not win).
It changes no settings and no policy. It answers three questions:

  1. Do the policies even select different samples? (overlap + class mix)
  2. Does the base model saturate, so added data barely helps regardless of
     which samples? (a random-data learning curve on the test set)
  3. Is the high-volatility regime simply harder / smaller? (regime breakdown)

Outputs: results/learning_curve.png, results/val_trajectory.png, and printed
tables. Uses seed 0 for the stochastic policies' selection sets.

Usage:  .venv/bin/python scripts/failure_analysis.py
"""

import sqlite3
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data, models  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

CLASS_NAMES = {0: "Up", 1: "Stat", 2: "Down"}


def selected_sets(db, policies, seed=0):
    out = {}
    for p in policies:
        q = ("SELECT sample_id FROM selections s JOIN runs r USING(run_id) "
             "WHERE r.policy=? AND r.seed=?")
        ids = pd.read_sql(q, db, params=[p, seed]).sample_id
        if len(ids):
            out[p] = set(ids)
    return out


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    df = pd.read_pickle(ROOT / cfg["data"]["processed_path"])
    db = sqlite3.connect(ROOT / cfg["harness"]["runs_db"])
    feats = models.feature_columns(cfg)
    threshold = cfg["volatility"]["threshold"]

    # ---- 1. Selection overlap and class mix across policies ----
    print("=" * 64)
    print("1. Do policies select different samples? (seed 0)")
    print("=" * 64)
    sets = selected_sets(db, ["B", "C", "D", "E", "E_minus_audit"])
    pols = list(sets)
    jac = pd.DataFrame(
        [[round(jaccard(sets[a], sets[b]), 3) for b in pols] for a in pols],
        index=pols, columns=pols)
    print("Jaccard overlap of selected sample sets:")
    print(jac.to_string())

    tgt = df.set_index("sample_id").target
    print("\nClass mix of selected samples (target distribution):")
    for p in pols:
        c = tgt.loc[list(sets[p])].value_counts(normalize=True).sort_index()
        print(f"  {p:14s} Up {c.get(0,0):.2f}  Stat {c.get(1,0):.2f}  Down {c.get(2,0):.2f}  (n={len(sets[p])})")
    # Stream pool class mix for reference.
    pool = df[df.day.isin(cfg["splits"]["stream"]) & (df.target >= 0)].target
    cp = pool.value_counts(normalize=True).sort_index()
    print(f"  {'stream pool':14s} Up {cp.get(0,0):.2f}  Stat {cp.get(1,0):.2f}  Down {cp.get(2,0):.2f}  (n={len(pool)})")

    # ---- 2. Saturation: random-data learning curve on the test set ----
    print("\n" + "=" * 64)
    print("2. Does the model saturate? Test macro-F1 vs # random stream samples added")
    print("=" * 64)
    Xtr, ytr, _ = models.split_xy(df, cfg["splits"]["train"], cfg)
    stream = df[df.day.isin(cfg["splits"]["stream"]) & (df.target >= 0)]
    Xs, ys = stream[feats].to_numpy(), stream.target.to_numpy()
    test = df[df.day.isin(cfg["splits"]["test"]) & (df.target >= 0)].copy()
    vol = data.classify_windows(df, cfg["splits"]["test"], cfg, threshold)
    test = test.merge(vol[["sample_id", "high_vol"]], on="sample_id", how="left")
    Xte, yte, hv = test[feats].to_numpy(), test.target.to_numpy(), test.high_vol.to_numpy()

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Xs))
    sizes = [0, 250, 1000, 2500, 4650, 10000, 20000, 40000, len(Xs)]
    curve = []
    print(f"  {'added':>7} {'overall F1':>11} {'high-vol F1':>12} {'low-vol F1':>11}")
    for k in sizes:
        idx = perm[:k]
        Xk = np.vstack([Xtr, Xs[idx]]) if k else Xtr
        yk = np.concatenate([ytr, ys[idx]]) if k else ytr
        m = models.make_v1(cfg).fit(Xk, yk)
        pred = m.predict(Xte)
        f_all = f1_score(yte, pred, average="macro")
        f_hi = f1_score(yte[hv], pred[hv], average="macro")
        f_lo = f1_score(yte[~hv], pred[~hv], average="macro")
        curve.append((k, f_all, f_hi, f_lo))
        print(f"  {k:>7} {f_all:>11.4f} {f_hi:>12.4f} {f_lo:>11.4f}")

    cur = np.array(curve)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cur[:, 0], cur[:, 1], "o-", label="overall")
    ax.plot(cur[:, 0], cur[:, 2], "s-", label="high-vol")
    ax.plot(cur[:, 0], cur[:, 3], "^-", label="low-vol")
    ax.axvline(4650, ls="--", color="grey", label="budget used (93x50)")
    ax.set_xlabel("random stream samples added to the 178k training set")
    ax.set_ylabel("test macro-F1")
    ax.set_title("Saturation check: more labelled data barely moves the model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "learning_curve.png", dpi=130)

    # ---- 3. Regime difficulty ----
    print("\n" + "=" * 64)
    print("3. Is high-volatility just harder / smaller?")
    print("=" * 64)
    print(f"  high-vol test snapshots: {int(hv.sum())}  ({hv.mean()*100:.1f}% of {len(hv)})")
    for name, mask in [("high-vol", hv), ("low-vol", ~hv)]:
        c = pd.Series(yte[mask]).value_counts(normalize=True).sort_index()
        print(f"  {name} class balance: Up {c.get(0,0):.2f}  Stat {c.get(1,0):.2f}  Down {c.get(2,0):.2f}")

    # ---- val-F1 trajectory during replay (per policy, mean over seeds) ----
    ups = pd.read_sql("SELECT r.policy, u.window, u.val_f1_after FROM updates u "
                      "JOIN runs r USING(run_id)", db)
    fig, ax = plt.subplots(figsize=(7, 4))
    for p in ["B", "C", "D", "E", "E_minus_audit"]:
        sub = ups[ups.policy == p].groupby("window").val_f1_after.mean()
        if len(sub):
            ax.plot(sub.index, sub.values, label=p, alpha=0.8)
    ax.set_xlabel("replay window"); ax.set_ylabel("candidate validation macro-F1")
    ax.set_title("Validation trajectory during replay (mean over seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "val_trajectory.png", dpi=130)
    print("\nWrote results/learning_curve.png and results/val_trajectory.png")


if __name__ == "__main__":
    main()
