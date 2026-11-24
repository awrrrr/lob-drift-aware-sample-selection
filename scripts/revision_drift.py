"""Revision R1: is the label-free drift score d_k a sensible drift indicator?

Post-hoc reviewer revision. Uses the stream days (7-8) and day 6 only -- never
the test days. For every replay window k it computes:

  d_k            label-free drift score (src/data.py: drift_score) = mean
                 snapshot volatility of the window's pool / frozen threshold
  V1 macro-F1    how well the frozen V1 does on that window's pool. This uses
                 the labels, AFTER the fact, as a diagnostic only; selection
                 never sees them.
  mean entropy   V1's mean predictive entropy on the pool (label-free)
  entropy ratio  mean entropy / V1's mean entropy on day 6 (a second, reported-
                 only indicator; not used by any policy)

and reports how d_k correlates with V1's error and with the entropy ratio.

Outputs: results/revision/drift_windows.csv, drift_trajectory.png,
drift_vs_v1.png, and a printed summary for LOG.md.

Usage:  .venv/bin/python scripts/revision_drift.py
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data, harness, models  # noqa: E402
from src.policies import entropy  # noqa: E402

OUT = ROOT / "results" / "revision"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1-2 (dataviz palette)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def per_window_table(df, cfg):
    feats = models.feature_columns(cfg)
    Xtr, ytr, _ = models.split_xy(df, cfg["splits"]["train"], cfg)
    Xva, _, _ = models.split_xy(df, cfg["splits"]["validation"], cfg)
    v1 = models.make_v1(cfg).fit(Xtr, ytr)
    h_val = entropy(v1.predict_proba(Xva)).mean()

    stream = harness.build_stream(df, cfg)
    day8_start = int(stream.loc[stream.day == cfg["splits"]["stream"][1], "stream_pos"].min())
    rows = []
    for k in range(int(stream.pool_window.max()) + 1):
        pool = stream[stream.pool_window == k]
        if pool.empty:
            continue
        d = data.drift_score(pool[["sample_id"] + feats + ["snap_vol"]], cfg)
        proba = v1.predict_proba(pool[feats].to_numpy())
        H = entropy(proba)
        y = pool.target.to_numpy()                     # diagnostic only, after the fact
        rows.append({
            "window": k, "n_pool": len(pool), "d_k": d,
            "v1_macro_f1": f1_score(y, proba.argmax(axis=1), average="macro"),
            "mean_entropy": H.mean(), "entropy_ratio": H.mean() / h_val,
            "day": int(pool.day.mode()[0]),
        })
    tab = pd.DataFrame(rows)
    return tab, h_val, day8_start // cfg["harness"]["window"]


def corr_line(a, b, name):
    r, p = pearsonr(a, b)
    rho, ps = spearmanr(a, b)
    return f"{name}: Pearson r = {r:+.3f} (p = {p:.3g}), Spearman rho = {rho:+.3f} (p = {ps:.3g})"


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    df = pd.read_pickle(ROOT / cfg["data"]["processed_path"])
    OUT.mkdir(parents=True, exist_ok=True)

    tab, h_val, day8_win = per_window_table(df, cfg)
    tab.to_csv(OUT / "drift_windows.csv", index=False)

    hi = tab.d_k > 1
    lines = [
        f"stream windows: {len(tab)}; windows with d_k > 1: {int(hi.sum())} "
        f"({hi.mean() * 100:.1f}%)  [day 7: {int(hi[tab.day == 7].sum())}, day 8: {int(hi[tab.day == 8].sum())}]",
        f"d_k: min {tab.d_k.min():.3f}, median {tab.d_k.median():.3f}, max {tab.d_k.max():.3f}",
        f"day-6 V1 mean entropy (reference): {h_val:.4f}",
        corr_line(tab.d_k, tab.v1_macro_f1, "d_k vs V1 macro-F1 (labels, diagnostic)"),
        corr_line(tab.d_k, tab.mean_entropy, "d_k vs V1 mean entropy"),
        corr_line(tab.d_k, tab.entropy_ratio, "d_k vs entropy ratio"),
        f"V1 macro-F1: d_k > 1 windows {tab.v1_macro_f1[hi].mean():.4f}  vs  "
        f"d_k <= 1 windows {tab.v1_macro_f1[~hi].mean():.4f}",
        f"entropy ratio: d_k > 1 windows {tab.entropy_ratio[hi].mean():.4f}  vs  "
        f"d_k <= 1 windows {tab.entropy_ratio[~hi].mean():.4f}",
    ]
    print("\n".join(lines))
    (OUT / "drift_summary.txt").write_text("\n".join(lines) + "\n")

    # --- trajectory: two panels sharing the window axis (no dual y-axis) ---
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    a1.plot(tab.window, tab.d_k, color=BLUE, linewidth=1.6)
    a1.axhline(1.0, color=INK2, linestyle="--", linewidth=1)
    a1.text(day8_win + 8, 1.03, "d = 1 (training upper quartile)", ha="left",
            va="bottom", fontsize=8, color=INK2)
    a1.set_ylabel("drift score d_k", fontsize=9, color=INK)
    a1.set_title("Label-free drift score across the update stream (days 7-8)",
                 fontsize=10, color=INK, loc="left")
    a2.plot(tab.window, tab.v1_macro_f1, color=ORANGE, linewidth=1.6)
    a2.set_ylabel("V1 macro-F1\n(diagnostic)", fontsize=9, color=INK)
    a2.set_xlabel("replay window", fontsize=9, color=INK)
    for ax in (a1, a2):
        style(ax)
        ax.axvline(day8_win - 0.5, color=INK2, linewidth=0.8)
    a1.text(day8_win, a1.get_ylim()[1], " day 8", va="top", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "drift_trajectory.png", dpi=150)

    # --- scatter: does high d_k coincide with V1 doing worse? ---
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(8, 3.4))
    b1.scatter(tab.d_k, tab.v1_macro_f1, s=18, color=BLUE, edgecolor="white", linewidth=0.6)
    b1.set_xlabel("drift score d_k", fontsize=9)
    b1.set_ylabel("V1 macro-F1 on window", fontsize=9)
    b1.set_title("d_k vs V1 error (labels used after the fact)", fontsize=9, loc="left")
    b2.scatter(tab.d_k, tab.entropy_ratio, s=18, color=BLUE, edgecolor="white", linewidth=0.6)
    b2.set_xlabel("drift score d_k", fontsize=9)
    b2.set_ylabel("entropy ratio (window / day 6)", fontsize=9)
    b2.set_title("d_k vs label-free entropy ratio", fontsize=9, loc="left")
    for ax in (b1, b2):
        style(ax)
        ax.axvline(1.0, color=INK2, linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(OUT / "drift_vs_v1.png", dpi=150)
    print("\nWrote results/revision/drift_windows.csv, drift_summary.txt, "
          "drift_trajectory.png, drift_vs_v1.png")


if __name__ == "__main__":
    main()
