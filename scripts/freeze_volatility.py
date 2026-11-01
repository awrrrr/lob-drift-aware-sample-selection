"""Freeze the volatility threshold on training days (step 5, leakage rule 4).

Computes window volatilities on days 1-5, reports the distribution and the
chosen-percentile threshold, and writes it into config.yaml volatility.threshold.
Also reports how days 6-10 windows split high/low at that threshold (for sanity;
no settings depend on days 7-10).

Usage:  .venv/bin/python scripts/freeze_volatility.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data  # noqa: E402

CFG_PATH = ROOT / "config.yaml"


def main():
    cfg = yaml.safe_load(open(CFG_PATH))
    df = pd.read_pickle(ROOT / cfg["data"]["processed_path"])

    train_wv = data.window_volatility(df, cfg["splits"]["train"], cfg).dropna()
    threshold = data.fit_volatility_threshold(df, cfg)
    pct = cfg["volatility"]["percentile"]
    print(f"Training-day window volatility (W={cfg['harness']['window']}, "
          f"vol_window={cfg['volatility']['vol_window']}): "
          f"n={len(train_wv)}  min={train_wv.min():.4f}  median={train_wv.median():.4f}  "
          f"max={train_wv.max():.4f}")
    print(f"Threshold = {pct}th percentile = {threshold:.6f}\n")

    print("High-volatility window share per split at this threshold:")
    for name, days in [("train 1-5", cfg["splits"]["train"]),
                       ("val 6", cfg["splits"]["validation"]),
                       ("stream 7-8", cfg["splits"]["stream"]),
                       ("test 9-10", cfg["splits"]["test"])]:
        wv = data.window_volatility(df, days, cfg).dropna()
        share = (wv > threshold).mean()
        print(f"  {name:12s}: {len(wv):3d} windows, {share * 100:5.1f}% high-vol")

    # Write threshold into config.yaml (textual edit to preserve comments).
    text = CFG_PATH.read_text()
    text = text.replace("  threshold: null     # frozen",
                        f"  threshold: {threshold:.6f}  # frozen")
    CFG_PATH.write_text(text)
    print(f"\nWrote volatility.threshold = {threshold:.6f} to config.yaml")


if __name__ == "__main__":
    main()
