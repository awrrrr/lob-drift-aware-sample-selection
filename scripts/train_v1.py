"""Step 2: fit the V1 candidates on days 1-5, select by day-6 macro-F1.

Usage:  .venv/bin/python scripts/train_v1.py

Prints a table of every grid point (fit/predict time, validation macro-F1 and
per-class F1), picks the best by validation macro-F1, and prints the config block
to freeze into config.yaml models.v1. It does NOT edit config.yaml itself.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import models  # noqa: E402

CANDIDATES = ["logreg", "knn", "svm_rbf"]
CLASS_NAMES = ["Up", "Stat", "Down"]


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    df = pd.read_pickle(ROOT / cfg["data"]["processed_path"])

    Xtr, ytr, _ = models.split_xy(df, cfg["splits"]["train"], cfg)
    Xva, yva, _ = models.split_xy(df, cfg["splits"]["validation"], cfg)
    print(f"train {Xtr.shape} (days {cfg['splits']['train']})   "
          f"val {Xva.shape} (day {cfg['splits']['validation']})   "
          f"features {Xtr.shape[1]}")
    print(f"train class balance: {np.round(np.bincount(ytr) / len(ytr), 3)}  (Up, Stat, Down)\n")

    results = []
    print(f"{'model':9s} {'params':34s} {'fit s':>7} {'pred s':>7} {'macroF1':>8} "
          f"{'F1 Up':>7} {'F1 Stat':>7} {'F1 Down':>7}")
    for name in CANDIDATES:
        for params in models.iter_grid(name, cfg):
            clf = models.build_model(name, params, cfg)
            t = time.time(); clf.fit(Xtr, ytr); fit_s = time.time() - t
            t = time.time(); pred = clf.predict(Xva); pred_s = time.time() - t
            macro = f1_score(yva, pred, average="macro")
            per = f1_score(yva, pred, average=None, labels=models.CLASSES)
            results.append({"model": name, "params": params, "fit_s": fit_s,
                            "pred_s": pred_s, "macro_f1": macro,
                            "f1_up": per[0], "f1_stat": per[1], "f1_down": per[2]})
            print(f"{name:9s} {json.dumps(params):34s} {fit_s:7.2f} {pred_s:7.2f} "
                  f"{macro:8.4f} {per[0]:7.3f} {per[1]:7.3f} {per[2]:7.3f}")

    res = pd.DataFrame(results).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    best = res.iloc[0]
    print("\nBest per model (by validation macro-F1):")
    for name in CANDIDATES:
        r = res[res.model == name].iloc[0]
        print(f"  {name:9s} macroF1 {r.macro_f1:.4f}  fit {r.fit_s:6.2f}s  "
              f"pred {r.pred_s:6.2f}s  params {r.params}")

    print(f"\nWinner: {best.model}  macroF1 {best.macro_f1:.4f}  params {best.params}")
    print("\nFull classification report on validation (winner):")
    clf = models.build_model(best.model, best.params, cfg).fit(Xtr, ytr)
    print(classification_report(yva, clf.predict(Xva),
                                labels=models.CLASSES, target_names=CLASS_NAMES, digits=3))

    out = ROOT / "data" / "processed" / "v1_selection.csv"
    res.to_csv(out, index=False)
    print(f"Saved all results to {out.relative_to(ROOT)}")
    print("\nTo freeze V1, set in config.yaml:")
    print(f"  models:\n    v1: {{model: {best.model}, params: {json.dumps(best.params)}}}")


if __name__ == "__main__":
    main()
