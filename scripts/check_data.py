"""Step 1: load FI-2010, recover days and stocks, print sanity checks.

Usage:  .venv/bin/python scripts/check_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data  # noqa: E402


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    raw_dir = ROOT / cfg["data"]["raw_dir"]

    # 1. Fold-sum check: Train_CF_{k+1} = Train_CF_k + Test_CF_k (row counts).
    section("1. Fold-sum check (row counts)")
    train_n = {k: data.count_columns(data.train_file(raw_dir, k)) for k in range(1, 10)}
    test_n = {k: data.count_columns(data.test_file(raw_dir, k)) for k in range(1, 10)}
    all_ok = True
    for k in range(1, 9):
        ok = train_n[k + 1] == train_n[k] + test_n[k]
        all_ok &= ok
        print(f"  Train_CF_{k + 1} = {train_n[k + 1]:>7}   Train_CF_{k} + Test_CF_{k} = "
              f"{train_n[k]:>7} + {test_n[k]:>6} = {train_n[k] + test_n[k]:>7}   {'OK' if ok else 'FAIL'}")
    print(f"  Fold-sum check: {'PASSED' if all_ok else 'FAILED'}")

    # 2. Load the ten days.
    section("2. Load days 1-10 and detect stock boundaries")
    df, info = data.load_all_days(raw_dir)
    for day in range(1, 11):
        print(f"  day {day:>2}: {info[day]['n']:>6} rows")
    print(f"  total : {len(df):>6} rows, {df.shape[1]} columns")

    # 3. Order check. Train_CF_9 holds days 1-9 in STOCK-major order:
    #    [stock 0: day 1..9][stock 1: day 1..9]... Labels are normalisation-free, so
    #    they must match exactly if our day and stock recovery is right.
    section("3. Order check: Train_CF_9 labels vs our days 1-9 in stock-major order")
    t9 = data.load_transposed(data.train_file(raw_dir, 9),
                              rows=[data.ASK1_COL, data.BID1_COL] + list(range(144, 149)))
    sm = df[df.day <= 9].sort_values(["stock_id", "day", "t"])
    match = (t9[:, 2:].astype(int) == sm[data.LABEL_COLS].values).all(axis=1).mean()
    print(f"  rows with all 5 labels equal: {match:.4%}   ({'PASSED' if match == 1 else 'FAILED'})")
    print("  (Day-major order matches only ~27%: training files are grouped by stock.)")

    # 3b. Put every day on one scale: z-scored with statistics of training days only.
    ref = cfg["data"]["scale_ref_fold"]
    section(f"3b. Rescale all days to the days 1-{ref} z-score scale (Train_CF_{ref})")
    maps = data.fit_scale_maps(raw_dir, df, ref)
    print(f"  {'file scale':>10} {'used by days':>13} {'median |a-1|':>13} {'median |b|':>11} {'max fit resid':>14}")
    for k, (a, b, r) in maps.items():
        used = [d for d in range(1, 11) if data.day_scale_fold(d) == k]
        print(f"  {'days 1-' + str(k):>10} {str(used):>13} {np.median(np.abs(a - 1)):>13.4f} "
              f"{np.median(np.abs(b)):>11.4f} {r:>14.1e}")
    df = data.rescale_to_fold(df, maps)
    t5 = data.load_transposed(data.train_file(raw_dir, ref))[:, :data.N_FEATURES]
    diff = np.abs(data.stock_major_days(df, range(1, ref + 1)) - t5).max()
    print(f"\n  check: rescaled days 1-{ref} vs Train_CF_{ref} file, max |difference| = {diff:.1e}"
          f"   ({'PASSED' if diff < 1e-4 else 'FAILED'})")
    print("  (Files store ~7 significant digits, so differences of ~1e-6 are rounding.)")

    # 4. Stock boundaries.
    section("4. Stock boundaries: 4 largest level-1 jumps per day vs the 5th largest")
    print("  jump = |d ask1| + |d bid1| between consecutive rows (z-score units)")
    print(f"  {'day':>3}  {'boundary rows':<28} {'smallest of top-4':>17} {'5th largest':>12} {'ratio':>7}")
    for day in range(1, 11):
        starts, jumps = info[day]["starts"], info[day]["jumps"]
        ratio = jumps[3] / jumps[4]
        cuts = str([int(s) for s in starts[1:]])
        print(f"  {day:>3}  {cuts:<28} {jumps[3]:>17.3f} {jumps[4]:>12.3f} {ratio:>7.1f}")

    # Stock identity across days: on the single Train_CF_9 scale, the overnight join
    # (last mid of stock s on day d -> first mid of stock s on day d+1) should be small.
    t9_mid = (t9[:, 0] + t9[:, 1]) / 2
    sizes = sm.groupby(["stock_id", "day"], sort=False).size().values
    ends = np.cumsum(sizes)
    joins = np.abs(t9_mid[ends[:-1]] - t9_mid[ends[:-1] - 1])
    same_stock = np.arange(len(sizes) - 1) % 9 != 8
    print(f"\n  overnight joins, same stock (n={same_stock.sum()}): max |jump in mid| = {joins[same_stock].max():.3f}")
    print(f"  joins between different stocks (n={(~same_stock).sum()}): min |jump in mid| = {joins[~same_stock].min():.3f}")

    section("5. Rows per stock per day")
    counts = df.pivot_table(index="day", columns="stock_id", values="sample_id", aggfunc="count")
    print(counts.to_string())

    section("6. Mean level-1 ask (z-score) per stock per day  (is stock order the same each day?)")
    levels = df.pivot_table(index="day", columns="stock_id", values="f0", aggfunc="mean")
    print(levels.round(3).to_string())
    ranks = levels.rank(axis=1).astype(int)
    print("\n  rank of price level within each day:")
    print(ranks.to_string())

    # 7. Labels.
    section("7. Label values and k=10 balance per day")
    for col in data.LABEL_COLS:
        print(f"  {col}: values {sorted(int(v) for v in df[col].unique())}")
    bal = df.pivot_table(index="day", columns="label_k10", values="sample_id", aggfunc="count")
    pct = bal.div(bal.sum(axis=1), axis=0) * 100
    bal.columns = [f"n_{c}" for c in bal.columns]
    pct.columns = [f"%_{c}" for c in pct.columns]
    print(pd.concat([bal, pct.round(1)], axis=1).to_string())

    # 8. Label meaning and alignment. move_s = mean(mid[t+s+1 .. t+s+h]) - mid[t+s],
    #    the k=10 style move starting s rows from t. If the stored label at row t
    #    were a forecast for row t, agreement would peak at s = 0.
    section("8. Label alignment (k=10): agreement of stored label with move starting at t+s")
    h = cfg["data"]["horizon_snapshots"]
    mid = data.mid_price_proxy(df)
    key = [df.day, df.stock_id]
    fut = mid.groupby(key).transform(lambda m: m[::-1].rolling(h, min_periods=h).mean()[::-1].shift(-1))
    move = fut - mid
    updown = df.label_k10 != 2
    print("  (Up/Down rows with a non-zero move only. 0.5 = no relation.)")
    for s in [-12, -10, -9, -8, -5, -2, 0, 2, 5]:
        m = move.groupby(key).shift(-s)
        v = updown & m.notna() & (m != 0)
        agree = ((m[v] > 0) == (df.label_k10[v] == 1)).mean()
        print(f"    s = {s:+3d}: {agree:.3f}")
    m = move.groupby(key).shift(9)
    print("\n  median |move| by label at s = -9 (Stationary should be ~0):")
    print("  " + m.abs().groupby(df.label_k10).median().round(5).to_string().replace("\n", "\n  "))

    # 9. Realigned target (user decision in step 1): target[t] = label_k10[t + 9].
    shift = cfg["data"]["target_shift"]
    section(f"9. Realigned target: target[t] = label_k10[t + {shift}] within each stock-day")
    df = data.add_target(df, "label_" + cfg["data"]["target"], shift, cfg["data"]["label_map"])
    has = df.target >= 0
    print(f"  rows without a target (last {shift} of each of 50 stock-days): {(~has).sum()}")
    print(f"  rows with a target: {has.sum()}")
    print("  agreement of target with move starting at t+s (should now peak at s = 0):")
    updown = has & (df.target != 1)
    for s in [-2, -1, 0, 1, 2]:
        m = move.groupby(key).shift(-s)
        v = updown & m.notna() & (m != 0)
        print(f"    s = {s:+3d}: {((m[v] > 0) == (df.target[v] == 0)).mean():.3f}")
    bal = df[has].pivot_table(index="day", columns="target", values="sample_id", aggfunc="count")
    pct = bal.div(bal.sum(axis=1), axis=0) * 100
    names = {0: "Up", 1: "Stat", 2: "Down"}
    bal.columns = [f"n_{names[c]}" for c in bal.columns]
    pct.columns = [f"%_{names[c]}" for c in pct.columns]
    print("\n  target balance per day:")
    print(pd.concat([bal, pct.round(1)], axis=1).to_string())

    # 10. Save the processed frame for later steps.
    out_path = ROOT / cfg["data"]["processed_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_path)
    section(f"Saved {len(df)} rows to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
