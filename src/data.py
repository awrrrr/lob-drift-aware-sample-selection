"""Loading FI-2010 (NoAuction, ZScore) and recovering days and stocks.

The raw files are transposed: 149 rows x N columns. Rows 0-143 are features,
rows 144-148 are labels for horizons k = 1, 2, 3, 5, 10.

Days are not marked in the files. They come from the anchored folds:
    Train_..._CF_1.txt  = day 1
    Test_..._CF_k.txt   = day k + 1   (k = 1..9)

Each day stacks five stocks back to back without a marker. We find the four
boundaries as the four largest jumps in the level-1 ask and bid prices.
"""

from pathlib import Path

import numpy as np
import pandas as pd

N_ROWS = 149
N_FEATURES = 144
N_STOCKS = 5

# Level-1 columns in the 40 raw LOB features: ask price, ask volume, bid price, bid volume.
ASK1_COL = 0
BID1_COL = 2

LABEL_COLS = ["label_k1", "label_k2", "label_k3", "label_k5", "label_k10"]
FEATURE_COLS = [f"f{i}" for i in range(N_FEATURES)]


def train_file(raw_dir, k):
    return Path(raw_dir) / f"Train_Dst_NoAuction_ZScore_CF_{k}.txt"


def test_file(raw_dir, k):
    return Path(raw_dir) / f"Test_Dst_NoAuction_ZScore_CF_{k}.txt"


def day_file(raw_dir, day):
    """Day 1 is the first training fold; day k + 1 is test fold k."""
    if day == 1:
        return train_file(raw_dir, 1)
    return test_file(raw_dir, day - 1)


def load_transposed(path, rows=None):
    """Read a raw file and return an (N x 149) array.

    If `rows` is given, only those raw rows are parsed and the result is
    (N x len(rows)). Each raw row is one long line of space-separated numbers.
    """
    wanted = set(range(N_ROWS)) if rows is None else set(rows)
    parsed = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i in wanted:
                parsed[i] = np.fromstring(line, sep=" ")
    if len(parsed) != len(wanted):
        raise ValueError(f"{path}: expected rows {sorted(wanted)}, found {sorted(parsed)}")
    lengths = {len(v) for v in parsed.values()}
    if len(lengths) != 1:
        raise ValueError(f"{path}: rows have different lengths {lengths}")
    order = sorted(wanted)
    return np.stack([parsed[i] for i in order], axis=1)


def count_columns(path):
    """Number of snapshots in a raw file (length of its first line)."""
    with open(path) as f:
        return len(f.readline().split())


def find_stock_boundaries(x, n_stocks=N_STOCKS):
    """Return (starts, jumps) for one day.

    x: (N x 149) array for the day. The jump at row i is the absolute change in
    level-1 ask plus level-1 bid between rows i and i + 1. The largest
    n_stocks - 1 jumps are taken as stock boundaries. `jumps` is the full
    sorted-descending jump list so the caller can report how clear the gap is.
    """
    ask = x[:, ASK1_COL]
    bid = x[:, BID1_COL]
    jump = np.abs(np.diff(ask)) + np.abs(np.diff(bid))
    cut_after = np.sort(np.argsort(jump)[::-1][: n_stocks - 1])
    starts = np.concatenate([[0], cut_after + 1])
    return starts, np.sort(jump)[::-1]


def build_day_frame(x, day, starts):
    """Turn one day's array into a DataFrame with day, stock_id and position."""
    df = pd.DataFrame(x[:, :N_FEATURES], columns=FEATURE_COLS)
    for j, col in enumerate(LABEL_COLS):
        df[col] = x[:, N_FEATURES + j].astype(int)
    stock_id = np.zeros(len(x), dtype=int)
    for s, start in enumerate(starts):
        stock_id[start:] = s
    df["day"] = day
    df["stock_id"] = stock_id
    # Snapshot position within (day, stock); time order inside one stock-day.
    df["t"] = df.groupby("stock_id").cumcount()
    return df


def load_all_days(raw_dir, n_days=10):
    """Load days 1..n_days. Returns (DataFrame, info) where info holds boundary details.

    Features are as stored, so each day is on its own fold's z-score scale
    (see day_scale_fold). Use rescale_to_fold to put them on one scale.
    """
    frames = []
    info = {}
    for day in range(1, n_days + 1):
        x = load_transposed(day_file(raw_dir, day))
        starts, jumps = find_stock_boundaries(x)
        frames.append(build_day_frame(x, day, starts))
        info[day] = {"starts": starts, "jumps": jumps, "n": len(x)}
    df = pd.concat(frames, ignore_index=True)
    # Global id in (day, stock, t) order. Stable across runs.
    df.insert(0, "sample_id", np.arange(len(df)))
    return df, info


# ---------------------------------------------------------------------------
# One common feature scale
#
# Train_CF_k and Test_CF_k are z-scored with statistics from days 1..k, so the
# day files are on different scales. Call "scale k" the one using days 1..k.
# Day 1 is on scale 1; day d >= 2 (Test_CF_{d-1}) is on scale d - 1.
#
# Two scales of the same raw value are related exactly by z_ref = a * z_k + b
# per feature. We find (a, b) from rows that appear in both Train_CF_k and
# Train_CF_ref: the training days 1..min(k, ref). Applying the map gives
# (raw - mean_ref) / std_ref, i.e. the data normalised with training-day
# statistics only. No statistic from later days survives the map.
# ---------------------------------------------------------------------------

def day_scale_fold(day):
    return max(1, day - 1)


def stock_major_days(df, days):
    """Features of `days` in Train_CF file order: stock 0 days..., stock 1 days..."""
    sub = df[df.day.isin(days)].sort_values(["stock_id", "day", "t"])
    return sub[FEATURE_COLS].values


def fit_affine(src, ref, const_tol=1e-9):
    """Per-feature least squares ref ~ a * src + b. Returns (a, b, max_residual).

    A feature that is constant in src is left unchanged (a = 1, b = 0); step 1
    found that such features are constant in every file.
    """
    a = np.ones(src.shape[1])
    b = np.zeros(src.shape[1])
    resid = np.zeros(src.shape[1])
    for j in range(src.shape[1]):
        if src[:, j].std() < const_tol:
            resid[j] = np.abs(ref[:, j] - src[:, j]).max()
            continue
        A = np.column_stack([src[:, j], np.ones(len(src))])
        coef, *_ = np.linalg.lstsq(A, ref[:, j], rcond=None)
        a[j], b[j] = coef
        resid[j] = np.abs(A @ coef - ref[:, j]).max()
    return a, b, resid.max()


def fit_scale_maps(raw_dir, df, ref_fold):
    """Maps from every scale k = 1..9 to scale `ref_fold`. Returns {k: (a, b, max_resid)}."""
    ref_file = load_transposed(train_file(raw_dir, ref_fold))[:, :N_FEATURES]
    ref_days = list(range(1, ref_fold + 1))
    maps = {}
    for k in range(1, 10):
        if k == ref_fold:
            maps[k] = (np.ones(N_FEATURES), np.zeros(N_FEATURES), 0.0)
            continue
        overlap = list(range(1, min(k, ref_fold) + 1))
        src = load_transposed(train_file(raw_dir, k))[:, :N_FEATURES]
        # Train_CF_k is stock-major over days 1..k; keep the rows of the overlap days.
        src_days = df[df.day <= k].sort_values(["stock_id", "day", "t"]).day.values
        ref_rows = df[df.day <= ref_fold].sort_values(["stock_id", "day", "t"]).day.values
        maps[k] = fit_affine(src[np.isin(src_days, overlap)], ref_file[np.isin(ref_rows, overlap)])
    return maps


def rescale_to_fold(df, maps):
    """Apply the scale maps in place: every day ends up on the reference scale."""
    for day in df.day.unique():
        a, b, _ = maps[day_scale_fold(day)]
        rows = df.day == day
        df.loc[rows, FEATURE_COLS] = df.loc[rows, FEATURE_COLS].values * a + b
    return df


def add_target(df, label_col, shift, label_map):
    """Add the forecasting target as column `target` (0/1/2, or -1 if none).

    The stored FI-2010 label at row r describes the move that started `shift`
    rows earlier (found in step 1: label_k10[r] matches
    mean(mid[r-8 .. r+1]) - mid[r-9]). So the forward target for row t is the
    stored label at row t + shift, within the same stock-day. The last `shift`
    rows of each stock-day have no target and get -1.
    """
    future = df.groupby(["day", "stock_id"])[label_col].shift(-shift)
    df["target"] = future.map(label_map).fillna(-1).astype(int)
    return df


def mid_price_proxy(df):
    """Level-1 mid price in z-score units (ask and bid are z-scored separately)."""
    return (df[f"f{ASK1_COL}"] + df[f"f{BID1_COL}"]) / 2.0


# ---------------------------------------------------------------------------
# Volatility condition (leakage rule 4: threshold fitted on training days only)
#
# Per-snapshot volatility = rolling std of the level-1 mid-price change over a
# fixed number of snapshots, within each stock-day (never across a boundary).
# A "window" is a fixed W-snapshot block of the replay stream (stream_pos // W);
# a window's volatility is the mean of its snapshots' volatility. A window is
# high-volatility if that mean exceeds a frozen threshold = a fixed percentile
# of the training-day window volatilities.
# ---------------------------------------------------------------------------

def snapshot_volatility(df, vol_window):
    """Rolling std of mid-price change over `vol_window` snapshots, per stock-day."""
    mid = mid_price_proxy(df)
    key = [df["day"], df["stock_id"]]
    dmid = mid.groupby(key).diff()
    return dmid.groupby(key).transform(
        lambda s: s.rolling(vol_window, min_periods=vol_window).std())


def stream_windows(df, days, W):
    """The given days in replay order with a positional window id (stream_pos // W)."""
    stream = (df[df.day.isin(days)]
              .sort_values(["day", "stock_id", "t"]).reset_index(drop=True).copy())
    stream["stream_pos"] = np.arange(len(stream))
    stream["window"] = stream["stream_pos"] // W
    return stream


def window_volatility(df, days, cfg):
    """Mean per-snapshot volatility for each positional window of `days`."""
    W = cfg["harness"]["window"]
    vol = snapshot_volatility(df, cfg["volatility"]["vol_window"])
    stream = stream_windows(df, days, W)
    stream["vol"] = vol.loc[
        df[df.day.isin(days)].sort_values(["day", "stock_id", "t"]).index].to_numpy()
    return stream.groupby("window")["vol"].mean()


def fit_volatility_threshold(df, cfg):
    """Frozen threshold: the chosen percentile of training-day window volatilities."""
    wv = window_volatility(df, cfg["splits"]["train"], cfg)
    return float(np.nanpercentile(wv.to_numpy(), cfg["volatility"]["percentile"]))


def classify_windows(df, days, cfg, threshold):
    """Map each snapshot of `days` to its window's volatility class.

    Returns a DataFrame with sample_id, window, win_vol, high_vol (bool).
    """
    W = cfg["harness"]["window"]
    stream = stream_windows(df, days, W)
    wv = window_volatility(df, days, cfg)
    stream["win_vol"] = stream["window"].map(wv)
    stream["high_vol"] = stream["win_vol"] > threshold
    return stream[["sample_id", "window", "win_vol", "high_vol"]]


# ---------------------------------------------------------------------------
# Drift indicator (post-hoc reviewer revision, R1)
#
# d_k = mean snapshot volatility of window k's selectable pool / the frozen
# volatility threshold. d_k > 1: the window is more volatile than the upper
# quartile of training-day windows, i.e. a high-volatility regime relative to
# training. It is label-free (leakage rule 2): snapshot volatility uses only
# current and past level-1 prices from the feature columns, and every pool row
# has already been observed when the pool is formed. The threshold was fitted
# on days 1-5 only (rule 4).
# ---------------------------------------------------------------------------

LABEL_COLUMNS = ("target",)


def drift_score(pool, cfg):
    """Label-free drift score d_k of one window's pool (see block comment above).

    `pool` must carry a `snap_vol` column (snapshot_volatility of each row).
    Raises if a label column is present: selection must never be handed labels.
    Rows whose volatility is not yet defined (the first vol_window rows of a
    stock-day) are ignored; a pool with no defined value scores 0.0 (calm).
    """
    bad = [c for c in pool.columns if c in LABEL_COLUMNS or str(c).startswith("label")]
    if bad:
        raise ValueError(f"drift_score must not see label columns: {bad}")
    v = pool["snap_vol"].to_numpy(dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return 0.0
    return float(v.mean() / cfg["volatility"]["threshold"])
