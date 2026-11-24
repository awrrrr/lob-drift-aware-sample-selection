# Revision log — post-hoc reviewer revision

Everything in this folder was added **after the step-5 test results were seen**
(post-hoc revision). New settings are chosen on day-6 validation (or on days 7-8
stream behaviour) only, frozen in `config.yaml` under `revision:` before any test
scoring, and test days are scored once per new run by `scripts/evaluate.py`.
The original step-5 outputs (`results/summary.csv`, `results/differences.csv`,
the `main`-tagged rows of `runs.db`, the frozen `config.yaml` values) are untouched.
Backup of runs.db before the revision: `data/processed/runs_pre_revision.db`.

## R0 — timing and setup (2026-11-24)

- Built: `runs.tag` column (migration-safe `ALTER TABLE`; all 18 original runs read
  back as `main`), `timings` table (`run_id, window, select_ms, retrain_ms`) written by
  `harness.replay`, `--tag` flag on `scripts/run_experiments.py`. Tests: migration keeps
  old rows as `main`; one timing row per update window; tag stored. 37 tests pass.
- Check: one full E seed-0 replay under the new harness (tag `revision_check`, never
  scored or aggregated) selected **exactly the same 4,650 samples** as the original
  E seed-0 run, so timing changes nothing about selection.
- Timing (E, seed 0, 93 windows): selection median 63.5 ms / p99 317 ms per window
  (model probabilities + policy); retrain median 5.1 s / p99 7.3 s per window (fit V1
  on train + selected, then validation gate). Total inside the loop: 8 s selecting,
  453 s retraining.
- Wall clock: **23.1 min** (1386 s), but only 635 CPU-seconds. The run was launched as a
  background process from the app and spent over half its time waiting (most likely
  scheduled at low priority; the original step-5 runs took ~11 min each). The untimed
  parts of the loop were measured separately and are negligible (pickle 0.3 s, stream
  0.1 s, per-pick lookups ~2 s per run). Remaining runs are launched from the Terminal
  panel instead.
- Rule applied: one run > 10 min, so **3 seeds** are used for every new variant below
  (instead of 5). The original 5-seed results for B, E, E-audit are reused as they are.
- Surprise worth noting: the full test suite was briefly run during the timed run
  (≈20 s of contention); that does not explain the 2x slowdown.

## R1 — drift indicator (reviewer priority i)

- Built: `data.drift_score(pool, cfg)` = mean snapshot volatility of window k's
  selectable pool / the volatility threshold frozen on days 1-5 (0.000510). Snapshot
  volatility is the rolling std (100 snapshots) of the level-1 mid-price change from
  current and past prices only, so the score is label-free and causal. The harness now
  attaches this `snap_vol` to each stream row and hands it to policies with the
  features; `_ids_feats` excludes it, so no existing policy sees a different feature
  matrix. `drift_score` raises if a `target`/`label*` column is present.
  Script: `scripts/revision_drift.py` (stream days 7-8 and day 6 only; no test data).
- Tests (`tests/test_revision.py`): score = mean vol / threshold; undefined volatility
  ignored; label columns refused; the harness hands policies `snap_vol` and never
  `target`. 42 tests pass.
- Results (`drift_summary.txt`, `drift_windows.csv`, `drift_trajectory.png`,
  `drift_vs_v1.png`):
  - 93 stream windows; **13 have d_k > 1 (14.0%)**: 5 on day 7, 8 on day 8.
    d_k ranges 0.21-1.67, median 0.69. The adaptive rule in R2 will therefore switch
    regime in a minority of windows.
  - **d_k vs V1 macro-F1 (labels, after the fact): r = +0.06 (p = 0.59), Spearman
    +0.08.** No relation. V1 is if anything slightly better in d_k > 1 windows
    (0.414 vs 0.395).
  - **d_k vs V1 mean entropy: r = -0.63 (p ≈ 2e-11).** Volatile windows are where V1
    is *more confident*. The entropy ratio (window / day-6 mean entropy, reported
    only) correlates identically (-0.63; it is the same quantity rescaled).
- Finding (not a reason to switch indicators, per plan): on this stream, volatility is
  a clean label-free regime indicator but **not a proxy for harmful drift** — high-d_k
  windows are more directional, so V1 is more confident there without being more
  wrong. This is consistent with the step-5 regime analysis (high-vol test windows
  are mostly Up/Down, Stationary 14%).

