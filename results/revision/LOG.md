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

## R2 — attempt notes (harness fixes found while launching the grid)

Attempt notes, kept because they changed the harness:
- First parallel launch: three of four grid runs failed `database is locked` — replay held
  one write transaction per run. Fixed: commits per window (`2a2aef2`).
- Second launch: three failed `UNIQUE constraint failed: runs.run_id` — same policy + seed
  started in the same second produce the same run_id. Fixed: uuid suffix; the seeded rng
  draw is kept so selections are unchanged (`417ea3b`).
- Third launch: runs starved each other — a window held the write lock across its ~5 s
  retrain. Fixed: selections are committed *before* labels are read, retraining holds no
  lock, timings + gate decision go in a second short transaction (`882da44`). Verified: E
  seed 0's first 5 windows are identical to the original run. Incomplete grid rows from the
  stalled attempt (3 runs, 2,250 selections) were deleted; the 18 `main` runs untouched.
- Throughput: with 4 parallel runs, retrain takes ~24 s per window vs 5.1 s alone (fitting
  on the 178k base is memory-bandwidth-bound on this laptop; memory pressure high). Total
  throughput is the same either way, ≈ 9 min of compute per run.

## R7 — novelty text (committed before R2 finished; the grid is compute-bound)

- `results/revision/novelty.md`, 398 words, for the report.
- Sources verified from full text: Zhdanov 2019 (arXiv:1901.05954): pre-filter βk most
  informative, weighted k-means, pick nearest the centre; reports most-informative-per-
  cluster performs the same. Žliobaitė, Bifet, Holmes, Pfahringer 2011 (JMLR W&CP 17):
  randomised uncertainty threshold η ~ N(1, δ), explicitly to "label the instances about
  which the classifier is very certain" so drift is not missed; DDM-based classifier
  replacement.
- **Honest consequence:** auditing confident predictions to catch drift is Žliobaitė et
  al.'s idea. What differs here is the explicit budget share, sampling only from confident
  predictions, and adapting the share to a measured drift score. The cluster step is a
  close variant of Zhdanov 2019 and is not claimed as new.
- Not verified: the 2014 TNNLS full text (Split strategy) was not accessible (IEEE page
  empty, ResearchGate 403); Split is described from secondary sources only, and the text
  says so.

## R6 — leakage statement

- `results/revision/leakage.md`: one row per rule (time-ordered splits, horizon carry-over,
  select-then-reveal, label-free drift score, train-only volatility threshold, test scored
  once, revision settings from validation), each pointing at the code and the test that
  enforces it. Ends with the honest caveat that the revision was designed after the step-5
  test results were known.
- New tests: `test_volatility_threshold_ignores_non_training_days` (making days 6-10 50x
  more volatile leaves the threshold unchanged); `drift_score` refuses `target`/`label*`
  columns (from R1); the harness never hands policies the target (from R1).

## R2 — drift-adaptive audit (reviewer priority ii) — frozen

- Built: policy `E_drift` (`src/policies.py: drift_adaptive_policy`): E with audit share `a_high` when d_k > 1, else `a_low`; B = 50 unchanged, so the total budget is equal across methods. Tests: exactly B unique pool ids at d_k above and below 1; audit count follows the rule (5 vs 20 picks for 0.1/0.4); same seed gives the same picks; refuses a pool containing `target`; E itself unchanged.
- Grid (seed 0, final day-6 macro-F1 of the replayed model, `scripts/revision_grid.py`, runs tagged `revision_grid:*`, never test-scored):

  | a_low | a_high | final val macro-F1 | accepted / rejected |
  |---|---|---|---|
  | 0.1 | 0.6 | 0.45756 | 93 / 0 |
  | 0.1 | 0.4 | 0.45750 | 93 / 0 |
  | 0.2 | 0.4 | 0.45737 | 93 / 0 |
  | 0.0 | 0.4 | 0.45731 | 93 / 0 |

- **Frozen: a_low = 0.1, a_high = 0.6** (`config.yaml: revision`), by the rule fixed before the grid ran (highest final validation macro-F1). Honest note: the whole grid spans 0.00025, i.e. noise level; the original fixed-20% E seed-0 run ended at 0.45752 on the same metric. The choice is procedural, not evidence that the adaptive share helps on validation.
- Activation: 13 of 93 stream windows have d_k > 1 (R1), so E_drift spends 30 audit picks in those 13 windows and 5 in the other 80: 390 + 400 = 790 audit picks in total, vs 930 for E's fixed 20%. E_drift therefore audits *less* than E overall, and its cluster step gets more of the budget in calm windows.

