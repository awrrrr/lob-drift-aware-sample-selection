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

## R3 — ablation (reviewer item 4)

- Built: `E_minus_cluster` (fixed 20% audit + plain top-entropy) and `E_drift_minus_cluster` (drift-adaptive audit + top-entropy) via `cluster=False` on `proposed_policy`. 9 new runs (3 variants x seeds 0-2), tag `revision`, all on config hash `0c0674a15911b0a6` and commit `bbcb7e1`; every run accepted all 93 updates.
- Reused, not rerun: A, B, C, D, E, E_minus_audit (tag `main`, step-5 config hash). Adding the `revision:` block changed the config hash, but `--skip-existing` keys on tag + hash and only new policies were launched, so nothing old was rerun or overwritten.
- Scored on test once: `scripts/evaluate.py --tag revision --skip-scored` (only the 9 new runs). Table: `results/revision/ablation.csv` (mean ± std over seeds; pp columns are **E_drift minus the row**, positive = E_drift better).

  | policy | seeds | high-vol macro-F1 | low-vol | overall | E_drift − row, high-vol (pp) |
  |---|---|---|---|---|---|
  | E_drift | 3 | 0.4149 ± 0.0002 | 0.4346 | 0.4339 | +0.000 |
  | E | 5 | 0.4153 ± 0.0004 | 0.4346 | 0.4340 | -0.040 |
  | E_minus_audit | 5 | 0.4154 ± 0.0003 | 0.4345 | 0.4338 | -0.049 |
  | E_minus_cluster | 3 | 0.4155 ± 0.0008 | 0.4344 | 0.4338 | -0.062 |
  | E_drift_minus_cluster | 3 | 0.4146 ± 0.0002 | 0.4343 | 0.4336 | +0.030 |
  | A | 1 | 0.4126 ± 0.0000 | 0.4345 | 0.4335 | +0.234 |
  | B | 5 | 0.4151 ± 0.0009 | 0.4346 | 0.4339 | -0.015 |
  | C | 1 | 0.4150 ± 0.0000 | 0.4343 | 0.4336 | -0.010 |
  | D | 1 | 0.4151 ± 0.0000 | 0.4355 | 0.4348 | -0.018 |

- **Result: E_drift does not beat random.** On the primary metric it is 0.4149, 0.015 pp *below* B (random) and 0.040 pp below E. Every updating variant lies in 0.4146-0.4155, inside seed noise (B's seed std is 0.0009 = 0.09 pp). Removing the audit, the clustering, or the drift adaptation changes high-vol macro-F1 by at most 0.06 pp in either direction. No component has a measurable effect at B = 50/window. The only clear effect is still updating vs not (A is 0.23 pp below E_drift).
- Consistent with R1: the drift score switches the audit share up in 13 windows where V1 was not making more errors, so auditing more there had little to correct.

## R4 — statistical variation (reviewer item 5)

- `scripts/revision_stats.py`: rebuilds each scored run's final model (27 runs) exactly as `src/evaluate.py` does and predicts the test days to get per-window confusion matrices. Check: recomputed high-vol macro-F1 equals the stored metric for every run (max gap 0.0e+00).
- Method: **paired bootstrap over test windows** (W = 1000-snapshot blocks, resampled with replacement to respect time dependence), 2,000 resamples, seed 0. In each resample every run is scored on the same windows; a policy's score is its **per-seed macro-F1 averaged over seeds**. Difference = E_drift minus the other, in pp. Per-seed values: `per_seed.csv`; intervals: `bootstrap.csv`.
- **Coarseness:** the test days have 85 windows, only **9 high-volatility**, so high-vol intervals are wide. They are honest about how little the primary condition can resolve.

  | E_drift − … | high-vol diff (pp) | 95% CI | overall diff (pp) | 95% CI |
  |---|---|---|---|---|
  | E | -0.040 | [-0.096, +0.020] | -0.007 | [-0.033, +0.020] |
  | E_minus_audit | -0.049 | [-0.114, +0.020] | +0.010 | [-0.011, +0.032] |
  | E_minus_cluster | -0.062 | [-0.172, +0.022] | +0.006 | [-0.059, +0.066] |
  | E_drift_minus_cluster | +0.030 | [-0.108, +0.145] | +0.026 | [-0.037, +0.088] |
  | A | +0.234 | [+0.029, +0.441] | +0.034 | [-0.055, +0.113] |
  | B | -0.015 | [-0.103, +0.064] | +0.000 | [-0.062, +0.059] |
  | C | -0.010 | [-0.204, +0.155] | +0.028 | [-0.043, +0.098] |
  | D | -0.018 | [-0.210, +0.193] | -0.091 | [-0.276, +0.088] |

- **Every interval against another updating policy contains 0.** The only one clear of zero is E_drift − A (fixed model) on high-vol: +0.23 pp, CI [+0.03, +0.44]. Updating helps; the choice of samples does not, at this budget.
- C, D and A are deterministic (no randomness in their selection), so their seed spread is 0 by construction, not by luck; they have one run each. E_drift and the two _minus_cluster variants have 3 seeds (R0 rule), B / E / E−audit keep their 5 step-5 seeds.

## R5 — overhead only (budget sweep not run)

- Per plan, the B ∈ {25,…,500} budget sweep was **not run** (deadline); it is future work.
- `overhead.csv` (from `timings` of runs already done) and `selection_benchmark.csv` (every policy timed on the same 93 stream pools under V1's probabilities, no retraining):

  | policy | selection ms / window (median) | p99 |
  |---|---|---|
  | B | 0.03 | 0.06 |
  | C | 0.07 | 0.08 |
  | D | 5.90 | 80.98 |
  | E | 55.34 | 167.05 |
  | E_minus_audit | 65.89 | 151.92 |
  | E_drift | 60.55 | 132.54 |
  | E_minus_cluster | 0.36 | 0.90 |
  | E_drift_minus_cluster | 0.39 | 0.97 |

- Retraining dominates: 5.1 s median per window when run alone (R0, E). The revision runs logged ~13 s because three ran in parallel and shared memory bandwidth, so `overhead.csv`'s retrain column for them overstates single-run cost. Selection is ≈1% of the per-window cost even for the clustering policies (55-66 ms); the cluster step is what costs (top-entropy variants: 0.4 ms). Prediction latency per snapshot is ≈0.06 ms p50 for every policy (same model class).
- Note on step-5 latency numbers: when `scripts/evaluate.py` was added (19 Nov) it re-scored the 18 step-5 runs once to verify it. All F1 / accuracy / update values were reproduced exactly, but latency is wall-clock, so `runs.db` now holds a re-measurement (e.g. A p50 0.0598 vs 0.0603 ms) that differs slightly from `results/summary.csv`. `summary.csv` was left untouched (ground rule 1).

