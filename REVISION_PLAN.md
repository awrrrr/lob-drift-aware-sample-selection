# Revision plan — reviewer comments (AIML mini project)

Hand this file to Claude Code in the repo root. Read `CLAUDE.md` first; it is still the protocol.

## Context

The faculty review approved the project "with minor revisions". The reviewer's **first priority** is:
(i) specify how drift is detected, and (ii) explain how the selection strategy changes when drift occurs.
Items on novelty, ablation and wider validation are mainly "before research publication", but the cheap parts are done here too.

Current state (do not re-derive): "drift-aware" is in the title but **nothing in the pipeline measures drift**. Volatility is used only to split test windows for scoring. Policy E reserves a *fixed* 20% audit share. Step 5 result: E does not win; all updating policies ≈ 0.415 high-vol macro-F1 vs A 0.4126, within seed noise. Failure analysis: B = 50/window is 2.6% of the base set, on the flat part of the learning curve.

## Deadline mode (submission is tomorrow)

Do the steps in this order and **commit after each**, so the repo is submittable whenever work stops:
**R0 → R1 → R2 → R7 → R6 → R3 → R4 → R8.** R5 (budget sweep) is **not** run; list it as future work in R8.
Compute limits:
- R0: time one full E run first. If one run takes more than 10 min, use 3 seeds everywhere below instead of 5 and say so in LOG.md.
- R2 grid: only (a_low, a_high) ∈ {(0.1, 0.4), (0.2, 0.4), (0.0, 0.4), (0.1, 0.6)}, seed 0.
- R3: E_drift at the seed count above; E_minus_cluster and E_drift_minus_cluster at 3 seeds. Reuse all existing runs; do not rerun A–E.
- R5 overhead table only: build it from the `timings` rows of runs you already did. No new runs for it.
- If a run fails or time runs short, write what exists, mark the gap in LOG.md, and move on. Never leave a half-finished change uncommitted.

## Ground rules for this revision

1. **The original step-5 results stay as they are.** Do not overwrite `results/summary.csv`, `results/differences.csv`, the existing `runs.db` rows or the frozen `config.yaml` values. New outputs go to `results/revision/`. New runs are tagged in `runs` (add a `tag` column, default `'main'`, new runs `'revision'`).
2. Everything added here comes **after test results were seen**. Say so plainly in every write-up ("post-hoc revision, settings chosen on validation only, test scored once"). Never tune anything on days 9–10.
3. New settings are chosen on **day 6 validation** (or on days 7–8 stream behaviour without test data), written to a new `revision:` block in `config.yaml` with a comment, then frozen **before** any test scoring.
4. The drift indicator must be **label-free at selection time** (leakage rule 2). Labels may be used afterwards only for diagnostics.
5. Run unattended. Do not stop for review between steps; instead append a short entry per step to `results/revision/LOG.md` (what was built, what was checked, anything surprising). Stop only if a step would break a rule above.
6. Keep code plain; the student defends it at a viva. One new function per policy in `src/policies.py`, same signature as the rest.
7. `pytest` must pass after every step. Commit after every step with a clear message.

## Steps

### R0. Timing and setup
- Add `tag` column to `runs` (migration-safe: `ALTER TABLE` if missing).
- Add timing to `harness.replay`: per window, log selection time (ms) and retrain time (ms) into a new `timings` table (`run_id, window, select_ms, retrain_ms`).
- Time one full run of policy E (seed 0). Record it in LOG.md. Use it to size R5.

### R1. Drift indicator (reviewer priority i)
- Define a per-window, label-free drift score: **d_k = (mean snapshot volatility of window k's pool) / volatility.threshold**, using the existing `snapshot_volatility` and the threshold frozen on days 1–5. d_k > 1 means the window is in a high-volatility regime relative to training. Add `drift_score(pool_rows, df, cfg)` to `src/data.py` (or reuse `window_volatility`); compute it only from features of rows already in the pool.
- **Justify it with evidence on train/validation/stream only (not test):**
  - Plot d_k across the stream windows (`results/revision/drift_trajectory.png`).
  - Diagnostic (labels used after the fact, stream days only): V1 macro-F1 and mean predictive entropy per stream window vs d_k. Report the correlation. This shows whether high d_k actually coincides with V1 doing worse, i.e. whether volatility is a reasonable proxy for harmful drift.
  - As a second, purely reported indicator, compute the mean predictive entropy ratio (window mean entropy / day-6 mean entropy). Do not use it for selection; report how it correlates with d_k.
- If d_k shows no relation to V1 error on the stream, say so in LOG.md and still continue; that is a finding, not a reason to switch indicators.

### R2. Drift-adaptive audit (reviewer priority ii, item 3)
- New policy **E_drift**: identical to E except the audit share depends on d_k:
  `audit_share_k = a_low if d_k <= 1 else a_high`, B unchanged (total budget equal across methods, as the reviewer requires). Rationale to write in the docstring: under drift, confident predictions are more likely to be confidently wrong (the case uncertainty sampling misses), so audit more of them; in calm windows spend more on uncertain samples.
- Choose `(a_low, a_high)` from a small grid, e.g. a_low ∈ {0.0, 0.1, 0.2}, a_high ∈ {0.2, 0.4, 0.6}, a_high ≥ a_low, by **final validation macro-F1** of the replayed model, seed 0. Freeze the choice in `config.yaml` under `revision:` with the grid results in LOG.md.
- Also note in LOG.md how many stream windows have d_k > 1; if very few, the adaptive rule barely activates — report that.
- Tests: E_drift returns exactly B unique pool ids; uses no labels; same seed → same picks; audit count matches the rule for a synthetic pool with d_k above and below 1.

### R3. Ablation (item 4)
Variants, all at the same B and window schedule:
| ID | Description |
|---|---|
| E_drift | full proposed: cluster diversity + audit + drift adaptation |
| E | no drift adaptation (fixed 20% audit) — already exists |
| E_minus_audit | no audit — already exists |
| E_minus_cluster | audit (fixed 20%) + plain top-entropy instead of PCA/k-means clustering |
| E_drift_minus_cluster | drift-adaptive audit + plain top-entropy |
- Implement the two `_minus_cluster` variants as thin wrappers (a flag on `proposed_policy` is fine).
- Run 5 seeds each for new variants. Reuse existing E, E_minus_audit, A, B, C, D runs (same config hash for the frozen part; confirm the hash logic does not force reruns, or rerun them under the revision tag if the harness change altered the hash — note which in LOG.md).
- Score on test once via `scripts/evaluate.py`; extend `make_tables.py` to write `results/revision/ablation.csv`: mean ± std over seeds and pp differences vs full E_drift, for high-vol, low-vol and overall macro-F1.

### R4. Statistical variation (item 5)
- For each comparison vs E_drift, report mean difference in pp, per-seed values, and a paired bootstrap 95% CI over test windows (seed-averaged predictions or per-seed then averaged; document which). 2,000 resamples, fixed seed.
- C and D are deterministic: state that their spread is 0 by construction, not by luck.

### R5. Budget sweep and overhead (item 5)
- Sensitivity analysis only; the primary result stays at B = 50.
- B ∈ {25, 50, 100, 250, 500} for A (once), B, C, D, E, E_drift. Seeds: 5 for B/E/E_drift if the R0 timing allows the whole sweep in about 8 hours of CPU; otherwise 3 seeds and say so. W and gate tolerance unchanged.
- Output `results/revision/budget_curve.png`: high-vol test macro-F1 vs B, one line per policy, error bars = seed std. This directly tests the failure-analysis claim that selection only matters at larger budgets.
- Overhead table `results/revision/overhead.csv`: per policy, median and p99 selection ms per window, retrain ms per window, and prediction latency (existing p50/p99).

### R6. Leakage statement (item 5)
Write `results/revision/leakage.md` (short): the time-ordered day splits, the horizon carry-over rule, label reveal after selection, test scored once, volatility threshold from train days, drift score from pool features only. Point to the test in `tests/` that enforces each. Add a test that `drift_score` raises or ignores any column named `target`/label.

### R7. Novelty text (item 1)
Write `results/revision/novelty.md` (plain prose, ≤ 400 words, for the student to paste into the report; the proposal is a `.pages` file, do not edit it). It must:
- State precisely how E_drift differs from (a) Liu et al. 2023 uncertainty × density, (b) Žliobaitė et al. 2014 — **check their "Split" strategy, which also reserves part of the budget for random labelling; the audit idea overlaps with it, so the difference must be stated honestly** (ours samples only from confident predictions and adapts the share to measured drift), (c) diverse batch active learning that clusters an uncertain shortlist (e.g. Zhdanov 2019, "Diverse mini-batch active learning"; verify the reference) — our cluster step is close to this, so do not claim it as new on its own.
- State the contribution as: a drift-adaptive confident-prediction audit, combined with cluster-diverse uncertainty sampling, evaluated under a fixed per-update budget on LOB data with a leakage-safe replay protocol — plus whatever the results actually show, including a null result.
- Do not overclaim. If E_drift does not beat random, say so in one sentence.

### R8. Wrap-up
- Update `DEMO.md` with a short "Revisions" section: what the drift indicator is, how E_drift changes selection, where the new tables/plots are.
- Optional, only if trivial: show d_k per window in the Streamlit dashboard.
- Final LOG.md entry: a 10-line summary of every reviewer item → what was done → file where the evidence is. Mark "another independent dataset" as **not done, future work** with one line on why (no comparable free labelled LOB dataset readily available; LOBSTER is paid).
- `pytest` green; final commit.

## Reviewer item → step

| Reviewer item | Step |
|---|---|
| 1 Clarify novelty | R7 |
| 2 Define drift detection (priority) | R1 |
| 3 Adaptive confidence auditing (priority) | R2 |
| 4 Ablation | R3 |
| 5 Budgets, variation, overhead, leakage, 2nd dataset | R4, R5, R6, R8 (2nd dataset: future work) |
