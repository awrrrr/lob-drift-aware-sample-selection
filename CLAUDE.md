# CLAUDE.md — LOB Drift-Aware Sample Selection (AIML mini project)

Read this file before every task. It is the experimental protocol. If a request conflicts with it, stop and ask before changing anything.

## What this project is

Predict short-horizon mid-price direction (Up / Stationary / Down) from limit order book snapshots, then study **which new samples a model should learn from when each update can only add B samples**. The contribution is a selection policy, tested fairly against baselines. The dashboard is a viewer, not the contribution.

- Base paper: Liu et al. 2023, *Online Active Learning for Drifting Data Streams*, IEEE TNNLS 34(1). It selects samples by uncertainty × local density.
- Earlier work: Žliobaitė et al. 2014, *Active Learning With Drifting Streaming Data*, IEEE TNNLS 25(1).
- Dataset: FI-2010, Ntakaris et al. 2018, *Journal of Forecasting*. CC BY 4.0.

Labels in LOB data are free once the horizon passes. The scarce resource is **samples per update**, not human labelling. Never describe this as human annotation.

## Data facts (verify in step 1, then trust)

- Use only the `NoAuction` + `ZScore` files. Data is already normalised; do not re-normalise across days (beyond the one-scale mapping below).
- Files are **transposed**: 149 rows × N columns. Rows 0–143 are features, rows 144–148 are labels for horizons k = 1, 2, 3, 5, 10 (in units of 10 events). Load, then transpose to N × 149.
- Label values are expected to be 1 = Up, 2 = Stationary, 3 = Down. **Check this against the paper / data before using it.** Map to 0/1/2 internally.
- Target: the **k = 10 label (row 148)**. One horizon only; do not tune across horizons.
- **Verified in step 1:** labels are 1 = Up, 2 = Stationary, 3 = Down. The stored labels are **backward-aligned**: `label_k10` at row r describes the move starting at row r − 9 (97.7% agreement; only 59% with the move starting at r). Decision (user, step 1): **realign**. The target for row t is the stored `label_k10` at row t + 9, within the same stock-day. The last 9 rows of each stock-day have no target (`target = -1`) and are never trained on or scored. The realigned target uses mid-prices up to t + 10, so it becomes usable after t + 10 (`horizon_snapshots` in `config.yaml`).
- **Training files are stock-major:** `Train_CF_k` = [stock 0: days 1..k][stock 1: days 1..k]… Test files are one day each, with the 5 stocks stacked. Day recovery uses only `Train_CF_1` + `Test_CF_1..9`.
- **Normalisation differs per fold:** `Train_CF_k` and `Test_CF_k` are z-scored with statistics from days 1..k, so the day files are on slightly different scales (median scale gap 1–6%). Decision (user, step 1): **rescale every day onto the `Train_CF_5` scale** (statistics of training days 1–5 only). We use exact per-feature affine maps fitted on rows of days 1–5 that appear in both files (residual ~1e-6). This undoes the per-fold statistics; it is not a new normalisation. `scale_ref_fold: 5` in `config.yaml`. Check: rescaled days 1–5 equal `Train_CF_5` to 4.4e-6.
- Nine features (f129–f131, f135–f137, f141–f143) are constant across all 10 days and carry no information.
- **Days are not labelled in the files.** Recover them from the anchored fold structure:
  - `Train_Dst_NoAuction_ZScore_CF_1.txt` = day 1
  - `Test_Dst_NoAuction_ZScore_CF_k.txt` = day k+1, for k = 1..9
  - Sanity check: rows(Train_CF_k+1) should equal rows(Train_CF_k) + rows(Test_CF_k).
- Each day file stacks **five stocks** back to back, and stock boundaries are not marked. Detect them (large discontinuities in level-1 price features) and store a `stock_id` column. Report how you detected them and the row counts per stock per day.

Step 1 output must print: rows per day, rows per stock per day, label balance per day, and confirm the fold-sum check. Do not start step 2 until the user has seen this.

## Splits (fixed, never shuffled)

| Days | Role | Allowed use |
|---|---|---|
| 1–5 | Train | Fit V1 classifiers. Fit any extra transform. Define volatility threshold. |
| 6 | Validation | Choose classifier and hyperparameters, update-gate decisions, policy settings. |
| 7–8 | Update stream | Replayed in time order; policies select samples from here. |
| 9–10 | Test | Final scoring only. Never used for any choice. |

Never shuffle across time. Never fit anything on days 7–10 except retraining on samples a policy selected from days 7–8.

## Leakage rules (non-negotiable)

1. A label at time t depends on future prices. A stream sample's label is only usable **after its horizon has passed**. When a window closes, samples within the last horizon steps of that window are not yet labelled: exclude them from that window's selectable pool and carry them into the next window.
2. Selection decisions use only predictions and features. **Never read labels before a sample is selected.** Log the selected IDs and reasons first, then reveal labels.
3. Test days are scored once per final run by an automated script. No settings change in response to test results.
4. The volatility threshold that defines "difficult" windows is computed on training days only, then frozen.

## Volatility condition

- Compute a rolling volatility proxy per stock from level-1 mid-price changes (from the feature columns) over a fixed number of snapshots.
- Threshold = a fixed percentile of the training-day distribution (start with the 75th). Windows above it = "high volatility", below = "low volatility".
- Document the exact window length and percentile in `config.yaml` before running any experiment.

## Classifiers (step 2)

- Logistic regression (baseline), SVM, KNN. Same features and splits for all.
- Small documented grid per model. Select by **validation macro-F1**.
- Use class weights for imbalance. For SVM, use proper probability calibration (training-only CV); never treat decision scores as probabilities.
- One classifier (V1) is then fixed for all policy experiments.

## Replay harness (step 3, the core)

- Replay days 7–8 in fixed windows of W snapshots. Each policy adds exactly B labelled samples per window.
- Retrain the candidate on (all training data + all samples selected so far). Keep old data; do not drop it.
- **Validation gate:** the candidate replaces the current model only if validation macro-F1 does not fall by more than a tolerance fixed in `config.yaml`. Otherwise keep the current model and log the rejection.
- Every policy starts from the same V1, sees the same stream, the same W, B and update schedule.
- W, B and the tolerance are chosen on validation behaviour, written to `config.yaml`, then frozen. Start values (illustrative only): W = 1000, B = 50.

## Policies (step 4)

All take the same inputs (current model, window features, already-selected IDs, seed) and return exactly B IDs with a reason per ID.

| ID | Policy | Rule |
|---|---|---|
| A | Fixed | No updates. |
| B | Random | B random samples from the selectable pool. |
| C | Uncertainty | Top-B by predictive entropy (or margin; pick one and document it). |
| D | Density × uncertainty | Window-based adaptation of Liu et al.'s selection: score = uncertainty × local density (kNN density in feature space), take top-B. Call it "adapted from", never "CogDQS". |
| E | Proposed | Reserve a fixed audit share (start at 20% of B) for random samples from **confident** predictions. Fill the rest: shortlist the 3× most uncertain, cluster the shortlist (k-means on PCA-reduced features), take the most uncertain per cluster. Documented fallback if a cluster is empty or the pool is small. No repeated IDs. |
| E−audit | Ablation | E with the audit share given to the cluster step. Isolates the audit's effect. |

## Evaluation (step 5)

- Primary: **macro-F1 on high-volatility test windows**, E vs B, C and D.
- Also: overall macro-F1, low-volatility macro-F1, per-class precision/recall, accuracy, number of accepted/rejected updates, prediction latency per snapshot (p50/p99, state what is included).
- 5 seeds per policy. Report mean and spread, and differences in percentage points.
- If E does not win, report that and examine failure cases. Never adjust the policy after seeing test results.

## Logging (build this before any policy)

SQLite database `runs.db`:
- `runs`: run_id, policy, seed, config hash, git commit, start/end time.
- `selections`: run_id, window, sample_id, reason, predicted probs, model version.
- `updates`: run_id, window, model version, parent version, accepted (bool), validation macro-F1 before/after.
- `metrics`: run_id, split, condition, metric, value.

Every number in the report must be reproducible from `runs.db` + `config.yaml` + the git commit.

## Repo layout

```
config.yaml          # all frozen settings
data/raw/            # FI-2010 files (gitignored)
src/data.py          # loading, transpose, day + stock recovery, volatility
src/models.py        # classifiers, calibration
src/policies.py      # A–E and ablation, one function each
src/harness.py       # replay, retrain, gate, logging
src/evaluate.py      # test scoring from runs.db
scripts/             # entry points: check_data, train_v1, run_experiments, make_tables
app/dashboard.py     # Streamlit, reads runs.db only
tests/               # leakage and budget tests
```

## Tests to write early

- Every policy returns exactly B unique IDs from the selectable pool.
- No selected ID is from days 9–10. No sample is used before its horizon has passed.
- Same seed → identical selections.
- Day recovery fold-sum check passes.

## Build order (one session each; stop for review after each)

1. Data loader, day + stock recovery, sanity printout.
2. V1 classifiers on train/validation.
3. Replay harness + logging + tests. No real policy yet (use B = random to exercise it).
4. Policies C, D, E, ablation.
5. Experiments over 5 seeds, results tables and plots.
6. Streamlit dashboard reading `runs.db`.

## Working style

- Python, pandas, NumPy, scikit-learn, Streamlit, SQLite. Laptop CPU only.
- Keep the code plain and readable; this is graded and defended at a viva.
- After each step, summarise what was built, what was checked, and anything that looked wrong. Flag surprises instead of working around them.
