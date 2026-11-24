# Leakage statement

How the experiment keeps future information out of every decision. Each rule
names the code that implements it and the test that fails if it breaks
(`pytest` runs them all).

| Rule | How it is enforced | Test |
|---|---|---|
| **Time-ordered splits.** Days 1-5 train, 6 validation, 7-8 update stream, 9-10 test. Never shuffled. | `config.yaml: splits`; `models.split_xy` selects by day; the stream is replayed in `(day, stock, t)` order (`harness.build_stream`). | `test_models.py::test_split_xy_excludes_missing_target_and_other_days`; `test_harness.py::test_no_selection_from_test_days_or_before_horizon` |
| **Labels only after their horizon.** The realigned target at t uses prices up to t + 10, so a stream sample joins the pool of window `(stream_pos + 10) // W`. The last 10 samples of a window carry into the next; samples without 10 rows left in their stock-day are never selectable. | `harness.build_stream` (`pool_window`, `selectable`); `data.add_target` shifts within a stock-day only. | `test_harness.py::test_selected_label_formed_within_its_window`; `test_harness.py::test_no_selection_from_test_days_or_before_horizon`; `test_data.py::test_add_target_shifts_within_stock_day` |
| **Select first, reveal labels after.** Policies receive ids, features, the current model's probabilities and the label-free `snap_vol`, never the target. Selected ids and reasons are logged, then labels are read for retraining. | `harness.replay` passes `pool[["sample_id"] + feats + ["snap_vol"]]`. | `test_revision.py::test_policy_pool_has_snap_vol_and_no_target`; `test_revision.py::test_drift_policies_never_see_labels` |
| **Drift score from pool features only.** d_k = mean snapshot volatility of the window's pool / the frozen threshold. Snapshot volatility uses current and past level-1 prices (backward rolling window, within a stock-day). | `data.drift_score` refuses any `target`/`label*` column. | `test_revision.py::test_drift_score_refuses_label_columns`; `test_revision.py::test_drift_score_is_mean_volatility_over_threshold` |
| **Volatility threshold from training days only**, then frozen (0.000510 in `config.yaml`). | `data.fit_volatility_threshold` uses `splits.train`; `scripts/freeze_volatility.py` writes it once. | `test_revision.py::test_volatility_threshold_ignores_non_training_days` |
| **Test days scored once, by script, with no setting changed in response.** The replay never reads days 9-10. `scripts/evaluate.py` rebuilds each run's final model from `runs.db` and scores it once; `--skip-scored` refuses to score a run twice; grid and check runs are never scored. | `src/evaluate.py` (`runs_to_score`, `evaluate_run`). | `test_revision.py::test_runs_to_score_filters_tag_unfinished_and_scored`; `test_evaluate.py::test_reconstruction_uses_selections_up_to_last_accepted_window` |
| **Revision settings chosen on validation.** E_drift's audit shares come from the final day-6 macro-F1 of seed-0 replays, frozen in `config.yaml: revision` before any revision run is test-scored. | `scripts/revision_grid.py`; grid runs are tagged `revision_grid:*`. | `test_revision.py::test_runs_to_score_filters_tag_unfinished_and_scored` (grid runs excluded from scoring) |

**Honest caveat.** The revision itself (drift score, E_drift, new ablations)
was designed *after* the step-5 test results were seen, in response to the
faculty review. No setting was tuned on days 9-10, but the choice of *which*
new variants to build was made knowing that E had not won. The original step-5
result stands unchanged as the primary pre-registered comparison.
