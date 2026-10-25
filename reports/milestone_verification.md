# Milestone verification — 2026-10-25

Audit of the supplied Milestone 0–9 checklist on branch `main` at `2ede2d1`.
The forecasting code on main is identical to the previously audited branch;
main additionally contains a Streamlit app and its UI requirements. The focused
cleanup was reapplied without restoring unrelated notebook kernel settings or
changing the app. The original stash remains intact.

Completion below means the requested deliverable and its checks are present;
existing code or a benchmark claim alone does not establish completion.
Machine-readable evidence is in [milestone_verification.json](milestone_verification.json).

## Completion status

| Milestone | Status | Verified evidence / remaining work |
| --- | --- | --- |
| 0 — Environment | Partial | A fresh Python 3.14.6 baseline installed all 59 pinned packages and passed the real-data smoke. LightGBM, PyTorch/GluonTS and FilterPy work in their separate environments. FilterPy 1.4.5 has no corresponding pinned dependency manifest. The baseline `python -m pytest` cannot collect the optional LightGBM tests. |
| 1 — Dataset understanding | Complete | `01_dataset_understanding.ipynb` executed; four source schemas, keys and 58,327,370 overlapping pre-holdout target cells checked. Selected cohort remains 240 series: 80 each low/medium/high. |
| 2 — EDA | Complete | `02_eda.ipynb` executed. Five findings, eight figures and volume/intermittency segmentation reproduced using only history through d_1801. |
| 3 — Preprocessing | Complete | Reusable modules and canonical Parquet exist. Development panel validated: 459,120 rows, 240 series, 1,913 daily rows per series, zero duplicate/missing indexed rows and zero missing targets. Proxy/unknown availability is explicit. |
| 4 — Experiment | Complete | Shared splits, prediction validation and MAE/RMSE/WAPE/pinball tests pass. Four origins, horizon 28 and final reservation remain unchanged. |
| 5 — Naive baselines | Complete | Both models rerun through the shared evaluator; each produces 26,880 exact-grid point forecasts with unavailable quantiles. Existing tests and plotting implementation pass. |
| 6 — Single-SKU Kalman | Partial | Filter/update mathematics, missing-observation behavior, forecasts, intervals, figures and tests work. The explicitly requested `06_kalman_single_sku.ipynb` is missing; code and CLI are not the same deliverable as that explanatory notebook. |
| 7 — All-SKU Kalman | Complete | All four variants have valid 26,880-row development outputs, quantiles, fit diagnostics and timing. Saved artifacts were independently checked again; mathematical/integration tests pass. |
| 8 — Features | Partial | All four feature datasets build, and existing tests pass. Additional checks expose target-day price use in training and incorrect missing-event flags. |
| 9 — LightGBM | Partial | Same-day tuning/backtest artifacts from identical forecasting code revalidate on main. The claimed horizon-aware training, saved fitted models and persisted tuning metadata are incomplete; M8 issues also affect this milestone. |

## Independent contract failures

These checks are now explicit strict expected failures in
`tests/test_milestone_contracts.py`. They document required fixes, not successful
milestone acceptance. An unexpected pass fails the test until its marker is
removed, so a future fix cannot silently leave the audit out of date.

1. **M8 target-day prices.** `_price_features_train()` forward-fills the price
   including the training label day's own price. Altering only that final day's
   price changes its `last_known_price` feature. A forecast made before that day
   cannot use that day's actual price under the configured conservative policy.
   Price inputs must be aligned to the historical forecast origin, as sales lags are.
2. **M8 missing events.** `_encode_calendar()` produces `[1, 1, 1, 0]` for
   `[None, pd.NA, NaN, ""]`; all four should mean no recorded event. Test actual
   missingness before string comparisons.
3. **M9 horizons.** `LightGBMForecaster.fit()` assigns NaN to every training
   horizon, while `predict()` supplies 1–28. The booster drops this all-missing
   feature. It cannot learn horizon-specific behavior. Build training examples
   for historical origins/horizons, or implement and document a consistent
   alternative multi-step strategy, then rerun model selection.

Additional M9 acceptance gaps found in the completed real run:

- No fitted booster or feature-encoder/model bundle is saved: only predictions,
  scores, metrics, feature importance, comparison and execution JSON.
- `best_params`, `wall_seconds` and `output_dir` are added to the returned metadata
  after the shared backtester has written its JSON. The saved execution file
  therefore lacks these added fields.
- The printed wall time starts after tuning, so it excludes the tuning run.
- The historical M9 findings' selected candidate and benchmark numbers did not
  reproduce in this audit. That document now carries an explicit audit notice.

## Actual development results

| Model | Pooled development WAPE ratio | Evidence in this audit |
| --- | ---: | --- |
| naive_last_value | 1.033204990 | Rerun on all four origins |
| naive_weekly_seasonal | 0.982750655 | Rerun on all four origins |
| kalman_single_sku | 0.837512342 | Saved forecasts/actual joins/metrics independently revalidated |
| kalman_local_level | 0.836776033 | Saved forecasts/actual joins/metrics independently revalidated |
| kalman_local_level_seasonal | 0.828557292 | Saved forecasts/actual joins/metrics independently revalidated |
| kalman_local_trend | 0.846587418 | Saved forecasts/actual joins/metrics independently revalidated |
| kalman_local_trend_seasonal | 0.830463711 | Saved forecasts/actual joins/metrics independently revalidated |
| lightgbm | 0.815841 | Same-day fit on identical code; saved outputs revalidated; M8/M9 contracts still require correction |

Kalman source and shared backtester are unchanged from the previously verified
full runs on 2026-10-24; they were not unnecessarily retrained during this audit.
The earlier same-day LightGBM run on identical forecasting code selected `num_leaves=63`, `learning_rate=0.05`,
`n_estimators=500`, with first-three-fold mean WAPE approximately 0.8148.
Its pooled MAE is 0.985352 and RMSE is 2.380505. These are measured values,
not a claim that its unresolved forecasting contracts have passed. The first
three folds also serve model selection, so their pooled score is not an
independent estimate of generalization.

All model outputs have 26,880 rows and maximum target day 1913. The reserved
final targets d_1914–d_1941 were not loaded or scored in this audit.

## Input cleanup

The unused competition template and its tracked duplicate were deleted.
Its loading/auditing/schema branches, synthetic fixture generation, notebook
content/output and setup/report references were removed. Readiness now needs
four original analytical CSVs plus four preparation artifacts, eight files total.
A new source-audit test and the updated smoke test exercise this smaller input
contract. Both notebooks were reexecuted and their existing kernel metadata was
preserved. The experiment configuration, selected-cohort CSV and canonical
Parquet remained byte-for-byte unchanged.

## Commands and observed outcomes

```bash
# A fresh baseline, installed from the existing local package cache.
uv venv /tmp/aml-main-audit-fresh --python .venv/bin/python
uv pip install --offline --python /tmp/aml-main-audit-fresh/bin/python -r requirements.txt
/tmp/aml-main-audit-fresh/bin/python -m scripts.smoke_test

# Full current dependency stack, including the optional LightGBM tests.
.venv-lightgbm/bin/python -m pytest --junitxml=/tmp/aml-main-milestone-audit.xml

# Targeted four-input source audit and smoke checks.
.venv/bin/python -m pytest tests/test_source_audit.py tests/test_smoke.py -v

# Feature generation rerun on main; model tuning/backtest command used earlier the same day on identical code.
.venv/bin/python -m scripts.build_features
.venv-lightgbm/bin/python -m scripts.run_lightgbm_forecaster --output-dir lightgbm

# Optional runtime checks, all passed in their own environments.
.venv-lightgbm/bin/python -m scripts.check_model_dependencies --stack lightgbm
.venv-deepar/bin/python -m scripts.check_model_dependencies --stack gluonts
```

Fresh installation, source checks, notebook execution, dependency compatibility,
feature generation, naive reruns, saved Kalman verification and LightGBM execution
all passed. The earlier same-day LightGBM-environment run passed 214 existing tests
with 12 documented Kalman all-zero-window warnings. The five targeted source/smoke
tests passed after adding the new source test. The final suite collected 218 tests: **215 passed, 3 strict expected failures,
zero unexpected failures/errors**, with 12 existing Kalman fallback warnings.
The three expected failures are the unresolved M8/M9 contracts described above;
they are not successful acceptance checks. Exact counts are also recorded in the JSON.

The baseline `.venv/bin/python -m pytest` was also actually attempted and failed
during collection because its isolated environment lacks LightGBM. This is
distinct from the model-enabled suite: use the documented dedicated environment
for the full current suite until optional-test collection is separated properly.

No claim is made that every listed milestone is complete. Remaining work is
the M0 dependency/test-entry-point cleanup, the M6 explanatory notebook, the M8
price/event fixes and the M9 training/persistence corrections, followed by new
M8/M9 benchmark runs.

Main-specific rerun: **215 passed, 3 strict expected failures**, with fresh-baseline
smoke, both notebook executions, development panel validation, both naive backtests,
feature generation and saved Kalman/LightGBM integrity checks passing. The added
Streamlit app passed syntax compilation; its UI runtime is outside this checklist.
