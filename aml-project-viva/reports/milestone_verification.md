# Milestone verification — 2026-10-25

Current `main` working tree, including the repaired M8/M9 implementation and the
four-input data cleanup. Machine-readable evidence and historical audit checks
are in [milestone_verification.json](milestone_verification.json).

## Current status

| Milestone | Status | Evidence / remaining work |
| --- | --- | --- |
| 0 — Environment | Partial | Fresh baseline installation and smoke pass. Optional LightGBM/PyTorch/GluonTS/FilterPy environments work. Default baseline pytest now handles optional LightGBM tests correctly. FilterPy 1.4.5 still lacks a pinned dependency manifest. |
| 1 — Understanding | Complete | Notebook executed; four source schemas, keys and pre-holdout overlap audit verified; deterministic 240-series cohort retained. |
| 2 — EDA | Complete | EDA notebook executed; five findings, eight figures and volume/intermittency segmentation reproduced through d_1801 only. |
| 3 — Preprocessing | Complete | Reusable pipeline and canonical Parquet exist. 459,120 development rows validate: 240 series × 1,913 complete daily rows, no duplicate/missing indexed rows. |
| 4 — Experiment | Complete | Shared 28-day/four-origin scoring, forecast schema, metrics and leakage tests pass; final reservation unchanged. |
| 5 — Naive baselines | Complete | Both models produce 26,880 exact-grid forecasts with passing tests and documented metrics/figures. |
| 6 — Single-SKU Kalman | Partial | Filtering, missing-observation behavior, forecasts, intervals and tests work. Requested `06_kalman_single_sku.ipynb` remains missing. |
| 7 — All-SKU Kalman | Complete | Four variants' saved forecasts/quantiles/metrics independently validated; synthetic tests pass. Source unchanged by the M8/M9 repair. |
| 8 — Features | Complete | Origin-safe prices, missing-event encoding, historical SNAP and direct origin/horizon features pass regression, parity and real-data checks. Version-2 matrices regenerated. |
| 9 — LightGBM | Complete | Genuine direct-horizon training, chronological pooled-WAPE tuning, full shared backtest, saved native models/feature state and persisted enriched metadata verified. |

## Repaired contracts

The three original audit failures are now ordinary regression tests, with no
expected-failure markers:

1. A training label day's actual price cannot change that example's features.
   One-step prices stop at t-1; direct examples freeze prices at their historical
   origin. Price ratios use seven calendar days, including missing listings.
2. Missing/blank event fields encode as absent, rather than positive flags.
3. M9 trains with observed horizons 1–28 from historical origin blocks, instead
   of an all-NaN training horizon. Sales lags requiring later observations remain
   NaN, even when those later values exist elsewhere in the offline history.

New checks independently compare direct training features with prediction
features built at each historical origin, mutate later sales/prices/SNAP, verify
missing labels and forecast row order, and round-trip saved feature/model states.

The runner also saves all four fitted boosters and their complete inference
state. Tuning records, resolved parameters, feature/package versions, artifact
paths and total/tuning/backtest times are persisted after the run completes.
Training-set early stopping has been removed: chronological tuning selects tree
count. The baseline comparison now includes all eight available models.

## Actual verification

| Check | Result |
| --- | --- |
| Complete LightGBM-environment suite | **231 passed**, zero skips/expected failures |
| Baseline suite without LightGBM | **211 passed, 2 explicit optional skips** |
| Setup smoke | Passed: 18 imports, eight required files, 240 series, four folds, 26,880 × 10 forecasts |
| Version-2 feature export | Four folds regenerated; training rows 430,080 / 436,800 / 443,520 / 450,240 |
| Real-data LightGBM tuning | Three candidates, first three development origins only |
| Full repaired LightGBM backtest | 26,880 predictions; exact keys/dates, nonnegative points and unavailable quantiles verified |
| Saved native model round trips | All four reloads reproduce their complete 6,720-row forecasts to 1e-12 tolerance |
| Training artifacts | Historical labels, observed horizons 1–28, frozen lag availability and feature schema verified |
| Metadata | Returned/saved equality tested; real selected candidate/tuning scope/timers/artifact paths checked |
| Final holdout | Not loaded or scored; maximum forecast target remains d_1913 |

Both full test runs emit the same 12 documented Kalman all-zero-window fallback
warnings. Optional baseline skips do not replace model testing: the dedicated
LightGBM environment runs the full suite without skips.

The repaired real-data model selected `num_leaves=31`, `learning_rate=0.05`,
`n_estimators=500`. Pooled development MAE is **0.922736**, RMSE **2.108136**,
and WAPE ratio **0.763996**. Last development fold WAPE is **0.760123**; that fold
was excluded from tuning. The first three folds also contributed to selection,
so pooled development results are not an independent generalization estimate.
See [M9 findings](m9_lightgbm_findings.md) for all candidates, folds, comparisons,
feature importance and complete artifact paths. Previous M8/M9 measurements are
superseded because their feature/training contracts were defective.

## Reproduction

```bash
.venv/bin/python -m scripts.build_features
.venv-lightgbm/bin/python -m scripts.run_lightgbm_forecaster
.venv-lightgbm/bin/python -m pytest --junitxml=/tmp/aml-m8-m9-full.xml
.venv/bin/python -m pytest --junitxml=/tmp/aml-m8-m9-baseline.xml
.venv/bin/python -m scripts.smoke_test
```

Raw data, cohort, experiment YAML and forecast splits remain unchanged. Both
existing notebooks were executed during the initial audit, and saved Kalman
outputs were independently revalidated; they were not unnecessarily refitted
for this feature/model repair. The Streamlit app remains unchanged and passed
syntax compilation; UI runtime is outside the supplied M0–M9 checklist.

## Input cleanup

The unused competition template and tracked duplicate are removed, together
with their loading/auditing branches, fixture generation, notebook outputs and
documentation references. Readiness requires four analytical CSVs and four
canonical preparation artifacts. The original stash remains intact.

The only remaining checklist gaps are M0's FilterPy dependency manifest and M6's
explanatory notebook; neither is part of this M8/M9 repair request.
