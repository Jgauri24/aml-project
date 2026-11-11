# Milestone 5: seasonal naive development findings

## Objective and implementation

Establish reproducible point baselines for the wider seasonal naive, Kalman,
LightGBM and DeepAR comparison. `SeasonalNaiveForecaster` implements the existing
`BaseForecaster`; fresh instances fit independently per series and origin.
The CLI delegates all prediction validation, actual joins and metrics to the
shared development engine. No new dependencies or data-preparation steps are required.

## Data and evaluation boundaries

Actual canonical-data development evaluation was executed for 2 models,
240 selected series,
origins [1801, 1829, 1857, 1885], and horizon 28.
Each model produces 26,880 forecasts.
The canonical Parquet reader is filtered through d_1913.
Final targets d_1914–d_1941
were not loaded, evaluated, scored or plotted. Cohort, seed, YAML, metrics and
split definitions remain unchanged.

## Model definitions and missing values

- **Last value:** y_hat(t+h) = y(s), where s is the latest finite observation
  at or before t. With no finite observations, forecast zero.
- **Weekly seasonal:** y_hat(t+h) = y(t-6+((h-1) mod 7)) for complete history.
  Freeze these seven calendar positions for the entire horizon. An unavailable
  position searches earlier observations on that weekday, then uses the latest
  overall observation, then zero if all history is unavailable.
- Zero sales remain valid. Nonfinite observations are unavailable; negative
  finite sales are rejected. Missing calendar rows retain calendar alignment.
  Each model records source days and fallback policies in `fit_summary.csv`.
- Both models return NaN q10/q50/q90. Pinball metrics remain NaN; no uncertainty
  intervals are fabricated.

## Actual development accuracy

| Model | Origin / scope | MAE | RMSE | WAPE ratio |
| --- | --- | ---: | ---: | ---: |
| naive_last_value | d_1801 | 1.231399 | 2.663493 | 1.039703 |
| naive_last_value | d_1829 | 1.252827 | 2.327917 | 1.038998 |
| naive_last_value | d_1857 | 1.250893 | 2.399467 | 1.071374 |
| naive_last_value | d_1885 | 1.256399 | 2.651678 | 0.986678 |
| naive_last_value | pooled | 1.247879 | 2.515066 | 1.033205 |
| naive_weekly_seasonal | d_1801 | 1.145982 | 2.524434 | 0.967584 |
| naive_weekly_seasonal | d_1829 | 1.183482 | 2.484446 | 0.981488 |
| naive_weekly_seasonal | d_1857 | 1.167560 | 2.566381 | 1.000000 |
| naive_weekly_seasonal | d_1885 | 1.250744 | 2.617791 | 0.982237 |
| naive_weekly_seasonal | pooled | 1.186942 | 2.548744 | 0.982751 |

naive_weekly_seasonal has the lower pooled development WAPE (0.982751). Selection uses pooled WAPE only; WAPE is a ratio.
Metrics pool observed series-days, rather than averaging per-series errors.
All-missing actuals and zero-denominator WAPE remain unavailable.

The weekly model preserves the last observed weekday pattern, while the last-value
model propagates a single day's level. Their comparison describes this cohort
and these development windows; it does not establish future or holdout performance.
Sparse sales and one unusual recent week can make either baseline inaccurate.

## Fallback audit

Counts cover all series/origin fits. Weekly counts refer to the seven frozen
positions per fit, rather than each repeated forecast day. Both sets of state
diagnostics are computed for every variant.

- `last_value`: last-value policies {'origin_observation': 960}; weekly fallback positions {'weekly_earlier_same_weekday': 0, 'weekly_latest_overall': 0, 'weekly_all_missing_zero': 0}.
- `weekly_seasonal`: last-value policies {'origin_observation': 960}; weekly fallback positions {'weekly_earlier_same_weekday': 0, 'weekly_latest_overall': 0, 'weekly_all_missing_zero': 0}.

## Reproducible figures and artifacts

Representatives are the alphabetically first existing selected series in each
volume group, using selection metadata fixed by d_1801:
`{"high": "FOODS_1_004__TX_2", "low": "FOODS_1_038__WI_2", "medium": "FOODS_1_026__CA_2"}`.
Each forecast plot uses the same first development origin d_1801,
the last 56 available training days and the configured forecast horizon.
Representatives are chosen independently of forecast errors.

- [m5_last_value_low.png](figures/m5_last_value_low.png)
- [m5_weekly_seasonal_low.png](figures/m5_weekly_seasonal_low.png)
- [m5_comparison_low.png](figures/m5_comparison_low.png)
- [m5_last_value_medium.png](figures/m5_last_value_medium.png)
- [m5_weekly_seasonal_medium.png](figures/m5_weekly_seasonal_medium.png)
- [m5_comparison_medium.png](figures/m5_comparison_medium.png)
- [m5_last_value_high.png](figures/m5_last_value_high.png)
- [m5_weekly_seasonal_high.png](figures/m5_weekly_seasonal_high.png)
- [m5_comparison_high.png](figures/m5_comparison_high.png)
- [m5_wape_by_origin.png](figures/m5_wape_by_origin.png)

Evaluator artifacts and source/fallback diagnostics: `data/processed/seasonal_naive/<variant>/`.
Comparison: `reports/m5_baseline_comparison.csv` (per-origin and pooled rows).
Integrity checks: `<variant>/integrity.json`.

## Reproduction and automated coverage

```bash
source .venv/bin/activate
python -m pytest tests/test_seasonal_naive.py -v
python -m pytest
python -m scripts.smoke_test
python -m scripts.run_seasonal_naive
```

Synthetic tests cover exact 28-day patterns, calendar gaps, zero/missing and
nonfinite observations, weekday/overall/zero fallbacks, schema and origin
checks, deterministic outputs, input immutability, factory isolation, shared
scoring, saved artifacts, CLI orchestration and exclusion of held-out targets.
The smoke check independently requires all four original raw CSVs, even though
these baselines consume the existing canonical panel. Its setup status must be
reported separately from the development evaluation.

All four required original raw CSVs are present and nonempty.

### Verification recorded on 2026-10-24

- `.venv/bin/python -m pytest tests/test_seasonal_naive.py -v`: **31 passed**.
- `.venv/bin/python -m pytest`: **173 passed**, with 12 existing Kalman
  all-zero-window fallback warnings.
- `.venv/bin/python -m scripts.run_seasonal_naive`: **passed**; both variants,
  53,760 total predictions, all four origins and ten figures generated.
- Independent saved-artifact checks: **passed** for exact cohort/origin/horizon
  coverage, dates, direct historical forecast values, evaluator actual joins,
  shared per-origin/pooled metrics, unavailable quantiles and final-holdout exclusion.
- `.venv/bin/python -m scripts.smoke_test`: **passed** after the user supplied
  `data/raw/sales_train_validation.csv`. Imports and required files
  passed; the shared dummy produced 26,880 × 10 predictions across 240 series
  and four folds, with maximum target day 1913 and `final_holdout_evaluated=false`.
- The supplied validation CSV's header was checked through d_1913. No prior
  validation-file source hash exists in the canonical panel metadata, so no
  historical byte-for-byte source match is claimed.
- `uv pip check --python .venv/bin/python` and `git diff --check`: **passed**.

Milestone 5 model implementation, development evaluation and repository setup
verification are complete.

## Limitations and completion checklist

Observed sales need not equal unconstrained customer demand. Neither model
infers stockouts, learns covariates or estimates uncertainty. Pooled errors can
be dominated by higher-volume series. No final-holdout conclusions are made.

- [x] Both deterministic adapters and fresh-fold factories implemented.
- [x] Explicit, recorded missing-observation policies; zeros preserved.
- [x] Shared backtester and configured development splits used unchanged.
- [x] Canonical-data evaluation and exact-grid integrity checks executed.
- [x] Per-origin/pooled metrics, model comparison and reproducible figures saved.
- [x] Synthetic tests and reproducible verification commands supplied.
- [x] Reserved final holdout excluded from this milestone.
- [x] Full repository setup smoke passed with all required original raw files present.

## Kalman compatibility verification before branch push

On 2026-10-24, the full suite passed again: **173 tests, zero failures, zero
skips**. The setup smoke and dependency checks also passed. Both Kalman runners
were then executed on the unchanged canonical cohort and four development folds:

```bash
.venv/bin/python -m pytest --junitxml=/tmp/aml-prepush-tests.xml
.venv/bin/python -m scripts.smoke_test
.venv/bin/python -m scripts.run_kalman_single_sku --output-dir verification_pre_push/kalman_single_sku
.venv/bin/python -m scripts.run_kalman_forecaster \
  --variants local_level local_level_seasonal local_trend local_trend_seasonal --n-jobs -1
```

All five Kalman runs completed successfully. Independent saved-artifact checks
confirmed the exact forecast grids, finite nonnegative points and ordered
quantiles, evaluator-only actual joins, shared metrics and configuration hashes.
Each run produced **26,880 forecasts** with maximum target day **1913** and
`final_holdout_evaluated=false`.

| Model | Verified pooled development WAPE |
| --- | ---: |
| kalman_single_sku | 0.837512342 |
| kalman_local_level | 0.836776033 |
| kalman_local_level_seasonal | 0.828557292 |
| kalman_local_trend | 0.846587418 |
| kalman_local_trend_seasonal | 0.830463711 |

These reproduce the recorded Kalman development results. Each multi-Kalman
variant records one documented all-zero-window fallback, with the other 959
series/origin fits converged. No Kalman source changes were needed. Generated
processed artifacts remain local and ignored by Git.
