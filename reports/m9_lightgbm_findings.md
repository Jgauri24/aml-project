# Milestone 9: repaired direct-horizon LightGBM

Verified on **2026-10-25** after correcting the M8/M9 audit findings. These
measurements replace the earlier findings whose price/event features and
all-missing training horizon did not satisfy the forecasting contract.

## Training and availability

One global L1 LightGBM model per development fold learns from all 240 selected
series. The model uses 26 base features plus a real integer `horizon`.
Historical origins are T-28, T-56, ... >=1, with disjoint 28-day label blocks
ending at T. Every observed label contributes once; earliest history provides
lag/rolling context. Every horizon 1–28 occurs in training.

For an example at historical origin s and horizon h, target calendar/events are
known ahead, while rolling sales, listed prices and historical SNAP are frozen
at s. A lag requiring observations after s is NaN even if that value appears
later in the offline historical table. Missing events are absent flags. Missing
labels are excluded and zero labels retained. The one-step compatibility API
also excludes the label day's actual price and SNAP.

This is direct multi-horizon prediction: no future actuals or recursive actual
updates are used. The standard experiment, raw data, cohort and seed remain
unchanged. Feature schema version 2 and origin/key columns make training
provenance auditable. See [M8 feature engineering](m8_feature_engineering.md).

## Rolling tuning

Three candidates were evaluated using the shared scorer on development origins
1801, 1829 and 1857. Selection now uses **pooled WAPE** over their observed
series-days, rather than an unweighted mean of fold ratios. Exact ties retain
the first candidate. The final development fold at origin 1885 is excluded from
parameter selection; the reserved final holdout is never loaded or scored.

| Candidate | num_leaves | learning_rate | n_estimators | Pooled tuning WAPE |
| --- | --- | --- | --- | --- |
| 1 | 31 | 0.05 | 500 | 0.765382 |
| 2 | 63 | 0.05 | 500 | 0.766221 |
| 3 | 127 | 0.03 | 700 | 0.768866 |

Selected: `num_leaves=31`, `learning_rate=0.05`,
`n_estimators=500`. Tree count is selected by chronological
validation. There is no misleading early-stopping callback on the training set.

## Actual development results

| Origin / scope | MAE | RMSE | WAPE ratio |
| --- | --- | --- | --- |
| d_1801 | 0.882133 | 2.210259 | 0.744809 |
| d_1829 | 0.930557 | 2.025056 | 0.771732 |
| d_1857 | 0.910340 | 2.024095 | 0.779695 |
| d_1885 | 0.967912 | 2.166537 | 0.760123 |
| pooled | 0.922736 | 2.108136 | 0.763996 |

Each model produces 26,880 finite nonnegative forecasts (240 series × four
origins × 28 days). WAPE is a ratio. Quantiles and pinball losses are unavailable
for this point-only model. The first three folds also contributed to selection;
their pooled score is a development comparison, not an independent estimate of
generalization. The last development fold is separate from tuning.

## Read-only baseline comparison

| Model | MAE | RMSE | WAPE ratio |
| --- | --- | --- | --- |
| lightgbm | 0.922736 | 2.108136 | 0.763996 |
| kalman_local_level_seasonal | 1.000711 | 2.053497 | 0.828557 |
| kalman_local_trend_seasonal | 1.003014 | 2.057588 | 0.830464 |
| kalman_local_level | 1.010637 | 2.061347 | 0.836776 |
| kalman_single_sku | 1.011527 | 2.077604 | 0.837512 |
| kalman_local_trend | 1.022487 | 2.097607 | 0.846587 |
| naive_weekly_seasonal | 1.186942 | 2.548744 | 0.982751 |
| naive_last_value | 1.247879 | 2.515066 | 1.033205 |

Baseline artifacts use the same configuration and development windows. Naive
and Kalman models were not changed or refitted for this repair. The measured
comparison does not establish final-holdout performance or causal effects.

## Feature importance

| Feature | Mean gain |
| --- | --- |
| rmean_28 | 452992.462 |
| rstd_28 | 413502.182 |
| rmean_14 | 334923.403 |
| last_known_price | 260829.630 |
| rstd_14 | 232965.425 |
| dayofweek | 219326.539 |
| rmean_7 | 85945.990 |
| item_id | 73657.149 |
| rstd_7 | 73508.827 |
| is_weekend | 35916.700 |
| price_change_ratio | 28399.472 |
| store_id | 18604.050 |
| weekofyear | 17837.115 |
| dayofmonth | 16093.566 |
| horizon | 8298.261 |

Gain is averaged across the four fitted folds. It describes model split usage,
not a causal influence or a proof of demand drivers. A genuine horizon feature
may be redundant with calendar features for this fixed origin cadence; its
presence and valid training range are independently verified.

## Saved models and provenance

`data/processed/lightgbm/` contains:

- `predictions.parquet`, `scored_predictions.parquet`, `metrics.csv`.
- `feature_importance.csv`, `fit_summary.csv`, `comparison_metrics.csv`.
- `tuning_results.json`: candidate parameters, per-fold and pooled shared metrics,
  selection origins and timings. Unavailable metrics are JSON null.
- `execution_metadata.json`: resolved/selected parameters, seed, package versions,
  strategy, feature schema, configuration hash, model paths and complete timings.
- `models/origin_1801/`, `origin_1829/`, `origin_1857/`, `origin_1885/`: native
  `model.txt` plus `state.json` with encoders, frozen origin features, recent
  historical sales, trained horizon and booster checksum.

All four model bundles were loaded in a separate process without fitting. Each
reproduced its complete saved 6,720-row forecast to 1e-12 tolerance. State/schema
and model checksum guards reject mismatched bundles; origin/horizon guards
prevent use of the wrong fitted state. No pickle or additional dependency is needed.

Measured total wall time **239.423 seconds**, including
**211.300 seconds** tuning and
**28.107 seconds** backtesting/model persistence.
The JSON is written after these fields are populated, so saved metadata matches
the returned object. Timings are machine/load dependent.

## Reproduce and load

```bash
.venv/bin/python -m scripts.build_features
.venv-lightgbm/bin/python -m scripts.run_lightgbm_forecaster
.venv-lightgbm/bin/python -m pytest
.venv/bin/python -m scripts.smoke_test
```

```python
from src.models.lightgbm_forecaster import LightGBMForecaster
model = LightGBMForecaster.load("data/processed/lightgbm/models/origin_1801")
predictions = model.predict(inputs)  # same fitted origin and trained horizon
```

The complete model-enabled suite passes **231 tests**, with no skips or expected
failures. The baseline environment passes **211 tests**, explicitly skipping
optional LightGBM checks. Existing Kalman all-zero-window warnings are unchanged.
Real-data verification checks exact keys/dates, nonnegative points, NaN quantiles,
actual joins, shared metrics, training label boundaries and lag availability.
All saved native boosters report observed horizons 1–28.

## Completion and limitations

- [x] Historical origin/horizon supervised training and matching inference.
- [x] Correct price/SNAP availability and missing-event handling.
- [x] Shared rolling-origin scoring and recorded tuning without final holdout.
- [x] Complete metrics/feature-importance comparison against all available baselines.
- [x] Native fitted models and complete feature states saved and reloaded.
- [x] Enriched tuning/timing metadata persisted and tested.
- [x] All three original audit regressions pass as ordinary tests.

M8 and M9 are complete under the supplied checklist. M0's FilterPy manifest and
M6's explanatory notebook remain separate outstanding deliverables. These models
predict observed sales, not unconstrained latent demand; zeros do not establish
stockouts. No uncertainty intervals or final-holdout conclusions are claimed.
