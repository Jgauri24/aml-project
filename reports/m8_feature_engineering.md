# Milestone 8: origin-safe forecasting features

The 2026-10-25 price-availability and missing-event audit failures are repaired.
The regression checks are ordinary passing tests, with no expected-failure markers.
Feature schema version **2** replaces the earlier one-step-only exported tables.

## Feature availability

There are 26 base features. Direct training adds an observed integer `horizon`,
so M9 fits 27 predictors. Feature encoders are fitted independently inside each
fold on `ForecastInputs.history`; the canonical cohort and YAML are unchanged.

| Feature | Availability at historical or prediction origin s |
| --- | --- |
| `lag_1`, `lag_7`, `lag_14`, `lag_28` | For target s+h, use sales at s+h-k only when h<=k; otherwise NaN, even if the value exists later in the historical training table. |
| `rmean_7/14/28`, `rstd_7/14/28` | Trailing observed sales ending at s, frozen for the whole horizon; sample standard deviation, explicit minimum observed count. |
| Calendar fields | Target-date weekday, month, year, day-of-month, ISO week, weekend and known event flags. |
| Event flags | Actual nulls, blank strings and textual null representations indicate no recorded event. None/pd.NA/NaN no longer become positive flags. |
| `last_known_price` | Most recent listed price at or before s, forward-filled only from the past. |
| `price_change_ratio` | As-of price at s divided by as-of price at s-7. The comparison is seven calendar days, not seven nonmissing price observations. Missing/invalid denominators remain NaN. |
| `snap` | Historical SNAP flag at s, frozen across the horizon. This is explicitly not the future target day's SNAP flag. |
| Static IDs | Fold-local integer codes for item/store/department/category/state; unknown levels map to -1. |

The backward-compatible `build_train_features()` one-step API uses sales,
prices and historical SNAP only through target day t-1. The direct model and
exporter use `build_direct_train_features()` instead. Unknown future columns
such as actual sales, prices and SNAP are ignored by the prediction builder.

## Historical origin/horizon examples

For fold origin T and horizon H, training origins are T-H, T-2H, ... >=1.
Each origin supplies the next H historical labels. These blocks do not overlap,
so an observed label contributes once, and their origin weekdays match the
forecast origin. The earliest history remains available for lag/rolling context.
For the standard experiment the first historical training origin is d_9.

Each training row has a real horizon 1..28, the target's known calendar, and
sales/price/SNAP summaries frozen at its historical origin. Later historical
sales are used only as labels. All training labels are at or before T. Missing
labels are omitted; zero labels are retained. A very short history can produce
a partial feature block, but M9 rejects fitting without observed labels for
every configured horizon.

Independent tests compare every direct training feature with a separately fitted
prediction builder at the same historical origin. They also mutate all later
sales, prices and SNAP and verify that earlier-origin features are unchanged.

## Regenerated datasets

Run from the repository root:

```bash
.venv/bin/python -m scripts.build_features
```

Outputs under `data/processed/features/`:

| Fold origin | Training rows | Training columns | Prediction rows | Prediction columns |
| --- | ---: | ---: | ---: | ---: |
| 1801 | 430,080 | 32 | 6,720 | 31 |
| 1829 | 436,800 | 32 | 6,720 | 31 |
| 1857 | 443,520 | 32 | 6,720 | 31 |
| 1885 | 450,240 | 32 | 6,720 | 31 |

Training files contain 27 predictors, the remaining four forecast keys and the
observed target. Prediction files contain 26 base predictors plus all five
forecast keys, including horizon and target date. The model appends that same
horizon as its 27th predictor. Keys preserve series, historical origin, horizon,
target day and target date, allowing independent alignment checks.

`feature_metadata.json` records schema version, strategy, configuration hash,
training-origin lists, horizon, feature names and actual row/column counts.
Regenerate version-1 feature tables before using them with the repaired model.
No raw preprocessing or cohort reselection is required.

## Verification

```bash
.venv-lightgbm/bin/python -m pytest tests/test_feature_engineering.py \
  tests/test_direct_horizon_features.py tests/test_milestone_contracts.py -v
```

Coverage includes current-day price exclusion, calendar-aligned price gaps,
missing events, multi-origin labels and lag availability, train/predict feature
parity, exclusion of later actuals, preserved row order, frozen historical SNAP,
missing-label handling, JSON feature-state round trips and origin validation.
The shared development horizon/cutoffs are unchanged; d_1914–d_1941 remains reserved.
