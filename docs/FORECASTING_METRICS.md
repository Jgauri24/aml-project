# Forecasting metrics

`src/evaluation/metrics.py` contains reusable metric functions and aggregation
helpers. Actuals are joined by the evaluation layer after forecasts are
validated. All point and probabilistic scores exclude rows with missing actuals;
a missing/nonfinite forecast for an observed actual is an error. Empty observed
groups return NA.

## Definitions and conventions

- **MAE**: `mean(|y - forecast|)`, in units.
- **RMSE**: `sqrt(mean((y - forecast)^2))`, in units.
- **WAPE**: `sum(|y - forecast|) / sum(|y|)`, kept as a ratio to match existing
  project reports. A zero actual-demand denominator returns NA, including a
  perfect all-zero forecast.
- **sMAPE**: `mean(200 * |y - forecast| / (|y| + |forecast|))`, reported as a
  percentage in the 0–200 range. A row with both actual and forecast equal to
  zero contributes zero; an empty group returns NA.
- **MASE**: mean absolute forecast error divided by the mean absolute seasonal
  naive error on training observations only. For daily retail demand the default
  seasonal period is 7: `mean(|y_t - y_(t-7)|)` over finite training pairs. The
  training window must retain calendar gaps as missing positions. Insufficient
  pairs or a zero scale returns NA. Backtesting attaches the scale from each
  fold's exact expanding or sliding training window and saves it with scored
  predictions; evaluation targets are never used for scaling.
- **Pinball loss** at quantile `q`: `mean(max(q*(y-p_q), (q-1)*(y-p_q)))` for
  q=0.1, 0.5, and 0.9. Point-only models receive NA for probabilistic metrics.
- **80% interval coverage**: fraction of observed actuals in `[q10, q90]`.
  **Average interval width** is `mean(q90-q10)`. Coverage is a proportion.
- **Weighted interval score (WIS)** uses the median and the central 80% interval:
  `(0.5*|y-q50| + 0.1*IS_0.2) / 1.5`, where
  `IS_alpha = (q90-q10) + (2/alpha)*(q10-y)*I(y<q10) +
  (2/alpha)*(y-q90)*I(y>q90)`. It is NA unless all ordered q10/q50/q90 values
  are present.

Forecasts must provide all three finite, nonnegative, ordered quantiles or leave
all three unavailable. Partial quantiles and crossed quantiles are rejected.

## Aggregation and export

`compute_metric_tables` returns portfolio, SKU/store, store, forecast-origin,
horizon-day, and (when supplied) demand-segment tables. Demand segments may be
provided as a `series_id` mapping or a table with `series_id` plus
`demand_segment` (or the project's `volume_group`). `series_metadata` supplies
`series_id`, `item_id`, and `store_id`. `save_metric_tables` writes one CSV or
Parquet file per aggregation level, e.g. `metrics_overall.csv`.

The shared backtester also adds sMAPE, MASE, interval coverage, width, and WIS to
its existing per-origin and pooled `metrics.csv`. Its `scored_predictions.parquet`
contains the fold-local `mase_scale` used for reproducibility. The small
four-model demonstration calculates these metrics on synthetic model forecasts;
it is an integration check, not a real-data benchmark.
