# Milestone 15: uncertainty diagnostics

Computed from saved row-level predictions and canonical observed sales. No model was retrained.

| model | n_observed | n_intervals | mae | pinball_q10 | pinball_q50 | pinball_q90 | coverage_80 | mean_interval_width_80 | lower_miss_rate | upper_miss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kalman_local_level | 26880 | 26880 | 1.0106 | 0.1315 | 0.5053 | 0.3366 | 0.8564 | 2.7936 | 0.033 | 0.1106 |
| kalman_local_level_seasonal | 26880 | 26880 | 1.0007 | 0.1318 | 0.5004 | 0.3306 | 0.8572 | 2.7626 | 0.0329 | 0.1099 |
| kalman_local_trend | 26880 | 26880 | 1.0225 | 0.1419 | 0.5112 | 0.3371 | 0.8537 | 2.7984 | 0.0369 | 0.1094 |
| kalman_local_trend_seasonal | 26880 | 26880 | 1.003 | 0.1347 | 0.5015 | 0.3304 | 0.8543 | 2.7643 | 0.0366 | 0.1091 |
| lightgbm | 26880 | 0 | 0.9227 | NA | NA | NA | NA | NA | NA | NA |
| naive_last_value | 26880 | 0 | 1.2479 | NA | NA | NA | NA | NA | NA | NA |
| naive_weekly_seasonal | 26880 | 0 | 1.1869 | NA | NA | NA | NA | NA | NA | NA |

## Evidence and interpretation

- All included models passed the same complete series/origin/horizon/date grid check; saved actuals were replaced by canonical actuals.
- Missing actuals are excluded equally for all models; zeros are observed outcomes. Counts are exported for every group.
- Coverage uses inclusive P10–P90 endpoints. Lower and upper misses separate overprediction from underprediction.
- Count-valued demand can produce coverage above 80% even with valid quantiles. The exported strict/non-strict CDF bounds help interpret ties at zero; excess coverage alone does not prove miscalibration.
- Volume terciles and trailing zero runs are computed from each origin’s training window only. Volume ties break by series ID.
- Paired SKU MAE differences use model A minus model B: negative favors A. Ties remain ties. These are descriptive comparisons, not independent-row significance tests.
- Point-only models retain NA uncertainty metrics. No latent demand or verified stockout labels are available.
- The final holdout is excluded. Shared evaluation keys do not by themselves establish that external checkpoints were trained without leakage; inspect their training provenance.

## Available and missing evidence

Models: kalman_local_level, kalman_local_level_seasonal, kalman_local_trend, kalman_local_trend_seasonal, lightgbm, naive_last_value, naive_weekly_seasonal.
Missing standard prediction files: deepar/predictions.parquet.
DeepAR versus Kalman conclusions require genuine saved DeepAR forecasts on the same complete grid.

## Outputs

`overall.csv`, `by_origin.csv`, `by_horizon.csv`, `by_week.csv`, `by_volume.csv`,
`by_volume_week.csv`, `by_series.csv`, `paired_series.csv`, `worst_series.csv`,
`history_labels.csv`, calibration plots and a source-hash manifest.
