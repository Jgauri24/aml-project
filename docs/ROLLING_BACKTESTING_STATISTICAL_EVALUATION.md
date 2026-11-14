# Rolling Backtesting and Statistical Evaluation

This guide describes the reusable evaluation subsystem in `src/evaluation/` and
its end-to-end runner. The evaluator is development-only: it keeps the
configured final holdout reserved and rejects requests to score it or use it
for tuning.

## 1. Rolling-origin backtesting

`splits.py` creates chronological folds. An origin is the last day whose
observations may be used for model fitting; the target dates are the following
`horizon` days. `backtesting.py` constructs a restricted `ForecastInputs`
object, checks the model forecast grid and schema, then joins actuals after
forecast validation. A forecast row is keyed by series, origin, horizon and
target date. A series is one SKU-store combination.

The default project experiment has 28-day forecasts at development origins
d_1801, d_1829, d_1857 and d_1885. The configured reserved final target period
is d_1914–d_1941. The end-to-end runner reads the origins from
`experiment.yaml`; `configs/evaluation_pipeline.yaml` can instead request
generated origins using `generate_origins`, `origin_start` and `step_size`.

Leakage controls include: training history ends at the origin; future targets
are absent from model inputs; the future covariate table is an explicit
known-calendar allowlist; historical quality flags are recomputed within each
fold; fold-specific MASE scales use training history only; duplicate keys,
wrong horizon/date grids, invalid values, and crossed/partial quantiles are
rejected. Actuals are joined only after predictions pass validation.

## 2. Expanding and sliding windows

An expanding window uses all eligible history up to each origin. A sliding
window uses only the configured trailing number of days and advances with each
origin. Select `evaluation.window_mode` as `expanding` or `sliding`; set
`training_window_days` for sliding mode. `step_size` controls generated-origin
spacing. The configured origin and window rules are recorded in the run
metadata and backtest outputs.

## 3. Model integration

All adapters implement `fit(inputs)`, `predict(inputs)` and the common
`return_forecasts(inputs)` entry point. The existing Seasonal Naive, Kalman,
LightGBM and DeepAR models are wrapped through factories in `model_suite.py`
and `pipeline.py`; their forecasting implementations remain model-owned.
Each origin gets training data through that origin and predictions for the
same SKU-store/date grid. DeepAR is fitted separately at each origin because
its current adapter requires an exact fitted origin and does not expose a safe
incremental update operation. Its P10/P50/P90 outputs are retained. The
deterministic models leave quantiles missing; no uncertainty is synthesized.

To add an adapter, implement the shared methods, return one row for each
requested series/date with `model`, identifiers, `origin`, `horizon`,
`target_day_num`, `target_date`, `y_pred`, `q10`, `q50`, `q90`, and register a
fresh-instance factory. For probabilistic outputs, provide all three finite,
nonnegative, ordered quantiles.

## 4. Metrics

`metrics.py` calculates point metrics (MAE, RMSE, WAPE, sMAPE, MASE) and
probabilistic metrics (P10/P50/P90 pinball loss, 80% interval coverage, mean
interval width and WIS when quantiles exist). It excludes missing actuals,
returns missing values for empty groups and undefined zero-denominator ratios,
and errors on invalid forecasts. WAPE is a ratio, sMAPE is a 0–200 percentage,
and MASE uses a seasonal-7 training-only scale. Full mathematical definitions
and edge-case conventions are in `FORECASTING_METRICS.md`.

Metric tables are aggregated by portfolio, SKU-store, store, forecast origin,
horizon day and demand segment. Demand segments in the central pipeline use
first-origin training means and do not depend on evaluation targets.

## 5. Statistical significance testing

`model_comparison.py` pairs forecasts only on identical series/origin/date/
horizon keys and verifies actuals match. It summarizes absolute- and
squared-loss differences and supported probabilistic losses. Pairwise loss
differences are model A minus model B; a negative value favors A.

The Diebold–Mariano procedure first averages loss differences within each
series and then across series for each origin. Thus daily SKU rows are not
treated as independent time-series replicates. It uses a Bartlett/Newey–West
long-run variance, overlap-aware lag, Harvey–Leybourne–Newbold small-sample
correction, Student-t reference and Holm adjustment. Fewer than the configured
minimum number of origins produces an explicit insufficient-origin result.
The cluster bootstrap resamples complete SKU-store series and reports a
percentile confidence interval. The assumptions and limitations are detailed
in `MODEL_COMPARISON.md`; significance should not be claimed from the smoke
example or from aggregate historical summaries without aligned row forecasts.

## 6. Computational benchmarking

Each fold records fit, prediction and total time, series count and inference
time per series. Where available, model-native size fields (DeepAR parameter
count, LightGBM tree count/serialized size, Kalman resident state estimate) are
also recorded. Timings vary by hardware, process load and dependency versions;
compare only runs made on comparable environments. Size fields have different
semantics and should not be treated as directly interchangeable.

## 7. Run the experiments

Run commands from the repository root with the project environment activated:

```bash
# Prepare/validate the processed demand panel from the project's raw inputs.
python -m src.data.preprocess

# Run the all-model synthetic integration demonstration (not retail results).
python -m examples.evaluation_pipeline_demo

# Run the configured development experiment on the processed real panel.
python -m src.evaluation.pipeline --config configs/evaluation_pipeline.yaml

# Validate evaluation contracts and metric/statistical behavior.
python3 -m pytest -q tests/test_backtesting.py tests/test_metrics.py \
  tests/test_model_comparison.py tests/test_model_suite.py \
  tests/test_data_pipeline_and_integration.py

# Run all project tests.
python3 -m pytest -q
```

The panel build requires the raw M5 data files described in the root README.
The current checkout has no `data/processed/demand_panel.parquet`, so the
configured real-data command exits with a clear missing-panel error until the
preprocessing command has valid raw inputs. The synthetic demo is useful for
interface and artifact checks only; its metrics must not be presented as
retail performance.

Each run writes to `reports/evaluation_runs/<experiment_id>/` by default. The
timestamp and configuration hash identify a run; set a caller-specified
experiment ID only when the output directory does not already exist. Avoid
changing model selection or tuning from final-holdout results.

## 8. Output files and schemas

The pipeline writes the following tree (CSV and Parquet are emitted for
dashboard metric tables and predictions when configured):

| Path | Purpose and grain | Principal fields |
| --- | --- | --- |
| `experiment_metadata.json` | Reproducibility manifest | experiment ID/time, dataset hash, config hashes, origins, horizon, window, models, seeds, metric summary, runtime, holdout policy |
| `backtest/<model>/predictions.parquet` | Unscored standardized forecasts | `model`, `series_id`, `origin`, `horizon`, `target_day_num`, `target_date`, `y_pred`, `q10`, `q50`, `q90` |
| `backtest/<model>/scored_predictions.parquet` | Model forecasts joined to evaluation outcomes | prediction schema plus `actual` and fold-local `mase_scale` |
| `backtest/<model>/metrics.csv` | Per-model pooled/per-origin scores | `model`, `scope`, `origin` (when per-fold), metric columns and observed counts |
| `backtest/<model>/backtesting_results.csv` | Analyst-facing result rows | SKU, store, forecast origin/date, horizon day, actual sales, model name, point forecast |
| `backtest/<model>/execution_metadata.json` | Per-fold fit/inference records | model, origin, fit/predict/total seconds, series count, optional size counters |
| `dashboard/overall_model_metrics.csv` | One row per model | model, scope, counts and configured metric columns |
| `dashboard/sku_store_metrics.csv` | Model × SKU-store | model, `series_id`, `item_id`, `store_id`, metrics |
| `dashboard/store_metrics.csv` | Model × store | model, `store_id`, metrics |
| `dashboard/forecast_origin_metrics.csv` | Model × origin | model, origin, metrics |
| `dashboard/horizon_metrics.csv` | Model × horizon day | model, horizon, metrics |
| `dashboard/demand_segment_metrics.csv` | Model × demand segment | model, segment, metrics |
| `dashboard/forecast_predictions.csv` | Model × series × origin × date | standardized forecast keys, point forecast, optional quantiles, actual, MASE scale, SKU/store IDs |
| `dashboard/prediction_intervals.csv` | Probabilistic forecasts only | standardized prediction fields restricted to rows with valid P10/P50/P90 |
| `dashboard/runtime_comparisons.csv` | Model × origin and model totals | fit/prediction/inference/total seconds, series count, available size counters |
| `dashboard/ranking_*.csv` and `rankings/*.csv` | Separate rankings | ranking-specific performance metric/stratum; no combined score |
| `statistics/paired_comparisons.csv` | Pair × loss | mean/median/SD difference, paired counts, missing-grid counts |
| `statistics/dm_tests.csv` | Pair × loss DM inference | statistic, p-values, HAC lag, origins, variance, status, interpretation |
| `statistics/bootstrap.csv` | Pair × metric CI | difference, CI bounds, confidence, iterations, series/row counts, zero-in-CI flag |
| `visualizations/figures/*.png` | 300-DPI plots | accuracy, horizons, SKU/store, intervals, runtime and model size where available |
| `visualizations/plotted_data/*.csv` | Reproducible figure inputs | same measured values and grouping keys used by each figure |
| `visualizations/visualization_metadata.json` | Figure inventory | figure names, plotted-data files and availability notes |

## Current executed evidence and limitations

The latest small run is
`reports/evaluation_pipeline_smoke/smoke-20261114T090728Z/`. It ran two
synthetic SKU-store series, four model adapters and one 28-day origin; all four
predicted 56 rows. It generated 14 dashboard CSVs, five ranking tables, three
statistical result tables, nine PNG figures, plotted data and metadata. These
are pipeline validation artifacts, **not** an experimental retail result.

Its actual (synthetic) overall point metrics were:

| Model | MAE | RMSE | WAPE | MASE |
| --- | ---: | ---: | ---: | ---: |
| DeepAR | 1.330357 | 1.551209 | 0.683486 | 2.240566 |
| Kalman local-level seasonal | 0.160055 | 0.279967 | 0.082230 | 0.803296 |
| LightGBM | 0.294112 | 0.467270 | 0.151103 | 1.013155 |
| Weekly seasonal naive | 0.232143 | 0.481812 | 0.119266 | 1.165094 |

These figures only confirm that outputs were calculated and written. With one
origin, DM significance is not supported; comparisons are low-sample smoke
diagnostics. No current unified retail metric table or paired retail
significance result can be reported because the validated panel is missing.
Older model-specific reports and aggregate summaries do not provide matching
row-level predictions for this pipeline and must not be combined to fabricate
such a table. The smoke run did not evaluate or tune on the holdout.

The smoke DM table marks every pair/loss comparison
`insufficient_origins` (1 observed origin versus the configured minimum of 8);
all DM statistics and p-values are therefore missing. For example, DeepAR
minus seasonal Kalman MAE difference is 1.170302, but this is a descriptive
paired loss difference only. The bootstrap table uses 100 iterations and only
two synthetic series, so its intervals demonstrate the export shape and must
not be interpreted as reliable inferential evidence.

The measured smoke fit-plus-predict totals for the two series were 0.0048 s
(weekly seasonal naive), 0.6885 s (seasonal Kalman), 0.0391 s (LightGBM) and
2.3375 s (DeepAR). The run also recorded 8 LightGBM trees / 9,443 serialized
bytes, 1,600 DeepAR trainable parameters and 2,576 estimated Kalman resident
state bytes. These figures are machine/run-specific and not directly
comparable as model-size definitions. The nine saved figures are:
`accuracy_metrics.png`, `horizon_accuracy.png`, `sku_store_analysis.png`,
`advanced_vs_seasonal_naive.png`, `store_model_mae.png`,
`probabilistic_forecasts_actual_median_intervals.png`,
`prediction_interval_coverage.png`, `prediction_interval_widths.png` and
`runtime_comparison.png`, all under the smoke run's
`visualizations/figures/` folder. The paired tables, bootstrap intervals,
runtime fields and plotted datasets are saved alongside them.

The focused evaluation/contract suite (backtesting, metrics, model comparison,
model suite and chronology/schema integration tests) completed with **66
passed**. `py_compile` also passed for the evaluation modules and demo. A full
repository pytest attempt was not clean in this checkout: the feature-dataset
builder test requires the absent `data/processed/panel_metadata.json`, and a
later native LightGBM test process terminated with a segmentation fault under
the current local runtime. These are reported separately from the passing
evaluation suite; the complete project test status therefore remains pending
on a prepared panel and a stable LightGBM runtime.

## Team handoff

### Member 4: cold-start and sparse-demand experiments

- Reuse the same `ForecastInputs`, fold origins, window configuration and
  prediction validator. Define eligibility for cold-start/sparse strata using
  training-only history at each origin; persist the cohort rule and counts.
- Keep the evaluation grid identical across compared models within each
  experiment. Do not assign segments from future actuals. Extend the panel
  metric aggregation only with explicit segment labels created from training
  data; preserve the current demand-segment output schema.
- Avoid treating SKU-day rows as independent in significance tests. Use the
  origin-level DM implementation and series-cluster bootstrap; report when
  origin/series counts are too small.

### Member 5: dashboard

- Consume the `dashboard/` tables, not model-specific intermediate files.
  Dashboard metric rows include `model`, `scope`, counts and metrics; entity
  dimensions are `series_id`/`item_id`/`store_id`, `origin`, `horizon` or
  `demand_segment` depending on the file.
- Use `forecast_predictions` for date-level exploration and
  `prediction_intervals` only for valid probabilistic rows. Quantile fields are
  null for deterministic forecasts. WAPE is a ratio, sMAPE a 0–200 percent,
  and coverage a 0–1 proportion.
- Use `runtime_comparisons` and `statistics` tables with status fields; do not
  interpret unavailable DM results as nonsignificance. Parquet is available
  for metric and forecast/interval tables where enabled; ranking/runtime and
  significance exports are CSV.

### Research report contributors

- Use real-data metric/statistics outputs only after the full configured run
  succeeds on the validated panel. Cite exact experiment ID and metadata.
- Report dataset/cohort, origins, horizon, window, model settings, metrics,
  DM aggregation/HAC/small-sample assumptions, bootstrap unit and runtime
  environment. Include paired-comparison limitations and uncertainty.
- Distinguish development results from a one-time final holdout evaluation;
  final holdout results remain pending, and must not feed model selection.
- Current executable evidence and actual dataset limitation are documented
  above. Do not report the synthetic metrics as retail findings.
