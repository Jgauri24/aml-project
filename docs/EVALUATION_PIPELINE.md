# End-to-end evaluation pipeline

The central development-evaluation workflow is `src/evaluation/pipeline.py`,
configured by `configs/evaluation_pipeline.yaml`. It composes the existing
validated panel loader, rolling-origin model suite, metric engine, paired model
comparison, runtime records, dashboard exports, and row-level visualizations.

## Run

After the processed panel and metadata have been built with the project data
preprocessing workflow, run from the repository root:

```bash
python -m src.evaluation.pipeline --config configs/evaluation_pipeline.yaml
```

For a small all-model smoke run:

```bash
python -m examples.evaluation_pipeline_demo
```

The smoke example uses two synthetic SKU-store series, one 28-day forecast
origin and reduced model settings. It checks output integration only and is not
a retail performance result. Production execution reads the configured
`demand_panel.parquet`; the panel loader checks its config and selected-series
hashes, and the runner applies structural validation before modeling.

## Central configuration

`evaluation_pipeline.yaml` controls the canonical panel and metadata paths,
series/item/store filters, forecast horizon and origins, expanding/sliding
training window, model-family enablement and adapter configuration, metrics,
seeds, bootstrap policy, and output root. Set `generate_origins: true` with an
`origin_start` and `step_size` to generate chronological folds. Otherwise the
configured origins from `experiment.yaml` are reused. `series_ids` filters
SKU-store combinations after panel validation. Demand segments are computed
from first-origin training means only; evaluation actuals do not affect segment
assignment.

Every fold uses the adapter's existing training behavior. A fresh model is
fitted through that fold's origin, then predicts that fold's future grid. The
DeepAR adapter enforces a matching fitted origin and exposes no safe incremental
update API, so it is fitted once per fold. It is not reused across origins.
Actuals are joined only after forecasts pass schema and date validation. MASE
scales come from each fold's exact training window. Model comparisons pair
matching series/origin/date/horizon rows and use the significance module's
origin-level DM test and series-cluster bootstrap.

## Outputs

Each run gets an experiment ID based on UTC creation time and a resolved
configuration hash (or a caller-provided ID). It records dataset SHA-256,
source metadata, model configuration, seed, origins, horizon, training window,
metric summary, runtime, and holdout policy in `experiment_metadata.json`.

`dashboard/` includes:

- `overall_model_metrics`, `sku_store_metrics`, `store_metrics`,
  `forecast_origin_metrics`, `horizon_metrics`, and `demand_segment_metrics`
  as CSV and Parquet tables.
- `forecast_predictions.parquet` and CSV, with standardized model, SKU-store,
  origin, date, horizon, actual, point forecast, quantiles, and fold-local MASE
  scale fields.
- `prediction_intervals` CSV and Parquet containing only valid P10/P50/P90 rows.
- `runtime_comparisons.csv` with per-fold and total fit, prediction, inference
  per SKU-store fold, runtime, series counts, and available model-size counters.
- Separate dashboard ranking files for point accuracy, horizon, segment,
  probabilistic score, and computational efficiency.

`statistics/` stores paired loss summaries, DM tests (with Holm-adjusted
p-values and explicit insufficient-origin statuses), and cluster-bootstrap
confidence intervals. `visualizations/figures/` stores 300-DPI figures and
`visualizations/plotted_data/` stores their CSV input datasets. The model suite's
per-adapter predictions, actuals, metrics, and backtesting results remain under
`backtest/` for traceability.

## Dashboard field definitions

| Table | Grain | Key fields |
| --- | --- | --- |
| `overall_model_metrics` | model | `model`, `mae`, `rmse`, `wape`, `smape`, `mase`, probabilistic metrics, `n_observed` |
| `sku_store_metrics` | model × SKU-store | `model`, `series_id`, `item_id`, `store_id`, metrics |
| `store_metrics` | model × store | `model`, `store_id`, metrics |
| `forecast_origin_metrics` | model × origin | `model`, `origin`, metrics |
| `horizon_metrics` | model × horizon day | `model`, `horizon`, metrics |
| `demand_segment_metrics` | model × segment | `model`, `demand_segment`, metrics |
| `forecast_predictions` | model × series × origin × target day | standardized identifiers, actual, point forecast, P10/P50/P90, MASE scale |
| `runtime_comparisons` | model × origin and total | fit/prediction/runtime seconds, series count, optional size counters |

Metric fields follow `docs/FORECASTING_METRICS.md`. WAPE is a ratio; sMAPE is a
percentage; coverage is a proportion. Missing probabilistic fields remain
missing for deterministic models.

## Holdout protocol

The pipeline is a development evaluator only. It rejects a configuration that
asks to score or tune against the reserved final holdout. Development origins
must finish by `forecast.final_origin`; its next horizon remains untouched.
After model and hyperparameter decisions are frozen using development results,
run a separate one-time final evaluation from history through the configured
final origin and score only the reserved target days. Do not feed final-holdout
metrics back into model selection or tuning.

The current workspace has no processed `data/processed/demand_panel.parquet`
or panel metadata, so the full configured retail run cannot execute until the
data-preprocessing step has produced them. The small smoke run demonstrates the
entire output workflow without claiming retail results.
