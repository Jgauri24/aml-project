# Rolling-origin backtesting

The shared engine is `src/evaluation/backtesting.py`; splits and training-window
boundaries are defined in `src/evaluation/splits.py`. It forecasts each configured
28-day target window from the calendar in the canonical panel. The current M5
origins (d_1801, d_1829, d_1857, d_1885) are retained and correspond to the
existing calendar dates. Forecast origins are day numbers mapped to the panel's
actual `date` column; future dates must be consecutive calendar days.

`experiment.yaml` configures `forecast.horizon`, explicit
`forecast.development_origins`, and the `backtest` options:

- `window_mode: expanding` starts at d_1 and includes all observations through
  the origin.
- `window_mode: sliding` requires `training_window_days` and gives the model only
  the final fixed number of historical days through the origin.
- `step_size` controls spacing when origins are generated. To generate origins,
  set `forecast.development_origins: null`, provide `backtest.origin_start`, and
  the engine steps forward up to the last origin whose horizon ends by
  `forecast.final_origin`. Existing explicit origins remain authoritative.
- `sku_ids` filters `item_id`; `store_ids` filters `store_id`. Null means all
  series already present in the supplied panel. Filters are intersected.

The engine gives models a fresh fold-local `ForecastInputs` with historical
rows only through the origin and an explicit allowlist of known future calendar
fields. It verifies complete training days, validates the forecast key grid,
and joins actuals only after forecasts are validated. The final reserved window
is removed before fold construction. The evaluator cannot prevent model code
from independently reading external future data, so model implementations must
continue to honor the input contract.

When `output_dir` is provided, the standard prediction and scored-prediction
artifacts are retained. The engine also writes `backtesting_results.csv` and
`backtesting_results.parquet` with `sku_id`, `store_id`, forecast-origin date and
day number, forecast date, horizon day, actual sales, model name, and point
forecast. `execution_metadata.json` records window settings, selected series,
and origin dates.

Run the independent synthetic demonstration with:

```bash
python -m examples.backtesting_demo
```

Run a small one-origin integration check for all four model adapters with:

```bash
python -m examples.model_suite_demo
```

The demo uses two synthetic SKU/store series, one 28-day forecast, and reduced
LightGBM/DeepAR settings. It is an interface and grid check, not a model quality
benchmark.

For a reusable suite run, `src.evaluation.model_suite.run_model_suite` accepts
a label-to-factory mapping and optionally writes one subdirectory per model plus
`all_model_predictions.parquet`, `all_model_metrics.csv`, and
`suite_metadata.json` at the suite output root.
