# Model integration contract

Use the existing data and evaluation code as the common boundary for model development. The seed, cohort, chronological windows, metric definitions and final-holdout reservation remain those in `experiment.yaml` and [the experiment protocol](../reports/experiment_protocol.md). Kalman and LightGBM adapters now use this contract; DeepAR remains a later milestone.

## Start here

From the repository root, in the baseline environment:

```bash
python -m pytest
python -m scripts.smoke_test
python -m examples.constant_forecaster --output-dir integration_example
```

The smoke command checks project imports, the four raw CSVs and four canonical preparation artifacts, then runs the existing dummy over all development folds **in memory**. It neither downloads/rebuilds data nor writes forecasts. With the established configuration it reports 240 series, four folds and 26,880 × 10 predictions; no target exceeds d_1913. Missing/stale artifacts cause a nonzero exit with a useful error. See [teammate setup](TEAMMATE_SETUP.md) if starting from a clone and [dependency compatibility](DEPENDENCIES.md) before installing model libraries.

## BaseForecaster and model factories

Import the shared types from `src.evaluation.backtesting`:

```python
from functools import partial
from src.evaluation.backtesting import BaseForecaster, ForecastInputs, run_backtest

# Your adapter subclasses BaseForecaster and implements:
# fit(self, inputs: ForecastInputs)
# predict(self, inputs: ForecastInputs) -> pandas.DataFrame

factory = partial(YourForecaster, seed=42)  # illustrative future adapter
predictions, metrics, execution = run_backtest(
    panel, cfg, factory, output_dir="your_model/run_001"
)
```

## M7 independent Kalman variants

`src.models.kalman_forecaster.KalmanForecaster` implements four independent
per-series state-space variants: `local_level`, `local_level_seasonal`,
`local_trend`, and `local_trend_seasonal`. It subclasses `BaseForecaster` and
returns the unchanged `PREDICTION_COLUMNS` schema with finite q10/q50/q90;
`y_pred == q50`. Every series is fitted independently within each fold—there
is no pooling, hierarchy, or state transfer across series or folds.

Use `make_kalman_factory(variant, n_jobs=N)` with the shared runner. The
development runner saves normal evaluator artifacts plus a per-series,
per-origin `fit_summary.csv`:

```bash
python -m scripts.run_kalman_forecaster --variants local_level local_level_seasonal --n-jobs -1
python -m scripts.run_kalman_forecaster --variants local_trend local_trend_seasonal --n-jobs -1
```

Outputs are under `data/processed/kalman_forecaster/<variant>/`. The command
loads only through `final_origin`, uses the existing `run_backtest`, and does
not evaluate the final holdout.

`factory` takes **no arguments** and must return a **fresh BaseForecaster instance** each time. Do not pass a fitted object, an estimator class that does not implement this contract, or a lambda returning a shared singleton. Constructor parameters belong in a closure or `functools.partial`. The engine invokes the factory once per development origin, then calls `fit(inputs)` followed by `predict(inputs)`. The return from `fit` is ignored; returning `self` is conventional.

A global-model adapter owns one global model per fold. A per-series adapter can own its collection of 240 independently fitted filters inside that one fold-local adapter. Both produce the same forecast table. Old fold models are released rather than retained by the engine; weak references still detect reuse of a live model. Model code must release any external GPU/cache/global references it creates itself.

All learned scalers, categorical encoders, feature selectors, likelihood parameters and model weights must be fitted **inside fit**, from the allowed historical table. Do not prefit on the complete panel or use the scored forecast window for early stopping. Hyperparameter tuning must remain within available history or a separately specified inner chronological procedure. Later adapters must seed their own library RNGs from the configured seed; the engine does not seed arbitrary third-party libraries for them.

## ForecastInputs

The mutable dataclass contains these fields:

| Field | Contents | Model use |
| --- | --- | --- |
| `history` | One row per series/day from d_1 through `origin`, with observed `units_sold`, static metadata, historical prices, calendar/SNAP fields and past-only quality flags. | Fit model state and causal lag/rolling transformations here. A missing target stays missing and needs an explicit adapter-local policy. |
| `static` | One row per series: `series_id, item_id, store_id, dept_id, cat_id, state_id`. | Encode with a fold-fitted mapping if necessary. |
| `future_known` | Requested series/date grid and the explicit known-calendar allowlist below. | Generate exactly these forecast keys; there are no target or actual-price columns. |
| `origin` | Last observed day number available for fitting. | First target day is `origin + 1`. |
| `horizon` | Number of future days requested (28 in this experiment). | Distinct from each output row's offset `horizon=1…28`. |

`future_known` has these exact columns:

```text
series_id, origin, horizon, target_day_num, target_date,
wm_yr_wk, weekday, wday, month, year,
event_name_1, event_type_1, event_name_2, event_type_2
```

Future dates/weekdays and announced calendar events are assumed known. Future true sales, actual prices, SNAP indicators, quality flags and inventory availability are excluded. Do not load them independently from raw CSVs or the canonical panel inside an adapter. Unknown new panel columns do not automatically become model features.

`history` contains categorical columns, nullable booleans and float32 targets/prices; it is not a ready-made numeric feature matrix. Only the adapter should convert these to library-specific inputs. Event nulls, missing prices and missing targets have different meanings. Do not confuse categorical codes with meaningful ordinal quantities, and do not fit a global encoder/scaler using future targets. For autoregressive prediction, use past observations and recursively generated predictions, never true future sales in a rolling statistic.

Inputs are separate copies from evaluation actuals. Treat them as read-only and make local feature copies. They are not a security sandbox: the engine cannot stop a model from directly reading forbidden files. `construct_inputs()` is the shared boundary; do not fork it for a particular model.

## Required forecast schema

Return a pandas DataFrame with one row per requested series/horizon, and these ten columns (`PREDICTION_COLUMNS`):

| Column | Meaning / requirement |
| --- | --- |
| `model` | Stable nonmissing model name, identical across the run's folds. |
| `series_id` | Canonical item/store ID exactly matching the requested grid. |
| `origin` | The fold origin day number; repeat for every forecast in the fold. |
| `horizon` | Integer offset from 1 to 28, not 28 on every row. |
| `target_day_num` | `origin + horizon`, aligned to calendar day. |
| `target_date` | Daily pandas timestamp matching the requested date; retain its datetime dtype. |
| `y_pred` | Finite, nonnegative point forecast for every requested row, including those with missing evaluation actuals. |
| `q10`, `q50`, `q90` | All NA for point-only models, or all finite/nonnegative/ordered `q10 <= q50 <= q90` over the complete output. |

Use `inputs.future_known[FUTURE_KEYS].copy()` to preserve the requested identifiers and dates. The validator requires exact key/date coverage and uniqueness, and rejects negative/nonfinite/nullable-missing points, crossed quantiles and partial quantile coverage. Output order may differ: actuals are joined by keys, never by row position. Return numeric predictions and the shared column order to keep saved artifacts straightforward. Additional columns are not part of the saved common schema.

Do not attach an `actual` column. The evaluator joins it **after** validating forecasts, retaining raw forecasts and scored forecasts separately. Do not invent quantiles for a deterministic model. A probabilistic adapter may later derive genuine predictive quantiles from its output distribution/samples; no such adapter is implemented yet. `y_pred` can represent a chosen mean or median and need not equal q50; document the point summary in the adapter.

## Executable minimal example

[`examples/constant_forecaster.py`](../examples/constant_forecaster.py) implements the entire interface without learning target values. It predicts a fixed value (default 1.0), returns NA quantiles, and rejects prediction before fitting the requested origin. It is an integration example only, not the seasonal naive baseline or a model-comparison result.

```bash
python -m examples.constant_forecaster --value 1 --output-dir integration_example/run_001
python -m pytest tests/test_model_integration.py
```

The zero-argument example factory is `partial(ExampleConstantForecaster, value=1.0)`. The integration test checks a synthetic three-series/two-fold run, saved evaluator-only actuals and unchanged input panel. Additional checks cover shuffled output alignment, quantile scoring through the whole engine, invalid schemas/factories/names and fold-model lifecycle.

## Load canonical data and call the shared engine

```python
from functools import partial
from src.config import load_config
from src.data.preprocess import load_panel
from src.evaluation.backtesting import run_backtest
from examples.constant_forecaster import ExampleConstantForecaster

cfg = load_config()
panel = load_panel(cfg, cutoff=cfg["forecast"]["final_origin"])
factory = partial(ExampleConstantForecaster, value=1.0)
predictions, metrics, execution = run_backtest(
    panel, cfg, factory, output_dir="example_constant/run_001"
)
```

`load_panel` checks configuration and selected-file provenance. Do not regenerate selection for a model branch. All comparisons must use the same canonical cohort and experiment YAML. Relative output directories are placed under `data/processed/`; absolute output paths are supported. With no `output_dir`, evaluation is in-memory only.

The original dummy command remains `python -m src.evaluation.backtesting`. The engine saves `predictions.parquet`, `scored_predictions.parquet`, `metrics.csv` and `execution_metadata.json` when a directory is supplied. Metrics are pooled MAE/RMSE/WAPE and pinball losses, both per fold and across all folds. Missing actuals are excluded; all-missing subsets and zero-denominator WAPE are NA. Point-only pinball scores are NA. See the unchanged metric definitions in the [protocol](../reports/experiment_protocol.md).

| Origin | Development targets |
| --- | --- |
| 1801 | 1802–1829 |
| 1829 | 1830–1857 |
| 1857 | 1858–1885 |
| 1885 | 1886–1913 |

The engine excludes final-holdout rows before constructing any development fold. **d_1914–d_1941 remains reserved**; do not call a private/final evaluation path, change the origins, or use holdout scores to choose models.

## Parallel ownership and review checklist

The M8/M9 direct-horizon implementation trains on historical origin blocks,
with real horizons 1–28 and exactly the same feature availability as prediction.
Prices and historical SNAP are frozen at each origin; future actual prices,
SNAP and sales remain excluded. Missing event values are not positive flags.

Run the complete model tests in the optional LightGBM environment:

```bash
.venv-lightgbm/bin/python -m pytest
.venv-lightgbm/bin/python -m scripts.run_lightgbm_forecaster
```

Each LightGBM fold saves `models/origin_<day>/model.txt` and `state.json` under
its output directory. The JSON includes category mappings, frozen origin
features, recent historical sales, trained horizon, parameters and a booster
checksum. Reloading does not refit encoders or read raw actuals:

```python
from src.models.lightgbm_forecaster import LightGBMForecaster
model = LightGBMForecaster.load("data/processed/lightgbm/models/origin_1801")
prediction = model.predict(inputs)  # inputs must request this fitted origin
```

The model rejects a different origin or a horizon beyond the one it trained on.
The base test suite explicitly skips optional LightGBM tests when that package
is absent; the dedicated environment runs all model checks. See the updated
[feature report](../reports/m8_feature_engineering.md) and
[LightGBM findings](../reports/m9_lightgbm_findings.md) for training cadence,
tuning boundaries and artifact schemas.

Develop separate adapters/tests on separate branches, each implementing the shared interface. Keep optional imports inside the adapter module so the baseline smoke/test suite never needs LightGBM, torch or GluonTS. Use a separate virtual environment for each dependency stack; see [compatibility notes](DEPENDENCIES.md).

Use unique output directories by model and run, for example `lightgbm/run_001` vs `deepar/run_001`. Never let simultaneous runs write the same directory or regenerate shared selection/panel artifacts. Adapters can read the same immutable canonical artifacts. Keep model-specific tuning settings separate from the established experiment YAML until a common change is agreed.

Before integrating an adapter, run all existing tests plus its own tests, the smoke command, and a development-only backtest. Review feature availability, inner tuning boundaries, library-specific seeds, missing-history behavior, categorical encoding and point/quantile interpretation. The default smoke uses the dummy and verifies framework/data readiness; it does not prove that a future adapter is correctly implemented or scientifically useful.
