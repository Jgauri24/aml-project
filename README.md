# Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models

Milestones **0–4** implement data understanding, cutoff-safe EDA, a reproducible daily panel and a shared development backtester. Milestone **5** adds last-value and weekly seasonal naive point baselines. Milestones **6–7** add independent per-series Kalman development models. The wider comparison includes these baselines, per-series Kalman, global LightGBM and probabilistic DeepAR; naive and Kalman development evaluations are documented below.

## Setup

Tested on macOS ARM64 with Python **3.14.6**; pinned dependencies are in `requirements.txt`. Use Python 3.14 to reproduce the tested environment. No GPU, deep-learning package or external data download is needed.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

With `uv`, the equivalent setup is:

```bash
uv venv .venv --python 3.14.6
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

The existing CSVs belong in `data/raw/`: `calendar.csv`, `sell_prices.csv`, `sales_train_validation.csv`, `sales_train_evaluation.csv`, `sample_submission.csv`. Do not concatenate the two overlapping sales files; evaluation is the sole canonical source. Raw CSVs and generated datasets are excluded from Git.

## Reproduce from the repository root

```bash
# M1: shapes, schemas, missingness, keys, overlap comparison, dictionary and selection
python -m src.data.understanding
# M3: deterministic selection + complete canonical panel + provenance/validation
python -m src.data.preprocess
# M2: EDA limited to d_1801 or earlier, regenerating figures/tables/findings
python -m src.data.eda
# M4: infrastructure-only constant-zero development backtest; never final holdout
python -m src.evaluation.backtesting
# M5: last-value and weekly seasonal naive baselines, reports and figures
python -m scripts.run_seasonal_naive
# M6 verified local-level baseline
python -m scripts.run_kalman_single_sku
# M7 independent variants; run two batches or all four together
python -m scripts.run_kalman_forecaster --variants local_level local_level_seasonal --n-jobs -1
python -m scripts.run_kalman_forecaster --variants local_trend local_trend_seasonal --n-jobs -1
python -m scripts.run_kalman_forecaster --variants local_level local_level_seasonal local_trend local_trend_seasonal --n-jobs -1
# M7 tests, including synthetic weekly-seasonality gate
python -m pytest tests/test_kalman_forecaster.py tests/test_kalman_seasonal_gate.py
# M8: build leakage-safe feature matrices for all backtest folds
python -m scripts.build_features
# M8 tests
python -m pytest tests/test_feature_engineering.py
# M9: train and evaluate LightGBM forecaster with rolling-validation tuning
python -m scripts.run_lightgbm_forecaster
# M9 tests
python -m pytest tests/test_lightgbm_forecaster.py
# Execute and save outputs in both notebooks using this environment's kernel
python scripts/execute_notebooks.py
# Unit, synthetic integration and leakage tests
python -m pytest --junitxml=reports/pytest_results.xml
```

The executable EDA consumes the canonical panel, so preprocessing precedes the EDA command even though the analysis is Milestone 2. All data modules accept `--config experiment.yaml`. The root YAML controls seed, cohort, cutoffs, horizon, availability and EDA settings; alternate YAML files may live in `configs/`. Relative paths always resolve from the repository root. If configuration or selection changes, rerun preprocessing; stale panel loads fail explicitly.

For IDE notebook execution, select `.venv/bin/python` as the kernel. The notebooks find the repository root themselves. Running `scripts/execute_notebooks.py` sets the kernel executable explicitly and saves executed outputs. The two notebooks call the same reusable implementations as the CLI, rather than duplicating pipeline logic.

## Model integration and teammate readiness

See [the model integration contract](docs/MODEL_INTEGRATION.md) for `BaseForecaster`, model factories, origin-safe inputs, prediction validation and isolated outputs. [Teammate setup](docs/TEAMMATE_SETUP.md) gives clone/data/artifact/reproduction steps. [Dependency compatibility](docs/DEPENDENCIES.md) records verified LightGBM/PyTorch/GluonTS setups; GluonTS needs a separate pandas-2 environment, and LightGBM on this Mac requires OpenMP. Baseline dependencies and experiment settings remain unchanged.

Run one read-only readiness command from the repository root:

```bash
python -m scripts.smoke_test
```

It checks imports and local data/artifacts, then runs the existing dummy over all development folds (26,880 predictions). It does not rebuild data, write results or evaluate the final holdout. A minimal constant integration example is available with `python -m examples.constant_forecaster --output-dir integration_example/run_001`; it is not an experimental baseline.

## Fixed experiment boundaries

- One item/store pair is one series. Seed **42**, **240 series**: 80 from each trailing-365-day volume tercile, selected using history through **d_1801 only**, with store/category diversity and no minimum positive-sales rule.
- EDA: **d_1–d_1801**, 2011-01-29–2016-01-03. No future target values affect selection or analysis.
- Development origins: **1801, 1829, 1857, 1885**, each with horizon **28**; evaluation targets end at **d_1913**.
- Final holdout: **d_1914–d_1941**, 2016-04-25–2016-05-22; stored in the canonical panel and mechanically checked, never analyzed, tuned on or development-scored.
- Forecast inputs contain history through origin, static metadata and an allowlist of known future calendar fields. Future actual sales, prices, SNAP and quality/availability flags are excluded. Actuals are attached only inside the evaluator.
- Pooled MAE/RMSE/WAPE and q=0.1/0.5/0.9 pinball losses. WAPE is a ratio; zero denominator is NA. No quantiles are fabricated for point-only models.

## Structure and outputs

```text
experiment.yaml                  Single experiment configuration
configs/                         Optional alternate configuration files
src/config.py                    Configuration checks and path handling
src/data/load_data.py            Schema-aware raw readers; early series filtering
src/data/select_skus.py           Deterministic cutoff-safe cohort selection
src/data/understanding.py         Source audits and dataset dictionary
src/data/preprocess.py            Wide-to-long panel, validated joins, past-only flags
src/data/validate.py              Panel integrity and missingness validation
src/data/eda.py                   Reproducible cutoff-safe statistics and plots
src/evaluation/splits.py          Expanding chronological windows; final reservation
src/evaluation/metrics.py         Explicit missingness and pooled metric definitions
src/evaluation/backtesting.py    Input boundary, BaseForecaster, factory and dummy runner
src/features/feature_engineering.py Leakage-safe feature builder and dataset exporter
src/models/lightgbm_forecaster.py Direct multi-horizon LightGBM forecaster
scripts/build_features.py         CLI runner for feature dataset creation
scripts/run_lightgbm_forecaster.py CLI runner for LightGBM training and tuning
notebooks/                       Executed dataset-understanding and EDA notebooks
scripts/execute_notebooks.py      Execute both notebooks with the active interpreter
tests/                           Synthetic unit/integration/leakage tests
reports/                         Findings, protocol, audits, test results and figures
data/raw/                        Existing original CSVs (ignored)
data/interim/selected_skus.csv    240 IDs, static fields and cutoff-safe selection stats
data/interim/selection_metadata.json
data/processed/demand_panel.parquet     465,840 × 28 canonical daily panel
data/processed/panel_metadata.json      Hashes, dtypes and validation results
data/processed/features/                M8 leakage-safe feature tables & metadata
data/processed/lightgbm/                M9 LightGBM predictions, metrics, feature importance
data/processed/dummy_backtest/          Forecasts, scored forecasts, metrics, execution metadata
```

Read [dataset dictionary](reports/dataset_dictionary.md), [EDA findings](reports/eda_findings.md), [experiment protocol](reports/experiment_protocol.md), [M8 feature report](reports/m8_feature_engineering.md), [M9 LightGBM findings](reports/m9_lightgbm_findings.md), and [implementation handoff](reports/implementation_handoff.md) for evidence, assumptions and learning questions. Reports and notebooks contain actual executed results; raw and processed data remain local.

Zero sales do not establish stockouts; missing prices do not establish unavailability. Price and calendar associations are descriptive, not causal. Models will predict observed sales unless a later protocol explicitly addresses latent demand.

## M5 naive baselines

`src/models/seasonal_naive.py` implements `SeasonalNaiveForecaster(BaseForecaster)`
with two independent point-forecast variants:

- `last_value` (`naive_last_value`): for every horizon h, predict y(s), where s
  is the latest finite observed sales day at or before the origin t.
- `weekly_seasonal` (`naive_weekly_seasonal`): predict
  y(t - 6 + ((h - 1) mod 7)), repeating the final seven **calendar days** unchanged
  across all 28 forecast days. Missing positions search backward on the same
  weekday, then use the latest overall finite observation.

Both models predict zero if no finite observations exist. Genuine zero sales
remain valid; nonfinite observations are unavailable and finite negative sales
are rejected. Calendar gaps preserve weekday alignment. Per-series/origin source
days and fallback counts/policies are recorded in `fit_summary.csv`. These models
estimate no uncertainty: q10/q50/q90 and pinball losses remain NaN.

Run from the repository root with the existing baseline environment:

```bash
source .venv/bin/activate
python -m scripts.run_seasonal_naive
python -m pytest tests/test_seasonal_naive.py -v
python -m pytest
python -m scripts.smoke_test

# Optional: choose a variant, alternate YAML or separate artifact directory.
python -m scripts.run_seasonal_naive --config experiment.yaml \
  --models weekly_seasonal --output-dir seasonal_naive_custom
```

The CLI loads the existing canonical panel through d_1913 using `load_panel`,
creates a fresh model for each fold and uses the shared backtester, metrics and
prediction validator. It preserves the seed, cohort and all experiment settings.
It neither reads nor scores reserved d_1914–d_1941 targets. For the standard
experiment, each variant produces 26,880 forecasts (240 series × 4 origins × 28 days).

Outputs:

- `data/processed/seasonal_naive/{last_value,weekly_seasonal}/`: shared
  `predictions.parquet`, `scored_predictions.parquet`, `metrics.csv`, and
  `execution_metadata.json`, plus `fit_summary.csv` and `integrity.json`.
- [Comparison CSV](reports/m5_baseline_comparison.csv): per-origin and pooled
  MAE/RMSE/WAPE; WAPE remains a ratio. Lowest pooled WAPE determines the winner;
  exact ties remain ties.
- [M5 findings](reports/m5_seasonal_naive_findings.md): actual development
  results, model interpretation, fallback audit and completion checklist.
- `reports/figures/m5_*.png`: last-value, weekly and combined forecast views for
  one representative per volume band, plus WAPE by origin. Representatives are
  chosen alphabetically within existing cutoff-safe cohort bands; plots use the
  first development origin, 56 training days and 28 development forecast days.

These are common reference points for Kalman, LightGBM and DeepAR under the same
evaluation contract. Observed sales need not equal unconstrained demand; zeros
do not establish stockouts. No final-holdout performance is claimed.

The current development evaluation can run from the existing canonical artifacts.
The separate setup smoke also requires all five original M5 CSVs in `data/raw/`;
if it names a missing file, supply that original file and rerun the smoke. With a
uv-created environment, use `uv pip install --python .venv/bin/python -r requirements.txt`
if dependencies need restoring, since that environment may not include pip.

## M7 Kalman status

M7 provides four `BaseForecaster` variants, all independently estimated for
each item/store series: `local_level`, `local_level_seasonal`, `local_trend`,
and `local_trend_seasonal`. The completed development run uses the four fixed
origins and does not score `d_1914–d_1941`.

With the pre-stated rule “lowest pooled development WAPE; then q50 pinball and
combined q10/q90 pinball only for an exact WAPE tie”, the verified winner is
`kalman_local_level_seasonal` (WAPE `0.828557`). It improves on the
nonseasonal local-level variant (`0.836776`); the two trend variants do not
win the same criterion. These are development-only findings, not final
holdout results.

Per-variant `predictions.parquet`, `scored_predictions.parquet`, `metrics.csv`,
`execution_metadata.json`, and `fit_summary.csv` are in
`data/processed/kalman_forecaster/<variant>/`. The M7 comparison CSVs and
figures are under `reports/` and `reports/figures/`. Milestone 5 now supplies
compatible naive baselines for comparison. Keep the reserved final holdout
unused while implementing the remaining global-model comparisons.

## M8 Feature Engineering status

Milestone 8 implements leakage-safe feature engineering for tabular gradient boosting models.
The module `src.features.feature_engineering.FeatureEngineer` processes `ForecastInputs`
and generates 26 predictor columns:
- **Lags**: 1, 7, 14, 28 days (for multi-step forecasting, unknown future observations are set to NaN)
- **Rolling statistics**: 7, 14, 28-day means and standard deviations
- **Calendar & events**: day of week, day of month, week of year, month, year, weekend flag, named event indicators
- **SNAP indicator**: state-specific SNAP benefits (conservatively 0 in future predictions)
- **Price features**: last known shelf price and 7-day price change ratio
- **Categorical encodings**: integer mappings for `item_id`, `store_id`, `dept_id`, `cat_id`, `state_id`

Feature matrices for all four development folds are exported to `data/processed/features/` with full provenance metadata in `feature_metadata.json`. Detailed findings and analysis are documented in [M8 feature report](reports/m8_feature_engineering.md).

## M9 LightGBM status

Milestone 9 implements a direct multi-horizon gradient boosting forecaster using LightGBM (`src.models.lightgbm_forecaster.LightGBMForecaster`).
Trained with L1 regression objective and rolling-validation hyperparameter selection across folds 1–3, the selected model (`num_leaves=127, learning_rate=0.03, n_estimators=700`) outperforms all statistical and heuristic baselines:

| Model | Model Family | Pooled MAE | Pooled RMSE | Pooled WAPE |
| :--- | :--- | :---: | :---: | :---: |
| **LightGBM (Tuned)** | **Global GBDT** | **0.9874** | **2.3995** | **0.8176** |
| `kalman_local_level_seasonal` | State-Space Filter | 1.0007 | 2.0535 | 0.8286 |
| `kalman_local_trend_seasonal` | State-Space Filter | 1.0030 | 2.0576 | 0.8305 |
| `kalman_local_level` | State-Space Filter | 1.0106 | 2.0613 | 0.8368 |
| `naive_weekly_seasonal` | Heuristic Baseline | 1.1869 | 2.5487 | 0.9828 |
| `naive_last_value` | Heuristic Baseline | 1.2479 | 2.5151 | 1.0332 |

Outputs and artifacts are saved in `data/processed/lightgbm/` (`predictions.parquet`, `scored_predictions.parquet`, `metrics.csv`, `feature_importance.csv`, `comparison_metrics.csv`). Comprehensive findings are documented in [M9 LightGBM findings](reports/m9_lightgbm_findings.md).


