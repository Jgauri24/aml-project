# Inventory Demand Forecasting via Kalman Filtering and Global DeepAR Models

Milestones **0–4** implement data understanding, cutoff-safe EDA, a reproducible daily panel and a shared development backtester. Milestones **6–7** add independent per-series Kalman development models. The intended wider comparison remains seasonal naive, per-series Kalman, global LightGBM and probabilistic DeepAR; only the Kalman work described below has been run here.

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
# M6 verified local-level baseline
python -m scripts.run_kalman_single_sku
# M7 independent variants; run two batches or all four together
python -m scripts.run_kalman_forecaster --variants local_level local_level_seasonal --n-jobs -1
python -m scripts.run_kalman_forecaster --variants local_trend local_trend_seasonal --n-jobs -1
python -m scripts.run_kalman_forecaster --variants local_level local_level_seasonal local_trend local_trend_seasonal --n-jobs -1
# M7 tests, including synthetic weekly-seasonality gate
python -m pytest tests/test_kalman_forecaster.py tests/test_kalman_seasonal_gate.py
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
notebooks/                       Executed dataset-understanding and EDA notebooks
scripts/execute_notebooks.py      Execute both notebooks with the active interpreter
tests/                           Synthetic unit/integration/leakage tests
reports/                         Findings, protocol, audits, test results and figures
data/raw/                        Existing original CSVs (ignored)
data/interim/selected_skus.csv    240 IDs, static fields and cutoff-safe selection stats
data/interim/selection_metadata.json
data/processed/demand_panel.parquet     465,840 × 28 canonical daily panel
data/processed/panel_metadata.json      Hashes, dtypes and validation results
data/processed/dummy_backtest/          Forecasts, scored forecasts, metrics, execution metadata
```

Read [dataset dictionary](reports/dataset_dictionary.md), [EDA findings](reports/eda_findings.md), [experiment protocol](reports/experiment_protocol.md), and [implementation handoff](reports/implementation_handoff.md) for evidence, assumptions and learning questions. Reports and notebooks contain actual executed results; raw and processed data remain local.

Zero sales do not establish stockouts; missing prices do not establish unavailability. Price and calendar associations are descriptive, not causal. Models will predict observed sales unless a later protocol explicitly addresses latent demand.

The next implementation task is **Milestone 5: a weekly seasonal naive forecaster behind BaseForecaster**, evaluated only on the four development origins.

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
figures are under `reports/` and `reports/figures/`. The next recommended task
is to implement and evaluate the compatible weekly seasonal-naive baseline;
do not use the reserved final holdout for model selection.
