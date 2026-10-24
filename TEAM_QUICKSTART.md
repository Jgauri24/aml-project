# Inventory Demand Forecasting — Team Quickstart

Forecast daily Walmart product-store sales over 28 days and compare seasonal naive, local Kalman filtering, LightGBM, and DeepAR.

**Status:** Milestones 0–9 (environment, dataset exploration, EDA, preprocessing, development backtesting, seasonal naive baselines, single/multi Kalman filters, leakage-safe feature engineering, and LightGBM gradient boosting) are implemented. Probabilistic deep learning (DeepAR) starts at Milestone 10.

## 1. Clone and install

Use Python **3.14.6** to reproduce the current data pipeline's tested environment (check compatibility separately before installing ML/deep-learning model libraries).

```bash
git clone --branch feat-engineering https://github.com/Jgauri24/aml-project.git
cd aml-project
python3.14 -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 2. Download the dataset

**M5 Forecasting Accuracy:** https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Download the competition dataset (Kaggle login / rules acceptance may be required). Extract these files into `data/raw/`:

- `calendar.csv`
- `sell_prices.csv`
- `sales_train_validation.csv`
- `sales_train_evaluation.csv`
- `sample_submission.csv`

Do **not** combine the two sales files: their histories overlap. Raw data is not committed to Git.

## 3. Generate model-ready data

Run from the repository root, with the environment activated:

```bash
python -m src.data.understanding          # Audit input data and select 240 series
python -m src.data.preprocess             # Generate cleaned daily panel
python -m src.data.eda                    # Rebuild EDA outputs (optional)
python -m src.evaluation.backtesting      # Dummy-model development backtest
python -m scripts.build_features          # Build M8 leakage-safe feature matrices
python -m scripts.run_lightgbm_forecaster # Train & evaluate M9 LightGBM model
python -m pytest                          # Check tests
```

Main outputs:

- `data/interim/selected_skus.csv` — selected item-store series
- `data/processed/demand_panel.parquet` — model-ready daily panel
- `data/processed/panel_metadata.json` — data validation and provenance
- `data/processed/features/` — M8 leakage-safe feature tables and metadata
- `data/processed/lightgbm/` — M9 LightGBM predictions, metrics, and feature importance
- `reports/eda_findings.md` — descriptive analysis
- `reports/m8_feature_engineering.md` — feature engineering findings and design
- `reports/m9_lightgbm_findings.md` — LightGBM model findings and comparison
- `reports/experiment_protocol.md` — experiment rules

For notebook outputs: `python scripts/execute_notebooks.py`.

## 4. Start model development

Implement a model using the common interface in `src/evaluation/backtesting.py` (`BaseForecaster` / `ForecastInputs`). **Reuse the existing backtester and metrics** rather than creating separate evaluation splits.

- **Target:** daily observed unit sales per item-store series
- **Horizon:** 28 days
- **Development origins:** `d_1801`, `d_1829`, `d_1857`, `d_1885`
- **Final holdout:** `d_1914`–`d_1941` — do not use during development
- **Metrics:** MAE, RMSE, WAPE, and pinball loss when quantiles are available

Never use future observed sales as model inputs. Zero observed sales do not prove a stockout. Future selling prices are not automatically known.

**Before opening a model PR:** run `python -m pytest`, document model-specific dependencies, and provide reproducible train/forecast commands. Python 3.14.6 is tested for the data pipeline, **not yet confirmed for all planned model libraries**.
