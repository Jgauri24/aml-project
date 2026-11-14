# Retail demand forecasting

A reproducible comparison of **weekly seasonal naive, seasonal Kalman,
global LightGBM and probabilistic DeepAR** on 240 Walmart M5 product-store
series. Forecast horizon: 28 days. Seed: 42.

## Run the dashboard

```bash
uv venv .venv --python 3.14.6
uv pip install --python .venv/bin/python -r requirements_ui.txt
.venv/bin/python -m streamlit run app.py
```

The packaged demonstration needs no raw dataset or model training. Select a
store and SKU, inspect history and forecasts, toggle real P10–P90 intervals,
and compare metrics, inference cost and demand segments. It shows historical
research forecasts, not current retail predictions.

## Final research outputs

- [IEEE research paper](artifacts/paper/AML_IEEE_Final_Report.pdf), with [LaTeX source](artifacts/paper/main.tex).
- [Editable research presentation](artifacts/presentation/Retail_Demand_Forecasting_Final_Reviewed.pptx), focused on milestones 10–16.
- [Final and development benchmark](artifacts/release/benchmark.csv).
- [Frozen model settings](configs/final_models.yaml) and [freeze manifest](artifacts/release/freeze.json).
- [Single final evaluation receipt](artifacts/release/final/receipt.json).
- [Executed uncertainty notebook](notebooks/uncertainty_analysis.ipynb).
- [Complete reproduction instructions](docs/REPRODUCING.md).

## Evaluation contract

The same 240 series use four development origins (1801, 1829, 1857, 1885),
each with 28 target days: 26,880 forecasts per model. A separate frozen final
run uses origin 1913 and targets 1914–1941: 6,720 forecasts per model. The final
window has now been evaluated once and its recorded outputs are committed.
The development backtester still rejects that reserved window.

Models receive only history, static identities and allowed future calendar
features. All predictions share exact series/origin/horizon/date keys. Actuals
join after forecasts exist. Missing observations never count as genuine zeros
in scoring. Only Kalman and DeepAR supply quantiles.

Earlier baseline results pool four origins; the earlier DeepAR architecture
ablation uses **only origin 1801**. The report and slides preserve this
scope difference. Earlier and final scores use different target windows and
are not a causal before/after improvement estimate.

## Reproduce experiments

Download the four original M5 CSVs into `data/raw/`, then follow
[docs/REPRODUCING.md](docs/REPRODUCING.md). The baseline environment prepares
canonical data. `requirements/release.txt` defines the custom-PyTorch,
LightGBM and viewer environment. GluonTS is optional and uses a separate
environment because its pandas requirements differ.

```bash
.venv/bin/python -m src.data.preprocess
.venv/bin/python -m scripts.smoke_test
# After installing the release environment:
.venv-release/bin/python -m scripts.test_all
```

Published evaluation artifacts cannot be silently overwritten. The guide
shows how to create an independent replication with its own freeze and receipt.
Raw CSVs, virtual environments and build intermediates are excluded from Git.

## Repository structure

```text
app.py                       Streamlit forecast viewer
configs/                     Experiment and frozen model settings
src/data/                    Canonical panel and origin-safe preparation
src/features/                Leakage-safe direct-horizon features
src/models/                  Naive, Kalman, LightGBM and custom PyTorch DeepAR
src/evaluation/              Shared backtester, validation and metrics
src/analysis/                Conditional uncertainty and demand analysis
src/release/                 Freeze and guarded final evaluation
src/ui/                      Read-only dashboard data access
scripts/                     Reproduction and packaging entry points
notebooks/                   EDA and executed uncertainty analysis
artifacts/release/           Checkpoints, forecasts, metrics and provenance
artifacts/paper/             Final IEEE paper, source and vector figures
artifacts/presentation/      Final editable slide deck and numerical evidence
reports/                     Earlier milestone evidence and analysis tables
docs/                        Setup, methodology and project handoff
```

## Interpretation

These models forecast observed sales. The dataset does not identify inventory,
lost sales or verified stockouts. Sparse-demand labels are descriptive.
Unseen-item cold starts and measured inventory-cost benefits remain untested.
Coverage must be read alongside interval width, quantile loss and discrete ties.

The [development history](docs/archive/DEVELOPMENT_HISTORY.md) preserves earlier
milestone commands and reported results. It describes the earlier state when
the holdout was still reserved; current release status is documented above.
