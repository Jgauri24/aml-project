# Reproduce the final forecasting project

## 1. Clone and run the packaged demonstration

```bash
git clone https://github.com/Jgauri24/aml-project.git
cd aml-project
uv venv .venv --python 3.14.6
uv pip install --python .venv/bin/python -r requirements_ui.txt
.venv/bin/python -m streamlit run app.py
```

The dashboard reads `artifacts/release/`. It displays real saved forecasts for
all 240 selected product-store series, recent historical sales, model metrics,
quantile bounds and demand segments. It requires no training, credentials or raw
dataset. Final forecasts concern April 25–May 22, 2016; they are a historical
holdout demonstration, not predictions of current retail conditions.

## 2. Prepare the original data for experiment replication

Download M5 Forecasting Accuracy from
<https://www.kaggle.com/competitions/m5-forecasting-accuracy/data> and accept the
competition's access terms if required. Put these four files in `data/raw/`:

- `calendar.csv`
- `sell_prices.csv`
- `sales_train_validation.csv`
- `sales_train_evaluation.csv`

Raw data is intentionally absent from Git. Do not concatenate the overlapping
sales histories. Build the canonical panel with the baseline pandas-3 environment:

```bash
.venv/bin/python -m src.data.preprocess
.venv/bin/python -m scripts.smoke_test
```

The fixed selection uses seed 42 and information through day 1801. All canonical
panel/selection hashes must agree before reproducing frozen experiments.

## 3. Install the all-model release environment

```bash
uv venv .venv-release --python 3.14.6
uv pip install --python .venv-release/bin/python -r requirements/release.txt
# macOS native prerequisite for LightGBM, if needed:
brew install libomp
.venv-release/bin/python -m scripts.check_model_dependencies --stack lightgbm
.venv-release/bin/python -m scripts.test_all
```

The test runner separates PyTorch and LightGBM test collection into different
processes. A monolithic pytest run on this Mac encountered a native OpenMP
segmentation fault after loading both library runtimes. The isolated suites and
the actual four-model training/evaluation runs are checked separately.

The custom DeepAR uses PyTorch directly. GluonTS is not required by this release.
Do not mix the separate pandas-2 GluonTS environment into the baseline environment.
`artifacts/release/environment.json` records the exact tested runtime versions.
The model environment uses one PyTorch thread, one LightGBM thread and four
Kalman fit workers. Timings therefore disclose different parallelization.

## 4. Inspect and independently replicate the frozen experiment

Frozen decisions live in `configs/final_models.yaml`. The committed
`artifacts/release/freeze.json` records source/configuration/dataset hashes.
Model selection used earlier development evidence. The primary model was
LightGBM. DeepAR's hidden-32 choice came from the earlier day-1801 ablation,
not an equally pooled four-origin experiment.

Published runs are protected from accidental overwrite. To perform your own
replication, choose a new empty output root in a new Python process:

```bash
.venv-release/bin/python - <<'PY'
from pathlib import Path
from src.release import evaluate as release
release.RELEASE = Path('artifacts/replication_001').resolve()
release.freeze()
release.evaluate('development')
PY

# A separate process avoids resetting the numerical runtime inside an existing one.
.venv-release/bin/python - <<'PY'
from pathlib import Path
from src.release import evaluate as release
release.RELEASE = Path('artifacts/replication_001').resolve()
release.evaluate('final')
PY
```

Each replication has its own freeze and scoring receipt. The final holdout has
already been disclosed in the published project, so a replication cannot honestly
claim new human blinding. Do not tune against its metrics or compare repeated
holdout runs to select settings. Exact floating-point/sample results may differ
across OS, library versions and CPU architectures.

For an original fresh release, the equivalent operations are:

```bash
python -m scripts.run_release freeze
python -m scripts.run_release development
python -m scripts.run_release final
```

They intentionally refuse to overwrite completed published runs. The final
runner saves all model predictions before the evaluator attaches holdout sales.
An exclusive `EVALUATION_STARTED.json` blocks automatic rescoring after scoring
begins. `receipt.json` records completion, source freeze hash and target limits.
The general development backtester still forbids final-holdout evaluation.

## 5. Artifacts and checkpoint interpretation

- `artifacts/release/{development,final}/predictions.parquet`: shared prediction schema.
- `scored_predictions.parquet`: the same forecasts joined to canonical actuals.
- `metrics.csv`, `metrics_by_week.csv`, `runtime.csv`: measured outcomes and costs.
- `models/origin_*/lightgbm/`: native booster and JSON feature state. Restore with `LightGBMForecaster.load(path)`.
- `models/origin_*/deepar/`: checkpoint, category mappings, scales and learning curve. `DeepARTrainer.load_checkpoint(path)` restores the trained network. Use `prepare_deepar_data` on matching origin-safe inputs and the stored category ordering to reconstruct inference tensors.
- `models/origin_*/{kalman,seasonal_naive}/model.joblib`: trained states. Load only trusted repository artifacts.
- `demo_history.parquet`, `segments.csv`: compact observed history and first-origin segment metadata for the viewer.
- `manifest.json`: file hashes, including predictions, models, configuration freeze and receipts.

The saved DeepAR weights support inference reconstruction; optimizer-resume state
is not included. All forecasts are nonnegative. Quantiles are genuine for Kalman
and DeepAR and unavailable for the other models.

## 6. Analysis and research outputs

```bash
.venv/bin/python -m scripts.run_uncertainty_analysis \
  --prediction artifacts/release/development/predictions.parquet \
  --output-dir reports/uncertainty_release
```

Open `notebooks/uncertainty_analysis.ipynb` for executed diagnostics. The final
paper and editable deck are under `artifacts/paper/` and
`artifacts/presentation/`. The IEEE LaTeX source references vector figures in
`artifacts/paper/figures/`. Compile with a standard TeX distribution containing
IEEEtran, for example:

```bash
cd artifacts/paper
pdflatex main.tex
pdflatex main.tex
```

`python -m scripts.package_release` rebuilds derived tables, demo history and
research figures from the saved evaluation outputs; it never retrains models or repeats the final evaluation. The document build script requires LaTeX and the packaged evidence JSON.
All numerical findings distinguish earlier development, current development and
final holdout scopes. The earlier DeepAR ablation is single-origin evidence.
