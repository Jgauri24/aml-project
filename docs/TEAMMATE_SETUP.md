# Clone, set up and reproduce

These commands target the verified **macOS ARM64 / Python 3.14.6** baseline. Linux/Windows and GPU training have not been executed in this review. Install Git and `uv` first; if using a different platform, verify its package/runtime availability rather than replacing the established dependency pins silently.

## Clone and build a separate environment

```bash
git clone --branch eda https://github.com/Jgauri24/aml-project.git
cd aml-project
uv venv .venv --python 3.14.6
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
python --version
python -m pytest
```

GitHub credentials/repository permission are required if access is restricted. Use your own authorized account; the original developer's SSH alias `github-mittalayushh` is machine-local and is not needed by teammates. The HTTPS clone above does not depend on it. A uv-created venv does not necessarily contain pip: use `uv pip --python ...` as shown rather than assuming `python -m pip` exists.

## Supply existing local M5 data

The five original CSVs are ignored by Git and therefore **not present in a fresh clone**. Obtain the already-downloaded, authorized M5 files from your local/team copy; do not expect notebooks to download data.

Replace `/path/to/existing/m5` below with the folder containing the original CSVs:

```bash
cp /path/to/existing/m5/calendar.csv data/raw/
cp /path/to/existing/m5/sell_prices.csv data/raw/
cp /path/to/existing/m5/sales_train_validation.csv data/raw/
cp /path/to/existing/m5/sales_train_evaluation.csv data/raw/
cp /path/to/existing/m5/sample_submission.csv data/raw/
```

Before preparation, `python -m scripts.smoke_test` intentionally fails with the missing processed-file names and preparation instructions. The unit/integration tests use synthetic fixtures and can run without M5 data.

## Choose one artifact preparation route

**Preferred for model collaborators:** copy the existing canonical artifacts byte-for-byte, preserving the same `experiment.yaml` and original raw CSVs:

```bash
cp /path/to/prepared/data/interim/selected_skus.csv data/interim/
cp /path/to/prepared/data/interim/selection_metadata.json data/interim/
cp /path/to/prepared/data/processed/demand_panel.parquet data/processed/
cp /path/to/prepared/data/processed/panel_metadata.json data/processed/
python -m scripts.smoke_test
```

`load_panel` rejects a changed config or selected CSV hash. Panel metadata also records source hashes for a source-integrity comparison; the smoke checks required-file presence and artifact config/selection provenance, not a fresh hash scan of all raw CSVs. Verify raw hashes against metadata when obtaining data from a different copy. Do not edit CSVs or recreate model-specific cohorts.

**If canonical artifacts are unavailable:** run the existing deterministic preparation command once in the **baseline** environment:

```bash
python -m src.data.preprocess
python -m scripts.smoke_test
```

This invokes the established selection and preprocessing pipeline. It uses evaluation sales only, never concatenates overlapping histories, and creates the expected 240 × 1,941 canonical panel. Do not run preprocessing from the optional GluonTS/pandas-2 environment; that environment is for consuming the shared panel and developing adapters.

## Reproduce earlier reports only when needed

The following existing commands regenerate M0–M4 outputs; model developers do not need to rerun them to integrate an adapter:

```bash
python -m src.data.understanding
python -m src.data.eda
python -m src.evaluation.backtesting
python scripts/execute_notebooks.py
python -m pytest --junitxml=/tmp/aml-tests.xml
```

Preprocessing must already have run (or artifacts must have been copied) before EDA/backtesting. The understanding/notebook commands can rewrite selection and reports; avoid running them concurrently with other development tasks. Keep unchanged artifacts from the same seed, data and dependency versions. Timestamps and timing metadata vary normally.

## Readiness and example integration

One command checks readiness and runs all four development folds without output writes:

```bash
python -m scripts.smoke_test
```

Expected summary: `status=passed`, 240 series, 4 folds, 26,880 prediction rows, 10 columns, maximum target day 1913, `final_holdout_evaluated=false`. A missing dependency/file, stale artifact or backtest contract violation exits nonzero. The smoke never fixes problems by changing the experiment or rebuilding data.

Run the minimal example separately if desired:

```bash
python -m examples.constant_forecaster --output-dir integration_example/run_001
```

See [MODEL_INTEGRATION.md](MODEL_INTEGRATION.md) for adapters/factories/output schemas and [DEPENDENCIES.md](DEPENDENCIES.md) for isolated model environments. Start your own branch after the integration-preparation changes are available:

```bash
git switch -c model/your-model
```

Use a distinct output directory per model/run. Keep the final d_1914–d_1941 holdout reserved and the development origins unchanged. Unit tests cannot substitute for reviewing an adapter's fitted preprocessing, feature availability and inner tuning choices.
