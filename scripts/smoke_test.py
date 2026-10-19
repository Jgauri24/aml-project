"""One read-only command: imports, required local files and full development dummy run."""
import argparse
import importlib
import json
import platform
from math import isnan
from pathlib import Path
import sys

# Also allow `python scripts/smoke_test.py` without requiring an editable install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROJECT_IMPORTS = [
    "numpy", "pandas", "matplotlib", "seaborn", "sklearn", "pyarrow", "yaml",
    "src.config", "src.data.load_data", "src.data.select_skus", "src.data.preprocess",
    "src.data.validate", "src.data.understanding", "src.data.eda",
    "src.evaluation.splits", "src.evaluation.metrics", "src.evaluation.backtesting",
    "examples.constant_forecaster",
]


def check_required_files(cfg):
    from src.config import data_path
    from src.data.load_data import EXPECTED_FILES
    required = [data_path(cfg, "raw") / name for name in EXPECTED_FILES]
    required.extend([
        data_path(cfg, "interim") / "selected_skus.csv",
        data_path(cfg, "interim") / "selection_metadata.json",
        data_path(cfg, "processed") / "demand_panel.parquet",
        data_path(cfg, "processed") / "panel_metadata.json",
    ])
    missing = [str(p) for p in required if not p.is_file() or p.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            "Missing or empty required files:\n" + "\n".join(missing)
            + "\nCopy the five original M5 CSVs to the configured raw directory. "
            "If processed artifacts are missing, run python -m src.data.preprocess. "
            "The smoke test does not download or rebuild data."
        )
    return len(required)


def run_smoke(cfg):
    for name in PROJECT_IMPORTS:
        importlib.import_module(name)
    from src.data.preprocess import load_panel
    from src.evaluation.backtesting import DummyForecaster, run_backtest
    from src.evaluation.splits import development_splits

    n_files = check_required_files(cfg)
    final_origin = cfg["forecast"]["final_origin"]
    # No holdout targets are returned by the guarded Parquet reader.
    panel = load_panel(cfg, cutoff=final_origin)
    n_series = cfg["selection"]["n_per_group"] * len(cfg["selection"]["groups"])
    if panel.series_id.nunique() != n_series or len(panel) != n_series * final_origin:
        raise ValueError("Development panel dimensions differ from the experiment configuration")
    folds = development_splits(cfg)
    predictions, metrics, metadata = run_backtest(panel, cfg, DummyForecaster)
    expected = n_series * len(folds) * cfg["forecast"]["horizon"]
    if len(predictions) != expected or predictions.target_day_num.max() > final_origin:
        raise ValueError("Unexpected forecast coverage or reserved holdout encountered")
    # Deterministic zero forecasts have WAPE=1 if any nonnegative sales are observed.
    pooled = metrics.loc[metrics.scope.eq("all_folds_pooled")].iloc[0]
    if not isnan(pooled.wape) and abs(pooled.wape - 1.0) > 1e-12:
        raise ValueError("Dummy WAPE sanity check failed")
    return {
        "status": "passed", "python": platform.python_version(),
        "project_imports": len(PROJECT_IMPORTS), "required_files": n_files,
        "series": n_series, "folds": len(folds), "prediction_rows": len(predictions),
        "prediction_columns": len(predictions.columns),
        "max_target_day": int(predictions.target_day_num.max()),
        "final_holdout_evaluated": metadata["final_holdout_evaluated"],
        "output_files_written": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    args = parser.parse_args()
    try:
        from src.config import load_config
        result = run_smoke(load_config(args.config))
    except (ImportError, FileNotFoundError, ValueError) as error:
        print(f"SMOKE FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
