"""Small end-to-end pipeline smoke run on two synthetic SKU-store series."""
from datetime import datetime, timezone

import pandas as pd
import yaml

from src.config import ROOT
from src.data.preprocess import add_quality_flags
from src.evaluation.pipeline import run_evaluation_pipeline
from examples.model_suite_demo import make_panel


def main() -> None:
    config_path = ROOT / "configs" / "evaluation_pipeline.yaml"
    pipeline_cfg = yaml.safe_load(config_path.read_text())
    pipeline_cfg["models"]["lightgbm"]["params"] = {
        "objective": "regression_l1", "metric": "mae", "n_estimators": 8,
        "num_leaves": 7, "learning_rate": .1, "min_child_samples": 2,
        "feature_fraction": 1., "bagging_fraction": 1., "bagging_freq": 0,
        "n_jobs": 1, "verbose": -1,
    }
    pipeline_cfg["models"]["deepar"]["config"].update({
        "context_length": 28, "hidden_dim": 8, "num_layers": 1,
        "batch_size": 2, "epochs": 1, "early_stopping_patience": 1,
        "val_window_length": 28, "windows_per_series_train": 2,
        "num_samples": 8, "device": "cpu",
    })
    pipeline_cfg["statistics"]["bootstrap_iterations"] = 100
    pipeline_cfg["outputs"]["root"] = "../reports/evaluation_pipeline_smoke"

    panel = add_quality_flags(make_panel())
    run_id = "smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = run_evaluation_pipeline(
        pipeline_cfg,
        panel=panel,
        experiment_id=run_id,
        overrides={"forecast_horizon": 28, "development_origins": [140],
                   "final_origin": 168, "final_end_day": 196,
                   "selection_cutoff_day": 140},
    )
    print(f"Output: {result['run_dir']}")
    print(f"Series: {result['metadata']['selected_series_count']}; models: {len(result['forecasts'])}")
    print(result["metric_tables"]["overall"][["model", "mae", "rmse", "wape", "mase"]].to_string(index=False))
    print(f"Dashboard tables: {len(list((result['run_dir'] / 'dashboard').glob('*.csv')))} CSV files")
    print(f"Figures: {len(list((result['run_dir'] / 'visualizations' / 'figures').glob('*.png')))} PNG files")
    print("Final holdout evaluated: false")


if __name__ == "__main__":
    main()
