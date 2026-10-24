"""Run the M9 LightGBM forecaster on development folds.

Examples:
    python -m scripts.run_lightgbm_forecaster
    python -m scripts.run_lightgbm_forecaster --no-tune
"""
from __future__ import annotations

import argparse
from src.config import load_config
from src.data.preprocess import load_panel
from src.models.lightgbm_forecaster import run_lightgbm_forecaster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiment.yaml")
    parser.add_argument("--output-dir", default="lightgbm",
                        help="Sub-directory under data/processed/")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip rolling-validation hyperparameter tuning")
    args = parser.parse_args()

    cfg = load_config(args.config)
    panel = load_panel(cfg, cutoff=cfg["forecast"]["final_origin"])
    print(f"Loaded {panel['series_id'].nunique()} series through d_{panel['day_num'].max()}")

    predictions, metrics, metadata = run_lightgbm_forecaster(
        panel, cfg,
        output_dir=args.output_dir,
        tune=not args.no_tune,
    )
    print("\nFinal metrics:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
