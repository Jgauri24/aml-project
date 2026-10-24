"""Command-line runner for Milestone 8: Feature Engineering.

Builds leakage-safe training and prediction feature matrices for all
development backtest folds.
"""
from __future__ import annotations

import argparse
import sys

from src.features.feature_engineering import build_feature_datasets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe feature datasets for M8.",
    )
    parser.add_argument(
        "--config",
        default="experiment.yaml",
        help="Path to experiment configuration YAML (default: experiment.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/features",
        help="Output directory for feature parquet files (default: data/processed/features).",
    )
    args = parser.parse_args(argv)

    print("Building M8 leakage-safe feature matrices...")
    meta = build_feature_datasets(config_path=args.config, output_dir=args.output_dir)
    print(f"Successfully generated {meta['n_folds']} fold feature datasets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
