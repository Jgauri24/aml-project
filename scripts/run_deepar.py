"""CLI runner for DeepAR global probabilistic forecasting model on development folds.

Examples:
    python -m scripts.run_deepar
    python -m scripts.run_deepar --epochs 25 --hidden-dim 64
    python -m scripts.run_deepar --context-length 84 --output-dir deepar_ctx84
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.config import load_config, data_path
from src.data.preprocess import load_panel
from src.evaluation.backtesting import run_backtest
from src.models.deepar_config import DeepARConfig
from src.models.deepar_forecaster import make_deepar_factory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_deepar_pipeline(
    panel: pd.DataFrame,
    cfg: dict,
    deepar_cfg: DeepARConfig,
    output_dir: str = "deepar",
    save_checkpoints: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Execute DeepAR backtesting pipeline across development folds."""
    dest_dir = data_path(cfg, "processed") / output_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Save DeepAR configuration
    deepar_cfg.to_yaml(dest_dir / "deepar_config.yaml")

    ckpt_dir = (dest_dir / "checkpoints") if save_checkpoints else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    factory = make_deepar_factory(deepar_cfg, checkpoint_dir=ckpt_dir)

    start_time = perf_counter()
    predictions, metrics, metadata = run_backtest(
        panel=panel,
        cfg=cfg,
        factory=factory,
        output_dir=output_dir,
    )
    total_duration = perf_counter() - start_time

    # Augment execution metadata with DeepAR specific details
    metadata["deepar_config"] = deepar_cfg.to_dict()
    metadata["deepar_execution_seconds"] = total_duration
    (dest_dir / "execution_metadata.json").write_text(json.dumps(metadata, indent=2))

    return predictions, metrics, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiment.yaml", help="Experiment configuration YAML")
    parser.add_argument("--output-dir", default="deepar", help="Sub-directory under data/processed/")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per fold")
    parser.add_argument("--context-length", type=int, default=56, help="Context sequence length (days)")
    parser.add_argument("--hidden-dim", type=int, default=64, help="LSTM hidden state dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--num-samples", type=int, default=200, help="Monte Carlo sample trajectories")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--distribution", default="negative_binomial", choices=["negative_binomial", "gaussian"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-checkpoints", action="store_true", help="Disable checkpoint saving")
    args = parser.parse_args()

    cfg = load_config(args.config)
    panel = load_panel(cfg, cutoff=cfg["forecast"]["final_origin"])
    logger.info(f"Loaded {panel['series_id'].nunique()} series through d_{panel['day_num'].max()}")

    deepar_cfg = DeepARConfig(
        context_length=args.context_length,
        prediction_length=cfg["forecast"]["horizon"],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        distribution=args.distribution,
        num_samples=args.num_samples,
        device=args.device,
        seed=args.seed,
    )

    predictions, metrics, metadata = run_deepar_pipeline(
        panel=panel,
        cfg=cfg,
        deepar_cfg=deepar_cfg,
        output_dir=args.output_dir,
        save_checkpoints=not args.no_checkpoints,
    )

    print("\nDeepAR Development Backtest Results:")
    print(metrics.to_string(index=False))
    print(f"\nTotal execution time: {metadata['deepar_execution_seconds']:.2f}s")
    print(f"Predictions shape: {predictions.shape}")


if __name__ == "__main__":
    main()

