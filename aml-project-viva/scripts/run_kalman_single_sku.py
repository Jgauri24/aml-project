"""
Experiment runner for Milestone 6: Single-SKU Kalman Filter.

Usage (from repository root, with .venv active):
    python scripts/run_kalman_single_sku.py
    python scripts/run_kalman_single_sku.py --config experiment.yaml
    python scripts/run_kalman_single_sku.py --output-dir kalman_single_sku/run_002

What this script does
---------------------
1. Loads the canonical development panel via the shared load_panel() call.
2. Runs the full development backtest (origins 1801, 1829, 1857, 1885) using
   run_backtest() with KalmanSingleSKU as the factory.
3. Prints the development metrics table (MAE, RMSE, WAPE, pinball losses).
4. Generates one forecast visualisation per volume band (low/medium/high),
   choosing the first available series from each band at fold origin=1885
   (the most recent fold, giving the longest training history).
5. Saves figures to reports/figures/kalman_single_sku_<series_id>.png.

The final holdout (d_1914–d_1941) is never evaluated; run_backtest() enforces
this through the development_splits configuration.
"""

import argparse
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import data_path, load_config
from src.data.preprocess import load_panel
from src.evaluation.backtesting import run_backtest
from src.evaluation.splits import development_splits
from src.models.kalman_single_sku import (
    KalmanSingleSKU,
    fit_local_level,
    forecast_local_level,
    kalman_filter,
    plot_single_sku_forecast,
    _P0_DIFFUSE,
    _initial_state_mean,
)


def _pick_representative_series(
    panel: pd.DataFrame, selected_csv: Path
) -> dict[str, str]:
    """Return {volume_band: series_id} — one representative per band."""
    meta = pd.read_csv(selected_csv)
    representatives = {}
    for band in ["low", "medium", "high"]:
        subset = meta.loc[meta["volume_group"] == band, "series_id"]
        if len(subset) == 0:
            continue
        # Pick the first one alphabetically for reproducibility
        sid = str(sorted(subset)[0])
        if sid in panel["series_id"].values or sid in panel["series_id"].cat.categories:
            representatives[band] = sid
    return representatives


def _make_forecast_plot(
    panel: pd.DataFrame,
    series_id: str,
    origin: int,
    horizon: int,
    figures_dir: Path,
) -> None:
    """Fit the filter on history up to origin, plot the 28-day forecast."""
    series_history = (
        panel.loc[
            (panel["series_id"] == series_id) & (panel["day_num"] <= origin)
        ]
        .sort_values("day_num")
        .reset_index(drop=True)
    )

    day_nums = series_history["day_num"].to_numpy()
    y = series_history["units_sold"].to_numpy(dtype=float)

    # Fit and run the filter (identical to what KalmanSingleSKU.fit does per series)
    mu0 = _initial_state_mean(y)
    Q, R, mu_T, P_T = fit_local_level(y, seed=42)
    fr = kalman_filter(y, Q, R, mu0=mu0, P0=_P0_DIFFUSE)

    # Forecast arrays
    y_hat, pred_vars = forecast_local_level(mu_T, P_T, Q, R, horizon=horizon)
    pred_std = np.sqrt(pred_vars)
    future_day_nums = np.arange(origin + 1, origin + horizon + 1)

    save_path = figures_dir / f"kalman_single_sku_{series_id.replace('__', '_')}.png"
    fig = plot_single_sku_forecast(
        series_id=series_id,
        day_nums=day_nums,
        y_history=y,
        mu_filtered=fr.mu_filtered,
        future_day_nums=future_day_nums,
        y_hat=y_hat,
        pred_std=pred_std,
        origin=origin,
        n_history_days=120,
        save_path=save_path,
    )
    plt.close(fig)
    print(f"  Saved figure: {save_path.relative_to(save_path.parents[3])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None,
                        help="Path to experiment YAML (default: experiment.yaml)")
    parser.add_argument("--output-dir", default="kalman_single_sku/run_001",
                        help="Output directory for artefacts (relative to data/processed/)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Load panel (development rows only; holdout excluded by load_panel) ──
    print("Loading canonical development panel …")
    panel = load_panel(cfg, cutoff=cfg["forecast"]["final_origin"])
    n_series = panel["series_id"].nunique()
    print(f"  {n_series} series × {panel['day_num'].nunique()} days loaded")

    # ── Run development backtest ─────────────────────────────────────────────
    print(f"\nRunning backtest → output: {args.output_dir}")
    factory = partial(KalmanSingleSKU, seed=cfg["seed"])
    predictions, metrics, metadata = run_backtest(
        panel, cfg, factory, output_dir=args.output_dir
    )

    print(f"\n{'─'*70}")
    print("Development metrics (pooled over nonmissing series-days):")
    print(f"{'─'*70}")
    print(metrics.to_string(index=False))
    print(f"{'─'*70}")
    print(f"Predictions: {predictions.shape}  |  "
          f"final_holdout_evaluated: {metadata['final_holdout_evaluated']}")
    print(f"Max target day: {metadata['max_target_day']}  "
          f"(must be <= {cfg['forecast']['final_origin']})")

    # ── Generate visualisations ──────────────────────────────────────────────
    figures_dir = data_path(cfg, "reports") / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Use the last development fold (largest training set)
    last_origin = max(cfg["forecast"]["development_origins"])
    horizon = cfg["forecast"]["horizon"]

    # Pick one representative series per volume band from selected_skus.csv
    selected_csv = data_path(cfg, "interim") / "selected_skus.csv"
    if selected_csv.exists():
        print(f"\nGenerating forecast visualisations (origin d_{last_origin}) …")
        representatives = _pick_representative_series(panel, selected_csv)
        for band, sid in representatives.items():
            print(f"  [{band:>6}] {sid}")
            _make_forecast_plot(panel, sid, last_origin, horizon, figures_dir)
    else:
        print(
            f"\nSkipping visualisations: {selected_csv} not found. "
            "Run python -m src.data.preprocess first."
        )


if __name__ == "__main__":
    main()
