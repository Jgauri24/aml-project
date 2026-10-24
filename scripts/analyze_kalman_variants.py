"""Create M6/M7 Kalman comparison CSVs and development-only figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import data_path, load_config
from src.data.preprocess import load_panel
from src.evaluation.metrics import score_predictions


MODEL_RUNS = [
    ("kalman_single_sku", Path("data/processed/kalman_single_sku/run_001")),
    ("kalman_local_level", Path("data/processed/kalman_forecaster/local_level")),
    ("kalman_local_level_seasonal", Path("data/processed/kalman_forecaster/local_level_seasonal")),
    ("kalman_local_trend", Path("data/processed/kalman_forecaster/local_trend")),
    ("kalman_local_trend_seasonal", Path("data/processed/kalman_forecaster/local_trend_seasonal")),
]
METRIC_COLUMNS = ["mae", "rmse", "wape", "pinball_q10", "pinball_q50", "pinball_q90"]
HORIZON_BUCKETS = [(1, 7, "days_1_7"), (8, 14, "days_8_14"), (15, 21, "days_15_21"), (22, 28, "days_22_28")]


def _group_metrics(frame: pd.DataFrame) -> dict:
    metrics = score_predictions(frame)
    metrics["missing_actuals"] = int(metrics["n_rows"] - metrics["n_observed"])
    metrics["observation_flag"] = "too_few_observed" if metrics["n_observed"] < 30 else "ok"
    return metrics


def _load_runs() -> tuple[pd.DataFrame, list[dict]]:
    combined, comparison_rows = [], []
    for expected_model, directory in MODEL_RUNS:
        scored = pd.read_parquet(directory / "scored_predictions.parquet")
        if scored["model"].nunique() != 1 or scored["model"].iloc[0] != expected_model:
            raise ValueError(f"Unexpected model identity in {directory}")
        metadata = json.loads((directory / "execution_metadata.json").read_text())
        fold_times = metadata["folds"]
        metrics = _group_metrics(scored)
        observed = scored.loc[scored["actual"].notna()]
        interval = observed["q90"] - observed["q10"]
        summary_path = directory / "fit_summary.csv"
        if summary_path.exists():
            fit_summary = pd.read_csv(summary_path)
            fallbacks = int(fit_summary["fallback"].sum())
            nonconverged = int((~fit_summary["convergence_status"].eq("converged")).sum())
            diagnostics_available = True
        else:
            fallbacks, nonconverged, diagnostics_available = np.nan, np.nan, False
        comparison_rows.append({
            "model": expected_model, **{name.upper() if name in {"mae", "rmse", "wape"} else name: metrics[name] for name in METRIC_COLUMNS},
            "empirical_coverage_q10_q90": float(((observed.actual >= observed.q10) & (observed.actual <= observed.q90)).mean()),
            "nominal_coverage_q10_q90": 0.80, "mean_interval_width": float(interval.mean()),
            "total_fit_seconds": float(sum(fold["fit_seconds"] for fold in fold_times)),
            "total_predict_seconds": float(sum(fold["predict_seconds"] for fold in fold_times)),
            "total_series_fits": int(len(fold_times) * scored["series_id"].nunique()),
            "fallbacks": fallbacks, "nonconverged_fits": nonconverged,
            "fit_diagnostics_available": diagnostics_available,
            "prediction_rows": len(scored), "observed_rows": int(observed.shape[0]),
            "max_target_day": int(metadata["max_target_day"]),
            "final_holdout_evaluated": bool(metadata["final_holdout_evaluated"]),
        })
        combined.append(scored.assign(model_label=expected_model))
    return pd.concat(combined, ignore_index=True), comparison_rows


def _breakdowns(scored: pd.DataFrame, selected: pd.DataFrame, reports: Path) -> None:
    horizon_rows = []
    for lower, upper, label in HORIZON_BUCKETS:
        for model, group in scored.loc[scored.horizon.between(lower, upper)].groupby("model_label", observed=True):
            horizon_rows.append({"model": model, "horizon_bucket": label, "horizon_start": lower, "horizon_end": upper,
                                 **_group_metrics(group)})
    pd.DataFrame(horizon_rows).to_csv(reports / "kalman_variant_by_horizon.csv", index=False)

    volume = selected[["series_id", "volume_group"]].copy()
    by_volume = scored.merge(volume, on="series_id", how="left", validate="many_to_one")
    if by_volume.volume_group.isna().any():
        raise ValueError("A scored forecast did not match selected_skus volume_group")
    volume_rows = [{"model": model, "volume_group": group_name, **_group_metrics(group)}
                   for (model, group_name), group in by_volume.groupby(["model_label", "volume_group"], observed=True)]
    pd.DataFrame(volume_rows).to_csv(reports / "kalman_variant_by_volume_band.csv", index=False)

    origin_rows = [{"model": model, "origin": int(origin), **_group_metrics(group)}
                   for (model, origin), group in scored.groupby(["model_label", "origin"], observed=True)]
    pd.DataFrame(origin_rows).to_csv(reports / "kalman_variant_by_origin.csv", index=False)


def _representatives(selected: pd.DataFrame) -> dict[str, str]:
    """Pre-registered rule: alphabetically first selected series in each existing volume group."""
    return {band: sorted(selected.loc[selected.volume_group.eq(band), "series_id"].astype(str))[0]
            for band in ("low", "medium", "high")}


def _forecast_figures(scored: pd.DataFrame, panel: pd.DataFrame, representatives: dict[str, str], figures: Path) -> list[Path]:
    paths = []
    styles = {"kalman_local_level": "C0", "kalman_local_level_seasonal": "C1",
              "kalman_local_trend": "C2", "kalman_local_trend_seasonal": "C3"}
    origin = 1885
    for band, series_id in representatives.items():
        history = panel.loc[(panel.series_id.astype(str).eq(series_id)) & (panel.day_num <= origin)].tail(120)
        future = scored.loc[(scored.series_id.astype(str).eq(series_id)) & (scored.origin.eq(origin)) &
                            (scored.model_label.ne("kalman_single_sku"))].sort_values("target_date")
        actual = future.loc[future.model_label.eq("kalman_local_level"), ["target_date", "actual"]]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(history.date, history.units_sold, color="0.45", linewidth=1, label="training actual")
        ax.plot(actual.target_date, actual.actual, color="black", linewidth=1.5, marker="o", markersize=3, label="development target actual")
        for model, group in future.groupby("model_label", observed=True):
            group = group.sort_values("target_date")
            color = styles[model]
            ax.fill_between(group.target_date, group.q10, group.q90, color=color, alpha=0.13)
            ax.plot(group.target_date, group.y_pred, color=color, linewidth=1.5, label=f"{model} q50; q10–q90 band")
        ax.axvline(history.date.iloc[-1], color="0.2", linestyle="--", linewidth=1)
        ax.set(title=f"{band.title()} band: {series_id}; origin d_{origin}", xlabel="Date", ylabel="Units sold")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        fig.autofmt_xdate()
        fig.tight_layout()
        path = figures / f"kalman_variants_{band}_{series_id.replace('__', '_')}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def _comparison_figures(comparison: pd.DataFrame, figures: Path) -> list[Path]:
    paths = []
    labels = comparison.model.tolist()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, comparison.WAPE)
    ax.set(title="Development WAPE by Kalman model", xlabel="Model", ylabel="WAPE ratio")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = figures / "kalman_variant_wape.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(comparison))
    width = 0.24
    for offset, column, label in [(-width, "pinball_q10", "q=0.1"), (0, "pinball_q50", "q=0.5"), (width, "pinball_q90", "q=0.9")]:
        ax.bar(positions + offset, comparison[column], width=width, label=label)
    ax.set(title="Development pinball loss by quantile", xlabel="Model", ylabel="Pinball loss")
    ax.set_xticks(positions, labels, rotation=25)
    ax.legend(title="Quantile")
    fig.tight_layout()
    path = figures / "kalman_variant_pinball.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> None:
    cfg = load_config()
    reports, figures = data_path(cfg, "reports"), data_path(cfg, "reports") / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    scored, comparison_rows = _load_runs()
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(reports / "kalman_variant_comparison.csv", index=False)
    selected = pd.read_csv(data_path(cfg, "interim") / "selected_skus.csv")
    _breakdowns(scored, selected, reports)
    panel = load_panel(cfg, cutoff=cfg["forecast"]["final_origin"])
    representatives = _representatives(selected)
    paths = _forecast_figures(scored, panel, representatives, figures) + _comparison_figures(comparison, figures)
    print(comparison.to_csv(index=False))
    print("Representatives:", representatives)
    print("Figures:", [str(path) for path in paths])


if __name__ == "__main__":
    main()
