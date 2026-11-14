"""Dashboard and publication plots generated from a pipeline's row-level forecasts."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import score_predictions


def _write_data(root: Path, name: str, data: pd.DataFrame) -> None:
    target = root / "plotted_data"
    target.mkdir(parents=True, exist_ok=True)
    data.to_csv(target / f"{name}.csv", index=False)


def _write_figure(root: Path, fig, name: str) -> None:
    target = root / "figures"
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _colors(models):
    cmap = plt.get_cmap("tab10")
    return {model: cmap(i % 10) for i, model in enumerate(sorted(models))}


def _metric_record(frame: pd.DataFrame) -> dict:
    return score_predictions(frame)


def generate_pipeline_visualizations(
    scored: pd.DataFrame,
    runtime: pd.DataFrame,
    output_dir: str | Path,
    *,
    baseline_model: str | None = None,
) -> dict:
    """Create accuracy, horizon, SKU/store, probabilistic and runtime plots.

    Expects evaluator-attached actuals, MASE scales and SKU/store columns. All
    values are calculated directly from that run's scored forecast rows.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {"model", "series_id", "item_id", "store_id", "origin", "horizon",
                "target_date", "actual", "y_pred", "q10", "q50", "q90"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"Visualization input is missing columns: {sorted(missing)}")
    models = sorted(scored.model.astype(str).unique())
    colors = _colors(models)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .22, "grid.linestyle": ":"})
    results = {}

    # Whole portfolio scores, including training-only MASE scale attached by backtesting.
    overall_rows = [{"model": model, **_metric_record(frame)}
                    for model, frame in scored.groupby("model", observed=True, sort=True)]
    overall = pd.DataFrame(overall_rows)
    _write_data(output_dir, "overall_metrics", overall)
    metric_specs = [("mae", "MAE", "Units"), ("rmse", "RMSE", "Units"),
                    ("wape", "WAPE", "Ratio"), ("smape", "sMAPE", "%"),
                    ("mase", "MASE", "Scaled error")]
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(19, 7), constrained_layout=True)
    for axis, (column, title, unit) in zip(axes, metric_specs):
        frame = overall.loc[pd.to_numeric(overall[column], errors="coerce").notna()].sort_values(column)
        axis.set_title(title, weight="bold")
        if frame.empty:
            axis.text(.5, .5, "Not available", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            continue
        y = np.arange(len(frame))
        axis.barh(y, frame[column], color=[colors[name] for name in frame.model])
        axis.set_yticks(y, frame.model, fontsize=7)
        axis.invert_yaxis()
        axis.set_xlabel(unit)
        if column == "smape":
            axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0f}%"))
        if column == "wape":
            axis.set_xlabel("Ratio (lower is better)")
        axis.grid(axis="y", visible=False)
    fig.suptitle("Rolling-origin accuracy across the selected cohort", fontsize=15, weight="bold")
    _write_figure(output_dir, fig, "accuracy_metrics.png")

    # Horizon buckets exactly as requested; each metric uses the same rows per model/bucket.
    bucket_edges = [(1, 7, "Days 1–7"), (8, 14, "Days 8–14"),
                    (15, 21, "Days 15–21"), (22, 28, "Days 22–28")]
    horizon_parts = []
    for lo, hi, label in bucket_edges:
        subset = scored.loc[scored.horizon.between(lo, hi)]
        for model, frame in subset.groupby("model", observed=True, sort=True):
            horizon_parts.append({"model": model, "horizon_bucket": label,
                                  "horizon_start": lo, "horizon_end": hi,
                                  **_metric_record(frame)})
    horizon = pd.DataFrame(horizon_parts)
    _write_data(output_dir, "horizon_metrics", horizon)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    for axis, metric, ylabel in zip(axes, ("mae", "rmse", "wape"),
                                    ("MAE (units)", "RMSE (units)", "WAPE (ratio)")):
        for model in models:
            subset = horizon.loc[horizon.model.eq(model)]
            axis.plot(subset.horizon_bucket, subset[metric], marker="o", lw=1.8,
                      label=model, color=colors[model])
        axis.set_title(metric.upper(), weight="bold")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="x", visible=False)
    axes[0].legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=7)
    fig.suptitle("Accuracy by forecast horizon", fontsize=14, weight="bold")
    _write_figure(output_dir, fig, "horizon_accuracy.png")

    # SKU-store summary distribution, best model, worst series, and baseline failures.
    sku_rows = []
    keys = ["model", "series_id", "item_id", "store_id"]
    for group_key, frame in scored.groupby(keys, observed=True, sort=True):
        sku_rows.append({**dict(zip(keys, group_key)), **_metric_record(frame)})
    sku = pd.DataFrame(sku_rows)
    _write_data(output_dir, "sku_store_model_metrics", sku)
    sku_box = [sku.loc[sku.model.eq(model), "mae"].dropna().to_numpy() for model in models]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    if any(len(values) for values in sku_box):
        parts = axes[0].boxplot(sku_box, patch_artist=True, showfliers=False, labels=models)
        for patch, model in zip(parts["boxes"], models):
            patch.set_facecolor(colors[model])
            patch.set_alpha(.75)
    axes[0].set_ylabel("SKU-store MAE (units)")
    axes[0].set_title("Error distribution across SKU-store series", weight="bold")
    axes[0].tick_params(axis="x", rotation=45, labelsize=7)

    winners = sku.loc[sku.groupby("series_id").mae.idxmin(), ["series_id", "item_id", "store_id", "model", "mae"]]
    winner_counts = winners.model.value_counts().reindex(models, fill_value=0)
    _write_data(output_dir, "best_model_per_sku_store", winners.sort_values(["item_id", "store_id"]))
    axes[1].bar(np.arange(len(models)), winner_counts.values, color=[colors[m] for m in models])
    axes[1].set_xticks(np.arange(len(models)), models, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Number of SKU-store series")
    axes[1].set_title("Best model by SKU-store (minimum MAE)", weight="bold")

    sku_mean = sku.groupby(["series_id", "item_id", "store_id"], as_index=False).mae.mean()
    worst = sku_mean.nlargest(min(20, len(sku_mean)), "mae")
    _write_data(output_dir, "worst_sku_store_series", worst)
    label = worst.item_id.astype(str) + " / " + worst.store_id.astype(str)
    axes[2].barh(np.arange(len(worst)), worst.mae, color="#B44B4B")
    axes[2].set_yticks(np.arange(len(worst)), label, fontsize=7)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Mean model MAE (units)")
    axes[2].set_title("Worst-performing SKU-store series", weight="bold")
    fig.suptitle("SKU-store performance analysis", fontsize=14, weight="bold")
    _write_figure(output_dir, fig, "sku_store_analysis.png")

    if baseline_model and baseline_model in set(sku.model):
        baseline = sku.loc[sku.model.eq(baseline_model), ["series_id", "mae"]].rename(columns={"mae": "baseline_mae"})
        advanced = sku.loc[~sku.model.eq(baseline_model), ["model", "series_id", "mae"]]
        failures = advanced.merge(baseline, on="series_id", how="inner", validate="many_to_one")
        failures["advanced_fails_to_beat_baseline"] = failures.mae.ge(failures.baseline_mae)
        summary = failures.groupby("model", as_index=False).advanced_fails_to_beat_baseline.agg(["sum", "count"]).reset_index()
        summary = summary.rename(columns={"sum": "series_not_beaten", "count": "series_compared"})
        _write_data(output_dir, "advanced_models_vs_seasonal_naive", failures)
        _write_data(output_dir, "advanced_models_failing_to_beat_baseline", summary)
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.bar(summary.model, summary.series_not_beaten, color=[colors[m] for m in summary.model])
        ax.set_ylabel("SKU-store series where advanced MAE ≥ baseline MAE")
        ax.set_title(f"Advanced models failing to beat {baseline_model}", weight="bold")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        _write_figure(output_dir, fig, "advanced_vs_seasonal_naive.png")

    # Store-wise model comparison and dashboard data.
    store_rows = []
    for (model, store_id), frame in scored.groupby(["model", "store_id"], observed=True, sort=True):
        store_rows.append({"model": model, "store_id": store_id, **_metric_record(frame)})
    store = pd.DataFrame(store_rows)
    _write_data(output_dir, "store_metrics", store)
    store_wide = store.pivot(index="store_id", columns="model", values="mae").reindex(columns=models)
    fig, ax = plt.subplots(figsize=(max(7, .8 * len(models)), max(5, .35 * len(store_wide))), constrained_layout=True)
    image = ax.imshow(store_wide.to_numpy(dtype=float), aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(models)), models, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(store_wide)), store_wide.index, fontsize=7)
    ax.set_title("Store-level MAE by model (units)", weight="bold")
    fig.colorbar(image, ax=ax, label="MAE (units)")
    _write_figure(output_dir, fig, "store_model_mae.png")

    # Coverage and width from genuine quantiles; no intervals are fabricated for point models.
    probability = []
    for model, frame in scored.groupby("model", observed=True, sort=True):
        valid = frame[["actual", "q10", "q50", "q90"]].notna().all(axis=1)
        valid &= np.isfinite(frame[["actual", "q10", "q50", "q90"]]).all(axis=1)
        part = frame.loc[valid]
        probability.append({"model": model, "n_intervals": len(part),
                            "coverage_80": (((part.actual >= part.q10) & (part.actual <= part.q90)).mean()
                                            if len(part) else np.nan),
                            "mean_interval_width_80": (part.q90 - part.q10).mean() if len(part) else np.nan})
    probability = pd.DataFrame(probability)
    _write_data(output_dir, "probabilistic_model_metrics", probability)

    # A data-derived high-demand SKU-store gives an interpretable actual/forecast detail view.
    unique_actual = scored[["series_id", "item_id", "store_id", "target_date", "actual"]].drop_duplicates()
    candidate_volume = unique_actual.groupby("series_id", as_index=False).actual.sum(min_count=1)
    chosen_series = str(candidate_volume.loc[candidate_volume.actual.idxmax(), "series_id"])
    series = scored.loc[scored.series_id.astype(str).eq(chosen_series)].copy()
    # For overlapping origins show the most recent valid forecast for each date.
    series = series.sort_values("origin").drop_duplicates(["model", "target_date"], keep="last")
    _write_data(output_dir, "representative_series_forecasts", series)
    fig, axis = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    first = series.sort_values("target_date").drop_duplicates("target_date")
    axis.plot(first.target_date, first.actual, color="#222222", lw=1.8, label="Actual")
    for model in models:
        part = series.loc[series.model.eq(model)].sort_values("target_date")
        axis.plot(part.target_date, part.q50.where(part.q50.notna(), part.y_pred),
                  color=colors[model], lw=1.3, label=f"{model} median/point")
        prob = part[["target_date", "q10", "q90"]].dropna()
        if not prob.empty:
            axis.fill_between(prob.target_date, prob.q10, prob.q90,
                              color=colors[model], alpha=.12, linewidth=0)
    axis.set_ylabel("Units sold")
    axis.set_title(f"Actual demand and forecasts — {chosen_series}", weight="bold")
    axis.legend(ncol=2, frameon=False, fontsize=7)
    _write_figure(output_dir, fig, "probabilistic_forecasts_actual_median_intervals.png")

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.bar(probability.model, probability.coverage_80,
           color=[colors[m] for m in probability.model])
    ax.axhline(.8, color="#222222", ls="--", lw=1, label="Nominal 80%")
    ax.set_ylabel("P10–P90 coverage")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.set_title("Empirical interval coverage (point models have no interval)", weight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(frameon=False)
    _write_figure(output_dir, fig, "prediction_interval_coverage.png")

    # Interval width remains separate from calibration and is NA for deterministic models.
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    avail = probability.loc[probability.mean_interval_width_80.notna()].sort_values("mean_interval_width_80")
    ax.barh(avail.model, avail.mean_interval_width_80, color=[colors[m] for m in avail.model])
    ax.set_xlabel("Mean P90 − P10 width (units)")
    ax.set_title("Prediction interval widths", weight="bold")
    _write_figure(output_dir, fig, "prediction_interval_widths.png")

    # Runtime table is generated by the backtester at every origin.
    runtime_total = runtime.loc[runtime.scope.eq("all_development_folds")].copy()
    _write_data(output_dir, "runtime_comparison", runtime_total)
    if not runtime_total.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
        run = runtime_total.sort_values("total_runtime_seconds")
        axes[0].barh(run.model, run.training_time_seconds, color="#4575B4", label="Training")
        axes[0].barh(run.model, run.prediction_time_seconds, left=run.training_time_seconds,
                     color="#91BFDB", label="Prediction")
        axes[0].set_xlabel("Total seconds across development folds")
        axes[0].set_title("Training and prediction time", weight="bold")
        axes[0].legend(frameon=False)
        per_series = runtime_total.dropna(subset=["inference_seconds_per_sku_store"])
        axes[1].barh(per_series.model, per_series.inference_seconds_per_sku_store, color="#66C2A5")
        axes[1].set_xlabel("Prediction seconds per SKU-store-fold")
        axes[1].set_title("Inference time per SKU-store series", weight="bold")
        _write_figure(output_dir, fig, "runtime_comparison.png")
        size_col = "trainable_parameters" if runtime_total.trainable_parameters.notna().any() else None
        if size_col:
            sized = runtime_total.dropna(subset=[size_col])
            _write_data(output_dir, "model_size", sized[["model", size_col]])

    summary = {"output_dir": str(output_dir), "figures": sorted(p.name for p in (output_dir / "figures").glob("*.png")),
               "plotted_datasets": sorted(p.name for p in (output_dir / "plotted_data").glob("*.csv")),
               "selected_detail_series_id": chosen_series,
               "probabilistic_models": probability.loc[probability.n_intervals.gt(0), "model"].tolist()}
    (output_dir / "visualization_metadata.json").write_text(__import__("json").dumps(summary, indent=2))
    return summary
