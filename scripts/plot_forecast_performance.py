"""Create publication-ready plots from checked-in experimental summaries.

Run from the repository root with ``python -m scripts.plot_forecast_performance``.
All plotted numeric data are read from reports/*.csv; unavailable analyses are
listed in the output availability report rather than being imputed.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "reports" / "forecast_performance" / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = PROJECT_ROOT
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "forecast_performance"
FIGURES = OUTPUT / "figures"
PLOT_DATA = OUTPUT / "plotted_data"

COLORS = {
    "Gradient Boosted Trees (LightGBM)": "#2878B5",
    "State-Space (Kalman)": "#E87500",
    "Deep Learning (DeepAR)": "#3A923A",
    "Baseline Heuristic": "#8A63B8",
}


def _save_data(name: str, frame: pd.DataFrame) -> None:
    PLOT_DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PLOT_DATA / f"{name}.csv", index=False)


def _save_figure(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
        "axes.titlesize": 11, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .22, "grid.linestyle": ":",
    })


def _model_color(row: pd.Series) -> str:
    return COLORS.get(row.get("family", ""), "#65717C")


def make_accuracy_figure() -> dict:
    source = pd.read_csv(REPORTS / "integrated_model_comparison.csv")
    metrics = [("mae", "MAE", "Mean absolute error (units)"),
               ("rmse", "RMSE", "Root mean squared error (units)"),
               ("wape", "WAPE", "WAPE (%)"),
               ("smape", "sMAPE", "Symmetric absolute percentage error (%)"),
               ("mase", "MASE", "Mean absolute scaled error")]
    available_rows, status_rows = [], []
    for col, title, unit in metrics:
        available = col in source.columns and pd.to_numeric(source.get(col), errors="coerce").notna().any()
        status_rows.append({"analysis": title, "available": bool(available),
                            "source_file": "integrated_model_comparison.csv",
                            "reason": "Metric not recorded in source summary" if not available else ""})
        if available:
            subset = source.loc[pd.to_numeric(source[col], errors="coerce").notna(),
                                ["model", "family", col]].rename(columns={col: "value"})
            subset["metric"], subset["unit"] = title, unit
            available_rows.append(subset)
    plotted = pd.concat(available_rows, ignore_index=True) if available_rows else pd.DataFrame()
    _save_data("accuracy_metrics", plotted)
    availability = pd.DataFrame(status_rows)

    fig, axes = plt.subplots(1, len(metrics), figsize=(20, 7), constrained_layout=True)
    for ax, (column, title, unit) in zip(axes, metrics):
        available = column in source.columns and pd.to_numeric(source.get(column), errors="coerce").notna().any()
        ax.set_title(title, weight="bold")
        if not available:
            ax.text(.5, .5, "Not reported in source\nsummary", ha="center", va="center",
                    transform=ax.transAxes, color="#5B6570", fontsize=9)
            ax.set_axis_off()
            continue
        frame = source.loc[pd.to_numeric(source[column], errors="coerce").notna()].copy()
        frame = frame.sort_values(column, ascending=True)
        y = np.arange(len(frame))
        ax.barh(y, frame[column], color=[_model_color(row) for _, row in frame.iterrows()])
        ax.set_yticks(y, frame.model.str.replace("_", " ", regex=False), fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(unit)
        ax.grid(axis="y", visible=False)
        if title == "WAPE":
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    family_handles = [plt.Line2D([0], [0], color=color, lw=6, label=family)
                      for family, color in COLORS.items()]
    fig.legend(handles=family_handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, -.04))
    fig.suptitle("Rolling-origin forecast accuracy by model", fontsize=15, weight="bold")
    _save_figure(fig, "accuracy_comparison.png")
    return {"accuracy": availability}


def make_horizon_figure() -> dict:
    path = REPORTS / "kalman_variant_by_horizon.csv"
    frame = pd.read_csv(path)
    order = ["days_1_7", "days_8_14", "days_15_21", "days_22_28"]
    labels = ["Days 1–7", "Days 8–14", "Days 15–21", "Days 22–28"]
    frame["horizon_bucket"] = pd.Categorical(frame.horizon_bucket, categories=order, ordered=True)
    frame = frame.sort_values(["model", "horizon_bucket"])
    frame["horizon_label"] = frame.horizon_bucket.map(dict(zip(order, labels))).astype(str)
    cols = [c for c in ("mae", "rmse", "wape") if c in frame]
    long = frame.melt(id_vars=["model", "horizon_bucket", "horizon_label"],
                      value_vars=cols, var_name="metric", value_name="value")
    _save_data("horizon_metrics_kalman_variants", long)
    models = frame.model.drop_duplicates().tolist()
    palette = plt.get_cmap("Oranges")
    colors = {model: palette(value) for model, value in
              zip(models, np.linspace(.48, .88, max(1, len(models))))}
    fig, axes = plt.subplots(1, len(cols), figsize=(17, 5.8), constrained_layout=True)
    for ax, metric in zip(axes, cols):
        for model in models:
            part = frame.loc[frame.model.eq(model)]
            ax.plot(part.horizon_label, part[metric], marker="o", ms=4, lw=1.8,
                    label=model.replace("kalman_", ""), color=colors[model])
        ax.set_title(metric.upper(), weight="bold")
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel({"mae": "MAE (units)", "rmse": "RMSE (units)",
                       "wape": "WAPE (ratio)"}.get(metric, metric))
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="x", visible=False)
    axes[0].legend(title="Kalman variant", frameon=False, fontsize=7)
    fig.suptitle("Forecast accuracy by horizon bucket — Kalman experiments", fontsize=14, weight="bold")
    _save_figure(fig, "forecast_horizon_kalman.png")
    return {"horizon": pd.DataFrame([{"analysis": "Horizon analysis", "available": True,
                                      "source_file": path.name,
                                      "reason": "Reported only for Kalman variants"}])}


def make_probabilistic_figure() -> dict:
    source = pd.read_csv(REPORTS / "integrated_model_comparison.csv")
    frame = source.loc[source.empirical_coverage.notna() & source.mean_interval_width.notna()].copy()
    plot_data = frame[["model", "family", "empirical_coverage", "nominal_coverage",
                       "mean_interval_width"]].copy()
    _save_data("probabilistic_coverage_and_width", plot_data)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    frame = frame.sort_values("empirical_coverage", ascending=True)
    y = np.arange(len(frame))
    colors = [_model_color(row) for _, row in frame.iterrows()]
    axes[0].barh(y, frame.empirical_coverage, color=colors, label="Empirical coverage")
    nominal = frame.nominal_coverage.dropna()
    if len(nominal) and nominal.nunique() == 1:
        axes[0].axvline(nominal.iloc[0], color="#252525", ls="--", lw=1.5,
                        label=f"Nominal {nominal.iloc[0]:.0%}")
    axes[0].set_yticks(y, frame.model.str.replace("_", " ", regex=False), fontsize=8)
    axes[0].set_xlabel("80% interval empirical coverage")
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    axes[0].set_title("Prediction interval coverage", weight="bold")
    axes[0].legend(frameon=False)
    axes[1].barh(y, frame.mean_interval_width, color=colors)
    axes[1].set_yticks(y, frame.model.str.replace("_", " ", regex=False), fontsize=8)
    axes[1].set_xlabel("Mean P90 − P10 width (units)")
    axes[1].set_title("Average prediction interval width", weight="bold")
    axes[1].grid(axis="y", visible=False)
    fig.suptitle("Probabilistic forecast summary", fontsize=14, weight="bold")
    _save_figure(fig, "probabilistic_coverage_width.png")
    return {"probabilistic": pd.DataFrame([{"analysis": "Aggregate P10–P90 coverage/width", "available": True,
                                           "source_file": "integrated_model_comparison.csv",
                                           "reason": "Date-level actual/median/quantile rows unavailable"}])}


def make_runtime_figure() -> dict:
    kalman = pd.read_csv(REPORTS / "kalman_variant_comparison.csv")
    deepar = pd.read_csv(REPORTS / "deepar_experiments_comparison.csv")
    kalman_plot = kalman[["model", "total_fit_seconds", "total_predict_seconds",
                          "total_series_fits"]].copy()
    kalman_plot = kalman_plot.rename(columns={"total_fit_seconds": "fit_seconds",
                                              "total_predict_seconds": "predict_seconds"})
    kalman_plot["prediction_seconds_per_series_fit"] = kalman_plot.predict_seconds / kalman_plot.total_series_fits
    kalman_plot["total_recorded_seconds"] = kalman_plot.fit_seconds + kalman_plot.predict_seconds
    kalman_plot["source_family"] = "Kalman pooled evaluation"
    deep_plot = deepar[["experiment", "fit_seconds", "predict_seconds", "trainable_parameters"]].copy()
    deep_plot = deep_plot.rename(columns={"experiment": "model", "trainable_parameters": "model_parameters"})
    deep_plot["total_recorded_seconds"] = deep_plot.fit_seconds + deep_plot.predict_seconds
    deep_plot["source_family"] = "DeepAR experiment"
    kalman_data = kalman_plot.rename(columns={"total_series_fits": "series_fits"})
    deep_data = deep_plot.copy()
    _save_data("runtime_kalman", kalman_data)
    _save_data("runtime_deepar", deep_data)

    fig, axes = plt.subplots(1, 4, figsize=(22, 7), constrained_layout=True)
    k = kalman_plot.sort_values("fit_seconds")
    y = np.arange(len(k))
    axes[0].barh(y, k.fit_seconds, color=COLORS["State-Space (Kalman)"], alpha=.85, label="Fit")
    axes[0].barh(y, k.predict_seconds, left=k.fit_seconds, color="#8CBAD9", alpha=.95, label="Prediction")
    axes[0].set_yticks(y, k.model.str.replace("kalman_", "", regex=False), fontsize=7)
    axes[0].set_xlabel("Recorded seconds (pooled evaluation)")
    axes[0].set_title("Kalman runtime", weight="bold")
    axes[0].legend(frameon=False)

    d = deep_plot.sort_values("fit_seconds")
    y = np.arange(len(d))
    axes[1].barh(y, d.fit_seconds, color=COLORS["Deep Learning (DeepAR)"], label="Fit")
    axes[1].barh(y, d.predict_seconds, left=d.fit_seconds, color="#94C994", label="Prediction")
    axes[1].set_yticks(y, d.model.str.replace("Exp_", "", regex=False), fontsize=7)
    axes[1].set_xlabel("Recorded seconds (experiment scope)")
    axes[1].set_title("DeepAR experiment runtime", weight="bold")
    axes[1].legend(frameon=False)

    per_series = kalman_plot.sort_values("prediction_seconds_per_series_fit")
    axes[2].barh(np.arange(len(per_series)), per_series.prediction_seconds_per_series_fit,
                 color="#8CBAD9")
    axes[2].set_yticks(np.arange(len(per_series)),
                       per_series.model.str.replace("kalman_", "", regex=False), fontsize=7)
    axes[2].set_xlabel("Prediction seconds per series-origin")
    axes[2].set_title("Kalman inference per SKU-fold", weight="bold")
    axes[2].text(.02, -.17, "Derived from total prediction time / recorded series fits.",
                 transform=axes[2].transAxes, fontsize=7)

    axes[3].barh(np.arange(len(d)), d.model_parameters, color=COLORS["Deep Learning (DeepAR)"])
    axes[3].set_yticks(np.arange(len(d)), d.model.str.replace("Exp_", "", regex=False), fontsize=7)
    axes[3].set_xlabel("Trainable parameters (count)")
    axes[3].set_title("DeepAR model size", weight="bold")
    axes[3].text(.02, -.17, "Parameter count is not available for other model families.",
                 transform=axes[3].transAxes, fontsize=7)
    fig.suptitle("Computational measurements reported by experiments", fontsize=14, weight="bold")
    _save_figure(fig, "computational_efficiency.png")
    return {"runtime": pd.DataFrame([
        {"analysis": "Kalman fit/prediction time", "available": True,
         "source_file": "kalman_variant_comparison.csv", "reason": "Pooled over recorded series fits"},
        {"analysis": "DeepAR fit/prediction time and parameter count", "available": True,
         "source_file": "deepar_experiments_comparison.csv", "reason": "Configuration experiments"},
        {"analysis": "Comparable all-family runtime/inference per SKU", "available": False,
         "source_file": "", "reason": "No common timing protocol or per-SKU LightGBM/baseline timings"},
    ])}


def make_availability_report() -> pd.DataFrame:
    rows = [
        {"analysis": "MAE / RMSE / WAPE model comparison", "available": True,
         "source_file": "integrated_model_comparison.csv", "reason": "Pooled retail evaluation results"},
        {"analysis": "sMAPE / MASE model comparison", "available": False,
         "source_file": "integrated_model_comparison.csv", "reason": "Not recorded; raw scored forecasts/training scales absent"},
        {"analysis": "Horizon buckets 1–7 through 22–28", "available": True,
         "source_file": "kalman_variant_by_horizon.csv", "reason": "Available for Kalman variants only"},
        {"analysis": "SKU error distributions / best and worst SKUs / naive comparison", "available": False,
         "source_file": "", "reason": "No per-series scored predictions"},
        {"analysis": "Store-level model performance", "available": False,
         "source_file": "", "reason": "No store identifier in aggregate performance tables"},
        {"analysis": "Date-level actual vs predicted median and P10–P90", "available": False,
         "source_file": "", "reason": "No row-level actuals and prediction quantiles"},
        {"analysis": "Aggregate probability coverage and width", "available": True,
         "source_file": "integrated_model_comparison.csv", "reason": "Reported aggregate interval summaries"},
        {"analysis": "Runtime/model size", "available": True,
         "source_file": "kalman_variant_comparison.csv; deepar_experiments_comparison.csv",
         "reason": "Partial; not a standardized cross-family benchmark"},
    ]
    frame = pd.DataFrame(rows)
    _save_data("analysis_availability", frame)
    return frame


def main() -> None:
    _style()
    available = {}
    available.update(make_accuracy_figure())
    available.update(make_horizon_figure())
    available.update(make_probabilistic_figure())
    available.update(make_runtime_figure())
    availability = make_availability_report()
    for key, frame in available.items():
        _save_data(f"availability_{key}", frame)
    (OUTPUT / "README.md").write_text(
        "# Forecast performance plots\n\n"
        "Figures and plotted datasets were generated from the checked-in experimental CSV summaries. "
        "No model values are embedded in the plotting script. See `plotted_data/analysis_availability.csv` "
        "for supported and unavailable analyses. The source folder contains pooled retail summaries, "
        "but no row-level scored predictions or demand panel; SKU/store residual plots and date-level "
        "actual-versus-quantile trajectories cannot be reconstructed. Runtime sources use different "
        "experiment scopes and should not be read as a standardized cross-family speed ranking.\n\n"
        "Run from the project root: `python -m scripts.plot_forecast_performance`.\n"
    )
    print(f"Saved figures: {len(list(FIGURES.glob('*.png')))}")
    print(f"Saved plotted datasets: {len(list(PLOT_DATA.glob('*.csv')))}")
    print(f"Available analyses: {int(availability.available.sum())}/{len(availability)}")
    print(f"Output directory: {OUTPUT}")


if __name__ == "__main__":
    main()
