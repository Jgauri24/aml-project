"""Run M5 point baselines, shared development scoring, diagnostics and figures.

Usage: python -m scripts.run_seasonal_naive [--models last_value weekly_seasonal]
Relative --output-dir paths resolve under the configured processed directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ROOT, data_path, load_config
from src.data.load_data import EXPECTED_FILES
from src.data.preprocess import load_panel
from src.evaluation.backtesting import (
    FUTURE_KEYS, PREDICTION_COLUMNS, construct_inputs, run_backtest, validate_predictions,
)
from src.evaluation.splits import development_splits
from src.models.seasonal_naive import VARIANTS, SeasonalNaiveForecaster


def validate_integrity(
    predictions: pd.DataFrame, metadata: dict, panel: pd.DataFrame, cfg: dict,
) -> dict:
    """Check the entire configured development grid using the shared validator."""
    folds = development_splits(cfg)
    expected_series = set(panel.series_id.astype(str))
    configured_series = cfg["selection"]["n_per_group"] * len(cfg["selection"]["groups"])
    expected_rows = len(expected_series) * len(folds) * cfg["forecast"]["horizon"]
    if list(predictions) != PREDICTION_COLUMNS or len(predictions) != expected_rows:
        raise ValueError("Unexpected forecast schema or development row count")
    if len(expected_series) != configured_series:
        raise ValueError("Canonical cohort size differs from the experiment configuration")
    if set(predictions.origin) != {fold.origin for fold in folds}:
        raise ValueError("Unexpected development origins")
    for fold in folds:
        forecast = predictions.loc[predictions.origin.eq(fold.origin)]
        if set(forecast.series_id.astype(str)) != expected_series:
            raise ValueError("Incomplete selected-series coverage")
        validate_predictions(forecast, construct_inputs(panel, fold).future_known, metadata["model"])
    if not predictions[["q10", "q50", "q90"]].isna().all().all():
        raise ValueError("Naive point models must leave all quantiles unavailable")
    if (metadata["final_holdout_evaluated"] is not False
            or predictions.target_day_num.max() > cfg["forecast"]["final_origin"]
            or metadata["max_target_day"] != int(predictions.target_day_num.max())
            or metadata["prediction_rows"] != len(predictions)):
        raise ValueError("Invalid development metadata or reserved final-holdout access")
    return {
        "series": len(expected_series), "origins": [fold.origin for fold in folds],
        "rows": len(predictions), "exact_grid": True, "point_only": True,
        "max_target_day": int(predictions.target_day_num.max()),
        "final_holdout_evaluated": False,
    }


def select_representatives(panel: pd.DataFrame, cfg: dict) -> dict[str, str]:
    """First alphabetical cohort member per pre-existing cutoff-safe volume band."""
    selected = pd.read_csv(data_path(cfg, "interim") / "selected_skus.csv")
    selected = selected.loc[selected.series_id.isin(panel.series_id.astype(str))]
    representatives = {}
    for band in cfg["selection"]["groups"]:
        candidates = sorted(selected.loc[selected.volume_group.eq(band), "series_id"].astype(str))
        if not candidates:
            raise ValueError(f"No representative available in volume group {band}")
        representatives[band] = candidates[0]
    return representatives


def create_figures(
    panel: pd.DataFrame, predictions: pd.DataFrame, comparison: pd.DataFrame,
    representatives: dict[str, str], cfg: dict,
) -> list[Path]:
    """Plot training history and development targets only, with fixed representatives."""
    cutoff = cfg["forecast"]["final_origin"]
    if panel.day_num.max() > cutoff or predictions.target_day_num.max() > cutoff:
        raise ValueError("M5 figures accept development rows only")
    figures = data_path(cfg, "reports") / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    origin = min(cfg["forecast"]["development_origins"])
    horizon = cfg["forecast"]["horizon"]
    colors = {"naive_last_value": "#2563eb", "naive_weekly_seasonal": "#db6d16"}
    labels = {"naive_last_value": "Last value", "naive_weekly_seasonal": "Weekly seasonal"}
    model_names = list(predictions.model.unique())
    paths = []
    for band, series_id in representatives.items():
        series = panel.loc[panel.series_id.astype(str).eq(series_id)].sort_values("day_num")
        history = series.loc[series.day_num.between(max(1, origin - 55), origin)]
        actual = series.loc[series.day_num.between(origin + 1, origin + horizon)]
        forecasts = predictions.loc[predictions.series_id.astype(str).eq(series_id) & predictions.origin.eq(origin)]
        selections = [(name.removeprefix("naive_"), [name]) for name in model_names]
        if len(model_names) > 1:
            selections.append(("comparison", model_names))
        # Shared limits make the three views directly comparable.
        maximum = pd.concat([history.units_sold, actual.units_sold, forecasts.y_pred]).max()
        for name, models in selections:
            fig, ax = plt.subplots(figsize=(12, 4.5))
            ax.plot(history.date, history.units_sold, color="0.5", linewidth=1.2, label="Observed training sales")
            ax.plot(actual.date, actual.units_sold, color="#111827", linewidth=1.4,
                    marker="o", markersize=3, label="Actual development sales")
            for model in models:
                forecast = forecasts.loc[forecasts.model.eq(model)].sort_values("horizon")
                ax.plot(forecast.target_date, forecast.y_pred, color=colors[model],
                        linewidth=1.7, linestyle="--", marker=".", label=labels[model])
            ax.axvline(history.date.iloc[-1], color="#6b7280", linestyle=":", label=f"Origin d_{origin}")
            ax.set(title=f"{band.title()} volume: {series_id} | {name.replace('_', ' ').title()}",
                   xlabel="Date", ylabel="Observed units sold", ylim=(0, max(1.0, float(maximum) * 1.1)))
            ax.grid(axis="y", alpha=0.2)
            ax.legend(loc="upper left", fontsize=8)
            fig.autofmt_xdate()
            fig.tight_layout()
            path = figures / f"m5_{name}_{band}.png"
            fig.savefig(path, dpi=cfg["eda"]["figure_dpi"])
            plt.close(fig)
            paths.append(path)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    origins = cfg["forecast"]["development_origins"]
    positions = np.arange(len(origins))
    width = 0.8 / len(model_names)
    for index, model in enumerate(model_names):
        scores = comparison.loc[comparison.model.eq(model) & comparison.scope.eq("fold_pooled")]
        values = scores.set_index("origin").reindex(origins).wape
        ax.bar(positions + (index - (len(model_names) - 1) / 2) * width,
               values, width=width, color=colors[model], label=labels[model])
    ax.set_xticks(positions, [f"d_{origin}" for origin in origins])
    ax.set(title="Naive baseline errors by development origin", xlabel="Forecast origin", ylabel="Pooled WAPE (ratio)")
    finite_wape = comparison.wape.loc[np.isfinite(comparison.wape)]
    ax.set_ylim(0, max(1.0, float(finite_wape.max()) * 1.18) if len(finite_wape) else 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = figures / "m5_wape_by_origin.png"
    fig.savefig(path, dpi=cfg["eda"]["figure_dpi"])
    plt.close(fig)
    return [*paths, path]


def comparison_conclusion(comparison: pd.DataFrame) -> str:
    """Use pooled WAPE only; explicitly retain ties and unavailable comparisons."""
    pooled = comparison.loc[comparison.scope.eq("all_folds_pooled")]
    if len(pooled) < 2:
        return "Only one baseline was requested; no between-model winner is selected."
    if not np.isfinite(pooled.wape.to_numpy(dtype=float)).all():
        return "Pooled WAPE is unavailable for at least one model; no winner is selected."
    best = pooled.loc[pooled.wape.eq(pooled.wape.min())]
    if len(best) > 1:
        return "The baselines tie on pooled development WAPE. MAE and RMSE are descriptive; no tiebreaker is applied."
    winner = best.iloc[0]
    return f"{winner.model} has the lower pooled development WAPE ({winner.wape:.6f})."


def write_findings(
    comparison: pd.DataFrame, summaries: pd.DataFrame, representatives: dict[str, str],
    paths: list[Path], output: Path, cfg: dict,
) -> None:
    """Write actual shared-engine results and the explicitly bounded interpretation."""
    rows = ["| Model | Origin / scope | MAE | RMSE | WAPE ratio |", "| --- | --- | ---: | ---: | ---: |"]
    for row in comparison.itertuples(index=False):
        label = "pooled" if row.scope == "all_folds_pooled" else f"d_{int(row.origin)}"
        rows.append(f"| {row.model} | {label} | {row.mae:.6f} | {row.rmse:.6f} | {row.wape:.6f} |")
    first_origin = min(cfg["forecast"]["development_origins"])
    fallback = summaries.groupby("variant", observed=True)[[
        "weekly_earlier_same_weekday", "weekly_latest_overall", "weekly_all_missing_zero",
    ]].sum()
    fallback_lines = []
    for variant, group in summaries.groupby("variant", observed=True):
        counts = group.last_value_fallback.value_counts().to_dict()
        fallback_lines.append(f"- `{variant}`: last-value policies {counts}; weekly fallback positions {fallback.loc[variant].to_dict()}.")
    figure_lines = [f"- [{path.name}](figures/{path.name})" for path in paths]
    artifact_label = output.relative_to(ROOT) if output.is_relative_to(ROOT) else output
    missing_raw = [name for name in EXPECTED_FILES
                   if not (data_path(cfg, "raw") / name).is_file()
                   or (data_path(cfg, "raw") / name).stat().st_size == 0]
    setup_status = ("Raw-file setup remains incomplete: missing "
                    + ", ".join(f"`{name}`" for name in missing_raw)
                    + ". Supply the original files in `data/raw/` and rerun the smoke command."
                    if missing_raw else f"All {len(EXPECTED_FILES)} required original raw CSVs are present and nonempty.")
    text = f"""# Milestone 5: seasonal naive development findings

## Objective and implementation

Establish reproducible point baselines for the wider seasonal naive, Kalman,
LightGBM and DeepAR comparison. `SeasonalNaiveForecaster` implements the existing
`BaseForecaster`; fresh instances fit independently per series and origin.
The CLI delegates all prediction validation, actual joins and metrics to the
shared development engine. No new dependencies or data-preparation steps are required.

## Data and evaluation boundaries

Actual canonical-data development evaluation was executed for {comparison.model.nunique()} models,
{cfg['selection']['n_per_group'] * len(cfg['selection']['groups'])} selected series,
origins {cfg['forecast']['development_origins']}, and horizon {cfg['forecast']['horizon']}.
Each model produces {int(comparison.loc[comparison.scope.eq('all_folds_pooled'), 'n_rows'].iloc[0]):,} forecasts.
The canonical Parquet reader is filtered through d_{cfg['forecast']['final_origin']}.
Final targets d_{cfg['forecast']['final_origin'] + 1}–d_{cfg['forecast']['final_end_day']}
were not loaded, evaluated, scored or plotted. Cohort, seed, YAML, metrics and
split definitions remain unchanged.

## Model definitions and missing values

- **Last value:** y_hat(t+h) = y(s), where s is the latest finite observation
  at or before t. With no finite observations, forecast zero.
- **Weekly seasonal:** y_hat(t+h) = y(t-6+((h-1) mod 7)) for complete history.
  Freeze these seven calendar positions for the entire horizon. An unavailable
  position searches earlier observations on that weekday, then uses the latest
  overall observation, then zero if all history is unavailable.
- Zero sales remain valid. Nonfinite observations are unavailable; negative
  finite sales are rejected. Missing calendar rows retain calendar alignment.
  Each model records source days and fallback policies in `fit_summary.csv`.
- Both models return NaN q10/q50/q90. Pinball metrics remain NaN; no uncertainty
  intervals are fabricated.

## Actual development accuracy

{chr(10).join(rows)}

{comparison_conclusion(comparison)} Selection uses pooled WAPE only; WAPE is a ratio.
Metrics pool observed series-days, rather than averaging per-series errors.
All-missing actuals and zero-denominator WAPE remain unavailable.

The weekly model preserves the last observed weekday pattern, while the last-value
model propagates a single day's level. Their comparison describes this cohort
and these development windows; it does not establish future or holdout performance.
Sparse sales and one unusual recent week can make either baseline inaccurate.

## Fallback audit

Counts cover all series/origin fits. Weekly counts refer to the seven frozen
positions per fit, rather than each repeated forecast day. Both sets of state
diagnostics are computed for every variant.

{chr(10).join(fallback_lines)}

## Reproducible figures and artifacts

Representatives are the alphabetically first existing selected series in each
volume group, using selection metadata fixed by d_{cfg['selection']['cutoff_day']}:
`{json.dumps(representatives, sort_keys=True)}`.
Each forecast plot uses the same first development origin d_{first_origin},
the last 56 available training days and the configured forecast horizon.
Representatives are chosen independently of forecast errors.

{chr(10).join(figure_lines)}

Evaluator artifacts and source/fallback diagnostics: `{artifact_label}/<variant>/`.
Comparison: `reports/m5_baseline_comparison.csv` (per-origin and pooled rows).
Integrity checks: `<variant>/integrity.json`.

## Reproduction and automated coverage

```bash
source .venv/bin/activate
python -m pytest tests/test_seasonal_naive.py -v
python -m pytest
python -m scripts.smoke_test
python -m scripts.run_seasonal_naive
```

Synthetic tests cover exact 28-day patterns, calendar gaps, zero/missing and
nonfinite observations, weekday/overall/zero fallbacks, schema and origin
checks, deterministic outputs, input immutability, factory isolation, shared
scoring, saved artifacts, CLI orchestration and exclusion of held-out targets.
The smoke check independently requires all {len(EXPECTED_FILES)} original raw CSVs, even though
these baselines consume the existing canonical panel. Its setup status must be
reported separately from the development evaluation.

{setup_status}

## Limitations and completion checklist

Observed sales need not equal unconstrained customer demand. Neither model
infers stockouts, learns covariates or estimates uncertainty. Pooled errors can
be dominated by higher-volume series. No final-holdout conclusions are made.

- [x] Both deterministic adapters and fresh-fold factories implemented.
- [x] Explicit, recorded missing-observation policies; zeros preserved.
- [x] Shared backtester and configured development splits used unchanged.
- [x] Canonical-data evaluation and exact-grid integrity checks executed.
- [x] Per-origin/pooled metrics, model comparison and reproducible figures saved.
- [x] Synthetic tests and reproducible verification commands supplied.
- [x] Reserved final holdout excluded from this milestone.
"""
    (data_path(cfg, "reports") / "m5_seasonal_naive_findings.md").write_text(text)


def run_experiment(
    cfg: dict, models: tuple[str, ...] | list[str] = VARIANTS,
    output_dir: str | Path = "seasonal_naive",
) -> pd.DataFrame:
    """Load only development data and produce bounded, model-specific M5 outputs."""
    if not models or len(set(models)) != len(models) or any(model not in VARIANTS for model in models):
        raise ValueError(f"Choose distinct models from {VARIANTS}")
    panel = load_panel(cfg, cutoff=cfg["forecast"]["final_origin"])
    if panel.day_num.max() > cfg["forecast"]["final_origin"]:
        raise ValueError("Panel loader returned reserved final-holdout rows")
    output = Path(output_dir)
    if not output.is_absolute():
        output = data_path(cfg, "processed") / output
    comparisons, forecasts, summaries = [], [], []
    for variant in models:
        fold_summaries = []
        class RecordingForecaster(SeasonalNaiveForecaster):
            def fit(self, inputs):
                super().fit(inputs)
                fold_summaries.append(self.fit_summary_.copy())
                return self

        def factory() -> SeasonalNaiveForecaster:
            return RecordingForecaster(variant)

        destination = output / variant
        predictions, metrics, metadata = run_backtest(panel, cfg, factory, output_dir=destination)
        # Validate the saved forecasts too, catching schema/date round-trip issues.
        saved = pd.read_parquet(destination / "predictions.parquet")
        pd.testing.assert_frame_equal(saved, predictions)
        integrity = validate_integrity(saved, metadata, panel, cfg)
        (destination / "integrity.json").write_text(json.dumps(integrity, indent=2))
        summary = pd.concat(fold_summaries, ignore_index=True)
        summary.to_csv(destination / "fit_summary.csv", index=False)
        summaries.append(summary.assign(variant=variant))
        comparisons.append(metrics)
        forecasts.append(predictions)
        print(f"{variant}: {len(saved):,} predictions; integrity passed; final holdout evaluated: False")
    comparison = pd.concat(comparisons, ignore_index=True)
    reports = data_path(cfg, "reports")
    reports.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(reports / "m5_baseline_comparison.csv", index=False)
    representatives = select_representatives(panel, cfg)
    paths = create_figures(panel, pd.concat(forecasts, ignore_index=True), comparison, representatives, cfg)
    write_findings(comparison, pd.concat(summaries, ignore_index=True), representatives, paths, output, cfg)
    print(comparison[["model", "origin", "scope", "mae", "rmse", "wape"]].to_string(index=False))
    print(comparison_conclusion(comparison))
    print(f"Artifacts: {output}\nComparison and findings: {reports}\nFigures generated: {len(paths)}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--models", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--output-dir", default="seasonal_naive")
    args = parser.parse_args()
    run_experiment(load_config(args.config), args.models, args.output_dir)


if __name__ == "__main__":
    main()
