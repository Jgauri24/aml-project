"""Central reproducible development-evaluation pipeline.

The pipeline composes the existing validated panel loader, rolling model suite,
forecast metrics, paired statistical comparisons, runtime records, dashboard
tables and row-level visualizations. The reserved final holdout is never passed
to a model or evaluated here.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

from src.config import ROOT, config_hash, data_path, load_config
from src.data.preprocess import load_panel
from src.data.validate import validate_panel
from src.evaluation.backtesting import attach_actuals
from src.evaluation.metrics import add_mase_scales, compute_metric_tables
from src.evaluation.model_comparison import compare_models, rank_models, save_comparison_results
from src.evaluation.model_suite import run_model_suite
from src.evaluation.splits import development_splits


SUPPORTED_METRICS = {"mae", "rmse", "wape", "smape", "mase", "pinball_q10",
                     "pinball_q50", "pinball_q90", "coverage_80", "mean_interval_width_80", "wis_80"}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_pipeline_config(config_path: str | Path | Mapping | None):
    if isinstance(config_path, Mapping):
        raw = deepcopy(dict(config_path))
        config_file = ROOT / "configs" / "evaluation_pipeline.yaml"
    else:
        config_file = Path(config_path) if config_path else ROOT / "configs" / "evaluation_pipeline.yaml"
        if not config_file.is_absolute():
            config_file = (Path.cwd() / config_file).resolve()
        raw = yaml.safe_load(config_file.read_text())
    base_path = Path(raw.get("base_config", "experiment.yaml"))
    if not base_path.is_absolute():
        base_path = (config_file.parent / base_path).resolve()
    base = load_config(base_path)
    return raw, base, config_file


def _resolve_config(raw: dict, base: dict, overrides: Mapping | None = None) -> tuple[dict, dict]:
    settings = deepcopy(raw.get("evaluation", {}))
    if overrides:
        settings.update(deepcopy(dict(overrides)))
    cfg = deepcopy(base)
    horizon = settings.get("forecast_horizon")
    if horizon is not None:
        cfg["forecast"]["horizon"] = int(horizon)
    origins = settings.get("development_origins")
    if origins is not None:
        cfg["forecast"]["development_origins"] = [int(x) for x in origins]
    if settings.get("generate_origins", False):
        cfg["forecast"]["development_origins"] = None
        if settings.get("origin_start") is not None:
            cfg["backtest"]["origin_start"] = int(settings["origin_start"])
    final_origin = settings.get("final_origin")
    if final_origin is not None:
        cfg["forecast"]["final_origin"] = int(final_origin)
    final_end = settings.get("final_end_day")
    if final_end is not None:
        cfg["forecast"]["final_end_day"] = int(final_end)
        cfg["observation"]["end_day"] = int(final_end)
    if settings.get("selection_cutoff_day") is not None:
        cfg["selection"]["cutoff_day"] = int(settings["selection_cutoff_day"])
        cfg["eda"]["cutoff_day"] = min(cfg["eda"]["cutoff_day"], int(settings["selection_cutoff_day"]))
    for key, target in (("window_mode", "window_mode"),
                        ("training_window_days", "training_window_days"),
                        ("step_size", "step_size"), ("sku_ids", "sku_ids"),
                        ("store_ids", "store_ids")):
        if key in settings and settings[key] is not None:
            cfg["backtest"][target] = settings[key]
    if settings.get("window_mode") == "expanding":
        cfg["backtest"]["training_window_days"] = None
    cfg["seed"] = int(settings.get("random_seed", cfg["seed"]))
    folds = development_splits(cfg)
    metrics = settings.get("metrics", sorted(SUPPORTED_METRICS))
    unknown = set(metrics) - SUPPORTED_METRICS
    if unknown:
        raise ValueError(f"Unsupported configured metrics: {sorted(unknown)}")
    if settings.get("final_holdout", {}).get("evaluate", False) or settings.get("final_holdout", {}).get("tuning", False):
        raise ValueError("Final holdout evaluation/tuning is disabled in the development pipeline")
    settings["metrics"] = list(metrics)
    settings["_resolved_origins"] = [fold.origin for fold in folds]
    settings["_resolved_horizon"] = int(cfg["forecast"]["horizon"])
    return cfg, settings


def _resolve_from_root(path_value: str | Path, config_file: Path) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (config_file.parent / path).resolve()


def _load_validated_panel(raw: dict, base_cfg: dict, config_file: Path,
                          panel_override: pd.DataFrame | None, run_cfg: dict) -> tuple[pd.DataFrame, dict]:
    if panel_override is not None:
        panel = panel_override.copy()
        info = {"source": "in_memory_panel_override", "validation": "structural validation"}
        validate_panel(panel, run_cfg["observation"]["start_day"], run_cfg["observation"]["end_day"])
        digest = hashlib.sha256(pd.util.hash_pandas_object(panel, index=True).values.tobytes()).hexdigest()
        info["dataset_sha256"] = digest
        return panel, info

    data_cfg = raw.get("dataset", {})
    panel_path = _resolve_from_root(data_cfg.get("panel_path", "../data/processed/demand_panel.parquet"), config_file)
    metadata_value = data_cfg.get("panel_metadata_path", "../data/processed/panel_metadata.json")
    metadata_path = _resolve_from_root(metadata_value, config_file)
    if not panel_path.is_file():
        raise FileNotFoundError(f"Validated processed panel is missing: {panel_path}")
    standard_path = (data_path(base_cfg, "processed") / "demand_panel.parquet").resolve()
    if panel_path == standard_path:
        panel = load_panel(base_cfg)
        validation = "load_panel verified config and selection hashes"
    else:
        panel = pd.read_parquet(panel_path)
        validate_panel(panel, base_cfg["observation"]["start_day"], base_cfg["observation"]["end_day"])
        validation = "structural panel validation"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    if data_cfg.get("expected_panel_metadata", True) and not metadata:
        raise FileNotFoundError(f"Processed panel metadata is required but missing: {metadata_path}")
    return panel, {"source": str(panel_path), "metadata_path": str(metadata_path),
                   "validation": validation, "dataset_sha256": _sha256_file(panel_path),
                   "panel_metadata": metadata}


def _model_factories(raw: dict, settings: dict) -> tuple[dict, dict]:
    # Import optional/heavy frameworks only after the canonical panel is validated.
    from src.models.deepar_config import DeepARConfig
    from src.models.deepar_forecaster import make_deepar_factory
    from src.models.kalman_forecaster import make_kalman_factory
    from src.models.lightgbm_forecaster import LightGBMForecaster
    from src.models.seasonal_naive import make_seasonal_naive_factory

    config = raw.get("models", {})
    seed = int(settings["random_seed"])
    horizon = int(settings["_resolved_horizon"])
    factories, model_configs = {}, {}
    naive = config.get("seasonal_naive", {})
    if naive.get("enabled", False):
        variant = naive.get("variant", "weekly_seasonal")
        factories["seasonal_naive"] = make_seasonal_naive_factory(variant)
        model_configs["seasonal_naive"] = {**naive, "seed": seed}
    kalman = config.get("kalman", {})
    if kalman.get("enabled", False):
        options = {k: v for k, v in kalman.items() if k not in {"enabled", "variant"}}
        options.setdefault("seed", seed)
        factories["kalman"] = make_kalman_factory(kalman.get("variant", "local_level_seasonal"), **options)
        model_configs["kalman"] = {**kalman, "seed": options["seed"]}
    lightgbm = config.get("lightgbm", {})
    if lightgbm.get("enabled", False):
        params = dict(lightgbm.get("params", {}))
        params.setdefault("random_state", seed)
        factories["lightgbm"] = lambda params=params: LightGBMForecaster(params)
        model_configs["lightgbm"] = {**lightgbm, "params": params, "random_state": seed}
    deepar = config.get("deepar", {})
    if deepar.get("enabled", False):
        deepar_values = dict(deepar.get("config", {}))
        deepar_values["prediction_length"] = horizon
        deepar_values["seed"] = seed
        deepar_config = DeepARConfig.from_dict(deepar_values)
        deepar_config.validate()
        factories["deepar"] = make_deepar_factory(deepar_config)
        model_configs["deepar"] = {**deepar, "config": deepar_config.to_dict()}
    if not factories:
        raise ValueError("At least one model must be enabled in the pipeline config")
    return factories, model_configs


def _apply_filters(panel: pd.DataFrame, settings: dict) -> pd.DataFrame:
    selected = panel.copy()
    if settings.get("series_ids"):
        selected = selected.loc[selected.series_id.astype(str).isin({str(x) for x in settings["series_ids"]})]
    if settings.get("sku_ids"):
        selected = selected.loc[selected.item_id.astype(str).isin({str(x) for x in settings["sku_ids"]})]
    if settings.get("store_ids"):
        selected = selected.loc[selected.store_id.astype(str).isin({str(x) for x in settings["store_ids"]})]
    if selected.empty:
        raise ValueError("Selected SKU/store series are absent from the validated panel")
    return selected


def _training_histories(panel: pd.DataFrame, cfg: dict) -> dict[int, pd.DataFrame]:
    return {fold.origin: panel.loc[panel.day_num.between(fold.train_start, fold.origin)].copy()
            for fold in development_splits(cfg)}


def _select_metric_columns(tables: dict[str, pd.DataFrame], selected: list[str]) -> dict[str, pd.DataFrame]:
    metric_names = sorted(SUPPORTED_METRICS)
    return {name: table[[column for column in table.columns
                         if column not in metric_names or column in selected]].copy()
            for name, table in tables.items()}


def _demand_segments(panel: pd.DataFrame, cfg: dict, method: str) -> dict[str, str]:
    first_fold = development_splits(cfg)[0]
    first_history = panel.loc[panel.day_num.between(first_fold.train_start, first_fold.origin)]
    means = first_history.groupby("series_id", observed=True).units_sold.mean().dropna()
    if method != "first_origin_training_mean_tertiles":
        raise ValueError(f"Unsupported demand segment method: {method}")
    n_bins = min(3, int(means.nunique()))
    if n_bins <= 1:
        return {str(series_id): "all" for series_id in means.index}
    labels = ["low", "high"] if n_bins == 2 else ["low", "medium", "high"]
    ranks = means.rank(method="first", pct=True)
    bins = pd.qcut(ranks, q=n_bins, labels=labels, duplicates="drop")
    return {str(series_id): str(segment) for series_id, segment in bins.items()}


def _runtime_dashboard(metadata: Mapping) -> pd.DataFrame:
    rows = []
    for model, model_meta in metadata.items():
        for fold in model_meta.get("folds", []):
            n_series = int(fold.get("n_series", 0))
            rows.append({"model": model, "origin": int(fold["origin"]), "scope": "fold",
                         "training_time_seconds": float(fold["fit_seconds"]),
                         "prediction_time_seconds": float(fold["predict_seconds"]),
                         "inference_seconds_per_sku_store": float(fold["prediction_seconds_per_series"]),
                         "total_runtime_seconds": float(fold["total_seconds"]),
                         "n_sku_store_series": n_series,
                         "trainable_parameters": fold.get("trainable_parameters"),
                         "tree_count": fold.get("tree_count"),
                         "serialized_model_bytes": fold.get("serialized_model_bytes"),
                         "resident_state_bytes": fold.get("resident_state_bytes")})
        model_rows = [row for row in rows if row["model"] == model]
        if model_rows:
            prediction_total = sum(row["prediction_time_seconds"] for row in model_rows)
            series_total = sum(row["n_sku_store_series"] for row in model_rows)
            rows.append({"model": model, "origin": pd.NA, "scope": "all_development_folds",
                         "training_time_seconds": sum(row["training_time_seconds"] for row in model_rows),
                         "prediction_time_seconds": prediction_total,
                         "inference_seconds_per_sku_store": prediction_total / series_total if series_total else np.nan,
                         "total_runtime_seconds": sum(row["total_runtime_seconds"] for row in model_rows),
                         "n_sku_store_series": series_total,
                         "trainable_parameters": next((row["trainable_parameters"] for row in model_rows
                                                        if pd.notna(row["trainable_parameters"])), np.nan),
                         "tree_count": next((row["tree_count"] for row in model_rows if pd.notna(row["tree_count"])), np.nan),
                         "serialized_model_bytes": next((row["serialized_model_bytes"] for row in model_rows
                                                         if pd.notna(row["serialized_model_bytes"])), np.nan),
                         "resident_state_bytes": next((row["resident_state_bytes"] for row in model_rows
                                                        if pd.notna(row["resident_state_bytes"])), np.nan)})
    return pd.DataFrame(rows)


def run_evaluation_pipeline(
    config_path: str | Path | Mapping | None = None,
    *,
    panel: pd.DataFrame | None = None,
    overrides: Mapping | None = None,
    experiment_id: str | None = None,
) -> dict:
    """Run development backtests, dashboard metrics, statistics and plots.

    ``panel`` is an explicit validated-panel override used by the small smoke
    example. Production runs load the hashed canonical processed panel.
    """
    pipeline_started = perf_counter()
    raw, base_cfg, config_file = _read_pipeline_config(config_path)
    cfg, settings = _resolve_config(raw, base_cfg, overrides)
    selected_panel, dataset_info = _load_validated_panel(raw, base_cfg, config_file, panel, cfg)
    selected_panel = _apply_filters(selected_panel, settings)
    # The runner's own filter contract is reused after the explicit series filter.
    if settings.get("series_ids"):
        cfg["backtest"]["sku_ids"] = None
        cfg["backtest"]["store_ids"] = None

    factories, model_configs = _model_factories(raw, settings)
    output_cfg = raw.get("outputs", {})
    output_root = Path(output_cfg.get("root", "../reports/evaluation_runs"))
    if not output_root.is_absolute():
        output_root = (config_file.parent / output_root).resolve()
    config_payload = {"pipeline": raw, "resolved_experiment": cfg}
    digest = hashlib.sha256(json.dumps(config_payload, sort_keys=True, default=str).encode()).hexdigest()
    generated_at = datetime.now(timezone.utc).isoformat()
    exp_id = experiment_id or settings.get("experiment_id") or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + digest[:10])
    run_dir = output_root / exp_id
    run_dir.mkdir(parents=True, exist_ok=False)
    suite_dir = run_dir / "backtest"
    started = perf_counter()
    forecasts, suite_metrics, model_metadata = run_model_suite(selected_panel, cfg, factories, output_dir=suite_dir)
    backtest_seconds = perf_counter() - started

    scored = pd.concat([attach_actuals(frame, selected_panel) for frame in forecasts.values()], ignore_index=True)
    histories = _training_histories(selected_panel, cfg)
    scored = add_mase_scales(scored, histories, seasonal_period=7)
    identity = selected_panel[["series_id", "item_id", "store_id"]].drop_duplicates()
    scored = scored.merge(identity, on="series_id", how="left", validate="many_to_one")
    segment_cfg = settings.get("demand_segments", {})
    segments = _demand_segments(selected_panel, cfg, segment_cfg.get("method", "first_origin_training_mean_tertiles"))
    metric_tables = compute_metric_tables(scored, demand_segments=segments, seasonal_period=7)
    metric_tables = _select_metric_columns(metric_tables, settings["metrics"])
    runtime = _runtime_dashboard(model_metadata)
    execution_long = runtime.loc[runtime.scope.eq("fold"), ["model", "training_time_seconds",
                                                              "prediction_time_seconds", "total_runtime_seconds"]].rename(
                                                                  columns={"training_time_seconds": "fit_seconds",
                                                                           "prediction_time_seconds": "predict_seconds",
                                                                           "total_runtime_seconds": "total_seconds"})
    model_names = list(forecasts)
    comparison = compare_models(
        scored, horizon=cfg["forecast"]["horizon"],
        origin_step=cfg.get("backtest", {}).get("step_size", cfg["forecast"]["horizon"]),
        n_bootstrap=int(raw.get("statistics", {}).get("bootstrap_iterations", 2000)),
        confidence=float(raw.get("statistics", {}).get("confidence", .95)),
        dm_min_origins=int(raw.get("statistics", {}).get("dm_min_origins", 8)),
        seed=int(raw.get("statistics", {}).get("seed", cfg["seed"]))) if len(model_names) > 1 else {
            "paired_comparisons": pd.DataFrame(), "dm_tests": pd.DataFrame(), "bootstrap": pd.DataFrame()}
    rankings = rank_models(scored, demand_segments=segments, execution=execution_long)

    dashboard = run_dir / output_cfg.get("dashboard_directory", "dashboard")
    dashboard.mkdir(parents=True, exist_ok=True)
    table_names = {"overall": "overall_model_metrics", "sku_store": "sku_store_metrics",
                   "store": "store_metrics", "origin": "forecast_origin_metrics",
                   "horizon": "horizon_metrics", "demand_segment": "demand_segment_metrics"}
    for key, filename in table_names.items():
        if key in metric_tables:
            metric_tables[key].to_csv(dashboard / f"{filename}.csv", index=False)
            if output_cfg.get("save_parquet", True):
                metric_tables[key].to_parquet(dashboard / f"{filename}.parquet", index=False)
    scored.to_parquet(dashboard / "forecast_predictions.parquet", index=False)
    if output_cfg.get("save_csv", True):
        scored.to_csv(dashboard / "forecast_predictions.csv", index=False)
    quantile_cols = ["q10", "q50", "q90"]
    interval_rows = scored.loc[scored[quantile_cols].notna().all(axis=1)].copy()
    interval_rows.to_csv(dashboard / "prediction_intervals.csv", index=False)
    if output_cfg.get("save_parquet", True):
        interval_rows.to_parquet(dashboard / "prediction_intervals.parquet", index=False)
    runtime.to_csv(dashboard / "runtime_comparisons.csv", index=False)
    comparison_files = save_comparison_results(comparison, run_dir / "statistics", "csv")
    ranking_files = save_comparison_results(rankings, run_dir / "rankings", "csv")
    for name, frame in rankings.items():
        frame.to_csv(dashboard / f"ranking_{name}.csv", index=False)

    visualization_dir = run_dir / output_cfg.get("visualization_directory", "visualizations")
    from src.evaluation.pipeline_visualizations import generate_pipeline_visualizations
    visualization_info = generate_pipeline_visualizations(
        scored, runtime, visualization_dir, baseline_model=next(
            (name for name in forecasts if name.startswith("naive_weekly_seasonal")), None))

    overall_records = metric_tables["overall"].to_dict("records") if "overall" in metric_tables else []
    metadata = {
        "experiment_id": exp_id, "created_at_utc": generated_at,
        "pipeline_config_path": str(config_file), "pipeline_config_sha256": digest,
        "base_experiment_config_sha256": config_hash(base_cfg), "resolved_backtest_config": cfg,
        "dataset": dataset_info, "selected_series_count": int(selected_panel.series_id.nunique()),
        "forecast_origins": settings["_resolved_origins"], "forecast_horizon": int(cfg["forecast"]["horizon"]),
        "training_window": {"mode": cfg["backtest"].get("window_mode", "expanding"),
                            "days": cfg["backtest"].get("training_window_days")},
        "models": model_configs, "model_names": model_names, "metrics_requested": settings["metrics"],
        "metric_summary": _json_safe(overall_records), "runtime_summary": _json_safe(runtime.to_dict("records")),
        "backtest_execution_seconds": backtest_seconds,
        "pipeline_execution_seconds_before_metadata_write": perf_counter() - pipeline_started,
        "statistical_settings": raw.get("statistics", {}),
        "final_holdout": {"evaluated": False, "used_for_tuning": False,
                           "origin": cfg["forecast"]["final_origin"],
                           "target_days": [cfg["forecast"]["final_origin"] + 1,
                                           cfg["forecast"]["final_origin"] + cfg["forecast"]["horizon"]]},
        "dashboard_files": [str(path.relative_to(run_dir)) for path in dashboard.iterdir()],
        "comparison_files": [str(path.relative_to(run_dir)) for path in comparison_files],
        "ranking_files": [str(path.relative_to(run_dir)) for path in ranking_files],
        "visualizations": visualization_info,
        "deep_ar_fit_strategy": "Fresh origin-specific model fit per fold; adapter requires exact fitted origin and has no safe incremental update API.",
    }
    (run_dir / "experiment_metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2, allow_nan=False))
    return {"run_dir": run_dir, "forecasts": forecasts, "scored_predictions": scored,
            "metric_tables": metric_tables, "comparison": comparison, "rankings": rankings,
            "runtime": runtime, "metadata": metadata, "suite_metrics": suite_metrics}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "evaluation_pipeline.yaml"))
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    result = run_evaluation_pipeline(args.config, experiment_id=args.experiment_id)
    print(f"Experiment: {result['metadata']['experiment_id']}")
    print(f"Output: {result['run_dir']}")
    print(f"Models: {', '.join(result['metadata']['model_names'])}")
    print(f"Origins: {result['metadata']['forecast_origins']}")
    print("Final holdout evaluated: false")


if __name__ == "__main__":
    main()
