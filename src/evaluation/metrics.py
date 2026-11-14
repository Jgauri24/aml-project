"""Reusable point and probabilistic forecasting metrics.

Missing actuals are excluded from point and probabilistic scores. A missing or
nonfinite prediction for an observed actual is an error. WAPE and MASE are
undefined when their scale denominator is zero. sMAPE assigns zero contribution
to a row where both actual and forecast are zero.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


def observed_pairs(y_true, y_pred):
    """Return observed actual/forecast pairs, rejecting invalid scored predictions."""
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y.ndim != 1 or p.shape != y.shape:
        raise ValueError("Expected equally sized one-dimensional arrays")
    if np.isinf(y).any():
        raise ValueError("Infinite actuals are invalid")
    observed = ~np.isnan(y)
    if not np.isfinite(p[observed]).all():
        raise ValueError("Every observed actual needs a finite prediction")
    return y[observed], p[observed]


def mae(y_true, y_pred):
    """Mean absolute error: mean(|actual - forecast|), in target units."""
    y, p = observed_pairs(y_true, y_pred)
    return float(np.abs(y - p).mean()) if len(y) else float("nan")


def rmse(y_true, y_pred):
    """Root mean squared error: sqrt(mean((actual - forecast)^2))."""
    y, p = observed_pairs(y_true, y_pred)
    return float(np.sqrt(np.square(y - p).mean())) if len(y) else float("nan")


def wape(y_true, y_pred):
    """WAPE: sum(|actual - forecast|) / sum(|actual|); zero denominator is NA."""
    y, p = observed_pairs(y_true, y_pred)
    denominator = np.abs(y).sum()
    return float(np.abs(y - p).sum() / denominator) if denominator else float("nan")


def smape(y_true, y_pred):
    """sMAPE in percent: mean(200*|y-p|/(|y|+|p|)); both-zero pairs count as 0."""
    y, p = observed_pairs(y_true, y_pred)
    if not len(y):
        return float("nan")
    denominator = np.abs(y) + np.abs(p)
    terms = np.divide(200.0 * np.abs(y - p), denominator,
                      out=np.zeros_like(denominator, dtype=float), where=denominator != 0)
    return float(terms.mean())


def mase_scale(y_train, seasonal_period: int = 1) -> float:
    """Mean absolute seasonal-naive training error at lag ``seasonal_period``.

    The input must contain training observations only, in chronological order,
    with missing calendar days represented as NaN rather than removed.
    """
    if isinstance(seasonal_period, bool) or int(seasonal_period) != seasonal_period or seasonal_period < 1:
        raise ValueError("seasonal_period must be a positive integer")
    values = np.asarray(y_train, dtype=float)
    if values.ndim != 1:
        raise ValueError("Training targets must be one-dimensional")
    if np.isinf(values).any():
        raise ValueError("Infinite training targets are invalid")
    period = int(seasonal_period)
    if len(values) <= period:
        return float("nan")
    earlier, later = values[:-period], values[period:]
    valid = np.isfinite(earlier) & np.isfinite(later)
    if not valid.any():
        return float("nan")
    scale = float(np.abs(later[valid] - earlier[valid]).mean())
    return scale if scale > 0 else float("nan")


def mase(y_true, y_pred, y_train, seasonal_period: int = 1):
    """MASE: mean(|test error|) divided by the seasonal-naive training MAE."""
    y, p = observed_pairs(y_true, y_pred)
    if not len(y):
        return float("nan")
    scale = mase_scale(y_train, seasonal_period)
    return float(np.abs(y - p).mean() / scale) if np.isfinite(scale) else float("nan")


def pinball_loss(y_true, y_pred, q):
    """Quantile loss: mean(max(q*(y-p), (q-1)*(y-p))) for 0 < q < 1."""
    if not 0 < q < 1:
        raise ValueError("Quantile must be between zero and one")
    y, p = observed_pairs(y_true, y_pred)
    residual = y - p
    return float(np.maximum(q * residual, (q - 1) * residual).mean()) if len(y) else float("nan")


def _validated_quantiles(scored: pd.DataFrame) -> pd.DataFrame | None:
    columns = ["q10", "q50", "q90"]
    if not set(columns).issubset(scored.columns):
        return None
    values = scored[columns].to_numpy(dtype=float)
    if np.isnan(values).all():
        return None
    if not np.isfinite(values).all():
        raise ValueError("Probabilistic forecasts must provide all three finite quantiles or none")
    q = scored[columns].astype(float)
    if (q.q10.lt(0).any() or q.q10.gt(q.q50).any() or q.q50.gt(q.q90).any()):
        raise ValueError("Quantiles must satisfy 0 <= q10 <= q50 <= q90")
    return q


def add_mase_scales(scored: pd.DataFrame, training_history,
                    seasonal_period: int = 7) -> pd.DataFrame:
    """Attach fold-local MASE denominators, derived only from supplied history.

    ``training_history`` can be a single fold's history, a full historical panel
    (filtered to each forecast origin), or a mapping from origin to fold history.
    For sliding windows, pass the already-windowed history for each origin as a
    mapping so older observations outside that window cannot set the scale.
    """
    required = {"series_id", "origin"}
    if not required.issubset(scored.columns):
        raise ValueError("Scored forecasts need series_id and origin for MASE scaling")
    if isinstance(training_history, Mapping):
        histories = {int(origin): frame for origin, frame in training_history.items()}
    else:
        histories = {int(origin): training_history for origin in scored.origin.dropna().unique()}
    scales = {}
    for origin, group in scored.groupby("origin", observed=True, sort=False):
        origin = int(origin)
        if origin not in histories:
            raise ValueError(f"No training history supplied for origin {origin}")
        hist = histories[origin]
        needed = {"series_id", "day_num", "units_sold"}
        if not needed.issubset(hist.columns):
            raise ValueError(f"Training history must include {sorted(needed)}")
        hist = hist.loc[hist.day_num.le(origin)]
        for series_id in group.series_id.astype(str).unique():
            series = hist.loc[hist.series_id.astype(str).eq(series_id)].sort_values("day_num")
            # Preserve gaps as NaN positions so lag 7 means seven calendar days.
            if series.empty:
                scale = float("nan")
            else:
                first, last = int(series.day_num.min()), int(series.day_num.max())
                daily = series.set_index("day_num").units_sold.reindex(range(first, last + 1)).to_numpy(dtype=float)
                scale = mase_scale(daily, seasonal_period)
            scales[(series_id, origin)] = scale
    result = scored.copy()
    result["mase_scale"] = [scales.get((str(series_id), int(origin)), np.nan)
                            for series_id, origin in zip(result.series_id, result.origin)]
    return result


def _mase_from_scored(scored: pd.DataFrame) -> tuple[float, int]:
    y, p = observed_pairs(scored["actual"], scored["y_pred"])
    if "mase_scale" not in scored or not len(y):
        return float("nan"), 0
    observed = scored.actual.notna().to_numpy()
    scale = pd.to_numeric(scored.mase_scale, errors="coerce").to_numpy(dtype=float)[observed]
    valid = np.isfinite(scale) & (scale > 0)
    if not valid.any():
        return float("nan"), 0
    return float((np.abs(y[valid] - p[valid]) / scale[valid]).mean()), int(valid.sum())


def score_predictions(scored: pd.DataFrame, training_history=None,
                      seasonal_period: int = 7) -> dict:
    """Score a scored-prediction frame; actuals are expected in ``actual``.

    Pass fold-local history to calculate MASE, or provide a previously attached
    ``mase_scale`` column. Existing callers without training data receive NA for
    MASE, which avoids accidentally scaling from evaluation actuals.
    """
    if not {"actual", "y_pred"}.issubset(scored.columns):
        raise ValueError("Scored predictions require actual and y_pred columns")
    if training_history is not None:
        scored = add_mase_scales(scored, training_history, seasonal_period)
    y, p = scored["actual"].to_numpy(), scored["y_pred"].to_numpy()
    scale_mase, n_mase = _mase_from_scored(scored)
    result = {"n_rows": len(scored), "n_observed": int(pd.notna(y).sum()),
              "mae": mae(y, p), "rmse": rmse(y, p), "wape": wape(y, p),
              "smape": smape(y, p), "mase": scale_mase, "n_mase": n_mase}
    quantiles = _validated_quantiles(scored)
    for name, q_value in [("q10", 0.1), ("q50", 0.5), ("q90", 0.9)]:
        result[f"pinball_{name}"] = (
            float("nan") if quantiles is None else pinball_loss(y, quantiles[name].to_numpy(), q_value))
    if quantiles is None:
        result.update(coverage_80=float("nan"), mean_interval_width_80=float("nan"), wis_80=float("nan"))
    else:
        observed = ~pd.isna(y)
        actual = np.asarray(y, dtype=float)[observed]
        lower = quantiles.q10.to_numpy()[observed]
        median = quantiles.q50.to_numpy()[observed]
        upper = quantiles.q90.to_numpy()[observed]
        if not len(actual):
            coverage = width = wis_value = float("nan")
        else:
            coverage = float(((actual >= lower) & (actual <= upper)).mean())
            width = float((upper - lower).mean())
            alpha = 0.2
            interval_score = (upper - lower
                              + (2 / alpha) * (lower - actual) * (actual < lower)
                              + (2 / alpha) * (actual - upper) * (actual > upper))
            # One central interval plus the median: weights alpha/2 and 1/2.
            wis = (0.5 * np.abs(actual - median) + (alpha / 2) * interval_score) / 1.5
            wis_value = float(wis.mean())
        result.update(coverage_80=coverage, mean_interval_width_80=width, wis_80=wis_value)
    return result


def compute_metric_tables(
    scored_predictions: pd.DataFrame,
    training_history=None,
    series_metadata: pd.DataFrame | None = None,
    demand_segments: pd.DataFrame | Mapping | None = None,
    seasonal_period: int = 7,
) -> dict[str, pd.DataFrame]:
    """Build portfolio, SKU/store, store, origin, horizon, and segment tables."""
    data = scored_predictions.copy()
    if training_history is not None and "mase_scale" not in data:
        data = add_mase_scales(data, training_history, seasonal_period)
    if "mase_scale" not in data:
        data["mase_scale"] = np.nan
    if series_metadata is not None:
        identity = series_metadata[["series_id", "item_id", "store_id"]].drop_duplicates()
        if identity.duplicated("series_id").any():
            raise ValueError("series_id must map to exactly one SKU/store pair")
        data = data.merge(identity, on="series_id", how="left", validate="many_to_one")
        if data[["item_id", "store_id"]].isna().any().any():
            raise ValueError("Some scored series have no SKU/store metadata")
    if demand_segments is not None:
        if isinstance(demand_segments, Mapping):
            segment = pd.DataFrame({"series_id": list(demand_segments),
                                    "demand_segment": list(demand_segments.values())})
        else:
            segment = demand_segments.copy()
            if "demand_segment" not in segment and "volume_group" in segment:
                segment = segment.rename(columns={"volume_group": "demand_segment"})
            if not {"series_id", "demand_segment"}.issubset(segment.columns):
                raise ValueError("Demand segments require series_id and demand_segment columns")
            segment = segment[["series_id", "demand_segment"]]
        if segment.duplicated("series_id").any():
            raise ValueError("Each series must have at most one demand segment")
        data = data.merge(segment, on="series_id", how="left", validate="many_to_one")

    def aggregate(keys: list[str], scope: str) -> pd.DataFrame:
        rows = []
        grouped = data.groupby(keys, dropna=False, observed=True, sort=True) if keys else [((), data)]
        for key, group in grouped:
            key_values = key if isinstance(key, tuple) else (key,)
            row = dict(zip(keys, key_values))
            row.update(scope=scope, **score_predictions(group, seasonal_period=seasonal_period))
            rows.append(row)
        return pd.DataFrame(rows)

    tables = {"overall": aggregate(["model"], "portfolio")}
    if {"series_id", "item_id", "store_id"}.issubset(data.columns):
        tables["sku_store"] = aggregate(["model", "series_id", "item_id", "store_id"], "sku_store")
    if "store_id" in data:
        tables["store"] = aggregate(["model", "store_id"], "store")
    if "origin" in data:
        tables["origin"] = aggregate(["model", "origin"], "forecast_origin")
    if "horizon" in data:
        tables["horizon"] = aggregate(["model", "horizon"], "horizon_day")
    if "demand_segment" in data:
        tables["demand_segment"] = aggregate(["model", "demand_segment"], "demand_segment")
    return tables


def save_metric_tables(tables: Mapping[str, pd.DataFrame], output_dir: str | Path,
                       file_format: str = "csv") -> list[Path]:
    """Save each aggregation table as CSV or Parquet using project conventions."""
    if file_format not in {"csv", "parquet"}:
        raise ValueError("file_format must be 'csv' or 'parquet'")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in tables.items():
        path = destination / f"metrics_{name}.{file_format}"
        if file_format == "csv":
            table.to_csv(path, index=False)
        else:
            table.to_parquet(path, index=False)
        paths.append(path)
    return paths
