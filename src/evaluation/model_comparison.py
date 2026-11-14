"""Paired, dependence-aware comparisons of rolling-origin forecasts.

All differences use ``model_a - model_b``: negative values favor model A for
loss metrics. Forecast rows are paired on series, origin, horizon, and date;
the DM test then aggregates per-series losses to one equally weighted value per
origin so daily SKU observations are never treated as independent replicates.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

PAIR_KEYS = ["series_id", "origin", "horizon", "target_date"]


def _paired_rows(predictions: pd.DataFrame, model_a: str, model_b: str):
    required = {"model", *PAIR_KEYS, "actual", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {sorted(missing)}")
    selected = predictions.loc[predictions.model.isin([model_a, model_b])].copy()
    for model, frame in selected.groupby("model", sort=False):
        if frame.duplicated(PAIR_KEYS).any():
            raise ValueError(f"Duplicate forecast keys for {model}")
        qcols = ["q10", "q50", "q90"]
        if set(qcols).issubset(frame.columns):
            qvalues = frame[qcols].to_numpy(dtype=float)
            if not np.isnan(qvalues).all():
                if not np.isfinite(qvalues).all():
                    raise ValueError(f"{model} must supply all finite quantiles or none")
                if (np.any(qvalues[:, 0] < 0) or np.any(qvalues[:, 0] > qvalues[:, 1])
                        or np.any(qvalues[:, 1] > qvalues[:, 2])):
                    raise ValueError(f"{model} quantiles must satisfy 0 <= q10 <= q50 <= q90")
    left = selected.loc[selected.model.eq(model_a)].drop(columns="model")
    right = selected.loc[selected.model.eq(model_b)].drop(columns="model")
    left = left.rename(columns={"actual": "actual_a", "y_pred": "pred_a",
                                "q10": "q10_a", "q50": "q50_a", "q90": "q90_a"})
    right = right.rename(columns={"actual": "actual_b", "y_pred": "pred_b",
                                  "q10": "q10_b", "q50": "q50_b", "q90": "q90_b"})
    merged = left.merge(right, on=PAIR_KEYS, how="outer", indicator=True, validate="one_to_one")
    n_unmatched_a = int(merged._merge.eq("left_only").sum())
    n_unmatched_b = int(merged._merge.eq("right_only").sum())
    paired = merged.loc[merged._merge.eq("both")].drop(columns="_merge").copy()
    both_missing = paired.actual_a.isna() & paired.actual_b.isna()
    different_actuals = (~both_missing) & (
        paired.actual_a.isna() | paired.actual_b.isna() |
        ~np.isclose(paired.actual_a.fillna(0), paired.actual_b.fillna(0), rtol=0, atol=0)
    )
    if different_actuals.any():
        raise ValueError("Paired forecasts have different/missing evaluation actuals")
    paired = paired.loc[~both_missing].copy()
    if paired.empty:
        raise ValueError(f"No common observed forecast rows for {model_a} and {model_b}")
    if not np.isfinite(paired.actual_a.to_numpy(dtype=float)).all():
        raise ValueError("Observed evaluation actuals must be finite")
    if not np.isfinite(paired[["pred_a", "pred_b"]].to_numpy(dtype=float)).all():
        raise ValueError("Every paired observed row needs finite point forecasts")
    paired["abs_loss_a"] = np.abs(paired.actual_a - paired.pred_a)
    paired["abs_loss_b"] = np.abs(paired.actual_a - paired.pred_b)
    paired["sq_loss_a"] = np.square(paired.actual_a - paired.pred_a)
    paired["sq_loss_b"] = np.square(paired.actual_a - paired.pred_b)
    return paired, n_unmatched_a, n_unmatched_b


def paired_loss_comparisons(predictions: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    """Summarize aligned row-level paired loss differences (A minus B)."""
    paired, unmatched_a, unmatched_b = _paired_rows(predictions, model_a, model_b)
    loss_columns = {"absolute_error": ("abs_loss_a", "abs_loss_b"),
                    "squared_error": ("sq_loss_a", "sq_loss_b")}
    rows = []
    for loss, (a_col, b_col) in loss_columns.items():
        diff = paired[a_col] - paired[b_col]
        rows.append({"model_a": model_a, "model_b": model_b, "loss": loss,
                     "mean_difference_a_minus_b": diff.mean(), "median_difference": diff.median(),
                     "sd_difference": diff.std(ddof=1), "n_paired_rows": len(diff),
                     "n_series": paired.series_id.nunique(), "n_origins": paired.origin.nunique(),
                     "unmatched_rows_a": unmatched_a, "unmatched_rows_b": unmatched_b})
    # Pinball losses are meaningful only when both models supply valid quantiles.
    for q, tau in (("q10", .1), ("q50", .5), ("q90", .9)):
        ca, cb = f"{q}_a", f"{q}_b"
        if ca not in paired or cb not in paired:
            continue
        valid = np.isfinite(paired[[ca, cb]].to_numpy(dtype=float)).all(axis=1)
        if not valid.any():
            continue
        actual = paired.loc[valid, "actual_a"].to_numpy(float)
        qa, qb = paired.loc[valid, ca].to_numpy(float), paired.loc[valid, cb].to_numpy(float)
        la = np.maximum(tau * (actual - qa), (tau - 1) * (actual - qa))
        lb = np.maximum(tau * (actual - qb), (tau - 1) * (actual - qb))
        diff = la - lb
        rows.append({"model_a": model_a, "model_b": model_b, "loss": f"pinball_{q}",
                     "mean_difference_a_minus_b": diff.mean(), "median_difference": np.median(diff),
                     "sd_difference": np.std(diff, ddof=1) if len(diff) > 1 else np.nan,
                     "n_paired_rows": len(diff), "n_series": paired.loc[valid, "series_id"].nunique(),
                     "n_origins": paired.loc[valid, "origin"].nunique(),
                     "unmatched_rows_a": unmatched_a, "unmatched_rows_b": unmatched_b})
    return pd.DataFrame(rows)


def _hac_long_run_variance(values: np.ndarray, max_lag: int) -> float:
    """Bartlett/Newey-West estimate of the variance of a stationary series."""
    x = np.asarray(values, dtype=float)
    x = x - x.mean()
    n = len(x)
    variance = float(np.dot(x, x) / n)
    for lag in range(1, min(max_lag, n - 1) + 1):
        gamma = float(np.dot(x[lag:], x[:-lag]) / n)
        variance += 2 * (1 - lag / (max_lag + 1)) * gamma
    return max(variance, 0.0)


def diebold_mariano_test(
    predictions: pd.DataFrame,
    model_a: str,
    model_b: str,
    *,
    loss: str = "absolute_error",
    horizon: int | None = None,
    origin_step: int | None = None,
    min_origins: int = 8,
) -> dict:
    """DM test on equally weighted per-origin loss differences.

    ``loss`` may be ``absolute_error``, ``squared_error`` or ``pinball_q10`` /
    ``pinball_q50`` / ``pinball_q90``. HAC uses Bartlett weights and a lag count
    for overlapping forecast blocks, ``max(0, ceil(horizon/origin_step)-1)``. An
    HLN finite-sample correction and Student-t reference are used when estimable.
    Fewer than ``min_origins`` blocks are deliberately reported as insufficient.
    """
    paired, ua, ub = _paired_rows(predictions, model_a, model_b)
    if loss in {"absolute_error", "mae"}:
        a, b = paired.abs_loss_a, paired.abs_loss_b
    elif loss in {"squared_error", "mse"}:
        a, b = paired.sq_loss_a, paired.sq_loss_b
    elif loss.startswith("pinball_"):
        q = loss.removeprefix("pinball_")
        tau = {"q10": .1, "q50": .5, "q90": .9}.get(q)
        if tau is None or f"{q}_a" not in paired or f"{q}_b" not in paired:
            raise ValueError("Pinball DM requires matching q10/q50/q90 forecasts")
        qa, qb = paired[f"{q}_a"].to_numpy(float), paired[f"{q}_b"].to_numpy(float)
        y = paired.actual_a.to_numpy(float)
        a = np.maximum(tau * (y - qa), (tau - 1) * (y - qa))
        b = np.maximum(tau * (y - qb), (tau - 1) * (y - qb))
    else:
        raise ValueError("loss must be absolute_error, squared_error, or pinball_q10/q50/q90")
    paired = paired.copy()
    paired["loss_diff"] = np.asarray(a) - np.asarray(b)
    # First average each series within each forecast origin, then weight series equally.
    origin_diff = paired.groupby(["origin", "series_id"], observed=True).loss_diff.mean()
    origin_diff = origin_diff.groupby(level="origin").mean().sort_index()
    d = origin_diff.to_numpy(float)
    n = len(d)
    h = int(horizon or paired.horizon.max())
    step = int(origin_step or (np.median(np.diff(origin_diff.index.to_numpy(dtype=float))) if n > 1 else h))
    overlap_lag = max(0, int(np.ceil(h / max(step, 1))) - 1)
    hac_lag = min(overlap_lag, max(0, n - 1))
    mean_diff = float(d.mean()) if n else np.nan
    lrv = _hac_long_run_variance(d, hac_lag) if n else np.nan
    correction = np.nan
    statistic = p_value = np.nan
    status = "insufficient_origins"
    if n >= min_origins and np.isfinite(lrv) and lrv > 0:
        # Effective h is the number of overlapping origin blocks, at least one.
        h_eff = max(1, overlap_lag + 1)
        correction_sq = (n + 1 - 2 * h_eff + h_eff * (h_eff - 1) / n) / n
        if correction_sq > 0:
            correction = float(np.sqrt(correction_sq))
            statistic = float(mean_diff / np.sqrt(lrv / n) * correction)
            p_value = float(2 * student_t.sf(abs(statistic), df=n - 1))
            status = "ok"
        else:
            status = "hln_correction_not_estimable"
    elif n >= min_origins:
        status = "zero_or_invalid_long_run_variance"
    if status == "ok":
        if p_value < .05:
            interpretation = f"Significant difference; {'model_a' if mean_diff < 0 else 'model_b'} has lower {loss} (uncorrected p<0.05)."
        else:
            interpretation = f"No significant difference detected at 0.05 for {loss}."
    else:
        interpretation = f"No significance conclusion: {status} (n_origins={n}, required={min_origins})."
    return {"model_a": model_a, "model_b": model_b, "loss": loss,
            "mean_loss_difference_a_minus_b": mean_diff, "dm_statistic": statistic,
            "p_value": p_value, "p_adjusted_holm": np.nan, "n_origins": n,
            "n_series": paired.series_id.nunique(), "hac_lag": hac_lag,
            "horizon": h, "origin_step": step, "hln_correction": correction,
            "long_run_variance": lrv, "unmatched_rows_a": ua, "unmatched_rows_b": ub,
            "status": status, "interpretation": interpretation}


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    valid = np.isfinite(p)
    vals = p[valid]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    adj_sorted = np.maximum.accumulate([(m - i) * value for i, value in enumerate(ranked)])
    adj_sorted = np.minimum(adj_sorted, 1.0)
    restored = np.empty(m)
    restored[order] = adj_sorted
    adjusted[valid] = restored
    return adjusted


def _performance_metric(frame: pd.DataFrame, metric: str) -> float:
    y = frame.actual.to_numpy(float)
    if metric == "mae":
        return float(np.mean(np.abs(y - frame.y_pred.to_numpy(float))))
    if metric == "rmse":
        return float(np.sqrt(np.mean(np.square(y - frame.y_pred.to_numpy(float)))))
    if metric == "wape":
        denom = np.abs(y).sum()
        return float(np.abs(y - frame.y_pred.to_numpy(float)).sum() / denom) if denom else np.nan
    if metric == "smape":
        pred = frame.y_pred.to_numpy(float)
        denom = np.abs(y) + np.abs(pred)
        return float(np.divide(200 * np.abs(y - pred), denom,
                               out=np.zeros_like(denom), where=denom != 0).mean())
    if metric.startswith("pinball_"):
        q = metric.removeprefix("pinball_")
        tau = {"q10": .1, "q50": .5, "q90": .9}.get(q)
        col = {"q10": "q10", "q50": "q50", "q90": "q90"}.get(q)
        if tau is None or col not in frame or not np.isfinite(frame[col].to_numpy(float)).all():
            return np.nan
        residual = y - frame[col].to_numpy(float)
        return float(np.maximum(tau * residual, (tau - 1) * residual).mean())
    if metric == "wis_80":
        cols = ["q10", "q50", "q90"]
        if not set(cols).issubset(frame) or not np.isfinite(frame[cols].to_numpy(float)).all():
            return np.nan
        low, med, high = (frame[c].to_numpy(float) for c in cols)
        alpha = .2
        score = high - low + (2 / alpha) * (low - y) * (y < low) + (2 / alpha) * (y - high) * (y > high)
        return float(np.mean((.5 * np.abs(y - med) + (alpha / 2) * score) / 1.5))
    raise ValueError(f"Unsupported comparison metric: {metric}")


def _paired_model_view(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
    rename = {"actual_a": "actual", f"pred_{suffix}": "y_pred"}
    rename.update({f"{q}_{suffix}": q for q in ("q10", "q50", "q90")})
    return frame.rename(columns=rename)


def bootstrap_performance_differences(
    predictions: pd.DataFrame,
    model_a: str,
    model_b: str,
    *,
    metrics: Sequence[str] = ("mae", "rmse", "wape"),
    n_bootstrap: int = 2000,
    confidence: float = .95,
    seed: int = 12345,
) -> pd.DataFrame:
    """Cluster bootstrap by SKU-store series, preserving each cluster's rows."""
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("n_bootstrap must be positive and confidence must lie in (0,1)")
    paired, ua, ub = _paired_rows(predictions, model_a, model_b)
    clusters = paired.series_id.drop_duplicates().to_numpy()
    if len(clusters) < 2:
        return pd.DataFrame([{"model_a": model_a, "model_b": model_b, "metric": m,
                              "difference_a_minus_b": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                              "n_bootstrap": n_bootstrap, "n_series": len(clusters),
                              "n_paired_rows": len(paired), "zero_in_ci": pd.NA,
                              "status": "fewer_than_two_series", "unmatched_rows_a": ua,
                              "unmatched_rows_b": ub} for m in metrics])
    rng = np.random.default_rng(seed)
    by_series = {s: paired.loc[paired.series_id.eq(s)] for s in clusters}
    alpha = (1 - confidence) / 2
    rows = []
    for metric in metrics:
        a_value = _performance_metric(_paired_model_view(paired, "a"), metric)
        b_value = _performance_metric(_paired_model_view(paired, "b"), metric)
        observed_diff = a_value - b_value
        draws = []
        for _ in range(n_bootstrap):
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            sample = pd.concat([by_series[s] for s in sampled], ignore_index=True)
            a = _performance_metric(_paired_model_view(sample, "a"), metric)
            b = _performance_metric(_paired_model_view(sample, "b"), metric)
            if np.isfinite(a) and np.isfinite(b):
                draws.append(a - b)
        if draws:
            lo, hi = np.quantile(draws, [alpha, 1 - alpha])
            status, zero = "ok", bool(lo <= 0 <= hi)
        else:
            lo = hi = np.nan
            status, zero = "metric_unavailable", pd.NA
        rows.append({"model_a": model_a, "model_b": model_b, "metric": metric,
                     "difference_a_minus_b": observed_diff, "ci_lower": lo, "ci_upper": hi,
                     "confidence": confidence, "n_bootstrap": n_bootstrap, "n_series": len(clusters),
                     "n_paired_rows": len(paired), "zero_in_ci": zero, "status": status,
                     "unmatched_rows_a": ua, "unmatched_rows_b": ub})
    return pd.DataFrame(rows)


def rank_models(
    scored_predictions: pd.DataFrame,
    *,
    series_metadata: pd.DataFrame | None = None,
    demand_segments: pd.DataFrame | Mapping | None = None,
    execution: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Create separate lower-is-better rankings; no combined score is formed."""
    data = scored_predictions.copy()
    if series_metadata is not None:
        data = data.merge(series_metadata[["series_id", "item_id", "store_id"]].drop_duplicates(),
                          on="series_id", how="left", validate="many_to_one")
    if demand_segments is not None:
        data = data.drop(columns="demand_segment", errors="ignore")
        if isinstance(demand_segments, Mapping):
            seg = pd.DataFrame({"series_id": list(demand_segments), "demand_segment": list(demand_segments.values())})
        else:
            seg = demand_segments.rename(columns={"volume_group": "demand_segment"})
        data = data.merge(seg[["series_id", "demand_segment"]], on="series_id", how="left", validate="many_to_one")

    def metric_rows(group_cols, scope, metric):
        groups = data.groupby(group_cols, observed=True, dropna=False, sort=True) if group_cols else [((), data)]
        out = []
        for key, frame in groups:
            keys = key if isinstance(key, tuple) else (key,)
            group_id = dict(zip(group_cols, keys))
            for model, model_frame in frame.groupby("model", observed=True, sort=True):
                try:
                    value = _performance_metric(model_frame.rename(columns={"actual": "actual", "y_pred": "y_pred"}), metric)
                except (ValueError, KeyError):
                    value = np.nan
                out.append({"scope": scope, **group_id, "model": model, "metric": metric, "value": value})
        result = pd.DataFrame(out)
        if not result.empty:
            result["rank"] = result.groupby(group_cols + ["metric"], dropna=False).value.rank(method="min", ascending=True)
        return result

    overall_metrics = ["mae", "rmse", "wape"]
    if "q10" in data:
        overall_metrics += ["pinball_q10", "pinball_q50", "pinball_q90", "wis_80"]
    overall = pd.concat([metric_rows([], "overall_accuracy", m) for m in overall_metrics], ignore_index=True)
    horizon = pd.concat([metric_rows(["horizon"], "horizon", "mae"),
                         metric_rows(["horizon"], "horizon", "rmse")], ignore_index=True) if "horizon" in data else pd.DataFrame()
    segment = pd.concat([metric_rows(["demand_segment"], "demand_segment", "mae"),
                         metric_rows(["demand_segment"], "demand_segment", "wape")], ignore_index=True) if "demand_segment" in data else pd.DataFrame()
    probabilistic = overall.loc[overall.metric.isin(["pinball_q10", "pinball_q50", "pinball_q90", "wis_80"])].copy()

    if execution is not None and not execution.empty:
        exec_rows = []
        for col in ("fit_seconds", "predict_seconds", "total_seconds"):
            if col in execution:
                mean_times = execution.groupby("model", as_index=False)[col].mean().rename(columns={col: "value"})
                mean_times["scope"], mean_times["metric"] = "computational_efficiency", col
                mean_times["rank"] = mean_times.groupby("metric").value.rank(method="min", ascending=True)
                exec_rows.append(mean_times)
        efficiency = pd.concat(exec_rows, ignore_index=True) if exec_rows else pd.DataFrame()
    else:
        efficiency = pd.DataFrame(columns=["model", "metric", "value", "rank"])
    return {"overall_accuracy": overall, "horizon_performance": horizon,
            "demand_segment_performance": segment, "probabilistic_performance": probabilistic,
            "computational_efficiency": efficiency}


def compare_models(
    predictions: pd.DataFrame,
    model_pairs: Sequence[tuple[str, str]] | None = None,
    *,
    horizon: int | None = None,
    origin_step: int | None = None,
    n_bootstrap: int = 2000,
    confidence: float = .95,
    dm_min_origins: int = 8,
    seed: int = 12345,
) -> dict[str, pd.DataFrame]:
    """Run paired summaries, DM tests and series-cluster bootstrap for pairs."""
    models = sorted(predictions.model.dropna().astype(str).unique())
    pairs = list(model_pairs or combinations(models, 2))
    paired_rows, dm_rows, boot_rows = [], [], []
    for a, b in pairs:
        paired_rows.append(paired_loss_comparisons(predictions, a, b))
        losses = ["absolute_error", "squared_error"]
        model_data = predictions.loc[predictions.model.isin([a, b])]
        if {"q10", "q50", "q90"}.issubset(model_data.columns):
            for q in ("q10", "q50", "q90"):
                if model_data.groupby("model")[q].apply(lambda s: s.notna().any()).all():
                    losses.append(f"pinball_{q}")
        dm_rows.extend(diebold_mariano_test(predictions, a, b, loss=loss,
                                           horizon=horizon, origin_step=origin_step,
                                           min_origins=dm_min_origins)
                       for loss in losses)
        boot_rows.append(bootstrap_performance_differences(
            predictions, a, b,
            metrics=("mae", "rmse", "wape", "smape", "pinball_q10", "pinball_q50", "pinball_q90", "wis_80"),
            n_bootstrap=n_bootstrap, confidence=confidence, seed=seed))
    dm = pd.DataFrame(dm_rows)
    if not dm.empty:
        dm["p_adjusted_holm"] = dm.groupby("loss", group_keys=False).p_value.transform(
            lambda values: pd.Series(_holm_adjust(values), index=values.index))
        valid = dm.status.eq("ok")
        dm.loc[valid, "interpretation"] = np.where(
            dm.loc[valid, "p_adjusted_holm"].lt(.05),
            np.where(dm.loc[valid, "mean_loss_difference_a_minus_b"].lt(0),
                     "Significant after Holm correction; " + dm.loc[valid, "model_a"].astype(str) + " has lower loss.",
                     "Significant after Holm correction; " + dm.loc[valid, "model_b"].astype(str) + " has lower loss."),
            "No significant difference detected after Holm correction.")
    return {"paired_comparisons": pd.concat(paired_rows, ignore_index=True) if paired_rows else pd.DataFrame(),
            "dm_tests": dm,
            "bootstrap": pd.concat(boot_rows, ignore_index=True) if boot_rows else pd.DataFrame()}


def save_comparison_results(results: Mapping[str, pd.DataFrame], output_dir: str | Path,
                            file_format: str = "csv") -> list[Path]:
    """Write named comparison/ranking tables as CSV or Parquet."""
    if file_format not in {"csv", "parquet"}:
        raise ValueError("file_format must be csv or parquet")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in results.items():
        path = root / f"{name}.{file_format}"
        frame.to_csv(path, index=False) if file_format == "csv" else frame.to_parquet(path, index=False)
        paths.append(path)
    return paths
