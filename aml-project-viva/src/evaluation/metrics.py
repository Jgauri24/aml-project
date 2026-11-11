"""Pooled metrics; missing actuals are excluded, missing predictions are not hidden."""
import numpy as np
import pandas as pd


def observed_pairs(y_true, y_pred):
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
    y, p = observed_pairs(y_true, y_pred)
    return float(np.abs(y - p).mean()) if len(y) else float("nan")


def rmse(y_true, y_pred):
    y, p = observed_pairs(y_true, y_pred)
    return float(np.sqrt(np.square(y - p).mean())) if len(y) else float("nan")


def wape(y_true, y_pred):
    y, p = observed_pairs(y_true, y_pred)
    denominator = np.abs(y).sum()
    return float(np.abs(y - p).sum() / denominator) if denominator else float("nan")


def pinball_loss(y_true, y_pred, q):
    if not 0 < q < 1:
        raise ValueError("Quantile must be between zero and one")
    y, p = observed_pairs(y_true, y_pred)
    residual = y - p
    return float(np.maximum(q * residual, (q - 1) * residual).mean()) if len(y) else float("nan")


def score_predictions(scored):
    y, p = scored["actual"].to_numpy(), scored["y_pred"].to_numpy()
    result = {"n_rows": len(scored), "n_observed": int(pd.notna(y).sum()),
              "mae": mae(y, p), "rmse": rmse(y, p), "wape": wape(y, p)}
    for name, q in [("q10", 0.1), ("q50", 0.5), ("q90", 0.9)]:
        result[f"pinball_{name}"] = (
            float("nan") if scored[name].isna().all() else pinball_loss(y, scored[name], q))
    return result
