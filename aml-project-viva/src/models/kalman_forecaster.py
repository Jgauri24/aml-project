"""Reusable linear-Gaussian state-space mathematics for M7 Kalman variants.

The state is ordered as ``[level, slope?, seasonal_0, ..., seasonal_5?]``.
The four systems are: ``local_level``: ``T=[1], Z=[1]``;
``local_level_seasonal``: ``T=diag(1, S), Z=[1, 1, 0, 0, 0, 0, 0]``;
``local_trend``: ``T=[[1, 1], [0, damping]], Z=[1, 0]``; and
``local_trend_seasonal``: ``T=diag_block([[1, 1], [0, damping]], S),
Z=[1, 0, 1, 0, 0, 0, 0, 0]``.  Here the 6x6 seasonal block S has ones on its
superdiagonal and an all--1 final row.  Q is diagonal in the same state order.
The seasonal coordinate at time t is the effect for that row's actual weekday,
followed by the next five weekdays; the observation adds that current
coordinate.  S advances this representation one calendar day, and creates the
seventh effect as the negative sum of the stored six.  Therefore at the first
observed row (even if earlier rows are missing), seasonal_0 matches that row's
weekday; after filtering the origin, the first forecast transition maps
seasonal_0 to the first forecast day's weekday effect.

Filtering follows the M6 convention: it first predicts with T and Q, then
updates with the observation.  Accordingly ``initial_state`` returns the
state immediately *before* the first calendar row.  It solves one transition
back from a state whose seasonal coordinate matches that first row's actual
weekday.  Forecast h=1 applies T once to the final filtered state, so its
first seasonal coordinate corresponds to the first forecast day.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import warnings
from typing import Mapping

import pandas as pd
from joblib import Parallel, delayed
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from src.evaluation.backtesting import BaseForecaster, FUTURE_KEYS, PREDICTION_COLUMNS, ForecastInputs


VARIANTS = (
    "local_level",
    "local_level_seasonal",
    "local_trend",
    "local_trend_seasonal",
)
MIN_PRED_VAR = 1e-4  # Same numerical floor used by the M6 local-level model.
_P0_DIFFUSE = 1e4
_P0_SLOPE = 1.0


def validate_variant(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown Kalman variant {variant!r}; expected one of {VARIANTS}.")
    return variant


def model_name(variant: str) -> str:
    return f"kalman_{validate_variant(variant)}"


def _has_trend(variant: str) -> bool:
    return "trend" in validate_variant(variant)


def _has_seasonal(variant: str) -> bool:
    return "seasonal" in validate_variant(variant)


def state_dimension(variant: str) -> int:
    return 1 + int(_has_trend(variant)) + 6 * int(_has_seasonal(variant))


def build_system(
    variant: str, variances: Mapping[str, float], damping: float = 0.98
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build ``(T, Z, Q, R)`` for one named variant.

    ``variances`` must provide nonnegative ``level`` and strictly positive
    ``observation`` entries, plus nonnegative ``slope`` and/or ``seasonal``
    entries when their states are present.  Q is diagonal in the stated state
    order.  Damping is fixed by the caller and is deliberately not estimated.
    """
    validate_variant(variant)
    if not np.isfinite(damping) or not 0.0 <= damping <= 1.0:
        raise ValueError("damping must be finite and in [0, 1]")
    required = ["level", "observation"]
    if _has_trend(variant):
        required.append("slope")
    if _has_seasonal(variant):
        required.append("seasonal")
    missing = [name for name in required if name not in variances]
    if missing:
        raise ValueError(f"Missing variance entries: {missing}")
    values = {name: float(variances[name]) for name in required}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("All variances must be finite")
    if values["observation"] <= 0 or any(values[name] < 0 for name in required if name != "observation"):
        raise ValueError("observation variance must be > 0 and state variances must be >= 0")

    n_state = state_dimension(variant)
    T = np.eye(n_state, dtype=float)
    Z = np.zeros(n_state, dtype=float)
    Z[0] = 1.0
    diagonal = [values["level"]]
    offset = 1
    if _has_trend(variant):
        T[0, 1] = 1.0
        T[1, 1] = damping
        diagonal.append(values["slope"])
        offset = 2
    if _has_seasonal(variant):
        Z[offset] = 1.0
        seasonal = np.zeros((6, 6), dtype=float)
        seasonal[:-1, 1:] = np.eye(5)
        seasonal[-1, :] = -1.0
        T[offset:offset + 6, offset:offset + 6] = seasonal
        diagonal.extend([values["seasonal"]] * 6)
    return T, Z, np.diag(diagonal), values["observation"]


@dataclass(frozen=True)
class InitialState:
    state: np.ndarray
    covariance: np.ndarray
    used_seasonal_fallback: bool


def initial_state(
    observations: np.ndarray,
    variant: str,
    weekday_positions: np.ndarray | None = None,
    damping: float = 0.98,
) -> InitialState:
    """Construct a finite diffuse prior from the first at most 28 observations.

    The level is their mean and trend slope is zero.  For seasonal variants,
    weekday deviations use the *calendar positions* of those observed values.
    We require at least seven observed values spanning all weekdays; otherwise
    every seasonal effect is zero (the stable documented fallback).  No target
    is imputed.  P0 is symmetric PSD: level and seasonal coordinates use M6's
    ``1e4`` diffuse variance, while the zero-initialised slope uses the modest
    fixed variance 1.0 so a transient first observation cannot create a large
    artificial long-horizon trend.
    """
    validate_variant(variant)
    y = np.asarray(observations, dtype=float)
    if y.ndim != 1 or not np.isfinite(y[~np.isnan(y)]).all():
        raise ValueError("observations must be one-dimensional with finite observed values")
    n_state = state_dimension(variant)
    observed = np.flatnonzero(~np.isnan(y))[:28]
    level = float(np.mean(y[observed])) if len(observed) else 0.0
    state_at_first = np.zeros(n_state, dtype=float)
    state_at_first[0] = level
    fallback = False

    if _has_seasonal(variant):
        if weekday_positions is None:
            raise ValueError("weekday_positions are required for seasonal variants")
        weekdays = np.asarray(weekday_positions)
        if weekdays.shape != y.shape or not np.issubdtype(weekdays.dtype, np.integer):
            raise ValueError("weekday_positions must be integer and match observations")
        if np.any((weekdays < 0) | (weekdays > 6)):
            raise ValueError("weekday_positions must use 0..6")
        # Consecutive panel rows must move forward exactly one weekday, even when y is NaN.
        if len(weekdays) > 1 and not np.all((np.diff(weekdays) % 7) == 1):
            raise ValueError("weekday_positions must advance by one calendar day per row")
        if len(observed) < 7 or len(np.unique(weekdays[observed])) < 7:
            fallback = True
        else:
            weekday_means = np.array([np.mean(y[observed][weekdays[observed] == day]) for day in range(7)])
            deviations = weekday_means - level
            deviations -= deviations.mean()  # Enforce the dummy-seasonal zero-sum constraint.
            first_day = int(weekdays[0])
            ordered = np.array([deviations[(first_day + step) % 7] for step in range(6)])
            seasonal_offset = 2 if _has_trend(variant) else 1
            state_at_first[seasonal_offset:seasonal_offset + 6] = ordered

    # M6 predicts before the first update.  Solve T x_prior = x_first so the
    # seasonal observation at the first row has the correct calendar phase.
    unit_variances = {"level": 1.0, "observation": 1.0}
    if _has_trend(variant):
        unit_variances["slope"] = 1.0
    if _has_seasonal(variant):
        unit_variances["seasonal"] = 1.0
    T, _, _, _ = build_system(variant, unit_variances, damping)
    prior = np.linalg.solve(T, state_at_first)
    covariance = np.eye(n_state, dtype=float) * _P0_DIFFUSE
    if _has_trend(variant):
        covariance[1, 1] = _P0_SLOPE
    return InitialState(prior, covariance, fallback)


@dataclass(frozen=True)
class FilterResult:
    states: np.ndarray
    covariances: np.ndarray
    innovations: np.ndarray
    innovation_variances: np.ndarray
    log_lik: float


def _validate_system(T: np.ndarray, Z: np.ndarray, Q: np.ndarray, R: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    T, Z, Q = np.asarray(T, dtype=float), np.asarray(Z, dtype=float), np.asarray(Q, dtype=float)
    R = float(R)
    if T.ndim != 2 or T.shape[0] != T.shape[1] or Z.shape != (T.shape[0],) or Q.shape != T.shape:
        raise ValueError("T must be square; Z and Q must match its state dimension")
    if not (np.isfinite(T).all() and np.isfinite(Z).all() and np.isfinite(Q).all() and np.isfinite(R)):
        raise ValueError("State-space matrices and R must be finite")
    if not np.allclose(Q, Q.T) or np.linalg.eigvalsh(Q).min() < -1e-12 or R <= 0:
        raise ValueError("Q must be symmetric PSD and R must be > 0")
    return T, Z, Q, R


def kalman_filter(
    observations: np.ndarray, T: np.ndarray, Z: np.ndarray, Q: np.ndarray, R: float,
    initial_mean: np.ndarray, initial_covariance: np.ndarray,
) -> FilterResult:
    """Run a validated predict-update filter; NaN observations skip only update."""
    T, Z, Q, R = _validate_system(T, Z, Q, R)
    y = np.asarray(observations, dtype=float)
    x = np.asarray(initial_mean, dtype=float).copy()
    P = np.asarray(initial_covariance, dtype=float).copy()
    n = len(Z)
    if y.ndim != 1 or x.shape != (n,) or P.shape != (n, n):
        raise ValueError("observations, initial_mean, and initial_covariance have incompatible dimensions")
    if not (np.isfinite(y[~np.isnan(y)]).all() and np.isfinite(x).all() and np.isfinite(P).all()):
        raise ValueError("Observed values and initial state must be finite")
    if not np.allclose(P, P.T) or np.linalg.eigvalsh(P).min() < -1e-12:
        raise ValueError("initial_covariance must be symmetric PSD")
    states, covariances = np.empty((len(y), n)), np.empty((len(y), n, n))
    innovations, innovation_variances = np.full(len(y), np.nan), np.full(len(y), np.nan)
    log_lik = 0.0
    eye = np.eye(n)
    for t, value in enumerate(y):
        x = T @ x
        P = T @ P @ T.T + Q
        P = (P + P.T) / 2.0
        if not np.isnan(value):
            innovation = value - Z @ x
            F = float(Z @ P @ Z + R)
            if not np.isfinite(F) or F <= 0:
                raise FloatingPointError("Innovation variance must be finite and positive")
            gain = (P @ Z) / F
            x = x + gain * innovation
            # Joseph form keeps the covariance PSD under floating-point arithmetic.
            update = eye - np.outer(gain, Z)
            P = update @ P @ update.T + np.outer(gain, gain) * R
            P = (P + P.T) / 2.0
            innovations[t], innovation_variances[t] = innovation, F
            log_lik -= 0.5 * (np.log(2.0 * np.pi * F) + innovation * innovation / F)
        states[t], covariances[t] = x, P
    return FilterResult(states, covariances, innovations, innovation_variances, float(log_lik))


def forecast(
    state: np.ndarray, covariance: np.ndarray, T: np.ndarray, Z: np.ndarray, Q: np.ndarray, R: float,
    horizon: int = 28,
) -> dict[str, np.ndarray]:
    """Iterate the state h=1..horizon and return clipped Gaussian forecasts."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    T, Z, Q, R = _validate_system(T, Z, Q, R)
    x, P = np.asarray(state, dtype=float).copy(), np.asarray(covariance, dtype=float).copy()
    if x.shape != Z.shape or P.shape != Q.shape or not np.isfinite(x).all() or not np.isfinite(P).all():
        raise ValueError("Forecast state and covariance have incompatible dimensions or non-finite values")
    means, variances = np.empty(horizon), np.empty(horizon)
    for h in range(horizon):
        x, P = T @ x, T @ P @ T.T + Q
        P = (P + P.T) / 2.0
        means[h] = Z @ x
        variances[h] = max(float(Z @ P @ Z + R), MIN_PRED_VAR)
    std = np.sqrt(variances)
    q50 = np.maximum(means, 0.0)
    q10 = np.minimum(np.maximum(means + norm.ppf(0.1) * std, 0.0), q50)
    q90 = np.maximum(np.maximum(means + norm.ppf(0.9) * std, 0.0), q50)
    return {"mean": means, "variance": variances, "q10": q10, "q50": q50, "q90": q90, "y_pred": q50}


# Estimation choices are fixed before cohort evaluation.  They are in scaled
# units, so the same deterministic choices apply fairly to each independent
# series regardless of its sales magnitude.
LOG_VARIANCE_BOUNDS = (-9.0, 5.0)
# A damped slope can legitimately be effectively deterministic.  Its lower
# bound is deliberately lower than other variances so the trend state does not
# inject artificial random-walk drift when the likelihood favours no trend.
SLOPE_LOG_VARIANCE_BOUNDS = (-20.0, 5.0)
OPTIMIZER_STARTS = (np.log([0.01, 1.0]), np.log([0.2, 0.2]))
FALLBACK_VARIANCES = {"level": 0.01, "slope": 0.001, "seasonal": 0.01, "observation": 1.0}
MIN_SCALE = 1e-3


@dataclass(frozen=True)
class SeriesFit:
    """Fold-local fitted result for a single series; no data are shared across series."""
    series_id: str
    scale: float
    T: np.ndarray
    Z: np.ndarray
    Q: np.ndarray
    R: float
    state: np.ndarray
    covariance: np.ndarray
    summary: dict


def _variance_names(variant: str) -> list[str]:
    names = ["level"]
    if _has_trend(variant):
        names.append("slope")
    if _has_seasonal(variant):
        names.append("seasonal")
    return names + ["observation"]


def _scale_from_window(y: np.ndarray) -> float:
    observed = y[~np.isnan(y)]
    if not len(observed):
        return 1.0
    return float(max(np.mean(np.abs(observed)), MIN_SCALE))


def _fallback_variances(variant: str) -> dict[str, float]:
    return {name: FALLBACK_VARIANCES[name] for name in _variance_names(variant)}


def _fit_one_series(
    series_id: str, history: pd.DataFrame, variant: str, fit_window: int | None, damping: float,
) -> SeriesFit:
    """Fit one independent series using exactly two deterministic L-BFGS-B starts."""
    data = history.sort_values("day_num").tail(fit_window) if fit_window is not None else history.sort_values("day_num")
    y_original = data["units_sold"].to_numpy(dtype=float)
    weekdays = ((data["wday"].to_numpy(dtype=int) - 1) % 7).astype(int)
    observed = int(np.count_nonzero(~np.isnan(y_original)))
    scale = _scale_from_window(y_original)
    y = y_original / scale
    initial = initial_state(y, variant, weekdays, damping)
    parameter_names = _variance_names(variant)
    n_parameters = len(parameter_names)
    fallback_reason: str | None = None
    attempts: list[dict] = []

    if observed == 0:
        fallback_reason = "all_nan_window"
        warnings.warn(f"{series_id}: all-NaN fit window; using fixed wide-uncertainty fallback.", RuntimeWarning, stacklevel=2)
    elif np.all(y_original[~np.isnan(y_original)] == 0.0):
        fallback_reason = "all_zero_window"
        warnings.warn(f"{series_id}: all-zero fit window; using stable zero-level fallback.", RuntimeWarning, stacklevel=2)

    chosen = None
    chosen_attempt_index: int | None = None
    if fallback_reason is None:
        def objective(log_values: np.ndarray) -> float:
            values = np.exp(np.asarray(log_values, dtype=float))
            variances = dict(zip(parameter_names, values, strict=True))
            try:
                T, Z, Q, R = build_system(variant, variances, damping)
                result = kalman_filter(y, T, Z, Q, R, initial.state, initial.covariance)
                value = -result.log_lik
                return float(value) if np.isfinite(value) else np.inf
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                return np.inf

        bounds = [SLOPE_LOG_VARIANCE_BOUNDS if name == "slope" else LOG_VARIANCE_BOUNDS
                  for name in parameter_names]
        for attempt_index, start_base in enumerate(OPTIMIZER_STARTS):
            start = np.resize(start_base, n_parameters)
            try:
                result = minimize(objective, start, method="L-BFGS-B", bounds=bounds,
                                  options={"maxiter": 250, "ftol": 1e-10, "gtol": 1e-6})
                objective_value = float(result.fun)
                attempt = {"success": bool(result.success), "message": str(result.message),
                           "objective": objective_value, "start": start.tolist(), "x": result.x.tolist()}
                attempts.append(attempt)
                if result.success and np.isfinite(objective_value) and (chosen is None or objective_value < chosen.fun):
                    chosen = result
                    chosen_attempt_index = attempt_index
            except Exception as exc:  # Optimizer/library failure is explicitly surfaced in diagnostics and fallback.
                attempts.append({"success": False, "message": f"exception: {type(exc).__name__}: {exc}", "objective": np.inf})
        if chosen is None:
            fallback_reason = "optimizer_failure"
            warnings.warn(f"{series_id}: both optimiser starts failed; using fixed fallback variances.", RuntimeWarning, stacklevel=2)

    if fallback_reason is not None:
        variances = _fallback_variances(variant)
        convergence_status, optimizer_message, objective_value = "fallback", fallback_reason, np.nan
    else:
        variances = dict(zip(parameter_names, np.exp(chosen.x), strict=True))
        convergence_status = "converged"
        optimizer_message = str(chosen.message)
        objective_value = float(chosen.fun)

    T, Z, Q, R = build_system(variant, variances, damping)
    result = kalman_filter(y, T, Z, Q, R, initial.state, initial.covariance)
    # Density transforms by 1/scale for each observed y: apply exactly once.
    log_lik_original = float(result.log_lik - observed * np.log(scale))
    aic = float(2 * n_parameters - 2 * log_lik_original) if fallback_reason is None else np.nan
    unscaled_variances = {name: float(variances[name] * scale * scale) for name in parameter_names}
    summary = {
        "series_id": series_id, "variant": variant,
        "fit_window_start": int(data["day_num"].iloc[0]), "fit_window_end": int(data["day_num"].iloc[-1]),
        "calendar_days_in_window": int(len(data)), "observed_targets": observed, "scale": scale,
        "level_variance": unscaled_variances["level"],
        "slope_variance": unscaled_variances.get("slope", np.nan),
        "trend_variance": unscaled_variances.get("slope", np.nan),
        "seasonal_variance": unscaled_variances.get("seasonal", np.nan),
        "observation_variance": unscaled_variances["observation"],
        "log_likelihood": log_lik_original, "aic": aic,
        # AIC counts only MLE variance parameters; fixed damping and diffuse P0 are not parameters.
        "estimated_parameter_count": n_parameters if fallback_reason is None else 0,
        "convergence_status": convergence_status, "optimizer_message": optimizer_message,
        "objective_value": objective_value, "fallback": fallback_reason is not None,
        "fallback_reason": fallback_reason, "seasonal_initialisation_fallback": initial.used_seasonal_fallback,
        "optimizer_attempts": json.dumps(attempts), "chosen_attempt_index": chosen_attempt_index,
    }
    return SeriesFit(series_id, scale, T, Z, Q, R, result.states[-1], result.covariances[-1], summary)


class KalmanForecaster(BaseForecaster):
    """Independent per-series Kalman MLE adapter for the M7 development backtest.

    The four linear-Gaussian state equations are defined at module level.  Each
    series estimates only its own diagonal process variances and observation
    variance, in log space, with two fixed L-BFGS-B starts and bounds [-9, 5]
    (slope [-20, 5] so a no-trend solution is representable).
    Damping is fixed in [0, 1], not optimised.  The default trailing 364-day
    window is a computational and recency choice, not a claim that older data
    are irrelevant; ``None`` uses all available fitting history.

    State initialisation uses the first 28 observed in-window targets and real
    calendar weekday positions.  All-NaN, all-zero, and optimiser-failure cases
    use documented fixed scaled variances.  Gaussian forecasts are an
    approximation to count data.  Clipping negative Gaussian quantiles to zero
    changes their nominal coverage, especially for sparse series; these are not
    claimed to be calibrated 80% intervals.
    """
    def __init__(self, variant: str, fit_window: int | None = 364, damping: float = 0.98,
                 seed: int = 42, n_jobs: int = 1) -> None:
        self.variant = validate_variant(variant)
        if fit_window is not None and (isinstance(fit_window, bool) or not isinstance(fit_window, int) or fit_window <= 0):
            raise ValueError("fit_window must be None or a positive integer")
        if not np.isfinite(damping) or not 0.0 <= damping <= 1.0:
            raise ValueError("damping must be finite and in the stable range [0, 1]")
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0:
            raise ValueError("n_jobs must be a nonzero joblib integer")
        self.fit_window, self.damping, self.seed, self.n_jobs = fit_window, float(damping), int(seed), n_jobs
        self._states: dict[str, SeriesFit] = {}
        self._fitted_origin: int | None = None
        self.fit_summary_ = pd.DataFrame()

    def fit(self, inputs: ForecastInputs) -> "KalmanForecaster":
        groups = [(str(sid), frame.copy()) for sid, frame in inputs.history.groupby("series_id", observed=True, sort=True)]
        runner = (delayed(_fit_one_series)(sid, frame, self.variant, self.fit_window, self.damping) for sid, frame in groups)
        fitted = Parallel(n_jobs=self.n_jobs)(runner) if self.n_jobs != 1 else [_fit_one_series(sid, frame, self.variant, self.fit_window, self.damping) for sid, frame in groups]
        self._states = {state.series_id: state for state in fitted}
        self._fitted_origin = inputs.origin
        self.fit_summary_ = pd.DataFrame([state.summary for state in fitted]).sort_values("series_id").reset_index(drop=True)
        return self

    def predict(self, inputs: ForecastInputs) -> pd.DataFrame:
        if self._fitted_origin is None:
            raise RuntimeError("Call fit() before predict().")
        if inputs.origin != self._fitted_origin:
            raise RuntimeError(f"Model was fitted on origin {self._fitted_origin} but predict() received origin {inputs.origin}.")
        pieces = []
        for sid, frame in inputs.future_known.groupby("series_id", observed=True, sort=True):
            state = self._states[str(sid)]
            frame = frame.sort_values("horizon").copy()
            result = forecast(state.state, state.covariance, state.T, state.Z, state.Q, state.R, int(frame["horizon"].max()))
            positions = frame["horizon"].to_numpy(dtype=int) - 1
            out = frame[FUTURE_KEYS].copy()
            out["model"] = model_name(self.variant)
            for key in ("y_pred", "q10", "q50", "q90"):
                out[key] = result[key][positions] * state.scale
            # Multiplication preserves order; retain an explicit guard against numerical noise.
            out["q10"] = np.minimum(out["q10"], out["q50"])
            out["q90"] = np.maximum(out["q90"], out["q50"])
            pieces.append(out)
        return pd.concat(pieces, ignore_index=True)[PREDICTION_COLUMNS]


def make_kalman_factory(variant: str, **kwargs):
    """Return the zero-argument fresh-model factory required by ``run_backtest``."""
    validate_variant(variant)
    return lambda: KalmanForecaster(variant=variant, **kwargs)
