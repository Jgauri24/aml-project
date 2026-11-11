"""
Local-level Kalman filter for single SKU-store daily sales series (Milestone 6).

Model
-----
Observation equation : y_t  = mu_t + eps_t,    eps_t ~ N(0, R)
State (level) equation: mu_t = mu_{t-1} + eta_t, eta_t ~ N(0, Q)

Q (process-noise variance) controls how rapidly the latent level drifts.
R (observation-noise variance) controls how noisy individual sales days are
around the level.  Both are estimated per-series via maximum likelihood
using the innovations representation of the Gaussian log-likelihood.

Single-SKU vs. Multi-SKU
-------------------------
"Single-SKU" refers to the estimation strategy: every series in the 240-series
cohort receives its own independently estimated (Q_i, R_i) pair.  No information
is shared across series during fitting. Milestone 7 extends this work with a
family of independent per-series Kalman variants; it is neither pooled nor
hierarchical.

Gaussian approximation and zero-clipping
-----------------------------------------
Predictive distributions are Gaussian: y_{T+h} | y_{1:T} ~ N(mu_T, V_h),
where V_h = P_{T|T} + h*Q + R grows linearly with the forecast horizon h.
Because retail sales are nonneg we clip all forecast values (y_pred, q10, q50,
q90) to >= 0.  This clip turns the Gaussian into a mixed distribution (point
mass at zero + continuous part above zero), so the nominal 80 % coverage of the
[q10, q90] interval is not guaranteed for sparse or near-zero series.  This is
a known limitation of the Gaussian approximation; a truncated-Gaussian or
count-distribution extension is a natural Milestone 7 or later improvement.

Point forecast
--------------
y_pred is set equal to q50 = max(0, mu_T).  For a Gaussian distribution the
mean equals the median, so this is unambiguous.  The point forecast is constant
across all 28 forecast horizons — a property of the random-walk state equation:
the level is expected to stay where it is, while variance accumulates with h.

Weekly seasonality — known limitation
--------------------------------------
The local-level model contains no seasonal component.  The state equation is a
pure random walk, so the filter tracks only a slowly-evolving level.  For M5
retail sales, the EDA found a median lag-7 autocorrelation of 0.08–0.32 across
volume bands, indicating that weekly patterns are present but vary in strength.
This model cannot exploit those patterns: it will systematically mis-forecast
series with strong day-of-week effects, particularly for the 28-day horizon
where four full weekly cycles are predicted.  A natural extension is the
local-level-plus-seasonal (LLT-S) state-space model; alternatively, Milestone 5
(Seasonal Naive) provides a dedicated weekly-seasonal baseline for comparison.
The omission is a deliberate scope boundary for Milestone 6, not an oversight.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from src.evaluation.backtesting import (
    FUTURE_KEYS,
    PREDICTION_COLUMNS,
    BaseForecaster,
    ForecastInputs,
)

# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants (documented for teammate reuse and viva defence)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME: str = "kalman_single_sku"

# Minimum predictive variance imposed after computing P_T + h*Q + R.
# Prevents zero-width intervals on constant or all-zero histories where
# the MLE may drive Q and R toward their lower bounds simultaneously.
MIN_PRED_VAR: float = 1e-4

# Diffuse prior for the initial state variance.  Large relative to typical
# daily sales (~0–10 units); ensures the filter is data-driven from the start
# and is insensitive to the choice of mu0 after a short burn-in.
_P0_DIFFUSE: float = 1e4

# Unconstrained optimisation operates in log-parameter space.
# exp(-10) ≈ 4.5e-5 prevents numerical collapse; exp(8) ≈ 3000 is generous
# even for high-volume, highly-volatile series.
_LOG_MIN: float = -10.0
_LOG_MAX: float = 8.0

# Fallback (Q, R) used when MLE fails or the series has no observations.
# These are weakly informative defaults: moderate drift, unit observation noise.
_Q_FALLBACK: float = 0.01
_R_FALLBACK: float = 1.0


def _initial_state_mean(y: np.ndarray) -> float:
    """Mean of the first ≤28 *non-NaN* observations; 0.0 if none exist.

    This is the single canonical formula for mu0 used by every forward Kalman
    pass in this module — the MLE objective, the final forecasting pass inside
    ``fit_local_level``, and the visualisation pass inside
    ``KalmanSingleSKU.fit``.  Keeping one implementation in one place prevents
    the three callers from drifting apart.

    Skipping NaN positions before slicing (``y[~np.isnan(y)][:28]``) is
    important for M5 series with a pre-launch period: a series whose first
    k observations are NaN has its first informative sales starting at position
    k.  Slicing positionally (``y[:28]``) and then applying ``nanmean`` would
    use only ``28-k`` observed values rather than 28, which underweights the
    early sales signal when k is large.
    """
    obs = y[~np.isnan(y)][:28]
    return float(np.mean(obs)) if len(obs) > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Mathematical core (pure functions; no pandas, no disk I/O)
# Milestone-7 teammate: import these functions directly.
# ─────────────────────────────────────────────────────────────────────────────

class FilterResult(NamedTuple):
    """Output of a single forward Kalman filter pass."""
    mu_filtered: np.ndarray   # posterior level means  mu_{t|t},   shape (T,)
    P_filtered: np.ndarray    # posterior level variances P_{t|t},  shape (T,)
    log_lik: float            # innovations log-likelihood (observed steps only)
    innovations: np.ndarray   # v_t = y_t - mu_{t|t-1};  NaN where y_t missing
    F: np.ndarray             # innovation variances F_t; NaN where y_t missing


def kalman_filter(
    y: np.ndarray,
    Q: float,
    R: float,
    mu0: float = 0.0,
    P0: float = _P0_DIFFUSE,
) -> FilterResult:
    """Forward Kalman filter for the local-level model.

    Prediction step (prior at time t)
    ----------------------------------
        mu_{t|t-1} = mu_{t-1|t-1}
        P_{t|t-1}  = P_{t-1|t-1} + Q

    Update step (posterior at time t, only when y_t is observed)
    -------------------------------------------------------------
        v_t        = y_t - mu_{t|t-1}               # innovation
        F_t        = P_{t|t-1} + R                  # innovation variance
        K_t        = P_{t|t-1} / F_t                # Kalman gain
        mu_{t|t}   = mu_{t|t-1} + K_t * v_t
        P_{t|t}    = (1 - K_t) * P_{t|t-1}         # uses P_{t|t-1}, NOT P_{t-1|t-1}

    Missing observation rule
    ------------------------
    When y[t] is NaN the update step is skipped entirely:
        mu_{t|t} = mu_{t|t-1},  P_{t|t} = P_{t|t-1}
    The variance therefore accumulates Q for every missing step, correctly
    reflecting growing uncertainty during gaps (Durbin & Koopman 2012, §4.10).

    Parameters
    ----------
    y   : (T,) float array; NaN entries are treated as missing observations.
    Q   : process-noise variance (>= 0); controls level drift speed.
    R   : observation-noise variance (> 0); controls sales scatter around level.
    mu0 : initial state mean (prior mean before t=0).
    P0  : initial state variance (prior); should be large (diffuse) when
          the true initial level is unknown.

    Returns
    -------
    FilterResult namedtuple.
    """
    T = len(y)
    mu_arr = np.empty(T)
    P_arr = np.empty(T)
    v_arr = np.full(T, np.nan)
    F_arr = np.full(T, np.nan)

    mu_prev = float(mu0)
    P_prev = float(P0)
    Q_f = float(Q)
    R_f = float(R)
    log_lik = 0.0
    _TWO_PI = 2.0 * np.pi

    for t in range(T):
        # --- Prediction step ---
        mu_pred = mu_prev          # state transition is identity
        P_pred = P_prev + Q_f     # P_{t|t-1}

        y_t = y[t]
        if y_t != y_t:  # NaN check — faster than np.isnan for Python scalars
            # Missing observation: carry forward without update
            mu_arr[t] = mu_pred
            P_arr[t] = P_pred
        else:
            # --- Update step ---
            F_t = P_pred + R_f                         # innovation variance
            K_t = P_pred / F_t                         # Kalman gain
            v_t = y_t - mu_pred                        # innovation

            mu_arr[t] = mu_pred + K_t * v_t
            P_arr[t] = (1.0 - K_t) * P_pred           # ← uses P_{t|t-1}

            # Gaussian innovations log-likelihood contribution
            log_lik -= 0.5 * (np.log(_TWO_PI * F_t) + v_t * v_t / F_t)
            v_arr[t] = v_t
            F_arr[t] = F_t

        mu_prev = mu_arr[t]
        P_prev = P_arr[t]

    return FilterResult(mu_arr, P_arr, log_lik, v_arr, F_arr)


def _neg_log_likelihood(log_params: np.ndarray, y: np.ndarray, mu0: float) -> float:
    """Negative log-likelihood in unconstrained log-parameter space.

    log_params[0] = log(Q),  log_params[1] = log(R).
    Parameterising as logs keeps Q, R strictly positive throughout
    optimisation without requiring inequality constraints or projection.

    mu0 is passed through to kalman_filter so that the likelihood is evaluated
    with the same initial-state mean used for the final forward pass in
    fit_local_level.  This removes the inconsistency of optimising under one
    prior mean and then extracting the filtered state under a different one.
    """
    Q = np.exp(np.clip(log_params[0], _LOG_MIN, _LOG_MAX))
    R = np.exp(np.clip(log_params[1], _LOG_MIN, _LOG_MAX))
    result = kalman_filter(y, Q, R, mu0=mu0)
    ll = result.log_lik
    return -ll if np.isfinite(ll) else 1e15


def fit_local_level(
    y: np.ndarray,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Estimate (Q, R) by MLE and return the final filtered state (mu_T, P_T).

    Uses L-BFGS-B with three starting points in log-space to reduce
    sensitivity to local optima.  The best finite solution is kept.

    Parameters
    ----------
    y    : (T,) float array; NaN entries are missing observations.
    seed : random seed (reserved for future stochastic restarts; currently
           controls the RNG used when randomly-perturbed starts are added).

    Returns
    -------
    (Q, R, mu_T, P_T) where Q and R are the MLE noise variances, and
    (mu_T, P_T) is the final filtered state after a full forward pass
    with the optimised parameters.

    Degenerate-series behaviour (documented for viva)
    --------------------------------------------------
    - All NaN (no observations)  : returns (_Q_FALLBACK, _R_FALLBACK, 0, P0_DIFFUSE).
    - All-zero history           : MLE converges to small R (exact zero prevented
                                   by the log-param lower bound); result is used.
    - Optimizer failure          : falls back to (_Q_FALLBACK, _R_FALLBACK) with
                                   a RuntimeWarning; filter still runs with defaults.
    - Nearly-constant / flat LL  : MLE converges but parameters may not be well-
                                   identified; MIN_PRED_VAR prevents zero-width
                                   predictive intervals downstream.

    Initial-state consistency
    -------------------------
    The empirical mean of the first ≤28 observed values is used as mu0 for both
    the MLE likelihood evaluations and the final forward pass.  Using the same
    mu0 in both places ensures the log-likelihood being maximised is the one
    whose filtered state is returned.  With a diffuse prior (P0 = _P0_DIFFUSE =
    1e4) the filter forgets mu0 within a handful of steps for long series, so
    the choice matters most for very short histories (T < 20); using the
    empirical mean rather than a fixed zero is a conservative improvement.
    """
    obs_mask = ~np.isnan(y)
    n_obs = int(obs_mask.sum())

    # --- Degenerate: no observations at all ---
    if n_obs == 0:
        warnings.warn(
            "Series has no observed values; returning fallback parameters.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _Q_FALLBACK, _R_FALLBACK, 0.0, float(_P0_DIFFUSE)

    # Initial state mean: mean of first ≤28 observed values (short burn-in window)
    mu0 = _initial_state_mean(y)

    # Three deterministic starting points covering main parameter regimes:
    #  (0, 0) → Q=R=1 (neutral), (-2, 0) → small Q (slow drift),
    #  (0, -2) → small R (high signal-to-noise)
    x0_list = [
        np.array([0.0, 0.0]),
        np.array([-2.0, 0.0]),
        np.array([0.0, -2.0]),
    ]

    best_nll: float = np.inf
    best_x: np.ndarray | None = None

    for x0 in x0_list:
        try:
            res = minimize(
                _neg_log_likelihood,
                x0=x0,
                args=(y, mu0),
                method="L-BFGS-B",
                options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6},
            )
            if np.isfinite(res.fun) and res.fun < best_nll:
                best_nll = res.fun
                best_x = res.x.copy()
        except Exception:
            continue

    if best_x is None:
        warnings.warn(
            "All MLE restarts failed; using fallback parameters.",
            RuntimeWarning,
            stacklevel=2,
        )
        Q, R = _Q_FALLBACK, _R_FALLBACK
    else:
        Q = float(np.exp(np.clip(best_x[0], _LOG_MIN, _LOG_MAX)))
        R = float(np.exp(np.clip(best_x[1], _LOG_MIN, _LOG_MAX)))

    # Final forward pass with optimal parameters to obtain (mu_T, P_T)
    result = kalman_filter(y, Q, R, mu0=mu0, P0=_P0_DIFFUSE)
    mu_T = float(result.mu_filtered[-1])
    P_T = float(result.P_filtered[-1])

    return Q, R, mu_T, P_T


def forecast_local_level(
    mu_T: float,
    P_T: float,
    Q: float,
    R: float,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute h-step-ahead point forecasts and predictive variances.

    Predictive distribution
    -----------------------
    y_{T+h} | y_{1:T} ~ N(mu_T,  V_h)   where V_h = P_{T|T} + h*Q + R

    The point forecast is mu_T for all h (the level is a random walk:
    its expected future value equals its current filtered estimate).
    The predictive variance V_h grows linearly with h because each
    additional step accumulates Q from the level random walk.

    Parameters
    ----------
    mu_T, P_T : final filtered state (posterior mean and variance at origin T)
    Q, R      : estimated noise variances
    horizon   : number of steps ahead; returns arrays indexed 0..horizon-1
                corresponding to h=1..horizon

    Returns
    -------
    y_hat     : (horizon,) constant array equal to mu_T
    pred_vars : (horizon,) predictive variances; floored at MIN_PRED_VAR
    """
    h_range = np.arange(1, horizon + 1, dtype=float)
    pred_vars = float(P_T) + h_range * float(Q) + float(R)
    pred_vars = np.maximum(pred_vars, MIN_PRED_VAR)
    y_hat = np.full(horizon, float(mu_T))
    return y_hat, pred_vars


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — BaseForecaster adapter
# ─────────────────────────────────────────────────────────────────────────────

class _SeriesState(NamedTuple):
    """Fitted state for one series, stored after fit()."""
    Q: float
    R: float
    mu_T: float
    P_T: float
    mu_filtered: np.ndarray   # full filtered-level history; used for plotting


class KalmanSingleSKU(BaseForecaster):
    """Per-series local-level Kalman filter adapter for the shared backtest engine.

    Each of the 240 series in the cohort receives an independently fitted
    (Q_i, R_i) pair estimated by MLE from that series' own history only.
    No parameters are shared across series.  This constitutes the Single-SKU
    design: "single" describes the estimation unit, not the number of series
    processed per fold.

    The adapter is fold-local: a fresh instance is created by the engine for
    each development origin, fitted on history up to that origin, and then
    released.  Temporal leakage is prevented by the engine's ``construct_inputs``
    which bounds ``inputs.history`` to ``day_num <= origin``.

    Parameters
    ----------
    seed : int
        Passed to ``fit_local_level`` for reproducibility of MLE restarts.

    Milestone-7 handoff
    -------------------
    M7 supplies independent per-series variants in ``kalman_forecaster.py``.
    They estimate their own parameters from each series' own history and do not
    pool or hierarchically share estimates.
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)
        self._states: dict[str, _SeriesState] = {}
        self._fitted_origin: int | None = None

    def fit(self, inputs: ForecastInputs) -> "KalmanSingleSKU":
        """Fit an independent local-level filter for every series.

        Iterates over all series in ``inputs.history`` (bounded by the engine
        to ``day_num <= inputs.origin``).  Missing target values (NaN) are
        handled inside ``kalman_filter`` by skipping the update step.
        """
        self._states = {}
        self._fitted_origin = inputs.origin

        for sid, grp in inputs.history.groupby("series_id", observed=True):
            y = grp.sort_values("day_num")["units_sold"].to_numpy(dtype=float)
            Q, R, mu_T, P_T = fit_local_level(y, seed=self.seed)

            # Extra filter pass to store the full filtered-level trajectory
            # (used only for visualisation; does not affect predictions).
            mu0 = _initial_state_mean(y)
            fr = kalman_filter(y, Q, R, mu0=mu0, P0=_P0_DIFFUSE)
            self._states[str(sid)] = _SeriesState(Q, R, mu_T, P_T, fr.mu_filtered)

        return self

    def predict(self, inputs: ForecastInputs) -> pd.DataFrame:
        """Produce 28-day point forecasts and Gaussian predictive quantiles.

        y_pred = q50 = max(0, mu_T).  For a Gaussian mean equals median.
        Quantiles q10 and q90 are Gaussian ± 1.282 * sigma_h, clipped to
        [0, ∞) and then clamped to maintain q10 <= q50 <= q90 after clipping.

        Returns
        -------
        DataFrame with exactly ``PREDICTION_COLUMNS``, one row per
        (series_id, horizon) requested in ``inputs.future_known``.
        """
        if self._fitted_origin is None:
            raise RuntimeError(
                "Call fit() before predict()."
            )
        if self._fitted_origin != inputs.origin:
            raise RuntimeError(
                f"Model was fitted on origin {self._fitted_origin} but predict() "
                f"received origin {inputs.origin}. Create a fresh instance per fold."
            )

        pieces: list[pd.DataFrame] = []

        for sid, grp in inputs.future_known.groupby(
            "series_id", sort=False, observed=True
        ):
            state = self._states[str(sid)]
            grp = grp.sort_values("horizon").copy()
            horizons = grp["horizon"].to_numpy(dtype=int)

            # Compute forecasts for all horizons at once
            y_hat_full, pred_vars_full = forecast_local_level(
                state.mu_T, state.P_T, state.Q, state.R, horizon=int(horizons.max())
            )
            pred_std_full = np.sqrt(pred_vars_full)

            # Select entries corresponding to the requested horizons (1-indexed)
            h_idx = horizons - 1
            mu_h = y_hat_full[h_idx]
            sig_h = pred_std_full[h_idx]

            # Gaussian quantiles
            raw_q10 = mu_h + norm.ppf(0.10) * sig_h   # ≈ mu - 1.282*sigma
            raw_q90 = mu_h + norm.ppf(0.90) * sig_h   # ≈ mu + 1.282*sigma

            # Clip to nonneg, then restore q10 <= q50 <= q90 ordering.
            # After clipping, q10 may equal q50 = 0 for near-zero series —
            # a valid degenerate case (zero-width interval at the floor).
            q50_h = np.maximum(0.0, mu_h)
            q10_h = np.minimum(np.maximum(0.0, raw_q10), q50_h)
            q90_h = np.maximum(np.maximum(0.0, raw_q90), q50_h)

            out = grp[FUTURE_KEYS].copy()
            out["model"] = MODEL_NAME
            out["y_pred"] = q50_h        # point forecast = clipped Gaussian median
            out["q10"] = q10_h
            out["q50"] = q50_h
            out["q90"] = q90_h
            pieces.append(out)

        return pd.concat(pieces, ignore_index=True)[PREDICTION_COLUMNS]


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Visualisation helper
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_sku_forecast(
    series_id: str,
    day_nums: np.ndarray,
    y_history: np.ndarray,
    mu_filtered: np.ndarray,
    future_day_nums: np.ndarray,
    y_hat: np.ndarray,
    pred_std: np.ndarray,
    origin: int,
    n_history_days: int = 120,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot historical sales, filtered level, and 28-day forecast with PI band.

    Parameters
    ----------
    series_id       : label used in the plot title.
    day_nums        : (T,) day numbers for history.
    y_history       : (T,) observed sales; NaN are rendered as gaps.
    mu_filtered     : (T,) posterior level means from ``kalman_filter``.
    future_day_nums : (horizon,) day numbers for the forecast period.
    y_hat           : (horizon,) point forecasts (constant = mu_T for local-level).
    pred_std        : (horizon,) predictive standard deviations.
    origin          : last observed day number (vertical dashed line).
    n_history_days  : how many trailing history days to display.
    save_path       : if given, saves the figure to this path.

    Returns
    -------
    matplotlib Figure (caller is responsible for closing it).
    """
    # Restrict history to trailing window for readability
    keep = day_nums >= (origin - n_history_days + 1)
    d_hist = day_nums[keep]
    y_hist = y_history[keep]
    mu_hist = mu_filtered[keep]

    fig, ax = plt.subplots(figsize=(12, 4))

    # Observed sales (dots + line, with NaN gaps)
    ax.plot(d_hist, y_hist, color="grey", linewidth=0.8, alpha=0.6,
            label="Observed sales")
    ax.scatter(d_hist[~np.isnan(y_hist)], y_hist[~np.isnan(y_hist)],
               s=8, color="grey", alpha=0.5, zorder=2)

    # Filtered level (posterior mean)
    ax.plot(d_hist, mu_hist, color="steelblue", linewidth=1.6,
            label="Filtered level μ_{t|t}")

    # 28-day forecast
    ax.plot(future_day_nums, y_hat, color="darkorange", linewidth=2.0,
            label="Forecast (μ_T)")

    # 10–90 % predictive interval
    q10 = np.maximum(0.0, y_hat + norm.ppf(0.10) * pred_std)
    q90 = np.maximum(0.0, y_hat + norm.ppf(0.90) * pred_std)
    ax.fill_between(future_day_nums, q10, q90,
                    color="darkorange", alpha=0.20,
                    label="80 % predictive interval (Gaussian, clipped)")

    # Origin line
    ax.axvline(origin, color="black", linestyle="--", linewidth=1.0,
               label=f"Origin d_{origin}")

    ax.set_xlabel("Day number")
    ax.set_ylabel("Units sold")
    ax.set_title(f"Kalman single-SKU forecast — {series_id}")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")

    return fig
