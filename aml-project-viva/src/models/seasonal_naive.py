"""Deterministic last-value and calendar-aligned weekly point forecasts."""
from __future__ import annotations

from functools import partial
import json
from typing import Callable

import numpy as np
import pandas as pd

from src.evaluation.backtesting import (
    BaseForecaster, ForecastInputs, FUTURE_KEYS, PREDICTION_COLUMNS,
)

VARIANTS = ("last_value", "weekly_seasonal")


class SeasonalNaiveForecaster(BaseForecaster):
    """Freeze historical values at fit time; missing weekdays use past-only fallbacks.

    Nonfinite observations are unavailable; genuine zeros remain valid. Finite
    negative sales are rejected. Day numbers, rather than observed-row offsets,
    preserve weekday alignment even for short or incomplete histories.
    """

    def __init__(self, variant: str = "weekly_seasonal") -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown naive variant {variant!r}; expected {VARIANTS}")
        self.variant = variant
        self._fitted_origin: int | None = None
        self._states: dict[str, tuple[float, np.ndarray]] = {}

    def fit(self, inputs: ForecastInputs) -> SeasonalNaiveForecaster:
        self._fitted_origin = None
        history = inputs.history[["series_id", "day_num", "units_sold"]].copy()
        if history.empty or history[["series_id", "day_num"]].isna().any().any():
            raise ValueError("History needs nonmissing series and day identifiers")
        days = history.day_num.to_numpy(dtype=float)
        if not np.isfinite(days).all() or not np.equal(days, np.floor(days)).all():
            raise ValueError("Historical day numbers must be finite integers")
        if (days < 1).any() or (days > inputs.origin).any():
            raise ValueError("History must contain only days at or before the fitted origin")
        if history.duplicated(["series_id", "day_num"]).any():
            raise ValueError("Duplicate series/day in history")
        states, summaries = {}, []
        for series_id, group in history.groupby("series_id", observed=True, sort=True):
            group = group.sort_values("day_num")
            values = group.units_sold.to_numpy(dtype=float, na_value=np.nan)
            if (values[np.isfinite(values)] < 0).any():
                raise ValueError("Observed sales must be nonnegative")
            valid = group.loc[np.isfinite(values)]
            latest = float(valid.units_sold.iloc[-1]) if len(valid) else 0.0
            latest_day = int(valid.day_num.iloc[-1]) if len(valid) else None
            pattern, source_days, policies = [], [], []
            for offset in range(7):
                source_day = inputs.origin - 6 + offset
                same_weekday = valid.loc[valid.day_num.mod(7).eq(source_day % 7)]
                if len(same_weekday):
                    day = int(same_weekday.day_num.iloc[-1])
                    value = float(same_weekday.units_sold.iloc[-1])
                    policy = "last_week" if day == source_day else "earlier_same_weekday"
                else:
                    day, value = latest_day, latest
                    policy = "latest_overall" if latest_day is not None else "all_missing_zero"
                pattern.append(value)
                source_days.append(day)
                policies.append(policy)
            states[str(series_id)] = (latest, np.array(pattern, dtype=float))
            summaries.append({
                "series_id": str(series_id), "origin": inputs.origin,
                "last_value": latest, "last_value_source_day": latest_day,
                "last_value_fallback": "all_missing_zero" if latest_day is None else (
                    "earlier_observation" if latest_day < inputs.origin else "origin_observation"),
                "unavailable_observations": int((~np.isfinite(values)).sum()),
                "weekly_last_week": policies.count("last_week"),
                "weekly_earlier_same_weekday": policies.count("earlier_same_weekday"),
                "weekly_latest_overall": policies.count("latest_overall"),
                "weekly_all_missing_zero": policies.count("all_missing_zero"),
                "weekly_source_days": json.dumps(source_days),
                "weekly_policies": json.dumps(policies),
            })
        self._states = states
        self.fit_summary_ = pd.DataFrame(summaries)
        self._fitted_origin = inputs.origin
        return self

    def predict(self, inputs: ForecastInputs) -> pd.DataFrame:
        if self._fitted_origin is None or self._fitted_origin != inputs.origin:
            raise RuntimeError("Fit this origin before predicting")
        prediction = inputs.future_known[FUTURE_KEYS].copy()
        missing = set(prediction.series_id.astype(str)) - self._states.keys()
        if missing:
            raise ValueError(f"Requested series missing from fitted history: {sorted(missing)}")
        horizons = prediction.horizon.to_numpy(dtype=float)
        if (not prediction.origin.eq(self._fitted_origin).all()
                or not np.isfinite(horizons).all()
                or not np.equal(horizons, np.floor(horizons)).all()
                or not ((horizons >= 1) & (horizons <= inputs.horizon)).all()
                or not prediction.target_day_num.eq(prediction.origin + prediction.horizon).all()):
            raise ValueError("Future grid must use the fitted origin and valid day/horizon offsets")
        prediction["model"] = f"naive_{self.variant}"
        prediction["y_pred"] = [
            self._states[str(sid)][0] if self.variant == "last_value"
            else self._states[str(sid)][1][(int(horizon) - 1) % 7]
            for sid, horizon in zip(prediction.series_id, prediction.horizon)
        ]
        prediction[["q10", "q50", "q90"]] = np.nan
        return prediction[PREDICTION_COLUMNS]


def make_seasonal_naive_factory(variant: str = "weekly_seasonal") -> Callable[[], SeasonalNaiveForecaster]:
    """Return a zero-argument constructor producing independent fold-local state."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown naive variant {variant!r}; expected {VARIANTS}")
    return partial(SeasonalNaiveForecaster, variant=variant)
