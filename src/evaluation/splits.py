"""Chronological expanding windows. Final holdout is deliberately not a runner option."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    origin: int
    horizon: int
    train_start: int = 1

    @property
    def target_days(self):
        return range(self.origin + 1, self.origin + self.horizon + 1)

    @property
    def training_days(self):
        return range(self.train_start, self.origin + 1)


def development_splits(cfg):
    horizon = cfg["forecast"]["horizon"]
    final_origin = cfg["forecast"]["final_origin"]
    backtest = cfg.get("backtest", {})
    origins = cfg["forecast"].get("development_origins")
    step_size = int(backtest.get("step_size", horizon))
    if step_size < 1:
        raise ValueError("Backtest step_size must be positive")
    if origins:
        origins = list(origins)
    else:
        start = backtest.get("origin_start")
        if start is None:
            raise ValueError("Configure development_origins or backtest.origin_start")
        origins = list(range(int(start), final_origin - horizon + 1, step_size))
    window_mode = backtest.get("window_mode", "expanding")
    window_days = backtest.get("training_window_days")
    if window_mode not in {"expanding", "sliding"}:
        raise ValueError("backtest.window_mode must be 'expanding' or 'sliding'")
    if window_mode == "sliding" and (window_days is None or int(window_days) < 1):
        raise ValueError("Sliding windows require positive backtest.training_window_days")
    if window_mode == "expanding" and window_days is not None:
        raise ValueError("training_window_days is only valid with sliding windows")
    folds = [Fold(int(origin), horizon,
                  max(1, int(origin) - int(window_days) + 1) if window_mode == "sliding" else 1)
             for origin in origins]
    if not folds or any(fold.origin + horizon > final_origin for fold in folds):
        raise ValueError("Development fold enters the reserved final holdout")
    if any(fold.origin < cfg["selection"]["cutoff_day"] for fold in folds):
        raise ValueError("Selection sees beyond a development origin")
    if [f.origin for f in folds] != sorted(set(f.origin for f in folds)):
        raise ValueError("Development origins must be unique and ordered")
    if any(f.train_start < 1 or f.train_start > f.origin for f in folds):
        raise ValueError("Training window must be nonempty and end at the forecast origin")
    return folds


def final_holdout_spec(cfg):
    """Documentation-only specification; the development runner never invokes this."""
    return Fold(cfg["forecast"]["final_origin"], cfg["forecast"]["horizon"])
