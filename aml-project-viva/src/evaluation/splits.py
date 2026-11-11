"""Chronological expanding windows. Final holdout is deliberately not a runner option."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    origin: int
    horizon: int

    @property
    def target_days(self):
        return range(self.origin + 1, self.origin + self.horizon + 1)


def development_splits(cfg):
    horizon = cfg["forecast"]["horizon"]
    final_origin = cfg["forecast"]["final_origin"]
    folds = [Fold(origin, horizon) for origin in cfg["forecast"]["development_origins"]]
    if not folds or any(fold.origin + horizon > final_origin for fold in folds):
        raise ValueError("Development fold enters the reserved final holdout")
    if any(fold.origin < cfg["selection"]["cutoff_day"] for fold in folds):
        raise ValueError("Selection sees beyond a development origin")
    if [f.origin for f in folds] != sorted(set(f.origin for f in folds)):
        raise ValueError("Development origins must be unique and ordered")
    return folds


def final_holdout_spec(cfg):
    """Documentation-only specification; the development runner never invokes this."""
    return Fold(cfg["forecast"]["final_origin"], cfg["forecast"]["horizon"])
