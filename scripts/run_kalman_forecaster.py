"""Run one or more independent M7 Kalman variants on development folds only.

Examples:
    python -m scripts.run_kalman_forecaster --variants local_level local_level_seasonal --n-jobs -1
    python -m scripts.run_kalman_forecaster --variants local_trend local_trend_seasonal --n-jobs -1
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.config import data_path, load_config
from src.data.preprocess import load_panel
from src.evaluation.backtesting import FUTURE_KEYS, PREDICTION_COLUMNS, construct_inputs, run_backtest
from src.evaluation.splits import development_splits
from src.models.kalman_forecaster import VARIANTS, KalmanForecaster, make_kalman_factory


def validate_integrity(predictions: pd.DataFrame, metadata: dict, panel: pd.DataFrame, cfg: dict) -> dict:
    """Assert exact development-grid coverage and the shared forecast contract."""
    folds = development_splits(cfg)
    expected = pd.concat([construct_inputs(panel, fold).future_known[FUTURE_KEYS] for fold in folds], ignore_index=True)
    assert list(predictions.columns) == PREDICTION_COLUMNS
    assert predictions["series_id"].nunique() == 240
    assert set(predictions["origin"].unique()) == {fold.origin for fold in folds}
    per_fold_series = predictions.groupby("origin", observed=True)["series_id"].nunique()
    assert per_fold_series.eq(240).all(), per_fold_series.to_dict()
    per_series_fold = predictions.groupby(["origin", "series_id"], observed=True).size()
    assert per_series_fold.eq(28).all(), "Each series/origin must have exactly 28 rows"
    assert len(predictions) == 26_880
    assert not predictions.duplicated(["series_id", "origin", "horizon"]).any()
    compared = expected.merge(predictions[FUTURE_KEYS], how="outer", on=FUTURE_KEYS,
                              validate="one_to_one", indicator=True)
    assert compared["_merge"].eq("both").all(), compared["_merge"].value_counts().to_dict()
    values = predictions[["y_pred", "q10", "q50", "q90"]].to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert predictions["y_pred"].ge(0).all()
    assert predictions["q10"].ge(0).all()
    assert (predictions["q10"] <= predictions["q50"]).all()
    assert (predictions["q50"] <= predictions["q90"]).all()
    assert np.array_equal(predictions["y_pred"].to_numpy(), predictions["q50"].to_numpy())
    assert metadata["final_holdout_evaluated"] is False
    assert metadata["max_target_day"] <= 1913
    assert predictions["target_day_num"].max() <= cfg["forecast"]["final_origin"]
    return {"unique_series": int(predictions["series_id"].nunique()), "origins": sorted(map(int, predictions["origin"].unique())),
            "rows": len(predictions), "expected_keys_exact": True, "finite_ordered_quantiles": True,
            "y_pred_equals_q50": True, "final_holdout_evaluated": metadata["final_holdout_evaluated"],
            "max_target_day": metadata["max_target_day"]}


def run_variant(panel: pd.DataFrame, cfg: dict, variant: str, n_jobs: int) -> tuple[dict, float, Path]:
    created_models: list[KalmanForecaster] = []
    base_factory = make_kalman_factory(variant, n_jobs=n_jobs)

    def collecting_factory() -> KalmanForecaster:
        model = base_factory()
        created_models.append(model)
        return model

    output_relative = f"kalman_forecaster/{variant}"
    started = perf_counter()
    predictions, _, metadata = run_backtest(panel, cfg, collecting_factory, output_dir=output_relative)
    elapsed = perf_counter() - started
    if len(created_models) != 4:
        raise RuntimeError(f"Expected one model per fold, created {len(created_models)}")
    integrity = validate_integrity(predictions, metadata, panel, cfg)
    summaries = []
    for model in created_models:
        summary = model.fit_summary_.copy()
        summary["origin"] = model._fitted_origin
        summaries.append(summary)
    destination = data_path(cfg, "processed") / output_relative
    summary = pd.concat(summaries, ignore_index=True).sort_values(["origin", "series_id"]).reset_index(drop=True)
    summary.to_csv(destination / "fit_summary.csv", index=False)
    fallback_rows = summary.loc[summary["fallback"] | ~summary["convergence_status"].eq("converged"),
                                ["origin", "series_id", "convergence_status", "fallback", "fallback_reason"]]
    report = {"variant": variant, "wall_seconds": elapsed, "integrity": integrity,
              "converged": int(summary["convergence_status"].eq("converged").sum()),
              "fallback": int(summary["fallback"].sum()), "nonconverged_or_fallback": fallback_rows.to_dict("records"),
              "output_dir": str(destination)}
    print(f"\n{variant}: {report}")
    return report, elapsed, destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    if args.n_jobs == 0:
        parser.error("--n-jobs must be nonzero")
    cfg = load_config()
    panel = load_panel(cfg, cutoff=cfg["forecast"]["final_origin"])
    print(f"Loaded {panel['series_id'].nunique()} series through d_{panel['day_num'].max()}")
    for variant in args.variants:
        run_variant(panel, cfg, variant, args.n_jobs)


if __name__ == "__main__":
    main()
