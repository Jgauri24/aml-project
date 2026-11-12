"""Evaluation module exports."""
from src.evaluation.backtesting import (
    BaseForecaster,
    ForecastInputs,
    FUTURE_KEYS,
    PREDICTION_COLUMNS,
    attach_actuals,
    construct_inputs,
    run_backtest,
    validate_predictions,
)
from src.evaluation.integration import (
    MODEL_FAMILIES,
    compute_comprehensive_metrics,
    load_all_model_benchmarks,
    verify_consistent_evaluation_inputs,
)
from src.evaluation.metrics import score_predictions
from src.evaluation.splits import Fold, development_splits, final_holdout_spec
from src.evaluation.splits_validation import (
    validate_chronological_splits,
    validate_forecast_inputs_chronology,
    validate_model_internal_splits,
)
from src.evaluation.standardization import standardize_prediction_output

__all__ = [
    "BaseForecaster",
    "ForecastInputs",
    "FUTURE_KEYS",
    "PREDICTION_COLUMNS",
    "attach_actuals",
    "construct_inputs",
    "run_backtest",
    "validate_predictions",
    "MODEL_FAMILIES",
    "compute_comprehensive_metrics",
    "load_all_model_benchmarks",
    "verify_consistent_evaluation_inputs",
    "score_predictions",
    "Fold",
    "development_splits",
    "final_holdout_spec",
    "validate_chronological_splits",
    "validate_forecast_inputs_chronology",
    "validate_model_internal_splits",
    "standardize_prediction_output",
]

