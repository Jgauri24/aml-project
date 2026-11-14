"""Evaluation module exports."""
from src.evaluation.backtesting import (
    BaseForecaster,
    ForecastInputs,
    FUTURE_KEYS,
    PREDICTION_COLUMNS,
    attach_actuals,
    construct_inputs,
    format_backtest_results,
    run_backtest,
    validate_predictions,
)
from src.evaluation.integration import (
    MODEL_FAMILIES,
    compute_comprehensive_metrics,
    load_all_model_benchmarks,
    verify_consistent_evaluation_inputs,
)
from src.evaluation.metrics import (
    add_mase_scales,
    compute_metric_tables,
    mae,
    mase,
    mase_scale,
    pinball_loss,
    rmse,
    save_metric_tables,
    score_predictions,
    smape,
    wape,
)
from src.evaluation.model_suite import run_model_suite
from src.evaluation.model_comparison import (
    bootstrap_performance_differences,
    compare_models,
    diebold_mariano_test,
    paired_loss_comparisons,
    rank_models,
    save_comparison_results,
)
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
    "format_backtest_results",
    "run_backtest",
    "validate_predictions",
    "MODEL_FAMILIES",
    "compute_comprehensive_metrics",
    "load_all_model_benchmarks",
    "verify_consistent_evaluation_inputs",
    "score_predictions",
    "mae",
    "rmse",
    "wape",
    "smape",
    "mase",
    "mase_scale",
    "pinball_loss",
    "add_mase_scales",
    "compute_metric_tables",
    "save_metric_tables",
    "run_model_suite",
    "paired_loss_comparisons",
    "diebold_mariano_test",
    "bootstrap_performance_differences",
    "compare_models",
    "rank_models",
    "save_comparison_results",
    "Fold",
    "development_splits",
    "final_holdout_spec",
    "validate_chronological_splits",
    "validate_forecast_inputs_chronology",
    "validate_model_internal_splits",
    "standardize_prediction_output",
]

