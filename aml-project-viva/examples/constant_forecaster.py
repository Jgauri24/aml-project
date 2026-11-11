"""Run a fixed-value integration example through the shared development engine."""
import argparse
from functools import partial

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.preprocess import load_panel
from src.evaluation.backtesting import (
    BaseForecaster, ForecastInputs, FUTURE_KEYS, PREDICTION_COLUMNS, run_backtest,
)


class ExampleConstantForecaster(BaseForecaster):
    """Illustrate the adapter contract only; do not use as a model comparison result."""

    def __init__(self, value=1.0):
        if not np.isfinite(value) or value < 0:
            raise ValueError("Example constant must be finite and nonnegative")
        self.value = float(value)
        self.fitted_origin = None

    def fit(self, inputs: ForecastInputs):
        # A real adapter fits encoders/scalers and model state from history here.
        # This example learns nothing and deliberately never reads target values.
        self.fitted_origin = inputs.origin
        return self

    def predict(self, inputs: ForecastInputs) -> pd.DataFrame:
        if self.fitted_origin != inputs.origin:
            raise RuntimeError("Fit a fresh example on this origin before predicting")
        prediction = inputs.future_known[FUTURE_KEYS].copy()
        prediction["model"] = "example_constant"
        prediction["y_pred"] = self.value
        prediction[["q10", "q50", "q90"]] = np.nan
        return prediction[PREDICTION_COLUMNS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--value", type=float, default=1.0)
    parser.add_argument("--output-dir", default="integration_example")
    args = parser.parse_args()
    cfg = load_config(args.config)
    panel = load_panel(cfg, cutoff=cfg["forecast"]["final_origin"])
    # partial is a zero-argument factory: the engine creates a new object per fold.
    factory = partial(ExampleConstantForecaster, value=args.value)
    predictions, metrics, _ = run_backtest(panel, cfg, factory, args.output_dir)
    print(f"Example forecasts: {predictions.shape}; development only")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
