# Milestone 15 — uncertainty calibration and paired failure diagnostics

This contribution extends the existing backtesting and aggregate metrics with
reproducible diagnostics of where probabilistic forecasts succeed or fail.
It consumes saved model predictions and never changes model training.

## Run

From the repository root:

```bash
.venv/bin/python -m scripts.run_uncertainty_analysis
.venv/bin/python -m pytest tests/test_uncertainty_analysis.py
```

Open `notebooks/uncertainty_analysis.ipynb` with the `.venv` kernel for the
executed analysis. The notebook reruns the same reusable implementation.
Outputs go to `reports/uncertainty_analysis/`. Only standard production artifact
locations are discovered: `data/processed/{lightgbm,deepar}/predictions.parquet`,
`kalman_forecaster/*/predictions.parquet`, and
`seasonal_naive/*/predictions.parquet`. Audit/test copies are excluded.

To include a teammate's saved DeepAR predictions, put them in the standard
DeepAR location or select the exact inputs explicitly:

```bash
.venv/bin/python -m scripts.run_uncertainty_analysis \
  --prediction data/processed/deepar/predictions.parquet \
  --prediction data/processed/kalman_forecaster/local_level_seasonal/predictions.parquet \
  --output-dir reports/uncertainty_deepar_vs_kalman
```

Each file needs the shared `PREDICTION_COLUMNS` schema and the **complete**
configured cohort, origins and horizons. Use `--config` for another experiment.
The canonical panel must have been built with that configuration. Files must
identify models consistently; duplicate exports of the same model are rejected.

## Added analysis

- Shared MAE, RMSE, WAPE, P10/P50/P90 pinball, 80% interval coverage, width and
  WIS, grouped overall, by origin, individual horizon, week, volume and volume × week.
- Lower-tail misses (`actual < P10`) and upper-tail misses (`actual > P90`).
  Coverage plus both miss rates equals one for observed targets.
- Quantile CDF bounds: fractions with `actual < Pq` and `actual <= Pq`.
  Demand is discrete: ties, especially at zero, mean exact 80% coverage is not
  guaranteed even for correctly specified count distributions. Interpret coverage
  jointly with interval width, tail balance, CDF bounds and pinball losses.
- Fold-local training volume terciles, history observation counts, zero fractions
  and trailing zero runs. These labels cannot see forecast targets. Terciles use
  deterministic rank with series ID as the tie breaker and can change by origin.
  A missing observation interrupts a zero run; it is never imputed to zero.
- Per-series losses, ten worst series per model, and all paired model MAE
  differences. Negative `mae_difference_a_minus_b` favors model A. Numerical ties
  (absolute tolerance 1e-12) are recorded explicitly. DeepAR/Kalman pairs appear
  automatically when both genuine forecast artifacts are available.
- Plots of weekly coverage, width, upper-tail misses and coverage by volume.
- Source SHA-256 hashes and configuration hash in `manifest.json`.

## Evaluation safeguards and limits

Forecasts are checked against a grid independently derived from the canonical
panel and configured development splits. Extra holdout rows, missing series,
duplicate keys, mismatched dates, nonfinite forecasts and partial/crossed
quantiles raise errors. All models use canonical actuals and identical missing
observation masks. Supplied saved actual columns are ignored.

Point-only models receive NA uncertainty metrics. The final holdout is never
scored; the run uses only days through the configured final origin. Saved models'
training provenance still needs separate review: matching prediction keys does
not establish leakage-free fitting. Origin and SKU errors are dependent, so
these tables do not claim independent-row statistical significance.

On this checkout, saved DeepAR row-level predictions are absent. The executed
notebook therefore reports Kalman intervals and point metrics for LightGBM and
naive models, and explicitly leaves the DeepAR comparison pending. Aggregate
DeepAR experiment metrics cannot reconstruct its per-SKU quantiles or coverage.
No model was retrained and no missing DeepAR results were synthesized.

The integration summary loader previously assigned every DeepAR experiment
coverage `0.824` and width `2.85`, regardless of saved evidence. These constants
and the repeated 82.4% narrative claim were removed. Optional measured values
are preserved when supplied; otherwise they remain NA.

## Contribution walkthrough

The work to review and explain is the validation boundary in `prepare_scored`,
the training-only grouping in `history_labels`, the tail/CDF calculations in
`diagnostic_metrics`, and the paired comparison in `analyze_uncertainty`.
The notebook and tests provide reproducible examples. This is a focused
Milestone 15 extension to the existing team implementation, not a replacement
for Members 1–4's model, pipeline or backtesting work.
