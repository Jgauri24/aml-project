# Paired model comparison and significance testing

`src/evaluation/model_comparison.py` compares standardized, evaluator-scored
forecast rows (`actual` must already be joined after prediction validation).
`compare_models` returns paired loss summaries, Diebold–Mariano (DM) results,
and cluster-bootstrap intervals. `rank_models` returns separate accuracy,
horizon, demand-segment, probabilistic, and runtime tables. No aggregate rank or
combined score is produced. `save_comparison_results` writes each table to CSV
or Parquet.

## Pairing and losses

Forecasts are paired by `series_id`, forecast origin, horizon day, and forecast
date. Evaluation actuals must match exactly. Unmatched forecast keys are
excluded from that pair and counted in the output; missing actuals on both sides
are excluded. Duplicate keys, mismatched actuals, and nonfinite predictions
are errors. Differences are always model A minus model B, so negative loss
differences favor A. Row-level summaries include the mean, median, standard
deviation, number of paired observations, series and origins for absolute error,
squared error, and pinball loss when both forecasts supply quantiles.

## DM test assumptions

Daily SKU rows are **not** treated as independent replicates. For each origin,
the module first averages loss differences within each SKU-store series and
then equally averages across series. The resulting sequence has one observation
per forecast origin. The DM test evaluates that origin-level sequence.

The long-run variance uses a Bartlett/Newey–West estimator. Its maximum lag is
`max(0, ceil(horizon / origin_step) - 1)`, reflecting overlap between forecast
blocks. The statistic uses the Harvey–Leybourne–Newbold finite-sample correction
and a Student-t reference with `n_origins - 1` degrees of freedom. Pairwise
p-values are Holm-adjusted within each loss family. By default, fewer than eight
origins yields an explicit `insufficient_origins` result and no p-value. This
conservative gate is especially important when only a handful of rolling folds
are configured. Even above the gate, the test assumes the origin-level loss
differential is sufficiently stationary; interpret results cautiously when the
number of independent origins is small or origins are irregularly spaced.

## Cluster bootstrap

The percentile bootstrap resamples SKU-store series with replacement and keeps
each selected series' complete set of forecast origins and dates together. This
preserves within-series dependence and reports the model A minus model B metric
difference, a 95% interval by default, iteration count, and whether zero is in
the interval. A single available series is reported as insufficient for a
cluster bootstrap. Results are conditional on the selected series population;
with few series the interval can be coarse and should not be interpreted as
population-wide certainty.

## Rankings and runtime

Accuracy rankings use MAE, RMSE, and WAPE; horizon and demand-segment rankings
are reported separately. Probabilistic rankings use available quantile pinball
losses and WIS. Runtime rankings are produced only if fold execution data with
`model`, `fit_seconds`, and/or `predict_seconds` is supplied. All ranks are
lower-is-better within each metric and stratum; metrics are never combined.

Run the reproducible synthetic integration example (which also writes result
tables under `reports/model_comparison_demo/`) with:

```bash
python -m examples.model_suite_demo
```

That demo generates fresh forecasts with the existing adapters. It is an
integration demonstration, not a retail-data significance study. The repository
currently contains historical aggregate model summary CSVs but not common
row-level forecasts for those runs, so those summaries cannot establish paired
significance. Run the backtester with all compared model outputs saved on the
same series, origins, and dates before drawing conclusions about retail demand.
