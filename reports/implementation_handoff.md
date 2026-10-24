# Implementation handoff

## M7 status and reproduction

M7 is complete: four independent per-series Kalman variants were evaluated on
the fixed development folds only. No final-holdout result or analysis is in
this handoff.

```bash
# verified M6 baseline
python -m scripts.run_kalman_single_sku
# M7 real development runs
python -m scripts.run_kalman_forecaster --variants local_level local_level_seasonal --n-jobs -1
python -m scripts.run_kalman_forecaster --variants local_trend local_trend_seasonal --n-jobs -1
# M7 comparison reports and figures
python -m scripts.analyze_kalman_variants
# complete test suite
python -m pytest
```

The executed full-suite result was 142 passed with 12 expected all-zero-window
fallback warnings. The M7 synthetic seasonal gate is included in that suite.

### M7 files and paths

- Model and tests: `src/models/kalman_forecaster.py`,
  `tests/test_kalman_forecaster.py`, and `tests/test_kalman_seasonal_gate.py`.
- Runners: `scripts/run_kalman_forecaster.py` and
  `scripts/analyze_kalman_variants.py`.
- Per-variant evaluator artifacts and 960-row fit summaries:
  `data/processed/kalman_forecaster/<variant>/`.
- Comparison and breakdowns: `reports/kalman_variant_comparison.csv`,
  `reports/kalman_variant_by_horizon.csv`,
  `reports/kalman_variant_by_volume_band.csv`, and
  `reports/kalman_variant_by_origin.csv`.
- Findings and figures: `reports/m7_kalman_findings.md` and
  `reports/figures/kalman_variant_*.png`.

Each M7 variant has 26,880 forecast rows (240 series × 4 origins × 28 days),
and its fit summary has 960 rows. Saved per-fold fit/predict totals are:

| Variant | Fit seconds | Predict seconds | Converged | Fallbacks |
| --- | ---: | ---: | ---: | ---: |
| local_level | 58.110396 | 1.223637 | 959 | 1 |
| local_level_seasonal | 209.475019 | 1.215572 | 959 | 1 |
| local_trend | 170.225579 | 1.224391 | 959 | 1 |
| local_trend_seasonal | 281.147637 | 1.281408 | 959 | 1 |

The generated metadata records per-fold fit/predict seconds shown above, but
does not persist a total command wall-clock duration. Every M7 variant had the
same all-zero-window fallback: `FOODS_1_102__CA_4` at d_1885. No
optimizer-failure fallback was recorded. Known issues are the Gaussian count
approximation, zero clipping, independent estimation, and material MLE runtime.
Next: add the compatible weekly seasonal-naive baseline; keep the final
holdout reserved.

## Historical M0–M4 handoff

## Status

| Milestone | Status | Executed evidence |
| --- | --- | --- |
| M0: environment/project | Complete | Local `.venv`, pinned requirements, importable modules, Git exclusions and commands documented. |
| M1: understanding/selection | Complete | Executed notebook 01; all raw key checks; 58,327,370 overlapping target cells match; deterministic 80/80/80 selection and dictionary. |
| M2: EDA | Complete | Executed notebook 02; eight labeled figures; descriptive statistics and five evidence/implication/limit findings, all through d_1801. |
| M3: preprocessing | Complete | CLI executed; full panel saved to Parquet and reread/revalidated; raw SHA-256 hashes unchanged. |
| M4: experiments/evaluation | Complete | Four development folds, dummy-only infrastructure run, 65 passing tests and explicit feature/holdout boundaries. |

No seasonal naive, Kalman, LightGBM or DeepAR implementation or training was performed. There are no remaining blockers within M0–M4.

## Files created or modified

- Setup: `README.md` (expanded), `requirements.txt`, `.gitignore`, `pytest.ini`, `experiment.yaml`, `configs/README.md`, package `__init__.py` files and data-directory `.gitkeep` placeholders.
- Reused/extended `notebooks/01_dataset_understanding.ipynb`; created `notebooks/02_eda.ipynb`. Both have saved successful executions. The previous full-width target previews were replaced with samples restricted to d_1801 or earlier.
- Data implementation: `src/config.py`; `src/data/load_data.py`, `select_skus.py`, `understanding.py`, `preprocess.py`, `validate.py`, `eda.py`.
- Evaluation implementation: `src/evaluation/splits.py`, `metrics.py`, `backtesting.py`.
- Execution/tests: `scripts/execute_notebooks.py`; `tests/conftest.py`, `test_data.py`, `test_metrics.py`, `test_backtesting.py`, `test_eda_and_config.py`.
- Reports: `dataset_dictionary.md`, `source_audit.json`, `eda_findings.md`, `eda_summary.json`, `experiment_protocol.md`, this handoff, `pytest_results.xml`, eight PNGs in `reports/figures/`, and local generated `eda_*.csv` statistics tables (ignored by Git).
- Local ignored data: `data/interim/selected_skus.csv`, `selection_metadata.json`; `data/processed/demand_panel.parquet`, `panel_metadata.json`; `data/processed/dummy_backtest/{predictions.parquet,scored_predictions.parquet,metrics.csv,execution_metadata.json}`.

Raw CSV files and generated datasets are not versioned. Useful source audits, notebooks, figures and written findings can be versioned. No raw file was modified; its hash was also checked against the earlier dataset-understanding task's snapshot.

## Actual data and cohort

| Source | Rows | Columns |
| --- | --- | --- |
| `calendar.csv` | 1,969 | 14 |
| `sell_prices.csv` | 6,841,121 | 4 |
| `sales_train_validation.csv` | 30,490 | 1,919 |
| `sales_train_evaluation.csv` | 30,490 | 1,947 |
| `sample_submission.csv` | 60,980 | 29 |

Both sales files contain 30,490 item/store pairs; evaluation covers d_1–d_1941, validation d_1–d_1913, calendar d_1–d_1969. There are 3,049 items, 10 stores, 7 departments, 3 categories and 3 states. Weekly prices have 282 distinct Walmart week keys; the file's candidate composite key is unique.

All 30,490 candidates satisfy the default complete-history eligibility rule through d_1801. Seed 42 selected **240 series: 80 low / 80 medium / 80 high**, with all 10 stores and all 3 categories in every band. Means use d_1437–d_1801; exact rank-pool limits, tie policy and sampling compromises are saved in selection metadata. Sparse/all-zero candidates are not excluded by a sales-volume rule.

- EDA/selection boundary: d_1801 = **2016-01-03**. EDA has 432,240 selected-series daily observations, starting 2011-01-29.
- Canonical observed panel: **465,840 × 28**, **2011-01-29–2016-05-22** (d_1–d_1941).
- Development targets: d_1802–d_1913, 2016-01-04–2016-04-24, across four 28-day folds.
- Reserved final targets: d_1914–d_1941, **2016-04-25–2016-05-22**, never analyzed or development-scored.

## Five EDA findings and interpretation

1. **Volume and relative noise differ.** Median historical mean sales/day are 0.145, 0.421, 1.574 for low/medium/high bands; median CVs are 3.275, 2.204, 1.460. Figure 01 and per-series statistics provide the evidence. Scale and intermittency can matter for global vs per-series fitting and pooled metrics. These descriptive groups do not establish future predictability or population-wide behavior.
2. **Weekly structure and persistence vary.** Equal-series normalized weekday peak/trough is 1.369; median lag-7 autocorrelation is 0.080/0.182/0.316 by band. Figures 02–05 show history, trailing means, weekday/month profiles and lag distributions. A weekly seasonal reference is a testable next step, but trends and availability uncertainty can inflate persistence; no accuracy gain is established.
3. **Zeros dominate many series.** 69.3% of selected observed days have zero sales; median zero fractions are 89.1%/74.4%/43.0%; the maximum run is 1,640 days. Descriptive patterns include 175 intermittent and 54 lumpy series. Figure 06 supports this. Count uncertainty and unstable percentage denominators deserve attention. Zeros cannot identify stockouts or latent demand; long runs include possible launch uncertainty.
4. **SNAP/event-day associations exist, with confounding.** SNAP-flagged-day normalized means relative to other days are {'CA': 1.048, 'TX': 1.06, 'WI': 1.08}. Figure 07 and named-event exposure tables support the comparisons. Announced events are plausible known future covariates. These contrasts are not causal effects, and future SNAP is excluded under the current conservative policy.
5. **Historical prices require careful interpretation.** 24.4% of safe-history rows lack a joined price; 159 series have usable within-series price/sales correlations with median -0.047. Figure 08 and per-series price counts/ranges provide evidence. Historical prices may be informative, but actual future prices are prohibited. Missing prices do not prove unavailable stock, and correlations are not price elasticities.

`eda_findings.md` expands each observation with evidence, forecasting implications and explicit limits. No causal claims, target imputation or inventory labels are made.

## Preprocessing validation

Validated again after the Parquet round trip: **240 series × 1,941 days**, **465,840 rows**, **0 duplicate series/day keys**, **0 missing indexed rows**, **0 missing targets**. Every selected series retains all daily timestamps. Calendar and price joins enforce many-to-one cardinality; every calendar day matches. Targets are nonnegative/finite, dates continuous, static metadata invariant, and zero/missing flags consistent.

Observed zeros remain observed zeros; target missingness is never forward-filled. `price_missing` stays separate from `possible_prelaunch` and from missing target/indexed rows. Prelaunch is a past-only uncertain heuristic, recomputed from fold history; availability is always unknown. No stockout labels are invented. In-memory panel size is 23.33 MB; Parquet uses compression and logical small/categorical dtypes. An all-null event-column Parquet edge case found during testing was fixed; dictionary string backend differences alone are not mistaken for value changes.

## Backtester and test evidence

`DummyForecaster` returns a constant zero solely to exercise the interface. Actual forecasts have **26,880 rows × 10 columns**, 6,720 rows at each of origins 1801, 1829, 1857 and 1885. Forecast keys, dates and offsets align exactly; the maximum target day is **1913**. Actuals are joined separately inside the evaluator and scored forecasts are stored separately from raw forecasts.

All-fold pooled sanity results: **MAE 1.207775**, **RMSE 2.846187**, **WAPE 1.0** (ratio), **pinball q10/q50/q90 = NA** because no quantiles were supplied. A zero prediction has absolute error equal to nonnegative actual sales, so WAPE = 1 whenever the denominator is positive. This is a plumbing sanity check, not baseline/model performance evidence. Hand-calculated quantile losses also agree with scikit-learn tests.

Final pytest run: **65 passed**, **0 failures**, **0 errors**, **0 skipped**, about **2.29 seconds**. `pytest_results.xml` records the run. Both notebooks executed without cell errors. Tests include raw readers, filtering, schema/key and domain failures, deterministic selection, missing/zero distinction, joins, continuity/static metadata, round trips, stale artifacts, exact folds, numerical metrics, quantile ordering, prediction grid alignment, fresh model instances and evaluator-only actual joins.

Leakage tests mutate future actuals, future prices, future flags and accidental future-derived columns; model inputs remain identical. A train-mean spy confirms fold-fitted state uses only historical targets. Mutating reserved final targets leaves every development forecast and metric unchanged. No final-holdout evaluator was invoked.

## Significant decisions, assumptions and residual risks

- **One canonical full panel, guarded consumers:** storing all observed dates avoids mismatched later preprocessing. Default development/EDA readers use explicit cutoff filters and reject stale config/selection; the holdout is present on disk but not in model inputs. Mechanical holdout validity checks are distinct from analytical inspection. This is an application boundary, not filesystem access control: later model code must use the supplied objects and avoid loading targets directly.
- **Selection before development:** a fixed, cutoff-safe, stratified cohort permits fair comparisons on identical series. Diversity is not population representativeness; all-history EDA means differ from trailing-mean ranking. Ties are deterministic, not distinct sales thresholds.
- **No learned preprocessing yet:** wide-to-long conversion, key joins, type choices and past-only audit flags do not fit future target parameters. Later fitted scalers/encoders/features belong inside each model's fold-local fit. Missing historical targets need an explicit model-local policy if encountered later.
- **Conservative feature availability:** announced dates/events are treated as known ahead; future actual prices, SNAP, sales, inventory and quality flags are excluded. Future price scenarios require a separately documented realistic assumption and tests. Excluding SNAP is stricter than could be possible with a published schedule.
- **Pooled evaluation:** every nonmissing series-day contributes equally to MAE/pinball; high-volume series can dominate scale-dependent metrics. WAPE is one pooled ratio, not an average of series ratios; zero-denominator and all-missing cases are NA. Point-only forecasts do not receive fabricated quantiles. Official hierarchical M5 WRMSSE is outside this scope.
- **Sales are not latent demand:** no stock, confirmed launch or stockout observations are present. Descriptive ADI/CV² categories and prelaunch flags remain heuristics.
- **Reproducibility:** selected IDs and numeric outputs are deterministic with pinned packages/seed/raw data. Execution timestamps and timing metadata intentionally vary. Tested environment: macOS ARM64, Python 3.14.6; other platforms were not executed here.

## Exact reproduction commands

From the repository root, using the already downloaded data:

```bash
uv venv .venv --python 3.14.6
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
python -m src.data.understanding
python -m src.data.preprocess
python -m src.data.eda
python -m src.evaluation.backtesting
python scripts/execute_notebooks.py
python -m pytest --junitxml=reports/pytest_results.xml
```

These are the executed pipeline/module entry points; preprocessing precedes the executable EDA because it produces its input artifact. Both notebooks call the same implementation and may regenerate identical reports/cohort files. Each data/evaluation module supports `--config`; the notebook runner uses the root configuration. `README.md` also gives a standard venv/pip setup alternative.

## Five questions to demonstrate understanding

1. Why would concatenating validation and evaluation sales double-count observations, and which exact observations are reserved from development?
2. How are eligible candidates ranked, ties broken and stores/categories represented without using sales after d_1801?
3. What distinguishes an observed zero, a missing target, a missing indexed day, a missing price and an uncertain prelaunch flag?
4. Which fields can a model see at origin d_1801, and why must both future actual prices and future target-derived rolling statistics be excluded?
5. How do pooled WAPE and per-series-averaged errors differ, why is an all-zero actual denominator undefined, and why can't point forecasts supply genuine pinball scores?

**Next specific task for Milestone 5:** implement a weekly seasonal naive `BaseForecaster` that repeats each series' final observed seven-day pattern over the 28-day horizon, add hand-checked horizon/alignment and missing-history-policy tests, and evaluate it on only the four development folds. Keep quantiles unavailable and leave the final holdout reserved. It has not been implemented in this handoff.
