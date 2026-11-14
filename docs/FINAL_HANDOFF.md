# Final project handoff

## Added in the final packaging contribution

- A Streamlit viewer backed by genuine saved forecasts for all 240 series,
  with store/SKU/origin selection, history, quantile intervals, model metrics,
  demand segmentation and CSV downloads.
- A frozen specification and source/data hashes recorded before final target
  scoring, with a separate one-shot final evaluator and completion receipt.
- Fresh four-origin development reproduction and a single final run for weekly
  seasonal naive, seasonal Kalman, LightGBM and the existing custom PyTorch DeepAR.
- Saved per-origin trained artifacts, predictions, runtimes, conditional metrics,
  checkpoint training curves and integrity manifests.
- An eight-page IEEE paper and six-slide editable presentation. Both preserve
  earlier results while distinguishing their actual evaluation scopes.
- A refreshed executed uncertainty notebook, focused unit tests and complete
  setup and independent-replication instructions.
- Removal of the duplicate `aml-project-viva` checkout, repeated smoke-run
  snapshots, duplicate calendar/deck files and obsolete UI implementation.
  Unique older reports moved into `docs/archive/`.

The teammates' model implementations remain intact. The contribution packages,
validates, evaluates and presents them rather than claiming a new forecasting
architecture.

## Final holdout results

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Weekly seasonal naive | 1.2570 | 2.6268 | 0.9583 |
| Seasonal Kalman | 1.0369 | 2.0565 | 0.7904 |
| LightGBM | 0.9476 | 2.1045 | 0.7224 |
| DeepAR | 0.9850 | 2.2056 | 0.7509 |

All rows use the same 6,720 observed targets at origin 1913. LightGBM was the
prespecified primary model and wins final MAE/WAPE. Kalman wins final RMSE.
These are historical observed-sales outcomes, not measured stockout reductions.

## Verification

The isolated numerical-stack runner passes 355 tests (332 core/LightGBM and
23 PyTorch). A monolithic mixed-runtime pytest run hit a macOS OpenMP native
segmentation fault; use `python -m scripts.test_all` as documented. Both actual
four-model evaluation runs completed. The release integrity checker verifies
75 artifact hashes and both forecast grids without fitting or scoring models.

The presentation contains nine native editable charts and three native tables.
Its final package/layout validation has zero findings or warnings. The IEEE
paper source and all final pages were rendered and inspected. The notebook
executed without errors. Dashboard behavior was tested with Streamlit AppTest;
native browser screenshot permissions were unavailable.
