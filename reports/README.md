# Research evidence

The **current verified benchmark and trained artifacts** are under
[`artifacts/release/`](../artifacts/release/benchmark.csv). The final paper and
presentation are linked from the repository README.

This directory preserves earlier milestone findings and their original scopes.
Some older integration summaries combined the single-origin DeepAR ablation
with four-origin baseline metrics and assigned unsupported calibration values.
Those historical tables are not the final benchmark. Use the release's saved
row-level forecasts, receipts and source hashes for current findings.

`uncertainty_analysis/` is the original seven-model diagnostic run;
`uncertainty_release/` is the new four-family development analysis including
real DeepAR predictions. The latter notebook does not retrain or score the
final holdout. Old smoke-run snapshots were removed because the final release
now provides a complete reproducible set of artifacts.
