# Model integration readiness review

Reviewed the existing `eda` branch on 2026-10-20. M0–M4 were not rebuilt, and experiment settings, raw data, selected cohort, panel, metadata, existing dummy output and established reports/notebooks were preserved. Tests that exercise preprocessing use temporary synthetic fixtures only.

## Deliverables

- [`docs/MODEL_INTEGRATION.md`](../docs/MODEL_INTEGRATION.md): existing interfaces, model factories, allowed information, required prediction schema, shared runner, output isolation and parallel ownership guidance.
- [`docs/TEAMMATE_SETUP.md`](../docs/TEAMMATE_SETUP.md): exact clone/environment commands; obtaining ignored local data; copying canonical artifacts vs explicitly preparing them when unavailable; reproduction and readiness steps.
- [`docs/DEPENDENCIES.md`](../docs/DEPENDENCIES.md), `requirements/lightgbm.txt`, `torch.txt`, `deepar.txt`, and `scripts/check_model_dependencies.py`: versioned optional setups and actual platform/dependency limitations. Baseline `requirements.txt` is unchanged.
- [`examples/constant_forecaster.py`](../examples/constant_forecaster.py): minimal fixed-value interface example and CLI; no target learning and no invented quantiles. It produced 26,880 × 10 development forecasts when run on the existing canonical data, with artifacts written only to a temporary output directory.
- [`scripts/smoke_test.py`](../scripts/smoke_test.py): single read-only command `python -m scripts.smoke_test`.
- `tests/test_model_integration.py` and `tests/test_smoke.py`: 20 additional test cases; detailed results in [`model_integration_pytest.xml`](model_integration_pytest.xml).
- `README.md`: linked integration/setup/compatibility guidance and the smoke command; `.gitignore`: excludes optional `.venv-*` environments.

## Verified fixes, without experiment-rule changes

New regression tests failed against the original backtester in two places:

1. A pandas nullable `Float64` point column containing `pd.NA` passed the finite-output validator because the boolean reduction skipped nulls. Validation now explicitly rejects missing points, consistent with the existing finite-point contract.
2. The engine held strong references to all previously fitted models to detect instance reuse. A weak-reference/lifecycle test confirmed that an old fold model remained live while constructing the next. Weak references now preserve reuse detection without retaining fitted weights, and fold-local models are released after scoring. The existing shared-singleton rejection test still passes.

No split, metric definition, feature allowlist, model interface or selection rule changed. Both fixes are confined to `src/evaluation/backtesting.py`.

## Actual validation

- Before edits: **65 / 65 existing tests passed**.
- After edits: **85 tests passed**, **0 failures**, **0 errors**, **0 skipped**; 4.06 seconds in the baseline environment.
- Smoke passed: **18 imports, 8 required files, 240 series, four folds, 26,880 × 10 forecasts**, maximum target day **1913**, `final_holdout_evaluated=false`, no result files written.
- Original final-holdout mutation/leakage tests remain in the full passing suite. New checks add shuffled output/actual alignment, evaluator quantile scoring, missing/schema/factory/name failures, fresh-origin example fitting, fold-model lifetime, input-panel preservation and smoke missing-file/CLI/non-write behavior.
- `uv pip check` passes for the unchanged baseline and both temporary optional environments. Baseline installed package inventory (59 distributions), `requirements.txt`, `experiment.yaml` and captured existing data/artifact hashes are unchanged. New Python sources compile and local Markdown links resolve.

## Compatibility findings

| Stack | Result |
| --- | --- |
| LightGBM 4.7.0 / Python 3.14.6 / baseline pandas 3.0.6 | Resolver/wheel installation pass. Initially import fails because this Mac lacks OpenMP. With official Homebrew libomp 23.1.3 extracted under `/tmp` and a process-local library path, import passes. No system-wide library installation or baseline package modification. |
| PyTorch 2.14.1 / Python 3.14.6 / baseline pandas 3.0.6 | Import, NumPy/tensor operation and full canonical-data smoke pass in an isolated environment. MPS is available here; CUDA is not. No accelerated model training attempted. |
| GluonTS 0.17.0 torch backend | Resolver rejects baseline pandas 3.0.6 because GluonTS requires pandas <3. A separate pinned pandas 2.3.3 / torch 2.14.1 / lightning 2.6.6 environment passes backend import, full canonical-data smoke and all 16 adapter integration cases. Use it to consume the baseline-built panel, not to regenerate M0–M4. |

Observed optional-stack warnings: GluonTS recommends optional faster JSON serialization; 13 warnings concern pandas-2/NumPy generic timedelta deprecation in synthetic fixture loading. They do not fail the integration checks. macOS ARM64 is the tested platform; Linux/Windows package resolution, CUDA, training convergence and future adapters are not certified by these import/readiness tests. Exact checks are in [`model_dependency_checks.json`](model_dependency_checks.json); primary sources and setup commands are linked in the dependency guide.

## Remaining checks for future adapters

The generic contract cannot prove a future adapter's statistical correctness. Each model branch still needs tests for its own lag/recursive-feature construction, inner tuning/early stopping, fitted encoders/scalers, missing-history handling, library seeds and model-specific numerical behavior. Probabilistic adapters additionally need genuine sample/quantile extraction and calibration diagnostics. GPU determinism/resource cleanup and shared-output collisions must be assessed for the chosen deployment; use unique per-model/run output directories now. Do not let adapter code load future targets directly, even though Python cannot sandbox filesystem access.

## Commands executed

```bash
.venv/bin/python -m pytest --junitxml=/tmp/aml-integration-before.xml
.venv/bin/python -m pytest --junitxml=reports/model_integration_pytest.xml
.venv/bin/python -m scripts.smoke_test
.venv/bin/python -m examples.constant_forecaster --output-dir /tmp/aml-example-integration-run
.venv/bin/python -m compileall -q examples scripts src tests
uv pip check --python .venv/bin/python
```

Optional checks used `/tmp/aml-model-integration-check` (baseline pins plus LightGBM/PyTorch) and `/tmp/aml-gluonts-integration-check` (standalone pandas-2 GluonTS stack). Commands to reproduce those stacks under local `.venv-*` directories are in `docs/DEPENDENCIES.md`. GluonTS was checked with `--stack gluonts`, the shared smoke, and `python -m pytest tests/test_model_integration.py`; the LightGBM runtime check used only a temporary extracted library path.

No final holdout evaluation and no Kalman, LightGBM or DeepAR implementation/training occurred. The next modeling task remains the separately scoped seasonal naive baseline.
