# Model dependency compatibility

Checked **2026-10-20**, on **macOS 26.6.2 ARM64, CPython 3.14.6**. The established `.venv` uses pandas 3.0.6 and NumPy 2.5.4. Its installed distributions and `requirements.txt` were kept unchanged. Optional libraries were installed into temporary virtual environments; no experimental model was instantiated or trained. These results are import/runtime/readiness checks, not evidence about future model training correctness or performance.

## Results

| Library / stack | Python/platform result | Existing project dependency result | Actual verification |
| --- | --- | --- | --- |
| LightGBM 4.7.0 | A Python-independent ARM64 macOS wheel is available and installs on Python 3.14.6. | Resolves alongside existing pandas/NumPy pins. | Initial import fails because this Mac lacks `libomp.dylib`. Import passes with official Homebrew libomp 23.1.3 supplied from a temporary extraction. No global runtime was installed. |
| PyTorch 2.14.1 | CPython 3.14 ARM64/macOS 14+ wheel installs and imports here. | Resolves alongside existing pandas/NumPy pins. | Torch/NumPy conversion and tensor sum pass; existing project smoke passes in the isolated environment. MPS is available on this Mac; CUDA is not. No GPU training was checked. |
| GluonTS 0.17.0 with torch extra | Metadata permits Python >=3.10,<3.15; the PyTorch backend imports on 3.14.6. | **Conflicts with baseline pandas 3.0.6:** GluonTS requires pandas >=1.4,<3. Do not install it into `.venv`. | A separate environment with pandas 2.3.3, NumPy 2.5.4, torch 2.14.1, lightning/pytorch-lightning 2.6.6 and pydantic 2.14.1 passes backend import, tensor check, canonical-panel smoke and 16 adapter integration tests. |

The pandas conflict is a dependency-resolution failure, not a claim that Python 3.14 itself is unsupported. The LightGBM issue is a native runtime prerequisite, not a missing Python wheel. A plain GluonTS import does not prove its torch backend works; verification explicitly imported `gluonts.torch.model.deepar` without constructing an estimator.

## Keep environments separate

All commands below run from the repository root. They create **new** environments and never replace baseline package versions. The added `.venv-*` Git exclusion keeps these local environments untracked. Use explicit interpreter paths so a currently activated baseline environment is not modified accidentally.

### LightGBM development environment

```bash
uv venv .venv-lightgbm --python 3.14.6
uv pip install --python .venv-lightgbm/bin/python -r requirements/lightgbm.txt
# On macOS, provide the required runtime if it is not already installed:
brew install libomp
.venv-lightgbm/bin/python -m scripts.check_model_dependencies --stack lightgbm
.venv-lightgbm/bin/python -m scripts.smoke_test
```

`requirements/lightgbm.txt` includes the existing baseline pins and adds LightGBM 4.7.0. The Homebrew installation above is a teammate setup command; it was **not** run during this review. To verify the dependency without changing this machine's system libraries, the review fetched the official Homebrew bottle, extracted it under `/tmp`, and ran the import with a process-local `DYLD_LIBRARY_PATH`:

```bash
HOMEBREW_NO_AUTO_UPDATE=1 brew fetch libomp
mkdir -p /tmp/aml-libomp-runtime
tar -xzf "$(brew --cache libomp)" -C /tmp/aml-libomp-runtime
DYLD_LIBRARY_PATH=/tmp/aml-libomp-runtime/libomp/23.1.3/lib \
  .venv-lightgbm/bin/python -m scripts.check_model_dependencies --stack lightgbm
```

The extraction subdirectory reflects the tested libomp version; use the version actually fetched if it changes. This is a diagnostic alternative, not a project-wide dynamic-library override. Normal development should provide a suitable OS runtime using the official installation guidance. Do not disable duplicate-OpenMP safeguards or globally rewrite dynamic-library search paths. Linux and Windows runtime setup has not been tested here.

### PyTorch-only development environment

```bash
uv venv .venv-torch --python 3.14.6
uv pip install --python .venv-torch/bin/python -r requirements/torch.txt
.venv-torch/bin/python -m scripts.check_model_dependencies --stack torch
.venv-torch/bin/python -m scripts.smoke_test
```

This uses the unchanged baseline pins plus torch 2.14.1. A torch-only setup is **not** a GluonTS setup: adding GluonTS still introduces the pandas conflict. Linux PyPI torch packages can pull substantial CUDA dependencies; use the official PyTorch selector for an appropriate CPU/CUDA installation if developing on Linux, then separately verify the requested pinned stack. No Linux/CUDA result is claimed here.

### GluonTS/PyTorch development environment

```bash
uv venv .venv-deepar --python 3.14.6
uv pip install --python .venv-deepar/bin/python -r requirements/deepar.txt
.venv-deepar/bin/python -m scripts.check_model_dependencies --stack gluonts
.venv-deepar/bin/python -m scripts.smoke_test
.venv-deepar/bin/python -m pytest tests/test_model_integration.py
```

`requirements/deepar.txt` is a standalone pinned model environment, **not** an overlay to combine with `requirements.txt`. It pins the resolved torch backend and shared-engine/test/plot dependencies, including pandas 2.3.3. It deliberately does not reproduce the baseline notebook toolchain. The full baseline test suite is run in `.venv`; 16 model-adapter integration tests and the full development smoke were additionally executed in the GluonTS environment.

Build/copy canonical Parquet and metadata using the **baseline** environment first; then consume those exact artifacts here. Do not regenerate selection, panel or established EDA under pandas 2. The existing preprocessing's string/categorical handling was established with pandas 3; moving the entire M0–M4 pipeline to pandas 2 would require a separate reviewed migration. Loading the unchanged canonical panel and running the shared development evaluator were verified with this separate environment.

Observed warnings in the GluonTS checks: the library suggests optional faster JSON serializers, and synthetic-fixture calendar loading produces 13 pandas-2/NumPy timedelta deprecation warnings in the adapter tests. Neither causes a failure. No serializer package or warning-suppression dependency was added. This review does not promise warning-free raw preprocessing in the alternate environment.

## Recheck after changing versions

`scripts/check_model_dependencies.py` checks only the selected optional stack and exits nonzero on missing/broken imports. It never constructs a LightGBM, Kalman or DeepAR model. Use it in the corresponding optional environment, not as a required baseline test:

```bash
python -m scripts.check_model_dependencies --stack lightgbm
python -m scripts.check_model_dependencies --stack torch
python -m scripts.check_model_dependencies --stack gluonts
```

Keep model-specific dependencies out of baseline requirements. Re-run the shared tests and smoke before claiming a new stack is compatible; resolver success alone does not catch missing OpenMP or backend import failures. Later training must separately establish reproducibility, inner validation boundaries and CPU/GPU behavior.

## Primary references

- [LightGBM official Python introduction](https://lightgbm.readthedocs.io/en/stable/Python-Intro.html) and [installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html).
- [PyTorch official local installation selector](https://pytorch.org/get-started/locally/).
- [GluonTS official installation guide](https://ts.gluon.ai/dev/getting_started/install.html).
- Release metadata used for exact constraints/wheel availability: [LightGBM 4.7.0](https://pypi.org/pypi/lightgbm/4.7.0/json), [torch 2.14.1](https://pypi.org/pypi/torch/2.14.1/json), [GluonTS 0.17.0](https://pypi.org/pypi/gluonts/0.17.0/json).
