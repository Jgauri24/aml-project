"""Import/tensor checks for optional environments; never construct or fit a model."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import platform


def check_stack(stack):
    versions = {name: metadata.version(name) for name in ["numpy", "pandas", "pyarrow"]}
    if stack == "lightgbm":
        module = importlib.import_module("lightgbm")
        versions["lightgbm"] = module.__version__
    else:
        import numpy as np
        import torch
        versions["torch"] = torch.__version__
        if torch.from_numpy(np.array([1, 2, 3], dtype="float32")).sum().item() != 6:
            raise RuntimeError("Torch/NumPy tensor interoperability failed")
        if stack == "gluonts":
            importlib.import_module("gluonts.torch.model.deepar")
            for name in ["gluonts", "lightning", "pytorch-lightning", "pydantic"]:
                versions[name] = metadata.version(name)
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", choices=["lightgbm", "torch", "gluonts"], required=True)
    args = parser.parse_args()
    try:
        versions = check_stack(args.stack)
    except (ImportError, OSError, RuntimeError, metadata.PackageNotFoundError) as error:
        print(json.dumps({"status": "failed", "stack": args.stack, "error": str(error)}, indent=2))
        return 1
    print(json.dumps({"status": "passed", "stack": args.stack, "python": platform.python_version(),
                      "platform": platform.platform(), "versions": versions,
                      "model_constructed_or_trained": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
