"""Run all tests in isolated numerical stacks to avoid macOS OpenMP collisions."""
from pathlib import Path
import subprocess
import sys

from src.config import ROOT


def main():
    neural=['tests/test_deepar.py','tests/test_data_pipeline_and_integration.py']
    common=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'tests').glob('test_*.py'))
            if str(p.relative_to(ROOT)) not in neural]
    for title, files in [('Core and LightGBM',common),('PyTorch DeepAR',neural)]:
        print(title,flush=True)
        subprocess.run([sys.executable,'-m','pytest',*files],cwd=ROOT,check=True)

if __name__=='__main__':
    main()
