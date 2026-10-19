"""Execute and save both notebooks using the current Python environment."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = [ROOT / 'notebooks' / name for name in ['01_dataset_understanding.ipynb', '02_eda.ipynb']]
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name='python3')
        # Explicit executable prevents IDE/system kernels from using another environment.
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        NotebookClient(notebook, km=manager, timeout=600,
                       resources={'metadata': {'path': str(path.parent)}}).execute(cleanup_kc=True)
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(f'Executed {path.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    main()
