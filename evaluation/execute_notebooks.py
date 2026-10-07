"""Execute selected notebooks with the dedicated GPU virtual environment.

Run with evaluation/.venv/bin/python evaluation/execute_notebooks.py NOTEBOOK.ipynb
from the repository root, selecting a name from notebook_paths.json.
Each completed cell is saved so long training does not hide earlier results.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import nbformat
from nbclient import NotebookClient
from render_notebook import export_html


def execute(name):
    root = Path(__file__).resolve().parent
    paths = json.loads((root / "notebook_paths.json").read_text())
    if name not in paths:
        raise ValueError("Choose a notebook from notebook_paths.json")
    path = root.parent / paths[name]
    if not path.is_file():
        raise FileNotFoundError(path)
    kernel_root = root / ".runtime-kernels"
    directory = kernel_root / "research-gpu"
    directory.mkdir(parents=True, exist_ok=True)
    specification = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                     "display_name": "Research GPU experiments", "language": "python",
                     "env": {"KERAS_BACKEND": "torch", "CUDA_VISIBLE_DEVICES": "0"}}
    (directory / "kernel.json").write_text(json.dumps(specification))
    manager = KernelManager(kernel_name="research-gpu",
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
    notebook = nbformat.read(path, as_version=4)
    # Old Colab widget state must not survive a fresh run with cleared outputs.
    notebook.metadata.pop("widgets", None)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    started = time.monotonic()

    def on_start(cell, cell_index, **kwargs):
        first = cell.source.splitlines()[0] if cell.source else "empty"
        print(f"{name}: cell {cell_index + 1}/{len(notebook.cells)} — {first[:100]}", flush=True)

    def on_executed(cell, cell_index, **kwargs):
        nbformat.write(notebook, path)

    client = NotebookClient(notebook, km=manager, timeout=10800,
                            resources={"metadata": {"path": str(path.parent)}},
                            on_cell_start=on_start, on_cell_executed=on_executed)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    html = export_html(notebook)
    preview = root / "runs" / "previews"
    preview.mkdir(parents=True, exist_ok=True)
    (preview / (path.stem + ".html")).write_text(html)
    print(f"Completed {name} in {time.monotonic() - started:.1f}s; saved outputs and HTML preview.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+")
    args = parser.parse_args()
    for notebook in args.notebooks:
        execute(notebook)
