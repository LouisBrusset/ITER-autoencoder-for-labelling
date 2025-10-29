# ITER Autoencoder for Labelling
Labelling datasets using a lightweight PyTorch autoencoder and latent-space analysis. This project provides a small FastAPI backend to create and manage synthetic datasets, train an autoencoder, and visualize reconstructions and latent-space projections to assist manual labelling.

## Overview

This repository contains a compact pipeline that demonstrates how to:

- generate synthetic datasets,
- load and manage datasets, 
- train a simple autoencoder model (PyTorch),
- save and manage model checkpoints,
- visualize reconstructions and 2D latent-space embeddings (UMAP),
- save human labels to a results folder.

The backend is a FastAPI application exposing endpoints for data creation, training, visualization and label persistence. A minimal frontend (static `index.html`) is provided for basic interaction.

## Table of Contents

1. [Overview](#overview)
2. [Package structure](#package-structure)
3. [Getting started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Clone the repository](#clone-the-repository)
    - [Create and activate a virtual environment](#create-and-activate-a-virtual-environment)
    - [Install dependencies](#install-dependencies)
    - [Run helper targets (clean/build/run)](#run-helper-targets-cleanbuildrun)
    - [Frontend](#frontend)
4. [File tree explanation](#file-tree-explanation)
5. [Uploaded files, notebooks and .npz structure](#uploaded-files-notebooks-and-npz-structure)
6. [Development dependencies](#development-dependencies)
7. [Contributing](#contributing)
8. [License](#license)
9. [Author](#author)
10. [Acknowledgments](#acknowledgments)
11. [References](#references)

## Package structure

Top-level packages and directories (short description):

- `src/autoencoder_for_labelling/` — Python package with the FastAPI app, routers, services, models and training code.
- `data/` — datasets (current and uploaded/synthetic). Not tracked in Git.
- `models/` — saved and current model checkpoints. Not tracked in Git.
- `results/` — saved label files (JSON & npz).
- `frontend/` — minimal static frontend (HTML/CSS/JS) to interact with the API.
- `src/tests/` — pytest test-suite for the API and modules.
- `Makefile`, `pyproject.toml` — development tasks and project metadata.

## Getting started

### 0. Prerequisites: 
Python 3.9+ 3.12- (3.11 recommended), git, uv

If `uv` is not available, install it via pip:

```bash
python -m pip install --user pipx
python -m pipx ensurepath
pipx install uv

# Check installation
pipx --version
uv --version
```

### 1. Clone the repository

```bash
git clone https://github.com/LouisBrusset/ITER-autoencoder-for-labelling.git
cd ITER-autoencoder-for-labelling
```

### 2. Create and activate a virtual environment

```bash
   # Using uv
   python -m pip install --user pipx
   python -m pipx ensurepath
   pipx install uv
   pipx --version
   uv --version
   uv venv --python 3.11
    
   # Using venv
   python -m venv .venv
   
   # On Windows
   source .venv\Scripts\activate
   
   # On Unix/MacOS
   source .venv/bin/activate
   ```

### 3. Install dependencies

   ```bash
   # Python venv
   pip install -e .

   # Using uv
   uv pip install -e .
   ```

Note: this repo uses `pyproject.toml`. If a `requirements.txt` is not present, you can install the project (`pip install .`) or install the runtime dependencies listed in `pyproject.toml`.

### 4. Run helper targets (clean/build/run)

Start the API server locally using the provided Makefile:

```bash
make clean
make run
```

The FastAPI backend should be available at http://127.0.0.1:8000

### 5. Frontend

Open `frontend/index.html` in a browser to interact with the API.

## File tree explanation

Below is a full view of the repository tree showing folders, subfolders and the important files (including `main.py` and `frontend/index.html`):

```
.
│
├── data/                    # datasets (generated at runtime)
│   ├── uploaded/
│   ├── synthetic/
│   └── current_dataset.npz  # (created when loading/producing datasets)
├── frontend/
│   ├── index.html
│   ├── styles/
│   │   └── style.css
│   └── scripts/
│       ├── script_checking.js
│       ├── script_classification.js
│       ├── script_data.js
│       ├── script_training.js
│       └── script_visualization.js
├── models/                  # model checkpoints (not tracked)
│   ├── saved_architechture/ # saved architecture JSON files
│   ├── saved_model/         # saved model checkpoints (.pth)
│   ├── current_architecture.json
│   └── current_model.pth    # (created when training a model)
├── notebooks/               # build datasets from MAST data
│   ├── mast_dataset_test.ipynb # MAST dataset test notebook
│   ├── mast_magnetics_data_constant_time.nc # MAST magnetics data from API scraping
│   └── [X]_mast_dataset_example.npz  # example datasets
├── results/
│   ├── labels/
│   ├── latents/
│   ├── projections2d/
│   └── reconstructions/
│
├── src/
│   ├── autoencoder_for_labelling/
│   │   ├── data/
│   │   │   └── dataset.py
│   │   ├── models/
│   │   │   └── autoencoder.py
│   │   ├── routers/
│   │   │   ├── checking_router.py
│   │   │   ├── data_router.py
│   │   │   ├── labels_router.py
│   │   │   ├── training_router.py
│   │   │   └── visualization_router.py
│   │   ├── services/
│   │   │   └── training_service.py
│   │   ├── training/
│   │   │   └── trainer.py
│   │   ├── __init__.py
│   │   └── main.py
│   └── tests/               # package-oriented tests
│       ├── test_checking_router.py
│       ├── test_data_router.py
│       ├── test_labels_router.py
│       ├── test_training_router.py
│       └── test_visualization_router.py
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── .gitignore
└── uv.lock
```

Notes:
- `src/autoencoder_for_labelling/main.py` is the FastAPI entrypoint used by the Makefile (module path `src.autoencoder_for_labelling.main:app`).
- `frontend/index.html` is the small static UI to interact with the API — open it in a browser while the backend runs.


## Uploaded files, notebooks and .npz structure

### a. Uploaded files and expected format

This project relies on compressed NumPy `.npz` files. For the application to work correctly (loading, training,
inference and visualization), a dataset file must contain at minimum the key `data`.

### b. `notebooks` directory

- The `notebooks/` folder contains example notebooks and scripts and
   example datasets (`*.npz`). Use these notebooks as templates to build `.npz` datasets compatible with the API.

Expected structure of a dataset `.npz` :

The dataset file consumed by the API and the visualization routes should follow this layout (types and shapes are
explicitly noted):

- `data` (required) — np.ndarray
   - shape: (N, D) where N = number of samples, D = input dimension
   - dtype: float32 or float64 (numeric 2-D feature matrix)

- `labels` (optional) — np.ndarray
   - shape: (N,)
   - dtype: integer (e.g. 0/1/2/...). Used for display and export of labels.

- `is_train` (optional) — np.ndarray
   - shape: (N,)
   - dtype: bool (True for training samples, False for validation samples). If absent, all samples are treated as
      training (`is_train=True`).

- `filename` (optional) — str
   - short human-readable identifier used when generating output filenames (e.g. `my_dataset`).

- `indices` (optional) — np.ndarray
   - shape: (M,) or (N,) depending on use
   - dtype: integer
   - used in some saved inference artifacts to map rows back to global dataset indices.

### c. Important notes

- The backend reads datasets with `np.load(..., allow_pickle=True)` and expects `d['data']` to exist. Any
   additional keys present are preserved and propagated into output artifacts (latents, projections, reconstructions)
   when available.

- Minimal example to save a compatible dataset from Python or a notebook:

```python
import numpy as np

# data: np.ndarray shape (N, D)
# labels: np.ndarray shape (N,) dtype=int (optional)
# is_train: np.ndarray shape (N,) dtype=bool (optional)

np.savez_compressed('data/current_dataset.npz', data=data, labels=labels, is_train=is_train, filename='my_dataset')
```

### d. Inference output files

- When running a full inference (`perform_full_inference`) the service writes files to
   `results/latents`, `results/projections2d` and `results/reconstructions`.

- Each inference `.npz` uses the main key `data` and includes metadata keys. Typical contents:
   - latents: `data` shape (N, latent_dim) — np.ndarray of floats
   - projection2d: `data` shape (N, 2) — np.ndarray of floats
   - reconstructions: `data` shape (N, D) — np.ndarray of floats
   - metadata keys: `indices` (np.ndarray of ints), `timestamp` (int), `filename` (str)
   - if present, `labels` (np.ndarray of ints) and `is_train` (np.ndarray of bools) are also included
   - latents files also include `latent_dim` (int)

### e. Compatibility and best practices

- Partial labels provided through the UI or API are merged into `data/current_dataset.npz` and, when possible,
   are propagated to current inference artifacts (`results/current_*.npz`) by matching the `indices` mapping.

- Avoid storing complex Python objects that are not simple ndarrays inside the primary keys — prefer numeric
   `ndarray` objects to ensure portability and correct behaviour across the API and frontend.

## Development dependencies

Development tools used by the project:

- pytest (test runner)
- pytest-cov (coverage)
- black (formatter)
- flake8 (linter)
- mypy (static typing)

You can install dev dependencies with pip (if using `pyproject.toml`) via an extra or by installing from the project:

```bash
uv pip install -e '.[dev]'
```

Or install individually:

```bash
# With uv
uv add pytest pytest-cov black flake8 mypy

# without uv
pip install pytest pytest-cov black flake8 mypy
```

### Running tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-quick

# Run specific test modules
pytest tests/test_data_router.py
pytest tests/test_training_router.py
```

### Code Formatting

```bash
# Format code (if using black)
black src/

# Check code style (if using flake8)
flake8 src/
```


### Type Checking

```bash
mypy src/
```

## Contributing

Contributions are welcome. A simple workflow:

1. Fork the repository on GitHub.
2. Create a feature branch: `git checkout -b feat/short-description`.
3. Make changes, add tests, run the test suite.
4. Commit and push to your fork: `git push origin feat/short-description`.
5. Open a Pull Request describing your changes and why they are useful.

Please follow the existing code style and add tests for new behavior. If you intend large refactors, open an issue first to discuss the design.

## License

This project is provided under the MIT License — see the `LICENSE` file for details.

## Author

**Louis Brusset**
- Email: louis.brusset@etu.minesparis.psl.eu
- Institution: École Nationale Supérieure des Mines de Paris
- Organization: ITER Organization (IO)

## Acknowledgments

- ITER Organization for providing the internship opportunity
- MAST team for data access and support
- École des Mines de Paris for academic supervision and providing knowledge

## References

- Project repository: https://github.com/LouisBrusset/ITER-autoencoder-for-labelling
- [ITER Organization](https://www.iter.org/)
- [MAST Experiment Documentation](https://mastapp.site/)
- [MAST VAE code experimentation by Samuel Jackson](https://github.com/samueljackson92/mast-signal-validation)

