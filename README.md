[![CI](https://github.com/ShozenD/cntmosaic/workflows/CI/badge.svg)](https://github.com/ShozenD/cntmosaic/actions)
[![codecov](https://codecov.io/gh/ShozenD/cntmosaic/graph/badge.svg?token=9U271V3D3H)](https://codecov.io/gh/ShozenD/cntmosaic)
[![Documentation](https://readthedocs.org/projects/cntmosaic-docs/badge/?version=latest)](https://cntmosaic-docs.readthedocs.io/en/latest/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

# Contact Mosaic

## What is Contact Mosaic?

Contact Mosaic (`cntmosaic`) is a Python package for analysing social contact patterns from
social contact data. It provides a set of tools to process, analyse, simulate, and visualise
social contact data. It also provides a set of models to infer social contact matrices from
real world social contact data. The models in `cntmosaic` are implemented using the
probabilistic programming language [Numpyro](https://num.pyro.ai/en/stable/index.html),
which enables both Hamiltonian Monte Carlo (HMC) based full Bayesian inference and fast
stochastic variational inference (SVI).

> **Note:** This is a preliminary v0.5 release intended for collaborators and early users.
> The API may change before a stable 1.0 release.

---

## Installation

`cntmosaic` v0.5 is installed directly from source.

#### Step 1: Clone the repository

```bash
git clone https://github.com/ShozenD/cntmosaic.git
cd cntmosaic
```

#### Step 2: Create a virtual environment and activate it

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

#### Step 3: Install the package

Because `cntmosaic` is installed from source, editable mode is recommended — changes from
`git pull` take effect immediately without reinstalling:

```bash
pip install -e .
```

Or to install without editable mode:
```bash
pip install .
```

#### Step 4: Verify installation

```bash
python -c "import cntmosaic; print(cntmosaic.__version__)"
```

---

## Tutorials

The following notebooks demonstrate key workflows in `cntmosaic`.
All notebooks are located in the [`tutorials/`](tutorials/) directory.

| Notebook | Description |
|---|---|
| [00 — Quickstart](tutorials/00_Quickstart.ipynb) | Walk through a complete workflow on the bundled POLYMOD Germany dataset: load survey data, fit a Bayesian contact model with SVI, and visualise the posterior contact intensity matrix. |
| [01 — Generalized Contact Matrices](tutorials/01_Tutorial_GenMix.ipynb) | Fit generalized contact matrices stratified by age and gender using POLYMOD data, covering both complete and partial observation scenarios. |
| [02 — SocialMix](tutorials/02_Tutorial_SocialMix.ipynb) | Estimate age-stratified social contact matrices from survey data using the SocialMix model with bootstrap inference. |
| [03 — Context-Specific Contact Matrices](tutorials/03_Tutorial_Prem_Context.ipynb) | Estimate household, school, work, and other contact matrices for the United States from the Epistorm-Mix survey using the `Prem2` model. |
| [04 — Feature Selection](tutorials/04_Tutorial_Feature_Selection.ipynb) | Select the most parsimonious set of demographic stratification variables for contact models using LOO-CV ELPD. |

---

## Citation

If you use `cntmosaic` in your research, please cite the following paper:

**Plain text:**

> Dan, S., van Dyk, D. A., Ling, Z., Mishra, S., & Ratmann, O. (2026).
> Bayesian Modeling and Prediction of Generalized Contact Matrices.
> *arXiv preprint* arXiv:2605.06742.
> https://doi.org/10.48550/arXiv.2605.06742

**BibTeX:**

```bibtex
@misc{dan2026bayesian,
  title         = {Bayesian Modeling and Prediction of Generalized Contact Matrices},
  author        = {Shozen Dan and David A. van Dyk and Zhi Ling and Swapnil Mishra and Oliver Ratmann},
  year          = {2026},
  eprint        = {2605.06742},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ME},
  doi           = {10.48550/arXiv.2605.06742},
  url           = {https://arxiv.org/abs/2605.06742}
}
```

---

## Contributing

Bug reports and feature requests are welcome. Please see [CONTRIBUTING.md](CONTRIBUTING.md)
for guidance, or open an issue directly:

- [Report a bug](https://github.com/ShozenD/cntmosaic/issues/new?template=bug_report.yml)
- [Request a feature](https://github.com/ShozenD/cntmosaic/issues/new?template=feature_request.yml)

---

## License

Copyright © 2026, Imperial College London. All rights reserved.

This project is licensed under the BSD 3-Clause License — see [LICENSE](LICENSE) for details.

---

## Testing

To run all unit tests from the root directory:
```bash
pytest
```

To run tests with a coverage report:
```bash
pytest --cov=cntmosaic --cov-report=html
```

To run tests in a specific module:
```bash
pytest cntmosaic/datasets/tests/
pytest cntmosaic/models/tests/
pytest cntmosaic/preprocess/tests/
pytest cntmosaic/sim/tests/
pytest cntmosaic/utils/tests/
```

---

## Imperial College HPC Setup

Special instructions for users of the [Imperial College HPC](https://icl-rcs-user-guide.readthedocs.io/en/latest/).

### Option A: Using Easybuild Python (Recommended for Batch Jobs)

#### Step 1: Connect and navigate

```bash
ssh username@login.hpc.ic.ac.uk
cd /path/to/your/workspace
git clone https://github.com/ShozenD/cntmosaic.git
cd cntmosaic
```

#### Step 2: Load Python module

```bash
module load tools/prod
module load Python/3.10.8-GCCcore-12.2.0
```

#### Step 3: Create virtual environment

```bash
virtualenv .venv
source .venv/bin/activate
```

#### Step 4: Install dependencies

```bash
pip install -r requirements_hpc.txt
pip install -e .
```

#### Step 5: Enable GPU (optional)

For GPU support on HPC, explicitly install JAX with CUDA:
```bash
pip install --upgrade "jax[cuda12]"
```

### Option B: Using Conda with JupyterHub (For Interactive Work)

#### Step 1: Initialise Conda

```bash
eval "$(~/miniforge3/bin/conda shell.bash hook)"
```

If you don't have Miniforge installed:
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

#### Step 2: Create Conda environment

```bash
conda create -n cntmosaic python=3.12 ipykernel jupyter_client
conda activate cntmosaic
```

#### Step 3: Install package

```bash
cd /path/to/cntmosaic
pip install -e .
```

#### Step 4: Register Jupyter kernel

```bash
python -m ipykernel install --user --name cntmosaic --display-name "Python 3.12 (cntmosaic)"
```

#### Step 5: Launch JupyterHub

1. Navigate to [jupyter.rcs.imperial.ac.uk](https://jupyter.rcs.imperial.ac.uk/)
2. Log in with your Imperial credentials
3. Start a new server
4. Select the **"Python 3.12 (cntmosaic)"** kernel from the launcher
