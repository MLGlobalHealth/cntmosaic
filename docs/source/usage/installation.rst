Installation
============

This page explains how to install ``cntmosaic`` on your computer.
If you are unsure which method to use, start with **Method 1** — it works on
any operating system and requires nothing beyond a working Python installation.

Prerequisites
-------------

- **Python 3.10 or later.** If you do not have Python, download it from
  `python.org <https://www.python.org/downloads/>`_.
- A **terminal** (macOS/Linux) or **Command Prompt / PowerShell** (Windows).

To check whether Python is already installed, open a terminal and run:

.. code-block:: bash

   python --version

You should see a version number such as ``Python 3.12.0``. If Python is not
found, follow the link above to install it first.

----

Method 1 — pip (recommended for most users)
---------------------------------------------

pip is a tool that installs Python packages and comes bundled with every
Python installation.

**Quick install:**

.. code-block:: bash

   pip install cntmosaic

**Recommended — install inside an isolated environment** to avoid conflicts
with other Python packages you may have installed:

.. code-block:: bash

   python -m venv cntmosaic-env
   source cntmosaic-env/bin/activate
   pip install cntmosaic

This creates a self-contained folder (``cntmosaic-env``) that holds only the
packages needed for this project. Once activated, you will see
``(cntmosaic-env)`` at the start of your terminal prompt.

.. note::

   On **Windows**, use this activation command instead:

   .. code-block:: bat

      cntmosaic-env\Scripts\activate

.. important::

   You need to activate the environment each time you open a new terminal
   window before using ``cntmosaic``.

----

Method 2 — conda-forge
-----------------------

`Conda <https://docs.conda.io>`_ is an alternative package manager that is
popular in the scientific community (used by Anaconda and Miniforge). If you
already use Conda, this method creates a self-contained environment in a
single command:

.. code-block:: bash

   conda create -n cntmosaic -c conda-forge python=3.12 cntmosaic
   conda activate cntmosaic

----

Method 3 — From source (advanced)
-----------------------------------

Use this method if you want the very latest unreleased changes, or if you
plan to contribute to ``cntmosaic``.

.. code-block:: bash

   git clone https://github.com/ShozenD/cntmosaic.git
   cd cntmosaic
   pip install -e ".[dev]"

The ``-e`` flag installs the package in *editable mode*, meaning any changes
you make to the source files take effect immediately without reinstalling.
The ``[dev]`` extra installs additional tools for testing and code quality
(pytest, black, mypy, etc.).

----

Verifying the Installation
---------------------------

After installation, open a Python session and run:

.. code-block:: python

   import cntmosaic
   print(cntmosaic.__version__)

If you see a version number (e.g. ``0.5.1``), the installation was successful.

----

Core Dependencies
------------------

``cntmosaic`` installs the following supporting libraries automatically — you
do not need to install them separately.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Package
     - Purpose
   * - JAX / JAXlib
     - Fast numerical computing
   * - NumPyro
     - Bayesian inference (MCMC and variational inference)
   * - NumPy
     - Array operations
   * - SciPy
     - Scientific computing utilities
   * - Pandas
     - Data manipulation and tabular data
   * - xarray
     - Labelled multi-dimensional arrays
   * - ArviZ
     - Bayesian model diagnostics and visualisation
   * - Altair
     - Interactive statistical charts
   * - Optax
     - Gradient-based optimisation (used internally during model fitting)
   * - scikit-learn
     - General machine-learning utilities
   * - tqdm
     - Progress bars during model fitting

----

Troubleshooting
----------------

**"ModuleNotFoundError: No module named 'jax'"**

The JAX library was not installed. Run:

.. code-block:: bash

   pip install jax jaxlib

If this fails with a CUDA-related error, your system may not have the
required GPU drivers. You can install the CPU-only version instead:

.. code-block:: bash

   pip install "jax[cpu]"

**Out-of-memory errors during model fitting**

Large MCMC runs can require significant RAM. Try switching to stochastic
variational inference (SVI), which is much more memory-efficient. If you
must use MCMC, you can reduce JAX's memory pre-allocation:

.. code-block:: python

   import os
   os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
   import jax   # import jax *after* setting the environment variable

**Getting help**

If you encounter an error not listed here, please open an issue on
`GitHub <https://github.com/ShozenD/cntmosaic/issues>`_ and include the
output of the following snippet so we can reproduce your environment:

.. code-block:: python

   import sys, platform, jax, numpyro
   print(f"Python:   {sys.version}")
   print(f"Platform: {platform.platform()}")
   print(f"JAX:      {jax.__version__}")
   print(f"NumPyro:  {numpyro.__version__}")
