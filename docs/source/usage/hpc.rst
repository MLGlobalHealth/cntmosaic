Imperial College HPC Setup
==========================

This page provides setup instructions for users of the
`Imperial College Research Computing Service <https://icl-rcs-user-guide.readthedocs.io/en/latest/>`_
(RCS) High-Performance Computing (HPC) cluster.

There are two ways to set up the environment depending on your needs:

- **Option A** — use the HPC's built-in Python module. This is the simplest
  path and is recommended for submitting batch jobs.
- **Option B** — create a Conda environment. This is required if you want to
  work interactively through
  `JupyterHub <https://jupyter.rcs.imperial.ac.uk/>`_.

----

Option A — Easybuild Python (batch jobs)
-----------------------------------------

**Step 1: Connect to the HPC**

.. code-block:: bash

   ssh username@login.hpc.ic.ac.uk

Replace ``username`` with your Imperial College username.

**Step 2: Load Python**

The HPC uses a system called *modules* to manage software versions. The
commands below load the tools you need. You must run these commands each time
you log in to a new session.

.. code-block:: bash

   module load tools/prod
   module load Python/3.10.8-GCCcore-12.2.0

If this exact Python version is unavailable, run ``module avail Python`` to
list all available versions and load the newest one.

**Step 3: Create a virtual environment**

A virtual environment is a self-contained folder that keeps your project's
packages separate from the system Python.

.. code-block:: bash

   virtualenv .venv
   source .venv/bin/activate

You must run ``source .venv/bin/activate`` each time you log in and want to
use ``cntmosaic``.

**Step 4: Install ``cntmosaic``**

.. code-block:: bash

   pip install cntmosaic

**Step 5: Verify the installation**

.. code-block:: bash

   python -c "import cntmosaic; print(cntmosaic.__version__)"

If you see a version number, the installation was successful.

**Step 6 (optional): Enable GPU acceleration**

If your job will run on a GPU node, install JAX with CUDA support:

.. code-block:: bash

   pip install --upgrade "jax[cuda12]"

----

Option B — Conda + JupyterHub (interactive work)
-------------------------------------------------

**Step 1: Connect to the HPC and initialise Conda**

.. code-block:: bash

   ssh username@login.hpc.ic.ac.uk
   eval "$(~/miniforge3/bin/conda shell.bash hook)"

This command activates the Conda package manager. If you see an error,
Miniforge may be installed in a different location — run ``which conda``
to find it, or follow the
`RCS Conda setup guide <https://icl-rcs-user-guide.readthedocs.io/en/latest/>`_.

If Miniforge is not yet installed, contact the RCS support team.

**Step 2: Create a Conda environment**

.. code-block:: bash

   conda create -n cntmosaic python=3.12 ipykernel jupyter_client
   conda activate cntmosaic

**Step 3: Install ``cntmosaic``**

.. code-block:: bash

   pip install cntmosaic

**Step 4: Register a Jupyter kernel**

This step makes your new environment available as an option inside JupyterHub:

.. code-block:: bash

   python -m ipykernel install --user --name cntmosaic \
       --display-name "Python 3.12 (cntmosaic)"

**Step 5: Verify the installation**

.. code-block:: bash

   python -c "import cntmosaic; print(cntmosaic.__version__)"

**Step 6: Launch JupyterHub**

1. Navigate to `jupyter.rcs.imperial.ac.uk <https://jupyter.rcs.imperial.ac.uk/>`_.
2. Log in with your Imperial College credentials.
3. Start a new server and select the **"Python 3.12 (cntmosaic)"** kernel from
   the launcher.

----

Getting Help
------------

For HPC-specific issues (job scheduling, storage, module availability), contact
the `RCS support team <https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/>`_.

For issues with ``cntmosaic`` itself, open an issue on
`GitHub <https://github.com/MLGlobalHealth/cntmosaic/issues>`_.
