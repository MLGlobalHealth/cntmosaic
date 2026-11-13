Quickstart
==========

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/MLGlobalHealth/cntmosaic/blob/main/docs/source/tutorials/00_Quickstart.ipynb
   :alt: Open In Colab

This guide walks through a complete contact matrix estimation workflow using the
bundled POLYMOD Germany dataset. You will load survey data, fit a Bayesian social contact
model with stochastic variational inference (SVI), and visualise the posterior contact
intensity matrix.

Installation
------------

**From PyPI:**

.. code-block:: bash

   pip install cntmosaic

**From conda-forge (recommended):**

.. code-block:: bash

   conda install -c conda-forge cntmosaic

**Using conda to create an isolated environment:**

.. code-block:: bash

   conda create -n cntmosaic -c conda-forge python=3.12 cntmosaic
   conda activate cntmosaic

Loading Contact Data
--------------------

``cntmosaic`` ships with a cleaned version of the German branch of the POLYMOD contact survey.
:func:`load_polymod_germany <cntmosaic.datasets.load_polymod_germany>` returns a
:class:`SurveyData <cntmosaic.datasets.SurveyData>` typed dict containing three
pandas DataFrames:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Contents
   * - ``"participants"``
     - One row per survey participant. Key columns: ``part_id``, ``part_age``, ``part_sex``.
   * - ``"contacts"``
     - One row per reported contact. Key columns: ``part_id``, ``cnt_age``, ``cnt_sex``.
   * - ``"population"``
     - Age–sex population counts. Key columns: ``age``, ``sex``, ``P``.

.. code-block:: python

   from cntmosaic.datasets import load_polymod_germany

   survey = load_polymod_germany()
   print(survey["participants"].head())
   print(survey["contacts"].head())
   print(survey["population"].head())

Preparing the Data
------------------

Raw DataFrames must be wrapped in validated container classes before being passed
to a model.  The containers check column types, handle missing values, and
standardise column names for downstream use.

Because we are fitting an age-only model (:class:`AgeMixFF <cntmosaic.models.AgeMixFF>`), the population counts need to be summed over gender first.

.. code-block:: python

   from cntmosaic.dataloader import (
       ContactSurveyLoader,
       ParticipantData,
       ContactData,
       PopulationData,
   )

   part_data = ParticipantData(
       data=survey["participants"],
       id_col="part_id",
       age_col="part_age",
   )

   cnt_data = ContactData(
       data=survey["contacts"],
       id_col="part_id",
       age_col="cnt_age",
   )

   # Aggregate over gender — AgeMixFF models age mixing only
   df_pop = (
       survey["population"]
       .groupby("age", observed=True)["P"]
       .sum()
       .reset_index()
   )
   pop_data = PopulationData(
       data=df_pop,
       age_col="age",
       size_col="P",
   )

   dataloader = ContactSurveyLoader.from_containers(part_data, cnt_data, pop_data)

Fitting a Model
---------------

:class:`AgeMixFF <cntmosaic.models.AgeMixFF>` is a semi-parametric Bayesian social contact
model that infers a 1-year age resolution contact matrix (the FF suffix stands
for *fine-fine*).  The model requires 1-year age information for both participants, contacts, and population counts.
For datasets with only coarse age information (e.g. 5-year age groups), the :class:`AgeMixCC <cntmosaic.models.AgeMixCC>` is appropriate.

When instantiating the model, it requires a ``priors`` dictionary with a ``"rate"`` key specifying a 2-D spatial prior for the log contact rate surface.

Here we use a penalized cubic B-spline prior (:class:`PSpline2D <cntmosaic.models.numpyro.priors.PSpline2D>`)
with 20 interior knots. 

.. code-block:: python

   from cntmosaic.models import AgeMixFF
   from cntmosaic.models.numpyro.priors import PSpline2D

   priors = {"rate": PSpline2D(prior_type="global", M=20, degree=3)}
   model = AgeMixFF(dataloader, priors=priors, likelihood="negbin")

**Running inference with SVI** (fast, recommended for exploration):

.. code-block:: python

   from jax.random import PRNGKey
   from numpyro.infer.autoguide import AutoNormal

   guide = AutoNormal(model.model)
   model.run_inference_svi(PRNGKey(0), guide=guide, num_steps=10_000)

.. note::

   Full Bayesian inference via MCMC is also available:

   .. code-block:: python

      model.run_inference_mcmc(
          PRNGKey(0),
          num_samples=1000,
          num_warmup=1000,
          num_chains=4,
      )

   MCMC produces exact posterior samples but is considerably slower than SVI.
   For a first exploration, SVI is a practical starting point.

Summarising the Posterior
--------------------------

:class:`ModelSummariser <cntmosaic.analysis.ModelSummariser>` computes credible
intervals for the estimated contact intensity matrix directly from the fitted model.

.. code-block:: python

   from cntmosaic.analysis import ModelSummariser

   summariser = ModelSummariser(model)
   summary = summariser.summarise_cint(alpha=0.05, measure="median")

``summarise_cint`` returns a dictionary keyed by stratum label.  For an unstratified
model like ``AgeMixFF`` the only key is ``"All->All"``.  Each value is a
:class:`ContactSummary <cntmosaic.analysis.ContactSummary>` with three attributes:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Attribute
     - Description
   * - ``.central``
     - Posterior median (or mean) contact intensity matrix, shape ``(A, A)``
   * - ``.lower``
     - Lower bound of the ``1 - alpha`` credible interval
   * - ``.upper``
     - Upper bound of the ``1 - alpha`` credible interval

.. code-block:: python

   cint = summary["All->All"]
   print(cint.central.shape)   # (A, A) — one cell per age-group pair
   print(cint.lower.min(), cint.upper.max())

Plotting the Contact Matrix
---------------------------

:func:`plot_mosaic <cntmosaic.vis.plot_mosaic>` visualises a contact intensity matrix
as an Altair heatmap.  Pass the ``numpy`` array from ``.central`` directly:

.. code-block:: python

   from cntmosaic.vis import plot_mosaic

   chart = plot_mosaic(
       summary["All->All"].central,
       title="Posterior contact intensity — POLYMOD Germany",
       xlabel="Age of participant",
       ylabel="Age of contact",
   )
   chart   # renders inline in a Jupyter notebook

The returned object is an ``altair.Chart``, which can be saved to HTML:

.. code-block:: python

   chart.save("contact_matrix.html")

.. note::

   :func:`plot_mosaic_pixilated <cntmosaic.vis.plot_mosaic_pixilated>` is designed for
   coarse-age models (e.g. :class:`AgeMixCC <cntmosaic.models.AgeMixCC>`,
   :class:`GenMixCC <cntmosaic.models.GenMixCC>`) where age-group bin boundaries are
   meaningful.  It requires :class:`AgeGroupSpecs <cntmosaic.utils.AgeGroupSpecs>` to
   be passed when creating the model.  For fine-age models such as ``AgeMixFF``,
   :func:`plot_mosaic <cntmosaic.vis.plot_mosaic>` is the right choice.

Next Steps
----------

* **Full tutorial** — see :doc:`../tutorials/01_Tutorial_GenMix` for a complete notebook
  demonstrating gender-stratified models, MCMC inference, and multi-panel
  visualisations with :class:`GenMixCC <cntmosaic.models.GenMixCC>`.
* **API reference** — :doc:`../api/models`, :doc:`../api/dataloader`,
  :doc:`../api/analysis`, :doc:`../api/visualization`.
* **Priors** — explore alternative 2-D spatial priors in :doc:`../api/priors`
  (``PSpline2D``, ``HSGP2D``, ``IGMRF2D``, ``vdKassteele2D``).
