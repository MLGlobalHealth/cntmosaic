Prediction
==========

The predict module provides tools for generating predictions from fitted contact matrix models.

Predictor
---------

.. autoclass:: cntmosaic.predict.Predictor
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__

Utility Functions
-----------------

.. autofunction:: cntmosaic.predict.z_marginals

.. autofunction:: cntmosaic.predict.sample_full_counts

.. autofunction:: cntmosaic.predict.counts_to_intensities

.. autofunction:: cntmosaic.predict.spectral_radius

Transportation Polytope Sampler
-------------------------------

.. autofunction:: cntmosaic.predict.sample_swap

.. autofunction:: cntmosaic.predict.sample_swap_path

.. autofunction:: cntmosaic.predict.sample_hit_run

.. autofunction:: cntmosaic.predict.check_margins

.. autofunction:: cntmosaic.predict.diagonal_is_feasible

Deprecated
----------

.. autofunction:: cntmosaic.predict.feasible_mixing_bounds

.. autofunction:: cntmosaic.predict.sample_eta

.. autofunction:: cntmosaic.predict.rtruncated_beta

.. autofunction:: cntmosaic.predict.rtruncated_dirichlet
