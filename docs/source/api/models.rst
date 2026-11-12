Models
======

The models module contains classes for estimating contact matrices from survey data
using Bayesian and classical approaches.

Abstract Base Class
-------------------

ContactModel
^^^^^^^^^^^^

.. autoclass:: cntmosaic.models.ContactModel
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__

GenMix Model Family
-------------------

These models estimate contact matrices stratified by one or more covariates
(e.g. age, sex) using a generalised mixing framework.

GenMix
^^^^^^

.. autoclass:: cntmosaic.models.GenMix
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

AgeMixCC
^^^^^^^^

.. autoclass:: cntmosaic.models.AgeMixCC
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

AgeMixFF
^^^^^^^^

.. autoclass:: cntmosaic.models.AgeMixFF
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

AgeMixFC
^^^^^^^^

.. autoclass:: cntmosaic.models.AgeMixFC
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

GenMixCC
^^^^^^^^

.. autoclass:: cntmosaic.models.GenMixCC
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

GenMixFF
^^^^^^^^

.. autoclass:: cntmosaic.models.GenMixFF
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

GenMixFC
^^^^^^^^

.. autoclass:: cntmosaic.models.GenMixFC
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

Classical Models
----------------

SocialMix
^^^^^^^^^

.. autoclass:: cntmosaic.models.SocialMix
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

PropMixFactorisation
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: cntmosaic.models.PropMixFactorisation
    :members:

SocialMixBootstrap
^^^^^^^^^^^^^^^^^^

.. autoclass:: cntmosaic.models.SocialMixBootstrap
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

Prem
^^^^

.. autoclass:: cntmosaic.models.Prem
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

vdKassteele
^^^^^^^^^^^

.. autoclass:: cntmosaic.models.vdKassteele
    :members:
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :special-members: __init__

Utility Functions
-----------------

.. autofunction:: cntmosaic.models.to_inference_data
