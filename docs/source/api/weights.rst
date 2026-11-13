Weights
=======

The weights module provides participant survey weights for the
:class:`~cntmosaic.models.SocialMix` estimator. Weighting is opt-in: build a
:class:`~cntmosaic.weights.WeightSpecs` from weight factors and pass it as
``SocialMix(..., weight_specs=...)``. The estimator computes the weights on its
own participants and cells (age group x participant strata), and rebuilds them
on every bootstrap resample.

Which factor should I use?
--------------------------

Ask what you know:

.. list-table::
   :header-rows: 1
   :widths: 45 20 35

   * - You know...
     - Use
     - Total weight carried by a category
   * - a weight for **each participant**, supplied with the survey
       (inverse inclusion probabilities, non-response adjustments)
     - :class:`~cntmosaic.weights.DesignWeights`
     - sum of the supplied weights
   * - a **fixed weight per category**, set by the design rather than by how
       many participants were sampled in it (e.g. the inverse sampling
       fraction of each region)
     - :class:`~cntmosaic.weights.CategoryWeights`
     - factor x number sampled: it grows with the sample count
   * - the **share** each category should have (a population distribution,
       the 5 : 2 weekday/weekend split, chosen country shares)
     - :class:`~cntmosaic.weights.PostStratWeights`
     - the target share, whatever the sample count

**Quick test:** if twice as many participants had been sampled in category X,
should X's total influence double? Yes: ``CategoryWeights``. No, it should
stay at its target share: ``PostStratWeights``.
:class:`~cntmosaic.weights.DayOfWeekWeights` and
:class:`~cntmosaic.weights.AgePostStratWeights` are ready-made
post-stratifications.

Where are the counts taken? (``within``)
----------------------------------------

The post-stratification factors depend on how many participants fall in each
category. ``within=`` sets where these counts are taken: over the whole sample
(``"sample"``) or within each cell, i.e. age group x participant strata
(``"cell"``). With ``"cell"`` the **same** targets are applied in every cell,
which is appropriate only when they hold within each age group (e.g. the 5:2
ratio of weekdays to weekend days), not for a composition that varies with age
(e.g. a national sex ratio). A cell that lacks a category cannot meet its
target and gives a :class:`~cntmosaic.weights.WeightWarning`.

.. list-table::
   :header-rows: 1
   :widths: 40 25 35

   * - Factor
     - Uses sample counts
     - Default ``within``
   * - :class:`~cntmosaic.weights.DesignWeights`,
       :class:`~cntmosaic.weights.CategoryWeights`
     - no
     - not applicable
   * - :class:`~cntmosaic.weights.PostStratWeights`
     - yes
     - ``"sample"``
   * - :class:`~cntmosaic.weights.DayOfWeekWeights`
     - yes
     - ``"cell"``
   * - :class:`~cntmosaic.weights.AgePostStratWeights`
     - yes
     - ``"sample"`` (the same weights as ``"cell"`` without participant strata)

Shares are taken among participants with a known value, so a missing value
(``on_missing="average"``) receives the average factor of 1.

:class:`~cntmosaic.weights.DayOfWeekWeights` reads the column ``dayofweek``
(kept with ``ParticipantData(aux_cols=...)``) and matches its values exactly
against POLYMOD's integer codes (0 = Sunday) unless ``weekday=`` and
``weekend=`` are given. Its docstring lists the assumptions about the data in
full: strings, case, missing-value placeholders and categoricals.

Several post-stratification factors are **not calibrated jointly**: each
matches its own margin on its own, and multiplying them moves each margin away
from its target. Matching several margins at once (raking) is not available.

Example
-------

.. code-block:: python

   from cntmosaic.dataloader import ParticipantData
   from cntmosaic.models import SocialMix
   from cntmosaic.weights import WeightSpecs, DayOfWeekWeights, AgePostStratWeights

   part = ParticipantData(df_part, id_col="part_id", age_col="part_age",
                          aux_cols="dayofweek")          # keep the weighting variable
   specs = WeightSpecs(DayOfWeekWeights(),             # 5:2 within every age group
                       AgePostStratWeights(pop_data), threshold=3)

   specs.preview(part, age_group_specs)                  # inspect before fitting
   sm = SocialMix(part, cnt_data, age_group_specs, pop_data, weight_specs=specs)
   sm.cint()                    # weighted
   sm.cint(weighted=False)      # unweighted, from the same fit
   sm.weights.cells             # effective sample size per cell
   sm.run_inference_bootstrap(n_boot=1000, random_state=1)   # weights rebuilt per resample

Specification
-------------

WeightSpecs
^^^^^^^^^^^

.. autoclass:: cntmosaic.weights.WeightSpecs
    :members:
    :show-inheritance:
    :special-members: __init__

Weight factors
--------------

DesignWeights
^^^^^^^^^^^^^

.. autoclass:: cntmosaic.weights.DesignWeights
    :members:
    :show-inheritance:

CategoryWeights
^^^^^^^^^^^^^^^

.. autoclass:: cntmosaic.weights.CategoryWeights
    :members:
    :show-inheritance:

PostStratWeights
^^^^^^^^^^^^^^^^

.. autoclass:: cntmosaic.weights.PostStratWeights
    :members:
    :show-inheritance:

DayOfWeekWeights
^^^^^^^^^^^^^^^^

.. autoclass:: cntmosaic.weights.DayOfWeekWeights
    :members:
    :show-inheritance:

AgePostStratWeights
^^^^^^^^^^^^^^^^^^^

.. autoclass:: cntmosaic.weights.AgePostStratWeights
    :members:
    :show-inheritance:

Diagnostics and warnings
------------------------

WeightResult
^^^^^^^^^^^^

.. autoclass:: cntmosaic.weights.WeightResult
    :members:

WeightWarning
^^^^^^^^^^^^^

.. autoclass:: cntmosaic.weights.WeightWarning
    :show-inheritance:

Writing new weight factors
--------------------------

A new factor subclasses :class:`~cntmosaic.weights.WeightFactor` (as a frozen
dataclass with a ``name`` field) and implements ``required_columns()`` and
``bind(ctx)``. ``bind`` checks the participant table once and returns a
picklable callable that evaluates the factor on row indices (which may repeat,
e.g. in a bootstrap resample).

.. autoclass:: cntmosaic.weights.WeightFactor
    :members:

.. autoclass:: cntmosaic.weights.BindContext
    :members:

.. autoclass:: cntmosaic.weights.BoundFactor
    :members: __call__
