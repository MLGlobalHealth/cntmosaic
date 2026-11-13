"""
Participant survey weights.

Weighting is opt-in: build a :class:`WeightSpecs` from weight factors and pass
it to an estimator (``SocialMix(..., weight_specs=...)``). The estimator
computes the weights on its own participants and cells, and on every
bootstrap resample. See ``docs/dev_docs/dev_plan_weights.md`` for the design.

Which factor should I use? Ask what you know:

- a weight for **each participant**, supplied with the survey →
  :class:`DesignWeights`;
- a **fixed weight per category**, set by the design → :class:`CategoryWeights`
  (a category's total weight grows with its sample count);
- the **share** each category should have → :class:`PostStratWeights`
  (a category's total weight is its target share, whatever its sample count);
  :class:`DayOfWeekWeights` and :class:`AgePostStratWeights` are ready-made
  post-stratifications.

Quick test: if twice as many participants had been sampled in category X,
should X's total influence double? Yes → ``CategoryWeights``; no →
``PostStratWeights``. Several post-stratification factors are not calibrated
jointly (that would be raking).

Public API
----------
WeightSpecs
    The specification: an immutable set of weight factors (and an optional
    truncation threshold), with ``preview()``.
DesignWeights, CategoryWeights, PostStratWeights, DayOfWeekWeights,
AgePostStratWeights
    Weight factors.
WeightResult
    Diagnostics: each factor's contribution, the final weights, per-cell
    effective sample sizes.
WeightWarning
    Warning category for weighting.
WeightFactor, BindContext, BoundFactor
    The extension hook for writing new weight factors.
"""

from ._factor import BindContext, BoundFactor, WeightFactor
from ._factors import CategoryWeights, DesignWeights
from ._poststrat import AgePostStratWeights, DayOfWeekWeights, PostStratWeights
from ._result import WeightResult
from ._specs import WeightSpecs
from ._warnings import WeightWarning

__all__ = [
    "WeightSpecs",
    "DesignWeights",
    "CategoryWeights",
    "PostStratWeights",
    "DayOfWeekWeights",
    "AgePostStratWeights",
    "WeightResult",
    "WeightWarning",
    "WeightFactor",
    "BindContext",
    "BoundFactor",
]
