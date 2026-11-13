"""Weight diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ._specs import WeightSpecs


@dataclass(frozen=True, eq=False)
class WeightResult:
    """The weights a fit uses, with each factor's contribution.

    Attributes
    ----------
    participants : pd.DataFrame
        One row per participant: ``id``; ``cell`` (the cell label); the
        participant columns the factors are computed from, under their
        ``part_*`` names (e.g. ``part_dayofweek``, ``part_age``,
        ``part_gender``), each once and in factor order; one column per
        factor, named after the factor; ``composite`` (the product
        of the factors, :math:`\\tilde w_i`); ``weight`` (the final weight
        :math:`w_i`: normalised so the weights in each cell sum to the cell's
        size, then truncated if a threshold is set); ``capped`` (whether the
        weight was capped by the threshold).
    cells : pd.DataFrame
        One row per cell: ``cell``, ``n``, ``min_w``, ``max_w``,
        ``n_capped``, ``n_eff`` (Kish effective sample size
        ``(sum w)^2 / sum w^2``) and ``deff`` (design effect ``n / n_eff``).
        Empty cells (possible only in ``WeightSpecs.preview()``) have ``n = 0``
        and NaN weight statistics.
    specs : WeightSpecs
        The specification that produced these weights.
    """

    participants: pd.DataFrame
    cells: pd.DataFrame
    specs: "WeightSpecs"

    def __repr__(self) -> str:
        cells = self.cells
        nonempty = cells[cells["n"] > 0]
        min_neff = nonempty["n_eff"].min() if len(nonempty) else np.nan
        threshold = "" if self.specs.threshold is None else f", threshold={self.specs.threshold:g}"
        return (
            f"WeightResult(factors={list(self.specs.names)}{threshold}: "
            f"{len(self.participants)} respondents in "
            f"{len(cells)} cells ({int((cells['n'] == 0).sum())} empty); "
            f"smallest n_eff {min_neff:.3g}; largest weight "
            f"{self.participants['weight'].max():.3g}; "
            f"{int(self.participants['capped'].sum())} capped)"
        )
