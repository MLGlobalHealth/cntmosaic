"""
The weight-factor contract.

A **weight factor** is one weighting step: it gives every participant a factor
:math:`f_i`, and the factors of all steps are multiplied into the composite
weight. A factor object holds configuration only, never data. Binding pairs it
with one participant table (see :class:`BindContext`) and returns a
:class:`BoundFactor` that evaluates the factor on any set of rows, e.g. the
full sample or a bootstrap resample.

This module is the extension hook for new factors: subclass
:class:`WeightFactor`, implement :meth:`~WeightFactor.required_columns` and
:meth:`~WeightFactor.bind`, and return a picklable bound factor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
import pandas as pd


@dataclass(frozen=True, eq=False)
class BindContext:
    """Everything a factor may use when binding.

    Part of the public extension hook: fields may be added in later versions,
    but existing fields keep their meaning.

    Attributes
    ----------
    part : pd.DataFrame
        Participants with a valid cell, with canonical ``part_*`` columns
        (``id``, ``part_age`` if given, ``part_age_grp``, strata, auxiliary
        columns). A private copy: factors must not modify it.
    cell : np.ndarray
        Integer cell code per participant, ``0 .. n_cells - 1``, aligned with
        ``part``. A cell is an age group crossed with the participant strata.
    n_cells : int
        Number of cells.
    cell_labels : tuple of str
        Label of each cell code.
    ids : np.ndarray
        Participant ids, aligned with ``part`` and ``cell``.
    """

    part: pd.DataFrame
    cell: np.ndarray
    n_cells: int
    cell_labels: Tuple[str, ...]
    ids: np.ndarray

    @property
    def n(self) -> int:
        """Number of participants."""
        return len(self.ids)


@runtime_checkable
class BoundFactor(Protocol):
    """A factor bound to one participant table.

    Calling it with integer row indices (into the bound table; indices may
    repeat, e.g. in a bootstrap resample) returns the factor :math:`f_i` for
    each entry of ``rows``: a float64 array, finite and strictly positive.
    Counts used by sample-dependent factors are taken over ``rows``, with
    multiplicity.

    Bound factors must be picklable: implement them as small classes (or
    ``functools.partial`` of module-level functions), not as closures.
    """

    def __call__(self, rows: np.ndarray) -> np.ndarray: ...


class WeightFactor(ABC):
    """One weighting step. Configuration only; holds no data.

    Subclasses are frozen dataclasses with a ``name: str`` field (the label of
    the factor in diagnostics; unique within a :class:`WeightSpecs`) and set
    the class variable :attr:`sample_dependent`.
    """

    #: True if the factor uses counts over the sample (and therefore changes
    #: between bootstrap resamples).
    sample_dependent: ClassVar[bool] = False

    name: str

    @abstractmethod
    def required_columns(self) -> Tuple[str, ...]:
        """Participant columns the factor reads (with or without ``part_``)."""

    def source_columns(self) -> Tuple[str, ...]:
        """Participant columns the factor is computed from (with or without
        ``part_``), shown next to the factor in ``WeightResult.participants``.

        Defaults to :meth:`required_columns`. Override it when the factor
        reads a column that it does not declare there.
        """
        return self.required_columns()

    @abstractmethod
    def bind(self, ctx: BindContext) -> BoundFactor:
        """Extension hook: check against ``ctx.part``, encode it, and return a
        picklable callable that evaluates the factor on row indices.

        All validation happens here, once, on the full sample. Raise
        ``ValueError`` (or ``KeyError`` for missing columns) with a clear
        message on invalid data.
        """


def resolve_column(columns: Sequence[str], col: str) -> Optional[str]:
    """Resolve a column name given with or without the ``part_`` prefix.

    Returns the matching column name in *columns*, or ``None`` if neither
    ``col`` nor ``part_{col}`` is present.
    """
    if col in columns:
        return col
    prefixed = f"part_{col}"
    if not col.startswith("part_") and prefixed in columns:
        return prefixed
    return None
