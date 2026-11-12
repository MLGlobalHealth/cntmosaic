"""
Weight factors that do not depend on the sample: DesignWeights and
CategoryWeights (dev plan: *Factor specifications*).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._factor import BindContext, WeightFactor, resolve_column
from ._table import Pairs, as_pairs, describe_unknown, encode, read_only


@dataclass(frozen=True, eq=False)
class _BoundValues:
    """Bound factor that looks up fixed per-participant values (picklable)."""

    values: np.ndarray

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        return self.values[rows]


def _column(ctx: BindContext, col: str) -> str:
    resolved = resolve_column(list(ctx.part.columns), col)
    if resolved is None:  # pragma: no cover - WeightSpecs checks columns first
        raise KeyError(f"Participant column '{col}' not found.")
    return resolved


@dataclass(frozen=True)
class DesignWeights(WeightFactor):
    """Design weights supplied with the survey: one weight per participant.

    ``f_i = x_i``, read from a positive numeric column kept on
    ``ParticipantData(aux_cols=...)``. This is the only way design weights are
    applied (dev plan D10).

    Parameters
    ----------
    col : str
        Weight column, with or without the ``part_`` prefix.
    name : str, optional
        Label in diagnostics; default ``"design:<col>"``.

    Raises
    ------
    ValueError
        At fit time, if the column is not numeric, has missing values, or has
        values that are not finite and strictly positive.

    Examples
    --------
    >>> part = ParticipantData(df, id_col="pid", age_col="age", aux_cols="design_w")
    >>> specs = WeightSpecs(DesignWeights("design_w"))
    """

    col: str
    name: Optional[str] = None

    sample_dependent = False

    def __post_init__(self) -> None:
        if not isinstance(self.col, str) or not self.col:
            raise TypeError("DesignWeights: col must be a non-empty string.")
        if self.name is None:
            object.__setattr__(self, "name", f"design:{self.col.removeprefix('part_')}")

    def required_columns(self) -> Tuple[str, ...]:
        return (self.col,)

    def bind(self, ctx: BindContext) -> _BoundValues:
        col = _column(ctx, self.col)
        values = ctx.part[col]
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            raise ValueError(
                f"{self.name}: design-weight column '{col}' must be numeric; "
                f"got dtype {values.dtype}."
            )
        missing = values.isna().to_numpy()
        if missing.any():
            raise ValueError(
                f"{self.name}: design-weight column '{col}' has {int(missing.sum())} "
                f"missing value(s) (participant ids {ctx.ids[missing][:10].tolist()})."
            )
        return _BoundValues(values.to_numpy(dtype=np.float64))


@dataclass(frozen=True)
class CategoryWeights(WeightFactor):
    """A fixed, known weight per category of a column.

    ``f_i = weights[x_i]``. Use it when the weight of a category is set by the
    design, not by how many participants were sampled in it (e.g. the inverse
    sampling fraction of each region): a category's total weight grows with
    its sample count. To make the weighted sample match a target mix instead,
    use :class:`PostStratWeights`.

    Parameters
    ----------
    col : str
        Category column, with or without the ``part_`` prefix.
    weights : dict or pd.DataFrame
        ``category -> weight`` (strictly positive). A DataFrame is read from
        ``key_col`` (default: ``col``) and ``value_col`` (default: the only
        other column). Stored as a read-only table; later changes to the
        input have no effect.
    key_col, value_col : str, optional
        DataFrame columns.
    name : str, optional
        Label in diagnostics; default ``"category:<col>"``.

    Notes
    -----
    Every observed non-missing value must be a key, and missing values raise:
    there is no neutral value on a fixed-factor scale.

    Examples
    --------
    >>> CategoryWeights("region", {"urban": 1.0, "rural": 2.5})
    >>> CategoryWeights("region", df_factors, key_col="region", value_col="w")
    """

    col: str
    weights: Union[Mapping, pd.DataFrame, Pairs]
    key_col: Optional[str] = field(default=None, compare=False, repr=False)
    value_col: Optional[str] = field(default=None, compare=False, repr=False)
    name: Optional[str] = None

    sample_dependent = False

    def __post_init__(self) -> None:
        if not isinstance(self.col, str) or not self.col:
            raise TypeError("CategoryWeights: col must be a non-empty string.")
        if self.name is None:
            object.__setattr__(self, "name", f"category:{self.col.removeprefix('part_')}")
        if not _is_pairs(self.weights):
            pairs = as_pairs(
                self.weights,
                default_key_col=self.col,
                key_col=self.key_col,
                value_col=self.value_col,
                what=f"{self.name} weights",
            )
            object.__setattr__(self, "weights", pairs)

    @property
    def table(self) -> Mapping[Hashable, float]:
        """Read-only ``category -> weight`` mapping."""
        return read_only(self.weights)

    def required_columns(self) -> Tuple[str, ...]:
        return (self.col,)

    def bind(self, ctx: BindContext) -> _BoundValues:
        col = _column(ctx, self.col)
        keys = [k for k, _ in self.weights]
        values = np.array([v for _, v in self.weights], dtype=np.float64)
        codes, missing, unknown = encode(ctx.part[col], {k: i for i, k in enumerate(keys)})
        if missing.any():
            raise ValueError(
                f"{self.name}: column '{col}' has {int(missing.sum())} missing value(s); "
                f"CategoryWeights has no weight for a missing category "
                f"(participant ids {ctx.ids[missing][:10].tolist()})."
            )
        if unknown:
            raise ValueError(
                f"{self.name}: values of '{col}' with no weight in the table: "
                f"{describe_unknown(unknown)}."
            )
        return _BoundValues(values[codes])


def _is_pairs(obj) -> bool:
    return isinstance(obj, tuple) and all(
        isinstance(p, tuple) and len(p) == 2 for p in obj
    )
