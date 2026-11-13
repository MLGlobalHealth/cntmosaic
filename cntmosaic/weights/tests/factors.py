"""
Minimal weight factors for testing the Core machinery (dev plan, Core item).

The real factors (DesignWeights, CategoryWeights, PostStratWeights, ...)
come in later checklist items; these exercise the WeightFactor contract.
Defined at module level so that they (and their bound forms) are picklable.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .._factor import BindContext, WeightFactor, resolve_column


@dataclass(frozen=True, eq=False)
class BoundValues:
    """A bound factor that looks up precomputed per-participant values."""

    values: np.ndarray

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        return self.values[rows]


@dataclass(frozen=True)
class ConstantFactor(WeightFactor):
    """Same factor for everyone."""

    value: float = 2.0
    name: str = "constant"

    def required_columns(self) -> Tuple[str, ...]:
        return ()

    def bind(self, ctx: BindContext) -> BoundValues:
        return BoundValues(np.full(ctx.n, float(self.value)))


@dataclass(frozen=True)
class LinearAgeFactor(WeightFactor):
    """f_i = intercept + slope * part_age (varies within age groups)."""

    intercept: float = 1.0
    slope: float = 0.5
    name: str = "linear_age"

    def required_columns(self) -> Tuple[str, ...]:
        return ("age",)  # resolves to part_age

    def bind(self, ctx: BindContext) -> BoundValues:
        col = resolve_column(list(ctx.part.columns), "age")
        ages = ctx.part[col].to_numpy(dtype=np.float64)
        return BoundValues(self.intercept + self.slope * ages)


@dataclass(frozen=True)
class ColumnFactor(WeightFactor):
    """f_i = value of an arbitrary column (used to test missing columns and
    invalid values)."""

    col: str = "missing"
    name: str = "column"

    def required_columns(self) -> Tuple[str, ...]:
        return (self.col,)

    def bind(self, ctx: BindContext) -> BoundValues:
        col = resolve_column(list(ctx.part.columns), self.col)
        return BoundValues(ctx.part[col].to_numpy(dtype=np.float64))


@dataclass(frozen=True, eq=False)
class _FailOnResample:
    n: int

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows)
        if len(rows) != self.n or not np.array_equal(rows, np.arange(self.n)):
            raise RuntimeError("injected bug in a weight factor")
        return np.ones(len(rows))


@dataclass(frozen=True)
class FailOnResampleFactor(WeightFactor):
    """Works on the full sample, raises RuntimeError on any resample (to check
    that bugs propagate out of the bootstrap)."""

    name: str = "fail_on_resample"

    def required_columns(self) -> Tuple[str, ...]:
        return ()

    def bind(self, ctx: BindContext) -> _FailOnResample:
        return _FailOnResample(ctx.n)
