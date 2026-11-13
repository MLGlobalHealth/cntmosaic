"""
Weight specifications.

:class:`WeightSpecs` is an immutable list of weight factors (plus an optional
truncation threshold): pure configuration, reusable across surveys and fits.
An estimator binds it to its own participants and cells (``WeightSpecs._bind``),
which returns a private :class:`_BoundWeightSpecs` that computes the weights
for any set of rows:

1. each factor gives :math:`f^{(k)}_i`;
2. the product is the composite weight :math:`\\tilde w_i = \\prod_k f^{(k)}_i`;
3. composite weights are normalised within cells,
   :math:`w_i = \\tilde w_i \\, n_{c(i)} / \\sum_{j : c(j) = c(i)} \\tilde w_j`,
   so the weights in each cell sum to the cell's size;
4. with a threshold :math:`\\tau`, weights are capped at :math:`\\tau` and
   re-normalised once, so they can end slightly above
   :math:`\\tau`.
"""

from __future__ import annotations

import numbers
import warnings
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ._factor import BindContext, BoundFactor, WeightFactor, resolve_column
from ._result import WeightResult
from ._warnings import WeightWarning, warn

if TYPE_CHECKING:
    from ..dataloader import ParticipantData
    from ..utils import AgeGroupSpecs

_MAX_IDS_SHOWN = 10


def _rebuild(factors, threshold):
    return WeightSpecs(*factors, threshold=threshold)


class WeightSpecs:
    """Specification of participant weights: an ordered set of weight factors.

    A ``WeightSpecs`` holds configuration only, no data, and is immutable. It
    is applied by passing it to an estimator, e.g.
    ``SocialMix(..., weight_specs=WeightSpecs(...))``; the estimator computes
    the weights on its own participants and cells, including on every
    bootstrap resample. Use :meth:`preview` to inspect the weights before
    fitting.

    Parameters
    ----------
    *factors : WeightFactor
        One or more weight factors. Their weights are multiplied together.
        Factor names must be unique.
    threshold : float, optional
        Truncation threshold :math:`\\tau > 1` on the normalised scale (the
        mean weight in every cell is 1): weights are capped at :math:`\\tau`
        and re-normalised once, so they can end slightly above it.

    Raises
    ------
    ValueError
        If no factor is given, two factors have the same name, or the
        threshold is not greater than 1.
    TypeError
        If an argument is not a :class:`WeightFactor`.

    Examples
    --------
    >>> specs = WeightSpecs(DayOfWeekWeights(), AgePostStratWeights(pop), threshold=3)
    >>> specs.preview(part_data, age_group_specs)          # inspect before fitting
    >>> sm = SocialMix(part_data, cnt_data, age_group_specs, pop_data, weight_specs=specs)
    """

    __slots__ = ("_factors", "_threshold")

    def __init__(self, *factors: WeightFactor, threshold: Optional[float] = None) -> None:
        if not factors:
            raise ValueError("WeightSpecs needs at least one weight factor.")
        for f in factors:
            if not isinstance(f, WeightFactor):
                raise TypeError(
                    f"WeightSpecs arguments must be WeightFactor instances; "
                    f"got {type(f).__name__}."
                )
        counts = Counter(f.name for f in factors)
        duplicates = sorted(n for n, c in counts.items() if c > 1)
        if duplicates:
            raise ValueError(
                f"Duplicate weight-factor names: {duplicates}. "
                f"Give each factor a unique name with name=..."
            )
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
                raise TypeError("threshold must be a number or None.")
            if not np.isfinite(threshold) or threshold <= 1:
                raise ValueError(
                    f"threshold must be greater than 1 (the mean weight in every cell "
                    f"is 1); got {threshold}."
                )
            threshold = float(threshold)
        object.__setattr__(self, "_factors", tuple(factors))
        object.__setattr__(self, "_threshold", threshold)

    # -- immutability, equality, pickling ---------------------------------

    def __setattr__(self, key, value):
        raise AttributeError("WeightSpecs is immutable.")

    def __delattr__(self, key):
        raise AttributeError("WeightSpecs is immutable.")

    def __eq__(self, other) -> bool:
        if not isinstance(other, WeightSpecs):
            return NotImplemented
        return self._factors == other._factors and self._threshold == other._threshold

    def __hash__(self) -> int:
        return hash((WeightSpecs, self._factors, self._threshold))

    def __reduce__(self):
        return (_rebuild, (self._factors, self._threshold))

    def __repr__(self) -> str:
        parts = [repr(f) for f in self._factors]
        if self._threshold is not None:
            parts.append(f"threshold={self._threshold:g}")
        return f"WeightSpecs({', '.join(parts)})"

    # -- public API ---------------------------------------------------------

    @property
    def factors(self) -> Tuple[WeightFactor, ...]:
        """The weight factors, in order."""
        return self._factors

    @property
    def names(self) -> Tuple[str, ...]:
        """Factor names, in order."""
        return tuple(f.name for f in self._factors)

    @property
    def threshold(self) -> Optional[float]:
        """Truncation threshold, or ``None``."""
        return self._threshold

    def preview(
        self,
        part_data: "ParticipantData",
        age_group_specs: "AgeGroupSpecs",
        adaptive_merge: bool = False,
    ) -> WeightResult:
        """Show the weights a ``SocialMix`` fit would use, without fitting.

        Uses the same participant cells as the fit (age groups, dropping of
        participants outside them, adaptive merging and stratum codes come
        from the same helper). It mirrors ``fit()``, not a later
        bootstrap-time merge. Where the fit is strict it is lenient: an empty
        participant age group (without ``adaptive_merge``) is reported as a
        cell with ``n = 0`` and a :class:`WeightWarning`, instead of raising.

        Parameters
        ----------
        part_data : ParticipantData
            Participants (not modified).
        age_group_specs : AgeGroupSpecs
            Age groups, as passed to ``SocialMix``.
        adaptive_merge : bool, default False
            As in ``SocialMix``.

        Returns
        -------
        WeightResult
        """
        from ..utils._participant_cells import build_participant_cells

        cells = build_participant_cells(
            part_data.data,
            part_data.get_strat_vars(prefix=True),
            age_group_specs,
            adaptive_merge=adaptive_merge,
            strict=False,
        )
        if cells.empty_age_groups:
            warn(
                f"Participant age group(s) {list(cells.empty_age_groups)} are empty in "
                f"at least one stratum; SocialMix would raise unless adaptive_merge=True. "
                f"The empty cells are reported with n = 0."
            )
        ctx = BindContext(
            part=cells.part.copy(),
            cell=cells.cell,
            n_cells=cells.n_cells,
            cell_labels=cells.cell_labels,
            ids=cells.ids,
        )
        return self._bind(ctx).result()

    # -- internal -----------------------------------------------------------

    def _bind(self, ctx: BindContext, warn_user: bool = True) -> "_BoundWeightSpecs":
        """Bind every factor to the participants and cells in *ctx* (internal).

        All checks run here, on the full sample: required columns, each
        factor's own checks, that every factor is finite and positive, and
        whether a factor has no effect (constant within every cell). With
        ``warn_user=False`` (a rebind of the same specs) warnings are not
        repeated.
        """
        columns = list(ctx.part.columns)
        missing: Dict[str, List[str]] = defaultdict(list)
        for f in self._factors:
            for col in f.required_columns():
                if resolve_column(columns, col) is None:
                    missing[col].append(f.name)
        if missing:
            detail = "; ".join(
                f"'{col}' (needed by {', '.join(names)})" for col, names in missing.items()
            )
            raise KeyError(
                f"Participant data is missing column(s) required for weighting: "
                f"{detail}. Keep weighting variables on ParticipantData(aux_cols=...)."
            )

        with warnings.catch_warnings():
            if not warn_user:
                warnings.simplefilter("ignore", WeightWarning)
            bound = tuple(f.bind(ctx) for f in self._factors)
            all_rows = np.arange(ctx.n)
            cell = np.asarray(ctx.cell, dtype=np.int64)
            for f, b in zip(self._factors, bound):
                values = np.asarray(b(all_rows))
                _check_factor_values(f.name, values, ctx.ids)
                if _constant_within_cells(values, cell, ctx.n_cells):
                    warn(
                        f"Weight factor '{f.name}' is constant within every cell "
                        f"(age group x participant strata), so it has no effect on the "
                        f"contact matrix."
                    )

        return _BoundWeightSpecs(
            specs=self,
            bound=bound,
            cell=cell,
            n_cells=int(ctx.n_cells),
            cell_labels=tuple(ctx.cell_labels),
            ids=np.asarray(ctx.ids),
            source=self._source_table(ctx),
        )

    def _source_table(self, ctx: BindContext) -> pd.DataFrame:
        """The participant columns the factors are computed from, in factor
        order and without duplicates, as resolved in ``ctx.part``."""
        resolved: List[str] = []
        for f in self._factors:
            for col in f.source_columns():
                name = resolve_column(list(ctx.part.columns), col)
                if name is not None and name != "id" and name not in resolved:
                    resolved.append(name)
        clashes = sorted(
            set(resolved) & (set(self.names) | {"cell", "composite", "weight", "capped"})
        )
        if clashes:
            raise ValueError(
                f"Weight-factor name(s) {clashes} clash with participant column(s) "
                f"shown in WeightResult.participants. Give the factor another name "
                f"with name=..."
            )
        return ctx.part[resolved].reset_index(drop=True)


def _check_factor_values(name: str, values: np.ndarray, ids: np.ndarray) -> None:
    """Raise ValueError unless *values* is one finite, positive number per row."""
    if values.shape != ids.shape:
        raise ValueError(
            f"Weight factor '{name}' returned {values.shape[0] if values.ndim else 0} "
            f"values for {ids.shape[0]} participants."
        )
    bad = ~np.isfinite(values) | (values <= 0)
    if bad.any():
        shown = ids[bad][:_MAX_IDS_SHOWN].tolist()
        more = "" if bad.sum() <= _MAX_IDS_SHOWN else f" (and {bad.sum() - _MAX_IDS_SHOWN} more)"
        raise ValueError(
            f"Weight factor '{name}' must be finite and strictly positive for every "
            f"participant; {int(bad.sum())} participant(s) are not: ids {shown}{more}. "
            f"To exclude participants, filter the data instead."
        )


def _constant_within_cells(values: np.ndarray, cell: np.ndarray, n_cells: int) -> bool:
    """True if *values* does not vary within any cell (relative tolerance)."""
    if len(values) == 0:
        return False
    lo = np.full(n_cells, np.inf)
    hi = np.full(n_cells, -np.inf)
    np.minimum.at(lo, cell, values)
    np.maximum.at(hi, cell, values)
    present = np.isfinite(lo)
    spread = (hi - lo)[present]
    scale = np.maximum(np.abs(hi[present]), 1e-300)
    return bool(np.all(spread <= 1e-12 * scale))


class _BoundWeightSpecs:
    """A :class:`WeightSpecs` bound to one participant table (private).

    Holds the bound factors, the cell code of each participant, the
    participant ids it was bound to and the participant columns the factors
    are computed from. Picklable as long as the bound factors are.
    """

    def __init__(
        self,
        specs: WeightSpecs,
        bound: Tuple[BoundFactor, ...],
        cell: np.ndarray,
        n_cells: int,
        cell_labels: Tuple[str, ...],
        ids: np.ndarray,
        source: Optional[pd.DataFrame] = None,
    ) -> None:
        self.specs = specs
        self.bound = bound
        self.cell = cell
        self.n_cells = n_cells
        self.cell_labels = cell_labels
        self.ids = ids
        self.source = pd.DataFrame(index=range(len(ids))) if source is None else source

    @property
    def n(self) -> int:
        return len(self.ids)

    def _rows(self, rows: Optional[np.ndarray]) -> np.ndarray:
        return np.arange(self.n) if rows is None else np.asarray(rows, dtype=np.int64)

    def factor_values(self, rows: Optional[np.ndarray] = None) -> np.ndarray:
        """Factor values, shape ``(n_factors, len(rows))``."""
        rows = self._rows(rows)
        return np.vstack([np.asarray(b(rows), dtype=np.float64) for b in self.bound])

    def composite(self, rows: Optional[np.ndarray] = None) -> np.ndarray:
        """Composite weights :math:`\\tilde w_i` (product of the factors)."""
        return np.prod(self.factor_values(rows), axis=0)

    def weights(self, rows: Optional[np.ndarray] = None) -> np.ndarray:
        """Final weights :math:`w_i` over *rows*: normalised within cells, then
        truncated if a threshold is set.

        Counts are taken over *rows* with multiplicity, so a participant drawn
        twice in a bootstrap resample counts twice.
        """
        rows = self._rows(rows)
        w, _ = _final_weights(
            self.composite(rows), self.cell[rows], self.n_cells, self.specs.threshold
        )
        return w

    def result(self) -> WeightResult:
        """Full-sample diagnostics."""
        values = self.factor_values()
        composite = np.prod(values, axis=0)
        weight, capped = _final_weights(
            composite, self.cell, self.n_cells, self.specs.threshold
        )
        labels = np.asarray(self.cell_labels, dtype=object)
        table = {"id": self.ids, "cell": labels[self.cell]}
        for col in self.source.columns:
            table[col] = self.source[col].array  # keeps the dtype, e.g. categorical
        for name, v in zip(self.specs.names, values):
            table[name] = v
        table["composite"] = composite
        table["weight"] = weight
        table["capped"] = capped
        return WeightResult(
            participants=pd.DataFrame(table),
            cells=_cell_table(weight, capped, self.cell, self.n_cells, labels),
            specs=self.specs,
        )


def _normalise_within_cells(
    weights: np.ndarray, cell: np.ndarray, n_cells: int
) -> np.ndarray:
    """Rescale *weights* so that the weights in each cell sum to its size."""
    cell_sum = np.bincount(cell, weights=weights, minlength=n_cells)
    cell_n = np.bincount(cell, minlength=n_cells)
    return weights * cell_n[cell] / cell_sum[cell]


def _final_weights(
    composite: np.ndarray, cell: np.ndarray, n_cells: int, threshold: Optional[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalise within cells, then cap at *threshold* and re-normalise once."""
    w = _normalise_within_cells(composite, cell, n_cells)
    if threshold is None:
        return w, np.zeros(len(w), dtype=bool)
    capped = w > threshold
    if capped.any():
        w = _normalise_within_cells(np.minimum(w, threshold), cell, n_cells)
    return w, capped


def _cell_table(
    weight: np.ndarray, capped: np.ndarray, cell: np.ndarray, n_cells: int, labels: np.ndarray
) -> pd.DataFrame:
    n = np.bincount(cell, minlength=n_cells)
    s1 = np.bincount(cell, weights=weight, minlength=n_cells)
    s2 = np.bincount(cell, weights=weight**2, minlength=n_cells)
    lo = np.full(n_cells, np.inf)
    hi = np.full(n_cells, -np.inf)
    np.minimum.at(lo, cell, weight)
    np.maximum.at(hi, cell, weight)
    empty = n == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = np.where(empty, np.nan, s1**2 / s2)
        deff = np.where(empty, np.nan, n / n_eff)
    return pd.DataFrame(
        {
            "cell": labels,
            "n": n,
            "min_w": np.where(empty, np.nan, lo),
            "max_w": np.where(empty, np.nan, hi),
            "n_capped": np.bincount(cell, weights=capped.astype(float), minlength=n_cells).astype(int),
            "n_eff": n_eff,
            "deff": deff,
        }
    )
