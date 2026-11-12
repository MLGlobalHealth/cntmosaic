"""
Post-stratification weight factors (dev plan: *Factor specifications*).

All three classes share :class:`_PostStratBase`, which holds the counting,
empty-group, unknown-value and ``within`` logic. For groups ``g`` with targets
``t_g`` and, over the evaluated rows, ``n_g`` participants in group ``g`` out
of the ``n`` participants with a known (non-missing) value, with
``T = sum of t_g over non-empty groups``:

    f_i = (t_g / T) / (n_g / n)        (target share / sample share)
    f_i = 1                            for a missing value, if on_missing="average"

The known factors average exactly 1, so a missing value receives the average
factor. With ``within="cell"`` the counts and ``T`` are taken within the
participant's cell instead, and the known factors average 1 in every cell.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, Hashable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._factor import BindContext, WeightFactor, resolve_column
from ._table import Pairs, as_pairs, describe_unknown, encode, norm_key, read_only
from ._warnings import warn

_ON_MISSING = ("raise", "average")
_WITHIN = ("sample", "cell")

GroupPairs = Tuple[Tuple[Hashable, Tuple[Hashable, ...]], ...]


def _check_literal(name: str, value: str, allowed: Sequence[str], owner: str) -> None:
    if value not in allowed:
        raise ValueError(f"{owner}: {name} must be one of {list(allowed)}; got {value!r}.")


@dataclass(frozen=True, eq=False)
class _BoundPostStrat:
    """Bound post-stratification factor (picklable).

    ``codes[i]`` is the group of participant ``i`` (``-1`` if missing);
    ``targets[g]`` the target of group ``g``.
    """

    codes: np.ndarray
    targets: np.ndarray
    cell: np.ndarray
    n_cells: int
    within: str

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        g = self.codes[rows]
        valid = g >= 0
        gv = g[valid]
        n_groups = len(self.targets)
        f = np.ones(len(rows), dtype=np.float64)

        if not valid.any():
            return f
        # Shares are taken among participants with a known value, so the
        # known factors average 1 and a missing value (factor 1) receives the
        # average weight.
        if self.within == "sample":
            n = int(valid.sum())
            n_g = np.bincount(gv, minlength=n_groups)
            T = self.targets[n_g > 0].sum()
            f[valid] = (self.targets[gv] / T) / (n_g[gv] / n)
        else:
            c = self.cell[rows]
            cv = c[valid]
            n_c = np.bincount(cv, minlength=self.n_cells)
            n_cg = np.bincount(
                cv * n_groups + gv, minlength=self.n_cells * n_groups
            ).reshape(self.n_cells, n_groups)
            T_c = (n_cg > 0).astype(np.float64) @ self.targets
            f[valid] = (self.targets[gv] / T_c[cv]) / (n_cg[cv, gv] / n_c[cv])
        return f


class _PostStratBase(WeightFactor):
    """Shared post-stratification logic (private).

    Subclasses provide the column, the groups and the targets through
    :meth:`_column_name` and :meth:`_group_targets`, and may transform the raw
    values (:meth:`_values`).
    """

    sample_dependent = True
    #: Warn when a user-declared group is empty in the full sample.
    _warn_empty_groups = True

    on_missing: str
    within: str
    name: str

    # -- hooks ---------------------------------------------------------------

    @abstractmethod
    def _column_name(self) -> Optional[str]:
        """Participant column (``None``: the subclass resolves it in _values)."""

    @abstractmethod
    def _group_targets(
        self, observed: pd.Series
    ) -> Tuple[Dict[Hashable, int], np.ndarray, List[Hashable]]:
        """Return ``(key -> group index, targets per group, group labels)``.

        *observed* holds the (transformed) participant values, so a subclass
        may restrict its targets to what is observed.
        """

    def _values(self, ctx: BindContext) -> pd.Series:
        col = resolve_column(list(ctx.part.columns), self._column_name())
        return ctx.part[col]

    # -- WeightFactor ----------------------------------------------------------

    def required_columns(self) -> Tuple[str, ...]:
        col = self._column_name()
        return () if col is None else (col,)

    def bind(self, ctx: BindContext) -> _BoundPostStrat:
        values = self._values(ctx)
        key_to_group, targets, labels = self._group_targets(values)
        codes, missing, unknown = encode(values, key_to_group)

        if unknown:
            expected = ", ".join(repr(k) for k in sorted(key_to_group, key=repr))
            raise ValueError(
                f"{self.name}: values with no group: {describe_unknown(unknown)}. "
                f"Every non-missing value must belong to a group; expected values: "
                f"{expected}. Values are matched exactly (case-sensitive; the string "
                f"'1' does not match the integer 1). Missing values must be NaN, None "
                f"or pd.NA: recode sentinels such as 'missing' first."
            )
        if missing.any() and self.on_missing == "raise":
            raise ValueError(
                f"{self.name}: {int(missing.sum())} participant(s) have a missing value "
                f"(ids {ctx.ids[missing][:10].tolist()}). Use on_missing='average' to "
                f"give them a factor of 1."
            )

        counts = np.bincount(codes[codes >= 0], minlength=len(targets))
        zero = (targets == 0) & (counts > 0)
        if zero.any():
            raise ValueError(
                f"{self.name}: participants fall in group(s) with target 0: "
                f"{[labels[g] for g in np.flatnonzero(zero)]}. Their factor would be 0; "
                f"filter them out instead."
            )
        if self._warn_empty_groups and (counts == 0).any():
            empty = [labels[g] for g in np.flatnonzero(counts == 0)]
            warn(
                f"{self.name}: group(s) {empty} have no participants; their targets "
                f"are left out of the total."
            )
        if self._warn_empty_groups and self.within == "cell":
            self._warn_empty_groups_in_cells(codes, targets, labels, ctx)

        return _BoundPostStrat(
            codes=codes,
            targets=np.asarray(targets, dtype=np.float64),
            cell=np.asarray(ctx.cell, dtype=np.int64),
            n_cells=int(ctx.n_cells),
            within=self.within,
        )

    def _warn_empty_groups_in_cells(
        self, codes: np.ndarray, targets, labels: List[Hashable], ctx: BindContext
    ) -> None:
        """Warn about cells in which a group with a positive target is empty.

        In such a cell the target mix cannot be met: the factors of the
        observed groups are constant within the cell and cancel, so the cell
        averages over the observed groups only.
        """
        n_groups = len(targets)
        valid = codes >= 0
        if not valid.any():
            return
        cell = np.asarray(ctx.cell, dtype=np.int64)
        n_cg = np.bincount(
            cell[valid] * n_groups + codes[valid], minlength=int(ctx.n_cells) * n_groups
        ).reshape(int(ctx.n_cells), n_groups)
        wanted = np.asarray(targets, dtype=np.float64) > 0
        observed = n_cg.sum(axis=1) > 0
        lacking = observed[:, None] & (n_cg == 0) & wanted[None, :]
        rows = np.flatnonzero(lacking.any(axis=1))
        if rows.size == 0:
            return
        details = "; ".join(
            f"{ctx.cell_labels[c]} (no {', '.join(str(labels[g]) for g in np.flatnonzero(lacking[c]))})"
            for c in rows
        )
        warn(
            f"{self.name}: with within='cell', {rows.size} cell(s) have no participants "
            f"in some group, so the target mix cannot be met there and these cells "
            f"average over the observed groups only: {details}. Consider merging age "
            f"groups, or within='sample'."
        )


def _as_codes(codes: Any) -> Tuple[Hashable, ...]:
    """A single code or an iterable of codes, as a tuple of canonical keys."""
    if isinstance(codes, (str, bytes)) or not hasattr(codes, "__iter__"):
        codes = (codes,)
    return tuple(norm_key(d) for d in codes)


def _normalise_groups(groups: Union[Mapping, GroupPairs], owner: str) -> GroupPairs:
    """``label -> keys`` mapping to a canonical tuple; overlapping groups raise."""
    if isinstance(groups, tuple):
        return groups
    if not isinstance(groups, Mapping) or not groups:
        raise TypeError(f"{owner}: groups must be a non-empty dict of label -> keys.")
    out = []
    seen: Dict[Hashable, Hashable] = {}
    for label, keys in groups.items():
        if isinstance(keys, (str, bytes)) or not hasattr(keys, "__iter__"):
            keys = [keys]
        normed = tuple(sorted({norm_key(k) for k in keys}, key=repr))
        if not normed:
            raise ValueError(f"{owner}: group {label!r} has no keys.")
        for k in normed:
            if k in seen:
                raise ValueError(
                    f"{owner}: groups overlap: key {k!r} is in both {seen[k]!r} and {label!r}."
                )
            seen[k] = norm_key(label)
        out.append((norm_key(label), normed))
    return tuple(sorted(out, key=lambda p: repr(p[0])))


def _explicit_group_targets(
    targets: Pairs, groups: Optional[GroupPairs], owner: str
) -> Tuple[Dict[Hashable, int], np.ndarray, List[Hashable]]:
    labels = [k for k, _ in targets]
    t = np.array([v for _, v in targets], dtype=np.float64)
    if groups is None:
        return {k: i for i, k in enumerate(labels)}, t, labels
    index = {label: i for i, label in enumerate(labels)}
    group_labels = [label for label, _ in groups]
    if set(group_labels) != set(labels):
        raise ValueError(
            f"{owner}: groups and targets must have the same labels; "
            f"groups {sorted(map(repr, group_labels))}, targets {sorted(map(repr, labels))}."
        )
    key_to_group = {k: index[label] for label, keys in groups for k in keys}
    return key_to_group, t, labels


@dataclass(frozen=True)
class PostStratWeights(_PostStratBase):
    """Post-stratification: make the weighted mix of a column match target shares.

    ``f_i = (t_g / T) / (n_g / n)``: target share over sample share, where
    ``n`` counts the participants with a known value. A category's total
    weight is its target share whatever its sample count, and a missing value
    (with ``on_missing="average"``) receives the average factor of 1.
    Use :class:`CategoryWeights` instead when you know a fixed weight per
    category.

    Parameters
    ----------
    col : str
        Column to post-stratify, with or without the ``part_`` prefix.
    targets : dict or pd.DataFrame
        ``group label -> target`` (non-negative; only ratios matter: totals,
        shares and population counts all work). A DataFrame is read from
        ``key_col`` and ``value_col``.
    groups : dict, optional
        ``group label -> keys`` (e.g. ``{"weekday": [1, 2, 3, 4, 5]}``). When
        omitted, each key of ``targets`` is its own group. Groups must not
        overlap. With a DataFrame ``targets`` and ``groups``, ``key_col`` must
        be given.
    key_col, value_col : str, optional
        DataFrame columns (``key_col`` defaults to ``col`` without groups).
    on_missing : {"raise", "average"}, default "raise"
        Missing values raise, or get a factor of 1 (the average weight).
    within : {"sample", "cell"}, default "sample"
        Match the target mix over the whole sample or within
        each cell (age group x participant strata). With ``"cell"`` the factor
        on its own reproduces the target mix within each cell, among
        participants with a known value, in cells where every group is present;
        a cell that lacks a group gives a :class:`WeightWarning`. ``"cell"``
        applies the *same* targets in every cell, so use it only when the
        targets hold within each age group (e.g. the 5:2 calendar ratio of
        weekdays and weekend days), not for a composition that varies with
        age (e.g. a national sex ratio).
    name : str, optional
        Label in diagnostics; default ``"poststrat:<col>"``.

    Notes
    -----
    Several post-stratification factors are not calibrated jointly: each
    matches its own margin on its own, and their product moves each margin
    away from its target. Matching several margins at once is raking.

    Examples
    --------
    >>> PostStratWeights("country", {"UK": 0.3, "DE": 0.4, "IT": 0.3})
    >>> PostStratWeights("dayofweek", {"weekday": 5, "weekend": 2},
    ...                  groups={"weekday": [1, 2, 3, 4, 5], "weekend": [0, 6]},
    ...                  on_missing="average")
    """

    col: str
    targets: Union[Mapping, pd.DataFrame, Pairs]
    groups: Optional[Union[Mapping, GroupPairs]] = None
    key_col: Optional[str] = field(default=None, compare=False, repr=False)
    value_col: Optional[str] = field(default=None, compare=False, repr=False)
    on_missing: Literal["raise", "average"] = "raise"
    within: Literal["sample", "cell"] = "sample"
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.col, str) or not self.col:
            raise TypeError("PostStratWeights: col must be a non-empty string.")
        if self.name is None:
            object.__setattr__(self, "name", f"poststrat:{self.col.removeprefix('part_')}")
        _check_literal("on_missing", self.on_missing, _ON_MISSING, self.name)
        _check_literal("within", self.within, _WITHIN, self.name)
        if self.groups is not None:
            object.__setattr__(self, "groups", _normalise_groups(self.groups, self.name))
        if not (isinstance(self.targets, tuple)):
            if isinstance(self.targets, pd.DataFrame) and self.groups is not None and self.key_col is None:
                raise ValueError(
                    f"{self.name}: with groups, the targets DataFrame is keyed by group "
                    f"label, so key_col must be given."
                )
            pairs = as_pairs(
                self.targets,
                default_key_col=None if self.groups is not None else self.col,
                key_col=self.key_col,
                value_col=self.value_col,
                what=f"{self.name} targets",
                allow_zero=True,
            )
            object.__setattr__(self, "targets", pairs)
        # Fail early on label mismatches.
        _explicit_group_targets(self.targets, self.groups, self.name)

    @property
    def table(self) -> Mapping[Hashable, float]:
        """Read-only ``group label -> target`` mapping."""
        return read_only(self.targets)

    def _column_name(self) -> str:
        return self.col

    def _group_targets(self, observed):
        return _explicit_group_targets(self.targets, self.groups, self.name)


@dataclass(frozen=True)
class DayOfWeekWeights(_PostStratBase):
    """Day-of-week weights: weekdays and weekend days in the ratio 5:2.

    Post-stratification with groups ``{"weekday": weekday, "weekend":
    weekend}`` and targets ``ratio``. By default the ratio is matched within
    each cell (age group x participant strata):

        f_i = (5/7) / (n_{c,wd} / (n_{c,wd} + n_{c,we}))   for a weekday report,
        f_i = (2/7) / (n_{c,we} / (n_{c,wd} + n_{c,we}))   for a weekend report,
        f_i = 1                                             for a missing day.

    Among participants with a known day the factors average 1, so a missing
    day receives the average factor.

    Parameters
    ----------
    col : str, default "dayofweek"
        Day column, with or without the ``part_`` prefix.
    weekday, weekend : code or iterable of codes
        Day codes of each group. A single code (e.g. ``"weekday"``) is allowed.
    ratio : tuple of float, default (5, 2)
        Targets of weekday and weekend.
    on_missing : {"average", "raise"}, default "average"
        Missing days get a factor of 1 (the average factor), or raise.
    within : {"cell", "sample"}, default "cell"
        Match 5:2 within each cell (age group x participant strata), or only
        over the whole sample. With ``"sample"`` the weekday
        share of an age group whose day mix differs from the sample's is
        pushed away from 5:2. With ``"cell"`` a cell that lacks weekday or
        weekend reports cannot meet the ratio; this gives a
        :class:`WeightWarning`, and the cell averages over the observed day
        type only.
    name : str, default "dayofweek"

    Notes
    -----
    **Assumptions about the data.**

    1. **Column.** The day is read from the participant column ``col``
       (default ``"dayofweek"``), given with or without the ``part_`` prefix.
       It must be kept on ``ParticipantData(aux_cols=...)``; otherwise fitting
       raises ``KeyError``.
    2. **Coding.** Observed values are matched *exactly* against the codes in
       ``weekday`` and ``weekend``. The default is POLYMOD's integer coding,
       0 = Sunday, ..., 6 = Saturday. Integral floats match integers (``1.0``
       matches ``1``), but strings are matched as they are and are
       case-sensitive: ``"1"`` does not match ``1`` and ``"Weekday"`` does not
       match ``"weekday"``. Labelled data are supported by giving the labels,
       e.g. ``weekday="weekday", weekend="weekend"``.
    3. **Unknown values raise.** A non-missing value in neither group raises
       ``ValueError`` (e.g. 7 under a 1-7 coding), so a Sunday coded 7 never
       silently receives the average weight.
    4. **Missing values.** Only NaN, None and pd.NA count as missing. A
       sentinel such as ``"missing"``, ``""`` or ``-1`` is an ordinary value
       and raises; recode it to NaN first.
    5. **Categorical columns.** The observed values are used, not the
       declared categories: unused categories are ignored.

    Examples
    --------
    >>> DayOfWeekWeights()                                   # POLYMOD coding, 0 = Sunday
    >>> DayOfWeekWeights(weekday=(1, 2, 3, 4, 5), weekend=(6, 7))   # 1 = Monday ... 7 = Sunday
    >>> DayOfWeekWeights(weekday="weekday", weekend="weekend")      # labelled column
    >>> DayOfWeekWeights(within="sample")                    # 5:2 over the whole sample only
    """

    col: str = "dayofweek"
    weekday: Union[Hashable, Tuple[Hashable, ...]] = (1, 2, 3, 4, 5)
    weekend: Union[Hashable, Tuple[Hashable, ...]] = (0, 6)
    ratio: Tuple[float, float] = (5.0, 2.0)
    on_missing: Literal["raise", "average"] = "average"
    within: Literal["sample", "cell"] = "cell"
    name: str = "dayofweek"

    def __post_init__(self) -> None:
        _check_literal("on_missing", self.on_missing, _ON_MISSING, self.name)
        _check_literal("within", self.within, _WITHIN, self.name)
        object.__setattr__(self, "weekday", _as_codes(self.weekday))
        object.__setattr__(self, "weekend", _as_codes(self.weekend))
        if len(self.ratio) != 2 or any(r <= 0 for r in self.ratio):
            raise ValueError(f"{self.name}: ratio must be two positive numbers.")
        object.__setattr__(self, "ratio", tuple(float(r) for r in self.ratio))
        _normalise_groups(self._groups(), self.name)  # overlap check

    def _groups(self) -> Dict[str, Tuple[int, ...]]:
        return {"weekday": self.weekday, "weekend": self.weekend}

    def _column_name(self) -> str:
        return self.col

    def _group_targets(self, observed):
        groups = _normalise_groups(self._groups(), self.name)
        targets = (("weekday", self.ratio[0]), ("weekend", self.ratio[1]))
        return _explicit_group_targets(targets, groups, self.name)


@dataclass(frozen=True)
class AgePostStratWeights(_PostStratBase):
    """Age post-stratification against a reference population.

    Post-stratification on single-year participant age (``floor(part_age)``)
    with targets ``P_a``: ``f_i = (P_a / P) / (n_a / n)``, where ``P`` sums the
    population over the ages present in the sample. Within a cell only
    ``P_a / n_a`` matters.

    Parameters
    ----------
    pop : PopulationData
        Reference population. Single-year ages are used directly; age bands
        (``age_min``/``age_max`` or interval groups) are spread uniformly over
        their single years, without rounding; strata are summed over. Only
        the resulting ``(age, P_a)`` pairs are stored.
    within : {"sample", "cell"}, default "sample"
        Count ``n_a`` over the whole sample, or within each cell. Without
        participant strata both give the same weights, since each single year
        of age lies in one age group; with strata, ``"cell"`` post-stratifies
        age separately within each stratum.
    name : str, default "age"

    Raises
    ------
    ValueError
        At construction, for a population band with a non-finite limit. At
        fit time, if participants have no single-year age, or an age with no
        population or a population of 0 (nobody is dropped silently).

    Examples
    --------
    >>> pop = PopulationData(df_pop, size_col="P", age_col="age")
    >>> AgePostStratWeights(pop)
    """

    pop: InitVar[Any] = None
    within: Literal["sample", "cell"] = "sample"
    name: str = "age"
    population: Tuple[Tuple[int, float], ...] = field(init=False, default=())

    on_missing = "raise"
    _warn_empty_groups = False

    def __post_init__(self, pop) -> None:
        _check_literal("within", self.within, _WITHIN, self.name)
        if pop is None:
            raise TypeError("AgePostStratWeights needs a PopulationData (pop=...).")
        object.__setattr__(self, "population", _single_year_population(pop, self.name))

    def __repr__(self) -> str:
        ages = [a for a, _ in self.population]
        span = f"ages {ages[0]}-{ages[-1]}" if ages else "empty"
        return (
            f"AgePostStratWeights(population=<{span}>, within={self.within!r}, "
            f"name={self.name!r})"
        )

    def _column_name(self) -> Optional[str]:
        return None  # checked in _values with a specific message

    def source_columns(self) -> Tuple[str, ...]:
        return ("part_age",)

    def _values(self, ctx: BindContext) -> pd.Series:
        if "part_age" not in ctx.part.columns:
            form = (
                "age ranges (age_min/age_max)"
                if "part_age_min" in ctx.part.columns
                else "age groups"
            )
            raise ValueError(
                f"{self.name}: needs single-year participant ages (ParticipantData "
                f"age_col), but the participants have {form}."
            )
        return np.floor(ctx.part["part_age"].astype(float))

    def _group_targets(self, observed):
        population = dict(self.population)
        ages = sorted(int(a) for a in pd.unique(observed.dropna()))
        uncovered = [a for a in ages if a not in population]
        if uncovered:
            lo, hi = min(population), max(population)
            raise ValueError(
                f"{self.name}: participant age(s) {uncovered[:10]} have no population "
                f"(population covers ages {lo}-{hi}; participants {ages[0]}-{ages[-1]}). "
                f"Nobody is dropped silently: extend the population or filter participants."
            )
        targets = np.array([population[a] for a in ages], dtype=np.float64)
        return {a: i for i, a in enumerate(ages)}, targets, ages


def _single_year_population(pop, owner: str) -> Tuple[Tuple[int, float], ...]:
    """``(age, P_a)`` pairs from a PopulationData, summed over strata."""
    df = getattr(pop, "data", None)
    if not isinstance(df, pd.DataFrame) or "P" not in df.columns:
        raise TypeError(f"{owner}: pop must be a PopulationData.")
    totals: Dict[int, float] = {}

    def add(age: int, size: float) -> None:
        totals[age] = totals.get(age, 0.0) + float(size)

    if "age" in df.columns:
        for age, size in zip(np.floor(df["age"].astype(float)), df["P"]):
            add(int(age), size)
    elif {"age_min", "age_max"} <= set(df.columns):
        for lo, hi, size in zip(df["age_min"], df["age_max"], df["P"]):
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise ValueError(f"{owner}: population band [{lo}, {hi}] is not finite.")
            years = range(int(math.ceil(lo)), int(math.floor(hi)) + 1)  # inclusive max
            for y in years:
                add(y, size / len(years))
    elif "pop_age_grp" in df.columns:
        for interval, size in zip(df["pop_age_grp"], df["P"]):
            lo, hi = interval.left, interval.right
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise ValueError(
                    f"{owner}: population band {interval} is open-ended; it cannot be "
                    f"spread over single years."
                )
            years = [y for y in range(int(math.floor(lo)), int(math.ceil(hi)) + 1) if y in interval]
            for y in years:
                add(y, size / len(years))
    else:
        raise ValueError(f"{owner}: population data has no recognised age column.")
    if any(v < 0 for v in totals.values()):
        raise ValueError(f"{owner}: population sizes must be non-negative.")
    return tuple(sorted(totals.items()))
