"""
Participant cells: the single source of truth for which cell each participant
belongs to.

A *cell* is a participant age group crossed with the participant strata. This
module bins ages, drops participants outside the age groups, merges empty age
groups (``adaptive_merge``) and computes cell codes. ``SocialMix`` (through its
validator) uses it, and so will ``WeightSpecs.preview()``, so that both always
agree on the cells (dev plan D21-D24).

It lives in ``cntmosaic.utils`` so that it can be imported without importing
the models package.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd

from ._age_bin_processor import AgeBinProcessor
from ._AgeGroupSpecs import AgeGroupSpecs
from ._strata import stratum_codes, stratum_dims, stratum_labels

ContainerT = TypeVar("ContainerT")


def copy_container(container: ContainerT, data: Optional[pd.DataFrame] = None) -> ContainerT:
    """Return a shallow copy of a data container with its own DataFrame.

    The container's preprocessing is **not** re-run, so every column survives
    (single-year ages, auxiliary columns, repeat and ambiguous-count columns).
    Used instead of rebuilding containers, which keeps only the declared
    columns (dev plan D24), and to give estimators working copies so they never
    modify the user's containers (Problem 11).

    Parameters
    ----------
    container : ParticipantData, ContactData or PopulationData
        Container to copy.
    data : pd.DataFrame, optional
        Replacement data. Defaults to a copy of ``container.data``.
    """
    new = copy.copy(container)
    new.data = (container.data if data is None else data).copy()
    return new


def bin_ages(ages: pd.Series, age_group_specs: AgeGroupSpecs) -> pd.Series:
    """Bin single-year ages into left-closed intervals ``[left, right + 1)``.

    Ages outside ``[left[0], right[-1] + 1)`` become NaN.
    """
    bin_edges = list(age_group_specs.left) + [age_group_specs.right[-1] + 1]
    intervals = [
        pd.Interval(left=lo, right=hi + 1, closed="left")
        for lo, hi in zip(age_group_specs.left, age_group_specs.right)
    ]
    return pd.cut(ages, bins=bin_edges, right=False, labels=intervals)


def _min_counts_per_age_group(df: pd.DataFrame, strat_cols: Sequence[str]) -> pd.Series:
    """Smallest participant count of each age group across all strata."""
    categories = pd.Index(df["part_age_grp"].cat.categories)
    if strat_cols:
        counts = df.groupby(list(strat_cols) + ["part_age_grp"], observed=False).size()
        mins = counts.groupby("part_age_grp", observed=False).min()
    else:
        mins = df.groupby("part_age_grp", observed=False).size()
    return mins.reindex(categories, fill_value=0)


def _empty_age_groups(df: pd.DataFrame, strat_cols: Sequence[str]) -> List[pd.Interval]:
    """Age groups that are empty in at least one stratum."""
    mins = _min_counts_per_age_group(df, strat_cols)
    return mins[mins == 0].index.tolist()


@dataclass(frozen=True, eq=False)
class ParticipantCells:
    """Participants with a valid cell, and the cell of each.

    Attributes
    ----------
    part : pd.DataFrame
        Copy of the participant data, restricted to participants with a valid
        cell, with ``part_age_grp`` assigned (after any merging).
    age_group_specs : AgeGroupSpecs
        Final age groups (differs from the input only if groups were merged).
    cell : np.ndarray
        Cell code per row of ``part``: ``stratum * C + age_group``, where the
        stratum code follows the stratum-label order.
    n_cells : int
        Number of cells, ``K_part * C``.
    cell_labels : tuple of str
        Label of each cell code, e.g. ``"F_A, [0, 10)"``.
    ids : np.ndarray
        Participant ids, aligned with ``part`` and ``cell``.
    n_dropped : int
        Participants dropped because their age is outside the age groups.
    empty_age_groups : tuple of pd.Interval
        Age groups still empty in some stratum (only with ``strict=False``).
    merged : bool
        Whether empty age groups were merged.
    """

    part: pd.DataFrame
    age_group_specs: AgeGroupSpecs
    cell: np.ndarray
    n_cells: int
    cell_labels: Tuple[str, ...]
    ids: np.ndarray
    n_dropped: int
    empty_age_groups: Tuple[pd.Interval, ...]
    merged: bool

    @property
    def age_groups(self) -> List[pd.Interval]:
        """Final age-group intervals, in code order."""
        return list(self.part["part_age_grp"].cat.categories)


def build_participant_cells(
    part: pd.DataFrame,
    strat_cols: Sequence[str],
    age_group_specs: AgeGroupSpecs,
    adaptive_merge: bool = False,
    strict: bool = True,
) -> ParticipantCells:
    """Assign every participant to a cell (age group x participant strata).

    Steps, in order:

    1. **Bin** single-year ages (``part_age``) into ``age_group_specs``, always
       re-binning on a copy; pre-binned ``part_age_grp`` is kept as given.
    2. **Drop** participants whose age is outside the age groups, with one
       ``UserWarning`` giving the count (dev plan D23).
    3. **Merge** age groups that are empty in any stratum when
       ``adaptive_merge=True``. Otherwise raise (``strict=True``) or report them
       in ``empty_age_groups`` (``strict=False``, used by previews).
    4. **Encode** cells as ``stratum * C + age_group``.

    Parameters
    ----------
    part : pd.DataFrame
        Participant data (canonical ``part_*`` columns). Not modified.
    strat_cols : sequence of str
        Participant stratification columns (e.g. ``["part_sex"]``), in label
        order.
    age_group_specs : AgeGroupSpecs
        Age groups.
    adaptive_merge : bool, default False
        Merge age groups that are empty in any stratum.
    strict : bool, default True
        Raise ``ValueError`` on empty age groups that are not merged.

    Returns
    -------
    ParticipantCells
    """
    strat_cols = list(strat_cols)
    df = part.copy()

    # 1. Bin.
    if "part_age" in df.columns:
        df["part_age_grp"] = bin_ages(df["part_age"], age_group_specs)
    elif "part_age_grp" not in df.columns:
        raise ValueError(
            "Participant data must have either a 'part_age' or a 'part_age_grp' column."
        )

    # 2. Drop participants outside the age groups (D23).
    outside = df["part_age_grp"].isna().to_numpy()
    n_dropped = int(outside.sum())
    if n_dropped:
        lo, hi = age_group_specs.left[0], age_group_specs.right[-1]
        warnings.warn(
            f"Dropped {n_dropped} participant(s) whose age lies outside the age "
            f"groups [{lo}, {hi}]. They are excluded from the estimate, the "
            f"bootstrap and the weights.",
            UserWarning,
            stacklevel=2,
        )
        df = df.loc[~outside].copy()
    if df.empty:
        raise ValueError("No participants remain after assigning age groups.")

    # 3. Empty age groups.
    specs = age_group_specs
    merged = False
    empty = _empty_age_groups(df, strat_cols)
    if empty and adaptive_merge:
        warnings.warn(
            f"Found empty participant age group(s): {empty}. "
            f"Merging age groups to enable contact intensity estimation.",
            UserWarning,
            stacklevel=2,
        )
        while empty:
            mins = _min_counts_per_age_group(df, strat_cols)
            merged_intervals = AgeBinProcessor.merge_zero_groups(
                list(mins.index), mins.values
            )
            df = AgeBinProcessor.reassign_age_groups(
                df,
                age_grp_col="part_age_grp",
                merged_intervals=merged_intervals,
                new_group_col="part_age_grp",
            )
            specs = specs.replace(
                age_min=[iv.left for iv in merged_intervals],
                age_max=[iv.right - 1 for iv in merged_intervals],
            )
            merged = True
            empty = _empty_age_groups(df, strat_cols)
    elif empty and strict:
        strata_info = (
            f" across {len(strat_cols)} stratification variable(s)" if strat_cols else ""
        )
        raise ValueError(
            f"\nCannot estimate contact intensity matrix: found empty participant "
            f"age group(s){strata_info}: {empty}. "
            f"\nEmpty age groups cause division by zero when computing contact intensities. "
            f"\nPlease either:\n"
            f"  1. Set adaptive_merge=True to automatically merge empty age groups, or\n"
            f"  2. Use coarser age bins that avoid empty groups, or\n"
            f"  3. Collect more participant data to fill all age groups."
        )

    # 4. Encode cells: stratum-major, age group minor.
    age_groups = list(df["part_age_grp"].cat.categories)
    n_age = len(age_groups)
    k = stratum_codes(df, strat_cols)
    cell = k * n_age + df["part_age_grp"].cat.codes.to_numpy().astype(np.int64)
    n_strata = int(np.prod(stratum_dims(df, strat_cols))) if strat_cols else 1
    labels = stratum_labels(df, strat_cols)
    cell_labels = tuple(f"{s}, {iv}" for s in labels for iv in age_groups)

    return ParticipantCells(
        part=df,
        age_group_specs=specs,
        cell=cell,
        n_cells=n_strata * n_age,
        cell_labels=cell_labels,
        ids=df["id"].to_numpy(),
        n_dropped=n_dropped,
        empty_age_groups=tuple(empty if not adaptive_merge else []),
        merged=merged,
    )
