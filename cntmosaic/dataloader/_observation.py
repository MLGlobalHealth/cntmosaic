"""
Free functions for building the observation grid and related arrays.

These functions were extracted from BaseLoader to make each concern
independently testable and to support the ContactSurveyLoader pipeline.
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._types import StratMode
from ._ColumnSpec import ColumnSpec
from ._utils import gaussian_smooth_by_group


def build_participant_counts(data: pd.DataFrame, col_spec: ColumnSpec) -> pd.DataFrame:
    """Aggregate participant counts (N) by age and stratification variables."""
    df_n = (
        data.groupby(col_spec.strat_vars_n, observed=False)
        .agg(N=(col_spec.id_col, "nunique"))
        .reset_index()
    )
    return df_n


def build_contact_offsets(
    data: pd.DataFrame, col_spec: ColumnSpec, smooth: bool
) -> pd.DataFrame:
    """
    Compute ambiguous contact offsets (V) by age and stratification variables.

    Returns a DataFrame with column 'V' where V = 1 when no ambiguous contacts
    are present, or V = 1 - z/(z+y) otherwise.
    """
    # No ambiguous contacts: V = 1.0 everywhere
    if col_spec.z is None:
        df_V = build_participant_counts(data, col_spec).drop(columns=["N"])
        df_V["V"] = 1.0
        return df_V

    df_z = (
        data[[col_spec.id_col] + col_spec.strat_vars_n + [col_spec.z]]
        .drop_duplicates()
        .groupby(col_spec.strat_vars_n, observed=False)[col_spec.z]
        .sum()
        .reset_index()
    )
    df_yz = (
        data[[col_spec.id_col] + col_spec.strat_vars_n + [col_spec.y]]
        .drop_duplicates()
        .groupby(col_spec.strat_vars_n, observed=False)[col_spec.y]
        .sum()
        .reset_index()
    )
    df_V = df_yz.merge(df_z, on=col_spec.strat_vars_n, how="left")
    df_V["V"] = 1 - df_V[col_spec.z] / (df_V[col_spec.z] + df_V[col_spec.y])
    df_V["V"] = np.where(df_V["V"] == 0, 1.0 / (df_V[col_spec.z] + 1.0), df_V["V"])
    df_V.fillna({"V": 1.0}, inplace=True)
    df_V = df_V.drop(columns=[col_spec.z, col_spec.y])

    if smooth:
        smooth_group_vars = [
            var for var in col_spec.strat_vars_n if var != col_spec.part_age
        ]
        if len(smooth_group_vars) > 0:
            df_V = gaussian_smooth_by_group(
                df_V,
                group_by=smooth_group_vars,
                target="V",
                cv=True,
                sort_by=col_spec.part_age,
            )
        else:
            df_V = gaussian_smooth_by_group(
                df_V, group_by=None, target="V", cv=True, sort_by=col_spec.part_age
            )
        df_V["V"] = df_V["V_gs"]

    return df_V


def build_contact_counts(
    data: pd.DataFrame,
    col_spec: ColumnSpec,
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate contact counts (y) by age and stratification variables.

    Parameters
    ----------
    data : pd.DataFrame
        Merged participant-contact DataFrame.
    col_spec : ColumnSpec
        Column mapping specification.
    weight_col : str or None, optional
        Name of a column in ``data`` containing per-row survey weights. When
        provided and the column is present, each contact record's ``y`` value
        is multiplied by its weight before summing, producing real-valued
        (weighted) contact counts suitable for quasi-likelihood inference.
        When ``None`` (default) a plain unweighted integer sum is computed.
    """
    if weight_col is not None and weight_col in data.columns:
        data = data.copy()
        data["_weighted_y"] = data[col_spec.y] * data[weight_col]
        df_y = (
            data.groupby(col_spec.strat_vars_y, observed=False)
            .agg({"_weighted_y": "sum"})
            .reset_index()
            .rename(columns={"_weighted_y": col_spec.y})
        )
    else:
        df_y = (
            data.groupby(col_spec.strat_vars_y, observed=False)
            .agg({col_spec.y: "sum"})
            .reset_index()
        )
    return df_y


def build_observation_grid(
    data: pd.DataFrame,
    col_spec: ColumnSpec,
    age_min: int,
    age_max: int,
    df_n: pd.DataFrame,
    df_y: pd.DataFrame,
    df_V: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full Cartesian observation grid by merging participant counts,
    contact counts, and offsets, then zero-filling missing cells.

    Parameters
    ----------
    data : pd.DataFrame
        Merged participant-contact DataFrame (used for category information).
    col_spec : ColumnSpec
        Column mapping specification.
    age_min, age_max : int
        Aligned age range bounds.
    df_n, df_y, df_V : pd.DataFrame
        Pre-computed participant counts, contact counts, and offsets.
    """
    unique_coords = {var: data[var].unique() for var in col_spec.strat_vars_y}
    if col_spec.part_age_grp:
        unique_coords[col_spec.part_age] = data[col_spec.part_age_grp].cat.categories
    else:
        unique_coords[col_spec.part_age] = np.arange(age_min, age_max + 1, dtype=int)

    if col_spec.cnt_age:
        unique_coords[col_spec.cnt_age] = np.arange(age_min, age_max + 1, dtype=int)
    elif col_spec.cnt_age_grp:
        unique_coords[col_spec.cnt_age_grp] = data[col_spec.cnt_age_grp].cat.categories

    index = pd.MultiIndex.from_product(
        unique_coords.values(), names=unique_coords.keys()
    )  # type: ignore
    df_full = pd.DataFrame(list(index), columns=unique_coords.keys())
    df_full = pd.merge(df_full, df_y, on=col_spec.strat_vars_y, how="left")
    df_full = pd.merge(df_full, df_n, on=col_spec.strat_vars_n, how="left")
    df_full = pd.merge(df_full, df_V, on=col_spec.strat_vars_n, how="left")

    # Restore categorical dtypes after merges
    if col_spec.cnt_age_grp:
        df_full[col_spec.cnt_age_grp] = pd.Categorical(
            df_full[col_spec.cnt_age_grp],
            categories=data[col_spec.cnt_age_grp].cat.categories,
            ordered=True,
        )
    if col_spec.part_age_grp:
        df_full[col_spec.part_age_grp] = pd.Categorical(
            df_full[col_spec.part_age_grp],
            categories=data[col_spec.part_age_grp].cat.categories,
            ordered=True,
        )
    if col_spec.part_strat_vars:
        for var in col_spec.part_strat_vars:
            categories = data[var].cat.categories
            df_full[var] = pd.Categorical(
                df_full[var], categories=categories, ordered=True
            )
    if col_spec.cnt_strat_vars:
        for var in col_spec.cnt_strat_vars:
            categories = data[var].cat.categories
            df_full[var] = pd.Categorical(
                df_full[var], categories=categories, ordered=True
            )

    df_full = df_full.dropna(subset=["N"])
    df_full = df_full[df_full["N"] > 0]
    df_full["V"] = df_full["V"].fillna(1.0)
    df_full["log_V"] = np.where(df_full["V"] > 0, np.log(df_full["V"]), 0.0)
    df_full["y"] = df_full["y"].fillna(0)

    return df_full


def construct_log_P(
    pop_df: pd.DataFrame,
    col_spec: ColumnSpec,
    strat_modes: Optional[Dict[str, StratMode]] = None,
) -> NDArray:
    """
    Construct log contact population sizes (log_P) from population data.

    ``log_P[pix, b]`` is the log population of contacts with age b in contact
    stratum ``pix``.  Only FULL-mode variables stratify the contact side, so
    when ``strat_modes`` is given the population is summed over every other
    variable, giving one row per combination of FULL-mode variables, ordered
    to match ``infer_strat_pixs`` (first variable slowest).  With no FULL-mode
    variables (PARTIAL-only or unstratified models) the single row is the
    total population.

    If ``strat_modes`` is None, one row per combination of all population
    stratification variables is returned (legacy behaviour).

    Returns shape (1, A) for a single row or (K, A) for K contact strata.
    A may be fine-age count or coarse-group count depending on population resolution.
    """
    age_dim = col_spec.pop_age or col_spec.pop_age_grp

    if strat_modes is not None:
        full_vars = [v for v, mode in strat_modes.items() if mode == StratMode.FULL]
        if not full_vars:
            P = (
                pop_df.groupby(age_dim, observed=False)[col_spec.P]
                .sum()
                .sort_index()
                .to_numpy()[np.newaxis, :]
            )  # shape (1, A) or (1, B)
        else:
            P = (
                pop_df.groupby(full_vars + [age_dim], observed=False)[col_spec.P]
                .sum()
                .unstack(age_dim)
                .fillna(1)
                .to_numpy()
            )  # shape (K, A) or (K, B)
        return np.log(P)

    if col_spec.pop_strat_vars:
        P = (
            pop_df.pivot(
                index=col_spec.pop_strat_vars,
                columns=age_dim,
                values=col_spec.P,
            )
            .fillna(1)
            .to_numpy()
        )  # shape (K, A) or (K, B)
    else:
        P = pop_df[col_spec.P].to_numpy()[np.newaxis, :]  # shape (1, A) or (1, B)

    return np.log(P)


def align_age_range(
    data: pd.DataFrame,
    pop_df: pd.DataFrame,
    col_spec: ColumnSpec,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Align the participant/contact age range to the population age range.

    Returns a (possibly filtered) copy of data plus the aligned min and max ages.
    Emits UserWarning when the sample age range differs from the population age range.
    """
    if col_spec.part_age_grp:
        cats = data[col_spec.part_age].cat.categories
        part_min_age = int(cats.left.min())
        part_max_age = int(cats.right.max() - 1)
    else:
        part_min_age = int(data[col_spec.part_age].min())
        part_max_age = int(data[col_spec.part_age].max())

    if col_spec.cnt_age:
        cnt_min_age = int(data[col_spec.cnt_age].min())
        cnt_max_age = int(data[col_spec.cnt_age].max())
    else:
        cnt_min_age = int(data[col_spec.cnt_age_grp].min().left)
        cnt_max_age = int(data[col_spec.cnt_age_grp].max().right - 1)

    if col_spec.pop_age:
        pop_min_age = int(pop_df[col_spec.pop_age].min())
        pop_max_age = int(pop_df[col_spec.pop_age].max())
    elif col_spec.pop_age_grp:
        grp_cats = pop_df[col_spec.pop_age_grp].cat.categories
        pop_min_age = int(grp_cats.left.min())
        pop_max_age = int(grp_cats.right.max()) - 1

    sample_min_age = min(part_min_age, cnt_min_age)
    sample_max_age = max(part_max_age, cnt_max_age)

    if sample_min_age != pop_min_age:
        warnings.warn(
            f"Sample minimum age ({sample_min_age}) differs from population minimum age "
            f"({pop_min_age}). Filtering sample data to match population (age >= {pop_min_age}).",
            UserWarning,
            stacklevel=3,
        )
        if col_spec.part_age_grp:
            mask = data[col_spec.part_age].apply(lambda iv: iv.left) >= pop_min_age
            data = data[mask].copy()
        else:
            data = data[data[col_spec.part_age] >= pop_min_age].copy()
        if data.empty:
            raise ValueError(
                f"After filtering to age >= {pop_min_age}, no data remains. "
                "Check age range compatibility."
            )
        min_age = pop_min_age
    else:
        min_age = sample_min_age

    if sample_max_age != pop_max_age:
        warnings.warn(
            f"Sample maximum age ({sample_max_age}) differs from population maximum age "
            f"({pop_max_age}). Using population maximum age ({pop_max_age}) for analysis.",
            UserWarning,
            stacklevel=3,
        )
        max_age = pop_max_age
    else:
        max_age = sample_max_age

    return data, min_age, max_age
