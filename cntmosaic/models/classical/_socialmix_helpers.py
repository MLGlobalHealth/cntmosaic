"""
Shared stateless helpers for classical contact models.

These functions are used by SocialMix, SocialMixValidator, SocialMixBootstrap,
and Prem to avoid duplicated logic across those classes.
"""

import itertools
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...utils import AgeGroupSpecs
from ...utils._strata import (  # noqa: F401  (re-exported)
    ravel_stratum_codes,
    stratum_codes,
    stratum_dims,
)


def infer_strat_mode(strat_vars_part: List[str], strat_vars_cnt: List[str]) -> str:
    """Infer stratification mode from participant and contact stratification variables.

    Returns
    -------
    str
        One of ``"single"``, ``"partial"``, ``"full"``, or ``"mixed"``.
    """
    if not strat_vars_part and not strat_vars_cnt:
        return "single"
    if not strat_vars_cnt:
        return "partial"
    if set(strat_vars_part) == set(strat_vars_cnt):
        return "full"
    return "mixed"


def assign_age_groups(
    df: pd.DataFrame,
    age_col: str,
    age_group_specs: AgeGroupSpecs,
    group_col: str,
) -> pd.DataFrame:
    """Bin a column of raw ages into coarse age groups defined by *age_group_specs*.

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame (not mutated; a copy is returned).
    age_col : str
        Name of the column containing raw integer ages.
    age_group_specs : AgeGroupSpecs
        Age stratification bins.
    group_col : str
        Name of the output column to add with ``pd.Interval`` labels.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with *group_col* added.
    """
    df = df.copy()
    bin_edges = age_group_specs.left + [age_group_specs.right[-1] + 1]
    intervals = [
        pd.Interval(left=l, right=r + 1, closed="left")
        for l, r in zip(age_group_specs.left, age_group_specs.right)
    ]
    df[group_col] = pd.cut(df[age_col], bins=bin_edges, right=False, labels=intervals)
    return df


def _side_labels(strat_vars: List[str], data, prefix: str) -> List[str]:
    """Return bare stratum labels for one side (participant or contact).

    Parameters
    ----------
    strat_vars : list of str
        Variable names (without prefix).
    data : ParticipantData or ContactData
        Container whose ``.data`` has columns ``"{prefix}_{var}"``.
    prefix : str
        ``"part"`` or ``"cnt"``.

    Returns
    -------
    list of str
        e.g. ``["M", "F"]`` or ``["All"]``.
    """
    if not strat_vars:
        return ["All"]
    category_lists = [
        data.data[f"{prefix}_{v}"].cat.categories.tolist() for v in strat_vars
    ]
    return [
        "_".join(str(x) for x in combo)
        for combo in itertools.product(*category_lists)
    ]


def population_by_contact_stratum(
    df_pop: pd.DataFrame,
    strat_vars_cnt: List[str],
    cnt_df: pd.DataFrame,
    n_age: int,
    age_col: str = "age_grp",
    size_col: str = "P",
) -> Optional[NDArray]:
    """Population sizes arranged as ``(K_cnt, D)`` in contact-stratum label order.

    Each population row is matched to a contact stratum **by category value**,
    so the result is correct whatever the order of the stratification variables
    in the population data and whatever the order of their categories.
    Population variables that are not contact stratification variables are
    summed over.

    Parameters
    ----------
    df_pop : pd.DataFrame
        Population data with a categorical *age_col* and a *size_col*.
    strat_vars_cnt : list of str
        Contact stratification variables (without prefix), in label order.
    cnt_df : pd.DataFrame
        Contact data with columns ``"cnt_{var}"``; its categories define the
        contact strata.
    n_age : int
        Number of age groups ``D``.

    Returns
    -------
    np.ndarray or None
        Array of shape ``(K_cnt, D)``, or ``None`` if the population data does
        not contain every contact stratification variable (the caller then
        decides how to proceed).
    """
    if any(v not in df_pop.columns for v in strat_vars_cnt):
        return None

    cnt_cols = [f"cnt_{v}" for v in strat_vars_cnt]
    dims = stratum_dims(cnt_df, cnt_cols)
    grouped = (
        df_pop.groupby(strat_vars_cnt + [age_col], observed=False)[size_col]
        .sum()
        .reset_index()
    )

    # Re-encode each population variable with the contact categories.
    code_arrays = [
        pd.Categorical(
            grouped[v].astype(object), categories=cnt_df[c].cat.categories
        ).codes
        for v, c in zip(strat_vars_cnt, cnt_cols)
    ]
    # Population categories that are not contact categories have no stratum.
    keep = np.all([codes >= 0 for codes in code_arrays], axis=0)

    k = ravel_stratum_codes([codes[keep] for codes in code_arrays], dims)
    d = grouped[age_col].cat.codes.to_numpy()[keep]
    values = grouped[size_col].to_numpy(dtype=np.float64)[keep]

    P = np.zeros((int(np.prod(dims)), n_age), dtype=np.float64)
    P[k, d] = values
    return P


def participant_labels(strat_vars_part: List[str], part_data) -> List[str]:
    """Bare source-side stratum labels, e.g. ``["M", "F"]`` or ``["All"]``."""
    return _side_labels(strat_vars_part, part_data, "part")


def contact_labels(strat_vars_cnt: List[str], cnt_data) -> List[str]:
    """Bare target-side stratum labels, e.g. ``["M", "F"]`` or ``["All"]``."""
    return _side_labels(strat_vars_cnt, cnt_data, "cnt")


def create_stratum_labels(
    K_part: int,
    K_cnt: int,
    strat_vars_part: List[str],
    strat_vars_cnt: List[str],
    part_data,
    cnt_data,
) -> List[str]:
    """Build ``"source->target"`` stratum labels in the same order as array indices.

    Parameters
    ----------
    K_part, K_cnt : int
        Number of participant / contact strata.
    strat_vars_part, strat_vars_cnt : list of str
        Stratification variable names (without prefix).
    part_data, cnt_data : ParticipantData / ContactData
        Container objects whose ``.data`` attribute has the categorical columns.

    Returns
    -------
    list of str
        Labels such as ``["All->All"]``, ``["M->All", "F->All"]``,
        ``["M->M", "M->F", "F->M", "F->F"]``.
    """
    part_combos = participant_labels(strat_vars_part, part_data)
    cnt_combos = contact_labels(strat_vars_cnt, cnt_data)
    return [f"{p}->{c}" for p in part_combos for c in cnt_combos]


def apply_reciprocity(
    cint_dict: Dict[str, NDArray],
    strat_mode: str,
    K_cnt: int,
    P: Optional[NDArray],
) -> Dict[str, NDArray]:
    """Apply population-weighted reciprocity adjustment to contact intensity matrices.

    For single/full stratification modes, enforces ``M[c,d]*P[c] == M[d,c]*P[d]``.
    No-op for partial and mixed modes (returns *cint_dict* unchanged).

    Parameters
    ----------
    cint_dict : dict
        Raw contact intensity matrices keyed by ``"source->target"`` label.
    strat_mode : str
        One of ``"single"``, ``"partial"``, ``"full"``, ``"mixed"``.
    K_cnt : int
        Number of contact strata (used to index *P* rows in full mode).
    P : NDArray or None
        Population sizes.  Shape ``(D,)`` for single/partial, ``(K_cnt, D)``
        for full.  Must not be ``None`` when *strat_mode* is ``"single"`` or
        ``"full"``.

    Returns
    -------
    dict
        Reciprocity-adjusted matrices (new arrays; originals are not mutated).
    """
    if strat_mode not in ("single", "full"):
        return cint_dict

    result: Dict[str, NDArray] = {}

    if strat_mode == "single":
        m = cint_dict["All->All"]
        numerator = m * P[:, np.newaxis] + m.T * P[np.newaxis, :]
        m_adj = numerator / (2 * P[:, np.newaxis])
        m_adj = np.nan_to_num(m_adj, nan=0.0, posinf=0.0, neginf=0.0)
        result["All->All"] = m_adj
        return result

    # Full stratification
    labels = list(cint_dict.keys())
    for idx, label in enumerate(labels):
        s_str, t_str = label.split("->")
        k_s = idx // K_cnt
        k_t = idx % K_cnt
        m_st = cint_dict[label]

        P_s = P[k_s, :] if P.ndim == 2 else P
        P_t = P[k_t, :] if P.ndim == 2 else P

        if s_str == t_str:
            # Within-stratum
            m_adj = (m_st * P_s[:, np.newaxis] + m_st.T * P_s[np.newaxis, :]) / (
                2 * P_s[:, np.newaxis]
            )
        else:
            # Between-stratum
            reverse_label = f"{t_str}->{s_str}"
            m_ts = cint_dict.get(reverse_label, np.zeros_like(m_st))
            P_s_safe = np.where(P_s > 0, P_s, 1.0)
            m_adj = (m_st + m_ts.T * P_t[np.newaxis, :] / P_s_safe[:, np.newaxis]) / 2
            m_adj = np.where(P_s[:, np.newaxis] > 0, m_adj, m_st)

        m_adj = np.nan_to_num(m_adj, nan=0.0, posinf=0.0, neginf=0.0)
        result[label] = m_adj

    return result


def validate_shared_strat_vars(strat_vars_shared: List[str], part_data, cnt_data) -> None:
    """Validate and align categories for variables present in both participant and contact data.

    Ensures both sides have identical category sets. When sets match but ordering
    differs, the contact side is reordered to match the participant side.

    Raises
    ------
    ValueError
        If both sides have different category values for any shared variable.
    """
    for var in strat_vars_shared:
        col_part = f"part_{var}"
        col_cnt = f"cnt_{var}"

        part_col = part_data.data[col_part]
        cnt_col = cnt_data.data[col_cnt]

        if not hasattr(part_col, "cat"):
            part_data.data[col_part] = part_col.astype("category")
            part_col = part_data.data[col_part]
        if not hasattr(cnt_col, "cat"):
            cnt_data.data[col_cnt] = cnt_col.astype("category")
            cnt_col = cnt_data.data[col_cnt]

        part_cats = list(part_col.cat.categories)
        cnt_cats = list(cnt_col.cat.categories)

        if set(part_cats) != set(cnt_cats):
            only_part = set(part_cats) - set(cnt_cats)
            only_cnt = set(cnt_cats) - set(part_cats)
            raise ValueError(
                f"Shared stratification variable '{var}' has different categories:\n"
                f"  Participant side: {part_cats}\n"
                f"  Contact side: {cnt_cats}\n"
                f"  Only in participants: {sorted(only_part) if only_part else 'None'}\n"
                f"  Only in contacts: {sorted(only_cnt) if only_cnt else 'None'}\n"
                f"For shared variables, both sides must have identical categories."
            )

        if part_cats != cnt_cats:
            cnt_data.data[col_cnt] = (
                cnt_data.data[col_cnt].cat.reorder_categories(part_cats, ordered=cnt_col.cat.ordered)
            )


def row_stratum_labels(
    df: pd.DataFrame, part_cols: List[str], cnt_cols: List[str]
) -> pd.Series:
    """Assign a ``"source->target"`` stratum label to each row in *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated data frame with stratification columns.
    part_cols : list of str
        Participant-side column names (e.g. ``["part_sex"]``).
    cnt_cols : list of str
        Contact-side column names (e.g. ``["cnt_sex"]``).

    Returns
    -------
    pd.Series
        Per-row label such as ``"M->F"`` or ``"All->All"``.
    """
    def _side(cols: List[str]) -> pd.Series:
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.Series(["All"] * len(df), index=df.index)
        result = df[present[0]].astype(str)
        for col in present[1:]:
            result = result + "_" + df[col].astype(str)
        return result

    return _side(part_cols) + "->" + _side(cnt_cols)
