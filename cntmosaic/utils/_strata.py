"""
Stratum codes shared by the estimators, the bootstrap and participant cells.

Composite stratum codes follow ``itertools.product`` order over the categories
of each stratification column: the **first column is the most significant**.
This is the order of the stratum labels, so a composite code ``k`` always
belongs to label ``labels[k]``. Each dimension is the number of categories
(including unused ones), not ``nunique()``.
"""

from __future__ import annotations

import itertools
from typing import List, Sequence

import numpy as np
import pandas as pd


def stratum_dims(df: pd.DataFrame, cols: Sequence[str]) -> List[int]:
    """Number of categories of each (categorical) stratification column."""
    return [len(df[c].cat.categories) for c in cols]


def ravel_stratum_codes(
    code_arrays: Sequence[np.ndarray], dims: Sequence[int]
) -> np.ndarray:
    """Combine per-variable category codes into one composite stratum code.

    Parameters
    ----------
    code_arrays : sequence of np.ndarray
        One integer code array per stratification variable (no negative codes).
    dims : sequence of int
        Number of categories of each variable, in the same order.

    Returns
    -------
    np.ndarray
        Composite codes in ``0 .. prod(dims) - 1``; first variable most
        significant.
    """
    if len(code_arrays) == 0:
        raise ValueError("ravel_stratum_codes needs at least one code array.")
    return np.ravel_multi_index(
        [np.asarray(c, dtype=np.int64) for c in code_arrays], dims=tuple(dims)
    ).astype(np.int64)


def stratum_codes(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Composite stratum code for every row of *df*, in label order.

    Returns zeros if *cols* is empty (a single stratum).
    """
    if len(cols) == 0:
        return np.zeros(len(df), dtype=np.int64)
    return ravel_stratum_codes(
        [df[c].cat.codes.to_numpy() for c in cols], stratum_dims(df, cols)
    )


def stratum_labels(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    """Stratum labels in code order, e.g. ``["F_A", "F_B", ...]`` or ``["All"]``."""
    if len(cols) == 0:
        return ["All"]
    return [
        "_".join(str(x) for x in combo)
        for combo in itertools.product(*(df[c].cat.categories for c in cols))
    ]
