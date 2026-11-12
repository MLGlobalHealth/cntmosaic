"""
Domain validation for participant survey data.

This module provides the validation logic for ParticipantData: checking
uniqueness of IDs, numeric constraints on ages, categorical structure of age
groups, and non-negativity of repeat and ambiguous-contact-count columns.
Keeping this logic here separates the validation concern from the
data-container and query API defined in ParticipantData.
"""

from typing import Optional

import pandas as pd


def validate_participant_data(
    data: pd.DataFrame,
    age_col: Optional[str],
    age_min_col: Optional[str],
    age_max_col: Optional[str],
    age_grp_col: Optional[str],
    repeat_col: Optional[str],
    amb_cnt_col: Optional[str],
) -> None:
    """
    Run all domain validation checks on a preprocessed participant DataFrame.

    Assumes the DataFrame has already been cleaned (no NaN rows) and that
    column names have been standardised by preprocess_participant_data.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed participant DataFrame with standardised column names.
    age_col : Optional[str]
        Original age column name; truthy when exact ages are in use.
    age_min_col : Optional[str]
        Original age_min column name; truthy when age ranges are in use.
    age_max_col : Optional[str]
        Original age_max column name; truthy when age ranges are in use.
    age_grp_col : Optional[str]
        Original age-group column name; truthy when age groups are in use.
    repeat_col : Optional[str]
        Original repeat-interview column name; truthy when present.
    amb_cnt_col : Optional[str]
        Original ambiguous-contact-count column name; truthy when present.

    Raises
    ------
    ValueError
        If duplicate participant IDs are found.
        If age / repeat / ambiguous-count values are invalid (negative or
        non-numeric).
    TypeError
        If age-group column is not categorical with pd.IntervalIndex categories.
    """
    _validate_unique_ids(data)

    if age_col:
        _validate_ages(data)

    if age_min_col and age_max_col:
        _validate_age_groups(data, "part_age_grp")

    if age_grp_col:
        _validate_age_groups(data, age_grp_col)

    if repeat_col:
        _validate_repeat(data)

    if amb_cnt_col:
        _validate_amb_cnt(data, amb_cnt_col)


# ---------------------------------------------------------------------------
# Internal helpers — one responsibility each
# ---------------------------------------------------------------------------


def _validate_unique_ids(data: pd.DataFrame) -> None:
    """Raise ValueError if any participant ID appears more than once."""
    duplicate_mask = data["id"].duplicated()
    if duplicate_mask.any():
        examples = data[duplicate_mask]["id"].head(5).tolist()
        n_duplicates = int(duplicate_mask.sum())
        raise ValueError(
            f"Found {n_duplicates} duplicate participant ID(s) in 'id' column.\n"
            f"  Examples of duplicates: {examples}\n"
            f"  Hint: each row should represent a unique participant."
        )


def _validate_ages(data: pd.DataFrame) -> None:
    """Raise ValueError if 'part_age' is non-numeric or contains negative values."""
    ages = data["part_age"]

    if not pd.api.types.is_numeric_dtype(ages):
        raise ValueError(
            f"Age column 'part_age' must contain numeric values.\n"
            f"  Current dtype: {ages.dtype}\n"
            f"  Hint: convert age to integer or float type."
        )

    if (ages < 0).any():
        negative_indices = data[ages < 0].index[:5].tolist()
        raise ValueError(
            f"Age column 'part_age' contains negative values.\n"
            f"  Rows with negative ages: {negative_indices}\n"
            f"  Values: {ages[ages < 0].head().tolist()}"
        )


def _validate_age_groups(data: pd.DataFrame, age_grp_col: str) -> None:
    """Raise TypeError if 'part_age_grp' is not categorical with IntervalIndex categories."""
    col = data["part_age_grp"]

    if not isinstance(col.dtype, pd.CategoricalDtype):
        raise TypeError(
            f"Column '{age_grp_col}' must be categorical with interval categories.\n"
            f"  Current type: {col.dtype}"
        )

    if not isinstance(col.cat.categories, pd.IntervalIndex):
        raise TypeError(
            f"Column '{age_grp_col}' must have pd.IntervalIndex categories.\n"
            f"  Got: {type(col.cat.categories)}"
        )


def _validate_repeat(data: pd.DataFrame) -> None:
    """Raise ValueError if 'part_repeat' is non-numeric or contains negative values."""
    repeats = data["part_repeat"]

    if not pd.api.types.is_numeric_dtype(repeats):
        raise ValueError(
            f"Repeat interview column 'part_repeat' must contain numeric values.\n"
            f"  Current dtype: {repeats.dtype}\n"
            f"  Hint: convert repeat interview to integer type."
        )

    if (repeats < 0).any():
        negative_indices = data[repeats < 0].index[:5].tolist()
        raise ValueError(
            f"Repeat interview column 'part_repeat' contains negative values.\n"
            f"  Rows with negative values: {negative_indices}\n"
            f"  Values: {repeats[repeats < 0].head().tolist()}"
        )


def _validate_amb_cnt(data: pd.DataFrame, amb_cnt_col: str) -> None:
    """Raise ValueError if amb_cnt_col is non-numeric or contains negative values."""
    grp_counts = data[amb_cnt_col]

    if not pd.api.types.is_numeric_dtype(grp_counts):
        raise ValueError(
            f"Group contact count column '{amb_cnt_col}' must contain numeric values.\n"
            f"  Current dtype: {grp_counts.dtype}\n"
            f"  Hint: convert group contact count to integer type."
        )

    if (grp_counts < 0).any():
        negative_indices = data[grp_counts < 0].index[:5].tolist()
        raise ValueError(
            f"Group contact count column '{amb_cnt_col}' contains negative values.\n"
            f"  Rows with negative values: {negative_indices}\n"
            f"  Values: {grp_counts[grp_counts < 0].head().tolist()}"
        )
