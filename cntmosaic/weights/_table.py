"""
Table input and key matching, shared by CategoryWeights and the
post-stratification factors (dev plan: *Table input*, *Key matching rules*).

Tables (dict or DataFrame) are converted to a canonical, hashable, picklable
sorted tuple of ``(key, value)`` pairs. Participant values are matched to
table keys with explicit rules:

- missingness is decided first, with ``isna()`` on the raw column;
- integral floats are normalised to int (``3.0`` matches key ``3``); other
  types are matched as they are (the string ``"3"`` does not match ``3``);
- an unknown non-missing value is never treated as missing;
- checks use the observed values, never ``.cat.categories``.
"""

from __future__ import annotations

import numbers
from types import MappingProxyType
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

Pairs = Tuple[Tuple[Hashable, float], ...]


def norm_key(value: Any) -> Hashable:
    """Canonical form of a key or participant value.

    Booleans stay booleans, integers become ``int``, integral finite floats
    become ``int``, other floats stay ``float``; everything else is unchanged.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        f = float(value)
        return int(f) if np.isfinite(f) and f.is_integer() else f
    return value


def _sort_key(pair: Tuple[Hashable, Any]):
    return (type(pair[0]).__name__, repr(pair[0]))


def as_pairs(
    table: Union[Mapping, pd.DataFrame],
    *,
    default_key_col: Optional[str],
    key_col: Optional[str] = None,
    value_col: Optional[str] = None,
    what: str = "table",
    allow_zero: bool = False,
) -> Pairs:
    """Convert a dict or DataFrame to a sorted tuple of ``(key, value)`` pairs.

    Parameters
    ----------
    table : Mapping or pd.DataFrame
        ``key -> value`` mapping, or a DataFrame with a key and a value column.
    default_key_col : str or None
        Key column used when *key_col* is not given (``None``: *key_col* is
        required for DataFrames).
    key_col, value_col : str, optional
        DataFrame columns. *value_col* defaults to the only other column.
    what : str
        Name used in error messages (e.g. ``"weights"``, ``"targets"``).
    allow_zero : bool
        Accept values of 0 (post-stratification targets); otherwise values
        must be strictly positive.

    Raises
    ------
    TypeError
        If *table* is neither a mapping nor a DataFrame.
    ValueError
        On missing or ambiguous columns, duplicate keys, missing,
        non-numeric, non-finite, negative (or zero) values, or an empty table.
    """
    if isinstance(table, pd.DataFrame):
        kcol = key_col if key_col is not None else default_key_col
        if kcol is None:
            raise ValueError(f"{what}: key_col must be given for a DataFrame here.")
        if kcol not in table.columns:
            raise ValueError(
                f"{what}: key column '{kcol}' not in DataFrame columns {list(table.columns)}."
            )
        if value_col is None:
            others = [c for c in table.columns if c != kcol]
            if len(others) != 1:
                raise ValueError(
                    f"{what}: cannot tell which column holds the values "
                    f"(candidates {others}); pass value_col=..."
                )
            vcol = others[0]
        else:
            vcol = value_col
            if vcol not in table.columns:
                raise ValueError(
                    f"{what}: value column '{vcol}' not in DataFrame columns "
                    f"{list(table.columns)}."
                )
        if table[kcol].isna().any():
            raise ValueError(f"{what}: key column '{kcol}' has missing values.")
        keys = [norm_key(k) for k in table[kcol].tolist()]
        values = table[vcol].tolist()
    elif isinstance(table, Mapping):
        keys = [norm_key(k) for k in table.keys()]
        values = list(table.values())
    else:
        raise TypeError(f"{what} must be a dict or a pandas DataFrame; got {type(table).__name__}.")

    if not keys:
        raise ValueError(f"{what} is empty.")
    duplicated = sorted({repr(k) for k in keys if keys.count(k) > 1})
    if duplicated:
        raise ValueError(f"{what} has duplicate keys: {duplicated}.")

    out: List[Tuple[Hashable, float]] = []
    for k, v in zip(keys, values):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
            raise ValueError(f"{what}: missing value for key {k!r}.")
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
            raise ValueError(f"{what}: value for key {k!r} is not numeric ({v!r}).")
        f = float(v)
        if not np.isfinite(f):
            raise ValueError(f"{what}: value for key {k!r} is not finite ({f}).")
        if f < 0 or (f == 0 and not allow_zero):
            bound = "non-negative" if allow_zero else "strictly positive"
            raise ValueError(f"{what}: value for key {k!r} must be {bound}; got {f}.")
        out.append((k, f))
    return tuple(sorted(out, key=_sort_key))


def read_only(pairs: Iterable[Tuple[Hashable, Any]]) -> Mapping:
    """Read-only mapping view of pairs."""
    return MappingProxyType(dict(pairs))


def encode(
    values: pd.Series, key_to_code: Dict[Hashable, int]
) -> Tuple[np.ndarray, np.ndarray, Dict[Hashable, int]]:
    """Map participant values to integer codes.

    Returns
    -------
    codes : np.ndarray
        Code of each value; ``-1`` where the value is missing or unknown.
    missing : np.ndarray of bool
        Where the raw value is missing (``isna``).
    unknown : dict
        Unknown non-missing values and how often each occurs.
    """
    missing = values.isna().to_numpy()
    raw = values.astype(object).to_numpy()
    codes = np.full(len(raw), -1, dtype=np.int64)
    unknown: Dict[Hashable, int] = {}
    for i in np.flatnonzero(~missing):
        key = norm_key(raw[i])
        code = key_to_code.get(key)
        if code is None:
            unknown[key] = unknown.get(key, 0) + 1
        else:
            codes[i] = code
    return codes, missing, unknown


def describe_unknown(unknown: Dict[Hashable, int], limit: int = 10) -> str:
    """Short description of unknown values with counts."""
    items = sorted(unknown.items(), key=lambda kv: -kv[1])
    shown = ", ".join(f"{k!r} (x{n})" for k, n in items[:limit])
    more = "" if len(items) <= limit else f", and {len(items) - limit} more"
    return shown + more
