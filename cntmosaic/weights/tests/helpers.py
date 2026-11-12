"""Small helpers shared by the weights tests."""

import warnings

import numpy as np
import pandas as pd

from .._factor import BindContext


def make_ctx(part: pd.DataFrame, cell=None, n_cells=None) -> BindContext:
    """BindContext over *part* (needs an ``id`` column); one cell by default."""
    cell = np.zeros(len(part), dtype=np.int64) if cell is None else np.asarray(cell)
    n_cells = int(cell.max()) + 1 if n_cells is None else n_cells
    return BindContext(
        part=part.reset_index(drop=True),
        cell=cell,
        n_cells=n_cells,
        cell_labels=tuple(f"c{k}" for k in range(n_cells)),
        ids=part["id"].to_numpy(),
    )


def factor_values(factor, part: pd.DataFrame, cell=None, rows=None, n_cells=None):
    """Bind *factor* to *part* and evaluate it (on all rows by default)."""
    ctx = make_ctx(part, cell, n_cells)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bound = factor.bind(ctx)
    return bound(np.arange(len(part)) if rows is None else np.asarray(rows))
