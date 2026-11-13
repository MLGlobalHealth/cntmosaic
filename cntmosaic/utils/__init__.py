from ._AgeGroupSpecs import AgeGroupSpecs
from ._AgeBins import AgeBins  # backward-compat alias
from ._utils import (
    pixilate,
    depixilate,
)
from ._matrix_utils import (
    symm_from_tril_ix_col,
    tril_ix_col,
)

__all__ = [
    "AgeGroupSpecs",
    "AgeBins",  # backward-compat alias
    "pixilate",
    "depixilate",
    "symm_from_tril_ix_col",
    "tril_ix_col",
]
