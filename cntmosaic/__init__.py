from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cntmosaic")
except PackageNotFoundError:
    __version__ = "unknown"

from . import dataloader, datasets, models, preprocess, sim, vis, weights
from .dataloader import ContactSurveyLoader
from ._types import StratMode

__all__ = [
    # Primary entry points
    "ContactSurveyLoader",
    "StratMode",
    # Submodules
    "dataloader",
    "datasets",
    "models",
    "preprocess",
    "sim",
    "vis",
    "weights",
]
