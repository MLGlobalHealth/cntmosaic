from ._base import ContactModel
from ._GenMix import GenMix
from ._AgeMixCC import AgeMixCC
from ._AgeMixFF import AgeMixFF
from ._AgeMixFC import AgeMixFC
from ._GenMixFF import GenMixFF
from ._GenMixFC import GenMixFC
from ._GenMixCC import GenMixCC
from ._vdKassteele import vdKassteele


def __getattr__(name: str):
    if name == "to_inference_data":
        from ._numpyro import to_inference_data

        return to_inference_data
    if name == "Prem":
        from .classical._Prem import Prem

        return Prem
    if name == "Prem2":
        from .classical._Prem2 import Prem2

        return Prem2
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Backward-compatibility re-exports
# SocialMix and related classes have moved to cntmosaic.models.classical.
# They are re-exported here so that existing code using
#   from cntmosaic.models import SocialMix
# continues to work without modification.
# ---------------------------------------------------------------------------
from .classical import BootstrapResults, PropMixFactorisation, SocialMix, SocialMixBootstrap

__all__ = [
    # Abstract base
    "ContactModel",
    # GenMix model family
    "GenMix",
    "AgeMixCC",
    "AgeMixFF",
    "AgeMixFC",
    "GenMixFF",
    "GenMixFC",
    "GenMixCC",
    # Other Bayesian models
    "Prem",
    "Prem2",
    "vdKassteele",
    # Classical models (re-exported for backward compatibility)
    "SocialMix",
    "SocialMixBootstrap",
    "BootstrapResults",
    "PropMixFactorisation",
    # Utilities
    "to_inference_data",
]
