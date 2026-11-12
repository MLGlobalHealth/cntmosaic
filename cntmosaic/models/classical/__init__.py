"""
cntmosaic.models.classical
==========================

Contact matrix models — both classical frequentist (SocialMix) and Bayesian
(Prem) implementations.

The primary frequentist model is :class:`SocialMix`, which estimates
age-structured contact matrices from survey data as (weighted) means of the
participants' reported contacts.  :class:`Prem` implements the Bayesian methodology
from Prem et al. (2017).

Public API
----------
DeterministicContactModel
    Abstract base class for deterministic contact models.
SocialMix
    Survey-based contact intensity and rate matrix estimator.
SocialMixBootstrap
    Bootstrap uncertainty quantification for SocialMix.
BootstrapResults
    Container for bootstrap estimation results.
PropMixFactorisation
    Result of :meth:`SocialMix.prop_mix_factorisation`.
Prem
    Bayesian contact matrix estimator (Prem et al. 2017).
Prem2
    Aggregated-counts reformulation of Prem.
"""

from ._base import DeterministicContactModel
from ._Prem import Prem
from ._Prem2 import Prem2
from ._SocialMix import SocialMix
from ._socialmix_bootstrap import BootstrapResults, SocialMixBootstrap
from ._socialmix_utils import PropMixFactorisation

__all__ = [
    "DeterministicContactModel",
    "Prem",
    "Prem2",
    "SocialMix",
    "SocialMixBootstrap",
    "BootstrapResults",
    "PropMixFactorisation",
]
