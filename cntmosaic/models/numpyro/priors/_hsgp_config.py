"""
Configuration rules for Hilbert space approximate Gaussian processes (HSGP).

The rules follow Riutort-Mayol et al. (2023), Section 4.3.1, which relate the
boundary factor ``c``, the number of basis functions ``m`` and the lengthscale
``ell`` through kernel-specific constants. All quantities are expressed on the
standardised input scale, where ``S`` is the half-range of the centred inputs
along one axis and the approximation domain is ``[-L, L]`` with ``L = c * S``.

For a kernel with constants ``(a1, a2)``:

- ``c >= max(a1 * ell / S, 1.2)``: the boundary must be wide enough for the
  *largest* lengthscale.
- ``m >= a2 * c / (ell / S) = a2 * L / ell``: there must be enough basis
  functions for the *smallest* lengthscale.

These are the same rules used by PyMC's ``approx_hsgp_hyperparams`` and by
EpiNow2's ``gp_opts()``. See ``docs/dev_docs/hsgp_configuration.md``.

References
----------
- Riutort-Mayol, G., Bürkner, P.-C., Andersen, M. R., Solin, A., & Vehtari, A. (2023).
  Practical Hilbert space approximate Bayesian Gaussian processes for probabilistic
  programming. Statistics and Computing, 33(1), 17.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

#: Kernel constants ``(a1, a2)`` keyed by Matérn smoothness ``nu``.
#: ``float('inf')`` denotes the squared exponential kernel.
#: Matérn 7/2 reuses the Matérn 5/2 constants, which is conservative because
#: smoother kernels need fewer basis functions. Matérn 1/2 has no published
#: constants and is deliberately absent.
HSGP_KERNEL_CONSTANTS: Dict[float, Tuple[float, float]] = {
    float("inf"): (3.2, 1.75),
    3 / 2: (4.5, 3.42),
    5 / 2: (4.1, 2.65),
    7 / 2: (4.1, 2.65),
}

#: Minimum boundary factor recommended by Riutort-Mayol et al. (2023).
MIN_BOUNDARY_FACTOR: float = 1.2

#: Default number of basis functions per axis for each grid type.
#: Per-axis boundary factors already differ between grids through the half-range
#: ``S`` of each standardised axis; this table is where per-grid ``M`` defaults
#: can diverge if needed.
GRID_DEFAULTS: Dict[str, Dict[str, List[int]]] = {
    "age-age": {"M": [40, 40]},
    "diff-age": {"M": [40, 40]},
}


def kernel_constants(nu: float) -> Tuple[float, float]:
    """
    Return the constants ``(a1, a2)`` for smoothness ``nu``.

    Raises
    ------
    ValueError
        If no published constants exist for ``nu`` (e.g. Matérn 1/2).
    """
    try:
        return HSGP_KERNEL_CONSTANTS[nu]
    except KeyError:
        raise ValueError(
            f"No HSGP configuration rule is available for nu={nu}. "
            f"Supported values are {sorted(HSGP_KERNEL_CONSTANTS)}. "
            "Pass an explicit C and lenscale_min instead."
        ) from None


def halfnormal_quantile(sigma: float, q: float) -> float:
    """Quantile ``q`` of a HalfNormal(``sigma``) distribution."""
    return float(sigma * norm.ppf((1.0 + q) / 2.0))


def hsgp_boundary_factor(
    lenscale_max: float, S: ArrayLike, nu: float
) -> np.ndarray:
    """
    Minimum boundary factor per axis for the largest lengthscale.

    ``c_d = max(a1 * lenscale_max / S_d, 1.2)``.

    Parameters
    ----------
    lenscale_max : float
        Largest lengthscale the approximation must represent (standardised scale).
    S : float or sequence of float
        Half-range of the centred, standardised inputs along each axis.
    nu : float
        Smoothness of the Matérn kernel (``inf`` for squared exponential).

    Returns
    -------
    np.ndarray
        Boundary factor for each axis.
    """
    a1, _ = kernel_constants(nu)
    S = np.atleast_1d(np.asarray(S, dtype=float))
    return np.maximum(a1 * lenscale_max / S, MIN_BOUNDARY_FACTOR)


def hsgp_max_lenscale_per_axis(
    C: ArrayLike, S: ArrayLike, nu: float
) -> np.ndarray:
    """
    Largest lengthscale a given boundary factor supports, per axis.

    Inverts ``c_d >= a1 * ell_d / S_d``: ``ell_max_d = c_d * S_d / a1``.
    """
    a1, _ = kernel_constants(nu)
    L = np.atleast_1d(np.asarray(C, dtype=float)) * np.atleast_1d(
        np.asarray(S, dtype=float)
    )
    return L / a1


def hsgp_max_lenscale(
    C: ArrayLike, S: ArrayLike, nu: float
) -> float:
    """
    Largest shared (isotropic) lengthscale a given boundary factor supports.

    Inverts ``c_d >= a1 * ell / S_d`` and takes the binding axis:
    ``ell_max = min_d(c_d * S_d / a1)``.
    """
    return float(np.min(hsgp_max_lenscale_per_axis(C, S, nu)))


def hsgp_min_lenscale_per_axis(
    L: ArrayLike, M: ArrayLike, nu: float
) -> np.ndarray:
    """
    Smallest lengthscale ``M_d`` basis functions on ``[-L_d, L_d]`` can represent, per axis.

    Inverts ``m_d >= a2 * L_d / ell_d``: ``ell_min_d = a2 * L_d / M_d``. This is the
    rule for a tensor-product kernel, whose 1D factors are approximated separately.
    """
    _, a2 = kernel_constants(nu)
    L = np.atleast_1d(np.asarray(L, dtype=float))
    M = np.atleast_1d(np.asarray(M, dtype=float))
    return a2 * L / M


def hsgp_min_lenscale(
    L: ArrayLike, M: ArrayLike, nu: float
) -> float:
    """
    Smallest shared (isotropic) lengthscale ``M`` basis functions on ``[-L, L]`` can represent.

    Inverts ``m_d >= a2 * L_d / ell`` and takes the binding axis:
    ``ell_min = a2 * max_d(L_d / M_d)``. The lengthscale is shared, so the axis
    with the coarsest basis decides.
    """
    return float(np.max(hsgp_min_lenscale_per_axis(L, M, nu)))


def hsgp_min_num_basis(
    L: ArrayLike, lenscale_min: ArrayLike, nu: float
) -> List[int]:
    """
    Minimum number of basis functions per axis for the smallest lengthscale.

    ``m_d = ceil(a2 * L_d / lenscale_min_d)``. ``lenscale_min`` may be a scalar
    (shared lengthscale) or one value per axis.
    """
    _, a2 = kernel_constants(nu)
    L = np.atleast_1d(np.asarray(L, dtype=float))
    lmin = np.broadcast_to(np.asarray(lenscale_min, dtype=float), L.shape)
    # Small tolerance so that exact integers are not bumped up by round-off.
    return [int(math.ceil(a2 * Ld / ld - 1e-9)) for Ld, ld in zip(L, lmin)]
