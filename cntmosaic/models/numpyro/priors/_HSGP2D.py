from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax import vmap
from jax.typing import ArrayLike
from numpyro import distributions as dist
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_matern

from ..._utils import age_age_grid, diff_age_age_grid, symm_from_tril_ix_row, tril_ix_row
from ._hsgp_config import (
    GRID_DEFAULTS,
    HSGP_KERNEL_CONSTANTS,
    MIN_BOUNDARY_FACTOR,
    halfnormal_quantile,
    hsgp_boundary_factor,
    hsgp_max_lenscale,
    hsgp_max_lenscale_per_axis,
    hsgp_min_lenscale,
    hsgp_min_lenscale_per_axis,
    hsgp_min_num_basis,
)
from ._Prior2D import Prior2D


ALLOWED_KERNELS = ("product", "isotropic")


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_positive_pair(x) -> bool:
    return (
        isinstance(x, (list, tuple))
        and len(x) == 2
        and all(_is_number(v) and v > 0 for v in x)
    )


def _as_pair(x) -> np.ndarray:
    """Broadcast a scalar or a length-2 sequence to a float array of shape (2,)."""
    return np.broadcast_to(np.asarray(x, dtype=float), (2,)).copy()


def validate_init_params(
    nu: float,
    C: Union[str, float, List[float]],
    M: Optional[Union[int, List[int]]],
    lenscale_scale: Union[float, List[float]] = 0.5,
    lenscale_max: Optional[Union[float, List[float]]] = None,
    lenscale_min: Optional[Union[float, List[float]]] = None,
    kernel: str = "product",
) -> None:
    """
    Validate initialization parameters for HSGP2D prior.

    Parameters
    ----------
    nu : float
        The smoothness parameter of the Matern kernel.
    C : 'auto', float or list of float
        The boundary inflation factors for the HSGP approximation.
    M : int, list[int] or None
        The number of eigenvalues and eigenfunctions to use in the HSGP approximation.
    lenscale_scale : float or list of float
        Scale of the (truncated) half-normal lengthscale prior. A list of two values
        (one per axis) is allowed for ``kernel='product'``.
    lenscale_max, lenscale_min : float, list of float or None
        Optional explicit bounds of the lengthscale prior. Lists of two values (one
        per axis) are allowed for ``kernel='product'``.
    kernel : {'product', 'isotropic'}
        Covariance structure of the 2D GP.

    Raises
    ------
    ValueError
        If nu is not one of [1/2, 3/2, 5/2, 7/2], if kernel is unknown, if C, M or
        the lengthscale settings are not positive, if per-axis lengthscale settings
        are used with the isotropic kernel, or if nu=1/2 is used without explicit C
        and lenscale_min.
    TypeError
        If C is not 'auto', a number or a list of two numbers, or if M is not
        None, an int or a list of two ints.
    """
    if kernel not in ALLOWED_KERNELS:
        raise ValueError(f"kernel must be one of {list(ALLOWED_KERNELS)}, got {kernel!r}.")

    if nu not in [1 / 2, 3 / 2, 5 / 2, 7 / 2]:
        raise ValueError("nu must be one of [1/2, 3/2, 5/2, 7/2]")

    if isinstance(C, str):
        if C != "auto":
            raise ValueError("C must be 'auto', a positive number or a list of two.")
    elif _is_number(C):
        if C <= 0:
            raise ValueError("C must be positive.")
    elif isinstance(C, list):
        if len(C) != 2 or not all(_is_number(c) for c in C) or any(c <= 0 for c in C):
            raise ValueError("C must be a list of two positive floats.")
    else:
        raise TypeError("C must be 'auto', a float or a list of two floats.")

    if M is None:
        pass
    elif isinstance(M, int) and not isinstance(M, bool):
        if M <= 0:
            raise ValueError("M must be a positive integer.")
    elif isinstance(M, list):
        if len(M) != 2 or any(m <= 0 for m in M):
            raise ValueError("M must be a list of two positive integers.")
    else:
        raise TypeError("M must be None, an integer or a list of two integers.")

    per_axis_ok = kernel == "product"
    settings = (
        ("lenscale_scale", lenscale_scale, False),
        ("lenscale_max", lenscale_max, True),
        ("lenscale_min", lenscale_min, True),
    )
    for name, value, optional in settings:
        if value is None and optional:
            continue
        if _is_number(value) and value > 0:
            continue
        if _is_positive_pair(value):
            if not per_axis_ok:
                raise ValueError(
                    f"{name} must be a single number for kernel='isotropic', which "
                    "has one shared lengthscale. Use kernel='product' for per-axis values."
                )
            continue
        expected = "a positive number" + (" or a list of two" if per_axis_ok else "")
        raise ValueError(f"{name} must be {expected}" + (" or None." if optional else "."))

    if lenscale_min is not None and lenscale_max is not None:
        if np.any(_as_pair(lenscale_min) >= _as_pair(lenscale_max)):
            raise ValueError("lenscale_min must be smaller than lenscale_max.")

    if nu == 1 / 2 and (isinstance(C, str) or lenscale_min is None):
        raise ValueError(
            "No published HSGP configuration rule exists for nu=1/2. "
            "Pass an explicit C and lenscale_min."
        )


class HSGP2D(Prior2D):
    """
    2D Hilbert space approximate Gaussian process (HSGP) prior for contact matrix estimation.

    This prior class implements flexible spatial smoothing for contact matrices using
    a 2D HSGP with a Matern kernel. It provides a computationally efficient
    alternative to traditional Gaussian process priors by approximating the GP using a finite
    set of eigenfunctions derived from the Laplacian operator.

    Kernels
    -------
    - **product** (default): tensor-product Matern kernel
      ``k(x, x') = sigma_f^2 k_nu(|x_1 - x_1'| / l_1) k_nu(|x_2 - x_2'| / l_2)``
      with one lengthscale per grid axis. The spectral weights are the outer product
      of the two 1D Matern spectral densities, so ``K = K_1 (x) K_2`` on a product
      grid. On the 'diff-age' grid, ``l_age > l_age_diff`` smooths along cohorts
      while keeping the assortative diagonal sharp.
    - **isotropic**: Matern kernel in the Euclidean distance on the standardised grid,
      ``k(x, x') = sigma_f^2 k_nu(||x - x'|| / l)``, with a single shared lengthscale.
      This is the 2D spectral density used by ``numpyro.contrib.hsgp`` and the
      behaviour of cntmosaic <= 0.5.0; it is kept to reproduce earlier fits.

    Both kernels use the same basis functions and are evaluated in Kronecker form,
    ``F = PHI_1 (sqrt(W) * Z) PHI_2^T`` on the rectangular grid spanned by the two
    axes, followed by a gather of the required grid cells. The full
    ``N x (M_1 M_2)`` basis matrix is never formed.

    Prior Types
    -----------
    - **global**: A single symmetric matrix for modeling global contact rate patterns.
      Only the lower triangular part of the matrix is modeled to ensure symmetry.

    - **partial**: Multiple independent and non-symmetric matrices for modeling different
      patterns in contact rates and intensities across different participant subgroups.
      Each matrix is fully modeled without symmetry constraints.

    - **full**: Multiple matrices for modeling contact rates and intensities across different
      participant and contact subgroups. Diagonal and off-diagonal elements are modeled
      separately to ensure reciprocity constraints.

    Sample sites
    ------------
    ``gp_scale`` (``gp_scale_diag``, ``gp_scale_non_diag`` for 'full') is the marginal
    variance ``sigma_f^2`` of the GP (it is passed as NumPyro's ``alpha``), with a
    HalfNormal(1) prior. ``gp_lenscale`` (and its 'full' counterparts) has a trailing
    axis of size 2 for ``kernel='product'`` (one lengthscale per grid axis, in grid
    column order) and no trailing axis for ``kernel='isotropic'``. ``gp_coefs`` has
    ``M_1 * M_2`` entries per event, ordered with the first axis slowest.

    Parameters
    ----------
    prior_type : {'global', 'partial', 'full'}
        The type of prior to use:
        - 'global': single symmetric matrix for global contact rates.
        - 'partial': multiple independent non-symmetric matrices.
        - 'full': multiple matrices with separate modeling of diagonal and off-diagonal elements.
    grid_type : {'age-age', 'diff-age'}, default='age-age'
        The type of age grid to use. The 'age-age' grid smoothes over age pairs, while the
        'diff-age' grid smoothes over age differences and contact ages. The grid axes are
        ``(age_1, age_2)`` and ``(age_diff, age)`` respectively.
    kernel : {'product', 'isotropic'}, default='product'
        Covariance structure of the 2D GP (see Kernels).
    nu : float, default=5/2
        The smoothness parameter of the Matern kernel. Common choices are:
        - 3/2: Sampled functions are once differentiable
        - 5/2: Sampled functions are twice differentiable
        - 7/2: Sampled functions are thrice differentiable (becomes similar to squared exponential kernel)
        - 1/2: Rough (Ornstein-Uhlenbeck) functions. No configuration rule exists, so
          ``C`` and ``lenscale_min`` must be given explicitly.
    C : 'auto', float or list of float, default='auto'
        The boundary inflation factors for each axis of the standardised grid. With
        'auto', ``C_d = max(a1 * lenscale_max_d / S_d, 1.2)`` so that the largest
        lengthscale in the prior is approximated accurately. An explicit value
        instead determines ``lenscale_max`` (see Notes).
    M : int, list[int] or None, default=None
        The number of eigenvalues and eigenfunctions per axis. ``None`` uses the
        default for the grid type (40 per axis). A higher number lowers the smallest
        representable lengthscale but increases computational cost.
    lenscale_scale : float or list of float, default=0.5
        Scale of the half-normal lengthscale prior, on the standardised grid scale
        (one unit equals one standard deviation of the grid axis). For
        ``kernel='product'`` a list of two values sets the scale per axis.
    lenscale_max : float, list of float or None, default=None
        Upper truncation point of the lengthscale prior. ``None`` uses the 95%
        quantile of HalfNormal(``lenscale_scale``) when ``C='auto'``, or the largest
        lengthscale supported by ``C`` when ``C`` is explicit. Per axis for
        ``kernel='product'``.
    lenscale_min : float, list of float or None, default=None
        Lower truncation point of the lengthscale prior. ``None`` uses the smallest
        lengthscale representable with ``M`` basis functions. Required when nu=1/2.
        Per axis for ``kernel='product'``.

    Notes
    -----
    Configuration follows Riutort-Mayol et al. (2023, Sec. 4.3.1), the rule used by
    PyMC's ``approx_hsgp_hyperparams`` and EpiNow2. On the standardised grid, with
    ``S_d`` the half-range of axis ``d`` and ``L_d = C_d * S_d``:

    - ``kernel='product'``: the rules apply to each 1D factor separately,
      ``C_d >= max(a1 * lenscale_max_d / S_d, 1.2)`` and
      ``lenscale_min_d = a2 * L_d / M_d``.
    - ``kernel='isotropic'``: the shared lengthscale must satisfy both axes,
      ``C_d >= max(a1 * lenscale_max / S_d, 1.2)`` and
      ``lenscale_min = a2 * max_d(L_d / M_d)``.

    with ``(a1, a2) = (4.5, 3.42)`` for nu=3/2 and ``(4.1, 2.65)`` for nu=5/2 and
    7/2. The lengthscale prior is a half-normal truncated to
    ``[lenscale_min, lenscale_max]``, so every lengthscale the prior allows is
    approximated accurately. The resolved values are available after
    ``set_age_bounds`` as ``C_``, ``M_``, ``L``, ``lenscale_min``, ``lenscale_max``
    (lists with one entry per axis; equal entries for the isotropic kernel) and, in
    years, ``lenscale_bounds_years``. See ``docs/dev_docs/hsgp_configuration.md``.

    Examples
    --------
    >>> import numpyro
    >>> from cntmosaic.models.numpyro.priors import HSGP2D
    >>> import jax.numpy as jnp
    >>>
    >>> # Global rate prior
    >>> prior = HSGP2D(prior_type="global", grid_type="age-age")
    >>> prior.set_age_bounds(0, 84)
    >>> def model_global():
    ...     f = prior.sample()
    ...     return f
    >>>
    >>> with numpyro.handlers.seed(rng_seed=0):
    ...     result = model_global()
    >>> # Resulting contact matrix shape
    >>> print(result.shape)  # symmetric matrix
    (85, 85)
    >>> print(bool(jnp.allclose(result, result.T)))
    True
    >>>
    >>> # Resolved HSGP settings and per-axis lengthscale bounds
    >>> print([round(c, 2) for c in prior.C_], prior.M_)
    [2.35, 2.35] [40, 40]
    >>> print([round(l, 3) for l in prior.lenscale_min], [round(l, 3) for l in prior.lenscale_max])
    [0.266, 0.266] [0.98, 0.98]

    See Also
    --------
    Prior2D : Base class for 2D priors.
    Spline2D : Spline with independent normal priors on coefficients.
    PSpline2D : Penalized B-splines with GMRF prior on coefficients.

    References
    ----------
    - Solin, A., & Särkkä, S. (2020). Hilbert space methods for reduced-rank Gaussian process regression.
      Statistics and Computing, 30(2), 419-446.
    - Riutort-Mayol, G., Bürkner, P.-C., Andersen, M. R., Solin, A., & Vehtari, A. (2023). Practical
      Hilbert space approximate Bayesian Gaussian processes for probabilistic programming. Statistics
      and Computing, 33(1), 17.
    - PyMC ``pymc.gp.hsgp_approx.approx_hsgp_hyperparams``; EpiNow2 ``gp_opts()`` (PR #1562).
    - Dan et al. (2023). Estimating fine age structure and time trends in human contact patterns from
      from coarse contact data: The Bayesian rate consistency model. PLOS Computational Biology. 19(6), e1011191.
    """

    pytree_aux_fields = (
        "kernel",
        "nu",
        "C",
        "M",
        "C_",
        "M_",
        "L",
        "lenscale_scale",
        "lenscale_min",
        "lenscale_max",
        "PHI_1",
        "PHI_2",
        "idx_all",
        "idx_sym",
        "symm_tril_ix",
    )

    def __init__(
        self,
        prior_type: str,
        grid_type: str = "age-age",
        nu: float = 5 / 2,
        C: Union[str, float, List[float]] = "auto",
        M: Optional[Union[int, List[int]]] = None,
        lenscale_scale: Union[float, List[float]] = 0.5,
        lenscale_max: Optional[Union[float, List[float]]] = None,
        lenscale_min: Optional[Union[float, List[float]]] = None,
        kernel: str = "product",
    ) -> None:
        validate_init_params(
            nu, C, M, lenscale_scale, lenscale_max, lenscale_min, kernel
        )
        super().__init__(grid_type, prior_type)
        self.kernel = kernel
        self.nu = nu
        self.C = C
        self.M = M
        self.lenscale_scale = lenscale_scale
        self._lenscale_max_user = lenscale_max
        self._lenscale_min_user = lenscale_min

    @property
    def axis_names(self) -> Tuple[str, str]:
        """Names of the two grid axes, in grid column order."""
        return ("age_1", "age_2") if self.grid_type == "age-age" else ("age_diff", "age")

    def set_age_bounds(self, min_age: int, max_age: int) -> None:
        """
        Set age range for the contact matrix, construct the grid, and build eigenfunction matrices.

        This method establishes the age boundaries, computes the number of age groups,
        initializes the grid structure, and constructs the eigenfunction matrices.
        Must be called before sampling from the prior.

        Parameters
        ----------
        min_age : int
            Minimum age (inclusive). Typically 0 for population-level contact matrices.
        max_age : int
            Maximum age (inclusive). Defines the upper bound of the oldest age group.

        The method performs the following steps:
        1. Sets age bounds and computes A
        2. Constructs grid points via _set_grid()
        3. Resolves C, M and the lengthscale bounds via _configure_hsgp()
        4. Builds the per-axis eigenfunction matrices via _set_eigenfunctions()
        """
        self.min_age = min_age
        self.max_age = max_age
        self.A = max_age - min_age + 1

        self._set_grid()
        self._configure_hsgp()
        self._set_eigenfunctions()

    def _configure_hsgp(self) -> None:
        """
        Resolve the boundary factors, basis sizes and lengthscale bounds (internal method).

        Uses the half-range ``S`` of each standardised grid axis (set by ``_set_grid``)
        and the rules in ``_hsgp_config``. Sets ``C_``, ``M_``, ``L``,
        ``lenscale_min`` and ``lenscale_max`` (the latter two as one value per axis).

        Raises
        ------
        ValueError
            If the resolved lower bound is not smaller than the upper bound, i.e.
            ``M`` is too small for the requested lengthscale range.
        """
        # Number of basis functions per axis: grid-type default or user value.
        if self.M is None:
            M_ = list(GRID_DEFAULTS[self.grid_type]["M"])
        elif isinstance(self.M, int):
            M_ = [self.M, self.M]
        else:
            M_ = list(self.M)

        if self.kernel == "product":
            C_, lmin, lmax = self._resolve_product(M_)
        else:
            C_, lmin, lmax = self._resolve_isotropic(M_)

        self.C_ = [float(c) for c in C_]
        self.M_ = M_
        self.L = [float(l) for l in np.asarray(C_) * self.S]
        self.lenscale_min = [float(l) for l in lmin]
        self.lenscale_max = [float(l) for l in lmax]

    def _explicit_C(self) -> np.ndarray:
        """Explicit boundary factors broadcast to both axes, warning below the floor."""
        C_ = np.broadcast_to(np.asarray(self.C, dtype=float), self.S.shape).copy()
        if np.any(C_ < MIN_BOUNDARY_FACTOR):
            warnings.warn(
                f"Boundary factor C={C_.tolist()} is below the recommended minimum "
                f"of {MIN_BOUNDARY_FACTOR}; the approximation may be inaccurate.",
                UserWarning,
                stacklevel=4,
            )
        return C_

    def _resolve_product(self, M_: List[int]):
        """Per-axis configuration for the tensor-product kernel."""
        S = self.S
        names = self.axis_names
        has_rule = self.nu in HSGP_KERNEL_CONSTANTS
        scale = _as_pair(self.lenscale_scale)
        lmax_user = None if self._lenscale_max_user is None else _as_pair(self._lenscale_max_user)
        lmin_user = None if self._lenscale_min_user is None else _as_pair(self._lenscale_min_user)
        quantile = np.array([halfnormal_quantile(s, 0.95) for s in scale])

        if isinstance(self.C, str):  # 'auto'
            lmax = quantile if lmax_user is None else lmax_user
            C_ = hsgp_boundary_factor(lmax, S, self.nu)
        else:
            C_ = self._explicit_C()
            lmax = lmax_user
            if has_rule:
                supported = hsgp_max_lenscale_per_axis(C_, S, self.nu)
                if lmax is None:
                    lmax = supported
                else:
                    over = lmax > supported * (1 + 1e-9)
                    if np.any(over):
                        detail = ", ".join(
                            f"{names[d]}: {lmax[d]:.3g} > {supported[d]:.3g}"
                            for d in np.flatnonzero(over)
                        )
                        warnings.warn(
                            f"lenscale_max exceeds the largest lengthscale supported by "
                            f"C={C_.tolist()} ({detail}). Increase C or use C='auto'.",
                            UserWarning,
                            stacklevel=4,
                        )
            elif lmax is None:
                lmax = quantile

        L = C_ * S
        lmin = lmin_user
        if has_rule:
            derived = hsgp_min_lenscale_per_axis(L, M_, self.nu)
            if lmin is None:
                lmin = derived
            else:
                under = lmin < derived * (1 - 1e-9)
                if np.any(under):
                    detail = ", ".join(
                        f"{names[d]}: {lmin[d]:.3g} < {derived[d]:.3g}"
                        for d in np.flatnonzero(under)
                    )
                    warnings.warn(
                        f"lenscale_min is below the smallest lengthscale representable "
                        f"with M={M_} ({detail}). Increase M.",
                        UserWarning,
                        stacklevel=4,
                    )

        empty = lmin >= lmax
        if np.any(empty):
            detail = ", ".join(
                f"{names[d]}: lenscale_min={lmin[d]:.3g} >= lenscale_max={lmax[d]:.3g}"
                for d in np.flatnonzero(empty)
            )
            hint = ""
            if has_rule:
                hint = (
                    f" At least M={hsgp_min_num_basis(L, lmax, self.nu)} basis "
                    "functions per axis are needed to represent the lengthscales up "
                    "to lenscale_max."
                )
            raise ValueError(
                f"The lengthscale range is empty ({detail}) with C={C_.tolist()} "
                f"and M={M_}." + hint
            )
        return C_, lmin, lmax

    def _resolve_isotropic(self, M_: List[int]):
        """Shared-lengthscale configuration for the isotropic kernel (legacy rules)."""
        S = self.S
        has_rule = self.nu in HSGP_KERNEL_CONSTANTS

        # Boundary factor per axis and the upper lengthscale bound.
        if isinstance(self.C, str):  # 'auto'
            lmax = self._lenscale_max_user
            if lmax is None:
                lmax = halfnormal_quantile(self.lenscale_scale, 0.95)
            C_ = hsgp_boundary_factor(lmax, S, self.nu)
        else:
            C_ = self._explicit_C()
            lmax = self._lenscale_max_user
            if has_rule:
                supported = hsgp_max_lenscale(C_, S, self.nu)
                if lmax is None:
                    lmax = supported
                elif lmax > supported * (1 + 1e-9):
                    warnings.warn(
                        f"lenscale_max={lmax:.3g} exceeds the largest lengthscale "
                        f"supported by C={C_.tolist()} ({supported:.3g}). Increase C "
                        "or use C='auto'.",
                        UserWarning,
                        stacklevel=4,
                    )
            elif lmax is None:
                lmax = halfnormal_quantile(self.lenscale_scale, 0.95)

        L = C_ * S

        # Lower lengthscale bound from the number of basis functions.
        lmin = self._lenscale_min_user
        if has_rule:
            derived_min = hsgp_min_lenscale(L, M_, self.nu)
            if lmin is None:
                lmin = derived_min
            elif lmin < derived_min * (1 - 1e-9):
                warnings.warn(
                    f"lenscale_min={lmin:.3g} is below the smallest lengthscale "
                    f"representable with M={M_} ({derived_min:.3g}). Increase M.",
                    UserWarning,
                    stacklevel=4,
                )

        if lmin >= lmax:
            hint = ""
            if has_rule:
                hint = (
                    f" At least M={hsgp_min_num_basis(L, lmax, self.nu)} basis "
                    "functions per axis are needed to represent any lengthscale "
                    f"up to {lmax:.3g}."
                )
            raise ValueError(
                f"The lengthscale range is empty: lenscale_min={lmin:.3g} >= "
                f"lenscale_max={lmax:.3g} with C={C_.tolist()} and M={M_}." + hint
            )
        return C_, _as_pair(lmin), _as_pair(lmax)

    @property
    def lenscale_bounds_years(self) -> Dict[str, Tuple[float, float]]:
        """
        Lengthscale bounds converted to years along each grid axis.

        Lengthscales are defined on the standardised grid, so their value in years
        differs between axes with different spread. Available after ``set_age_bounds``.

        Returns
        -------
        dict
            ``{axis_name: (lower, upper)}``. Axis names are ``('age_1', 'age_2')``
            for the 'age-age' grid and ``('age_diff', 'age')`` for the 'diff-age' grid.
        """
        return {
            name: (float(lo * sd), float(hi * sd))
            for name, lo, hi, sd in zip(
                self.axis_names, self.lenscale_min, self.lenscale_max, self.X_sd
            )
        }

    def _lenscale_prior(self) -> dist.Distribution:
        """Half-normal lengthscale prior truncated to the representable range.

        Batch shape (2,) (one lengthscale per axis) for the product kernel and a
        scalar for the isotropic kernel.
        """
        if self.kernel == "product":
            return dist.TruncatedNormal(
                0.0,
                jnp.asarray(_as_pair(self.lenscale_scale)),
                low=jnp.asarray(self.lenscale_min),
                high=jnp.asarray(self.lenscale_max),
            )
        return dist.TruncatedNormal(
            0.0, self.lenscale_scale, low=self.lenscale_min[0], high=self.lenscale_max[0]
        )

    def _set_grid(self) -> None:
        """
        Intialize and scale the grid points for the HSGP approximation (internal method).

        This method constructs the grid points based on the specified grid type
        ('age-age' or 'diff-age') and standardizes them such that they can be used
        as inputs to the eigenfunction calculations.

        Notes
        -----
        For 'age-age' grid: Creates full A x A coordinate pairs
        For 'diff-age' grid: Creates age difference representation

        Raises
        ------
        ValueError
            If grid_type is not one of 'age-age' or 'diff-age'.
        """
        if self.grid_type == "age-age":
            X = age_age_grid(self.A)
        elif self.grid_type == "diff-age":
            X = diff_age_age_grid(self.A)
        else:
            raise ValueError("grid_type must be 'age-age' or 'diff-age'")

        self.X_raw = X
        self.X_mean = X.mean(axis=0)
        self.X_sd = X.std(axis=0)
        Xn = (X - self.X_mean) / self.X_sd
        self.S = np.abs(Xn).max(axis=0)  # half-range of each standardised axis

        self.X = Xn
        self._tril_idx = tril_ix_row(self.A)
        self.X_tril = Xn[self._tril_idx]
        self.symm_tril_ix = symm_from_tril_ix_row(self.A)

    def _set_eigenfunctions(self) -> None:
        """
        Build the per-axis eigenfunction matrices and gather indices (internal method).

        Every grid point lies on the rectangular grid spanned by the unique values of
        each axis. The 2D eigenfunctions are products of 1D eigenfunctions, so the GP
        is evaluated on that rectangle as ``PHI_1 B PHI_2^T`` and the required cells
        are gathered with ``idx_all`` (all A*A points, row-major) or ``idx_sym``
        (lower triangle mirrored to a symmetric matrix).
        """
        inverse, phis = [], []
        for d in range(2):
            u, inv = np.unique(self.X_raw[:, d], return_inverse=True)
            u_std = (u - self.X_mean[d]) / self.X_sd[d]
            phis.append(
                np.asarray(eigenfunctions(x=u_std, ell=self.L[d], m=self.M_[d]))
            )
            inverse.append(inv.reshape(-1))
        self.PHI_1, self.PHI_2 = phis  # (n_1, M_1), (n_2, M_2)
        flat = inverse[0] * self.PHI_2.shape[0] + inverse[1]
        self.idx_all = flat
        self.idx_sym = flat[self._tril_idx][self.symm_tril_ix]
        self.num_basis = self.M_[0] * self.M_[1]

    def _evaluate(self, coefs: ArrayLike, idx: np.ndarray) -> ArrayLike:
        """
        Evaluate the GP at grid cells ``idx`` from weighted coefficients (internal method).

        Parameters
        ----------
        coefs : array, shape (..., M_1, M_2)
            Square-root spectral weights times standard-normal coefficients.
        idx : array of int
            Flat indices into the rectangular (n_1 x n_2) evaluation grid.

        Returns
        -------
        array, shape (..., len(idx))
        """
        G = self.PHI_1 @ coefs @ self.PHI_2.T  # (..., n_1, n_2)
        G = G.reshape(G.shape[:-2] + (-1,))
        return G[..., idx]

    def _sqrt_weights(self, alpha, length) -> ArrayLike:
        """
        Square root of the spectral weights on the (M_1, M_2) basis grid (internal method).

        Parameters
        ----------
        alpha : float
            Marginal variance of the GP (``gp_scale``).
        length : array of shape (2,) or float
            Per-axis lengthscales (product kernel) or the shared lengthscale
            (isotropic kernel), on the standardised grid scale.

        Returns
        -------
        array, shape (M_1, M_2)
        """
        if self.kernel == "product":
            s = [
                diag_spectral_density_matern(
                    nu=self.nu,
                    alpha=1.0,
                    length=length[d],
                    ell=self.L[d],
                    m=self.M_[d],
                    dim=1,
                )
                for d in range(2)
            ]
            return jnp.sqrt(alpha * s[0][:, None] * s[1][None, :])
        spd = diag_spectral_density_matern(
            nu=self.nu, alpha=alpha, length=length, ell=self.L, m=self.M_, dim=2
        )
        return jnp.sqrt(spd).reshape(self.M_[0], self.M_[1])

    def _coefs(self, name: str, sqrt_w: ArrayLike, batch: Tuple[int, ...]) -> ArrayLike:
        """Sample flat standard-normal coefficients and weight them (internal method)."""
        beta = numpyro.sample(
            name, dist.Normal(0.0, 1.0), sample_shape=batch + (self.num_basis,)
        )
        return sqrt_w * beta.reshape(batch + (self.M_[0], self.M_[1]))

    def sample_global(self) -> ArrayLike:
        # sample GP scale (marginal variance) and lengthscale parameters
        sigma = numpyro.sample("gp_scale", dist.HalfNormal(1.0))
        lenscale = numpyro.sample("gp_lenscale", self._lenscale_prior())

        sqrt_w = self._sqrt_weights(sigma, lenscale)
        coefs = self._coefs("gp_coefs", sqrt_w, ())

        f = self._evaluate(coefs, self.idx_sym)
        return f.reshape((self.A, self.A))

    def sample_partial(self) -> ArrayLike:
        # Sample GP scale (marginal variance) and lengthscale parameters for each event
        sigma = numpyro.sample(
            "gp_scale", dist.HalfNormal(1.0), sample_shape=(self.event_dim,)
        )
        lenscale = numpyro.sample(
            "gp_lenscale", self._lenscale_prior(), sample_shape=(self.event_dim,)
        )

        sqrt_w = vmap(self._sqrt_weights)(sigma, lenscale)  # (event_dim, M_1, M_2)
        coefs = self._coefs("gp_coefs", sqrt_w, (self.event_dim,))

        f = self._evaluate(coefs, self.idx_all)
        f = f.reshape((self.event_dim, self.A, self.A))
        return self.loc + f

    def sample_full(self) -> ArrayLike:
        # Sample GP scale parameters for diagonal and non-diagonal elements
        sigma_diag = numpyro.sample(
            "gp_scale_diag",
            dist.HalfNormal(1.0),
            sample_shape=(self.event_dim_diag,),
        )
        lenscale_diag = numpyro.sample(
            "gp_lenscale_diag",
            self._lenscale_prior(),
            sample_shape=(self.event_dim_diag,),
        )

        sigma_non_diag = numpyro.sample(
            "gp_scale_non_diag",
            dist.HalfNormal(1.0),
            sample_shape=(self.event_dim_non_diag_eff,),
        )
        lenscale_non_diag = numpyro.sample(
            "gp_lenscale_non_diag",
            self._lenscale_prior(),
            sample_shape=(self.event_dim_non_diag_eff,),
        )

        sqrt_w_diag = vmap(self._sqrt_weights)(sigma_diag, lenscale_diag)
        sqrt_w_non_diag = vmap(self._sqrt_weights)(sigma_non_diag, lenscale_non_diag)

        coefs_diag = self._coefs("gp_coefs_diag", sqrt_w_diag, (self.event_dim_diag,))
        coefs_non_diag = self._coefs(
            "gp_coefs_non_diag", sqrt_w_non_diag, (self.event_dim_non_diag_eff,)
        )

        f_diag = self._evaluate(coefs_diag, self.idx_sym)
        f_diag = f_diag.reshape((self.event_dim_diag, self.A, self.A))

        f_non_diag = self._evaluate(coefs_non_diag, self.idx_all)
        f_non_diag = f_non_diag.reshape((self.event_dim_non_diag_eff, self.A, self.A))

        # Allocate diagonal and off-diagonal elements into full K×K contact matrix grid
        f = self._assemble_full_prior_blocks(f_diag, f_non_diag)
        return self.loc + f

    def sample(self) -> ArrayLike:
        """
        Sample from the HSGP2D prior.

        Generates contact matrix samples by drawing from the specified HSGP prior type
        ('global', 'partial', or 'full') and applying the appropriate transformations.

        Returns
        -------
        f : array
            Sampled contact matrix with shape depending on prior_type:
            - 'global': (A, A) - symmetric matrix
            - 'partial': (event_dim, A, A) - asymmetric, event_dim matrices
            - 'full': (event_dim, A, A) - with separate diagonal/off-diagonal structure

        Examples
        --------
        >>> import numpyro
        >>> from jax import random
        >>> import jax.numpy as jnp
        >>>
        >>> # Global prior - symmetric contact matrix
        >>> prior = HSGP2D(prior_type='global')
        >>> prior.set_age_bounds(0, 50)
        >>>
        >>> with numpyro.handlers.seed(rng_seed=42):
        ...     sample = prior.sample()
        >>>
        >>> print(sample.shape)
        (51, 51)
        >>> print(bool(jnp.allclose(sample, sample.T)))  # symmetric
        True
        >>>
        >>> # Partial prior: one asymmetric matrix per event
        >>> prior_partial = HSGP2D(prior_type='partial')
        >>> prior_partial.set_age_bounds(0, 40)
        >>> prior_partial.set_event_dim(2)  # 2 categories
        >>> prior_partial.set_loc(0.0)
        >>>
        >>> with numpyro.handlers.seed(rng_seed=42):
        ...     sample_partial = prior_partial.sample()
        >>>
        >>> print(sample_partial.shape)
        (2, 41, 41)
        >>>
        >>> # Full prior: diagonal and off-diagonal blocks
        >>> prior_full = HSGP2D(prior_type='full')
        >>> prior_full.set_age_bounds(0, 30)
        >>> prior_full.set_event_dim(3)  # 3 categories
        >>> prior_full.set_loc(0.0)
        >>> with numpyro.handlers.seed(rng_seed=42):
        ...     sample_full = prior_full.sample()
        >>>
        >>> print(sample_full.shape)
        (9, 31, 31)
        """

        if self.prior_type == "global":
            return self.sample_global()

        elif self.prior_type == "partial":
            return self.sample_partial()

        elif self.prior_type == "full":
            return self.sample_full()

        else:
            raise ValueError(
                f"Invalid prior_type: {self.prior_type}. "
                f"Must be one of ['global', 'partial', 'full']"
            )
