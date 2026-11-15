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
    hsgp_min_lenscale,
    hsgp_min_num_basis,
)
from ._Prior2D import Prior2D


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def validate_init_params(
    nu: float,
    C: Union[str, float, List[float]],
    M: Optional[Union[int, List[int]]],
    lenscale_scale: float = 0.5,
    lenscale_max: Optional[float] = None,
    lenscale_min: Optional[float] = None,
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
    lenscale_scale : float
        Scale of the (truncated) half-normal lengthscale prior.
    lenscale_max, lenscale_min : float or None
        Optional explicit bounds of the lengthscale prior.

    Raises
    ------
    ValueError
        If nu is not one of [1/2, 3/2, 5/2, 7/2], if C, M or the lengthscale
        settings are not positive, or if nu=1/2 is used without explicit C and
        lenscale_min.
    TypeError
        If C is not 'auto', a number or a list of two numbers, or if M is not
        None, an int or a list of two ints.
    """
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

    if not _is_number(lenscale_scale) or lenscale_scale <= 0:
        raise ValueError("lenscale_scale must be a positive number.")
    for name, value in (("lenscale_max", lenscale_max), ("lenscale_min", lenscale_min)):
        if value is not None and (not _is_number(value) or value <= 0):
            raise ValueError(f"{name} must be a positive number or None.")
    if (
        lenscale_min is not None
        and lenscale_max is not None
        and lenscale_min >= lenscale_max
    ):
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
    a 2D HSGP characterized by the Matern kernel. It provides a computationally efficient
    alternative to traditional Gaussian process priors by approximating the GP using a finite
    set of eigenfunctions derived from the Laplacian operator.

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


    Parameters
    ----------
    prior_type : {'global', 'partial', 'full'}
        The type of prior to use:
        - 'global': single symmetric matrix for global contact rates.
        - 'partial': multiple independent non-symmetric matrices.
        - 'full': multiple matrices with separate modeling of diagonal and off-diagonal elements.
    grid_type : {'age-age', 'diff-age'}, default='age-age'
        The type of age grid to use. The 'age-age' grid smoothes over age pairs, while the
        'diff-age' grid smoothes over age differences and contact ages.
    nu : float, default=5/2
        The smoothness parameter of the Matern kernel. Common choices are:
        - 3/2: Sampled functions are once differentiable
        - 5/2: Sampled functions are twice differentiable
        - 7/2: Sampled functions are thrice differentiable (becomes similar to squared exponential kernel)
        - 1/2: Rough (Ornstein-Uhlenbeck) functions. No configuration rule exists, so
          ``C`` and ``lenscale_min`` must be given explicitly.
    C : 'auto', float or list of float, default='auto'
        The boundary inflation factors for each axis of the standardised grid. With
        'auto', ``C_d = max(a1 * lenscale_max / S_d, 1.2)`` so that the largest
        lengthscale in the prior is approximated accurately. An explicit value
        instead determines ``lenscale_max`` (see Notes).
    M : int, list[int] or None, default=None
        The number of eigenvalues and eigenfunctions per axis. ``None`` uses the
        default for the grid type (40 per axis). A higher number lowers the smallest
        representable lengthscale but increases computational cost.
    lenscale_scale : float, default=0.5
        Scale of the half-normal lengthscale prior, on the standardised grid scale
        (one unit equals one standard deviation of the grid axis).
    lenscale_max : float or None, default=None
        Upper truncation point of the lengthscale prior. ``None`` uses the 95%
        quantile of HalfNormal(``lenscale_scale``) when ``C='auto'``, or the largest
        lengthscale supported by ``C`` when ``C`` is explicit.
    lenscale_min : float or None, default=None
        Lower truncation point of the lengthscale prior. ``None`` uses the smallest
        lengthscale representable with ``M`` basis functions. Required when nu=1/2.

    Notes
    -----
    Configuration follows Riutort-Mayol et al. (2023, Sec. 4.3.1), the rule used by
    PyMC's ``approx_hsgp_hyperparams`` and EpiNow2. On the standardised grid, with
    ``S_d`` the half-range of axis ``d`` and ``L_d = C_d * S_d``:

    - ``C_d >= max(a1 * lenscale_max / S_d, 1.2)``
    - ``lenscale_min = a2 * max_d(L_d / M_d)``

    with ``(a1, a2) = (4.5, 3.42)`` for nu=3/2 and ``(4.1, 2.65)`` for nu=5/2 and
    7/2. The lengthscale prior is a half-normal truncated to
    ``[lenscale_min, lenscale_max]``, so every lengthscale the prior allows is
    approximated accurately. The resolved values are available after
    ``set_age_bounds`` as ``C_``, ``M_``, ``L``, ``lenscale_min``, ``lenscale_max``
    and, in years, ``lenscale_bounds_years``. See
    ``docs/dev_docs/hsgp_configuration.md``.

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
    >>> # Resolved HSGP settings and lengthscale bounds
    >>> print([round(c, 2) for c in prior.C_], prior.M_)
    [2.35, 2.35] [40, 40]
    >>> print(round(prior.lenscale_min, 3), round(prior.lenscale_max, 3))
    0.266 0.98

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
        "nu",
        "C",
        "M",
        "C_",
        "M_",
        "L",
        "lenscale_scale",
        "lenscale_min",
        "lenscale_max",
        "PHI",
        "symm_tril_ix",
    )

    def __init__(
        self,
        prior_type: str,
        grid_type: str = "age-age",
        nu: float = 5 / 2,
        C: Union[str, float, List[float]] = "auto",
        M: Optional[Union[int, List[int]]] = None,
        lenscale_scale: float = 0.5,
        lenscale_max: Optional[float] = None,
        lenscale_min: Optional[float] = None,
    ) -> None:
        validate_init_params(nu, C, M, lenscale_scale, lenscale_max, lenscale_min)
        super().__init__(grid_type, prior_type)
        self.nu = nu
        self.C = C
        self.M = M
        self.lenscale_scale = lenscale_scale
        self._lenscale_max_user = lenscale_max
        self._lenscale_min_user = lenscale_min

    def set_age_bounds(self, min_age: int, max_age: int) -> None:
        """
        Set age range for the contact matrix, construct the grid, and build eigenfunction matrix.

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
        4. Builds eigenfunction matrices via _set_eigenfunctions()
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
        ``lenscale_min`` and ``lenscale_max``.

        Raises
        ------
        ValueError
            If the resolved lower bound is not smaller than the upper bound, i.e.
            ``M`` is too small for the requested lengthscale range.
        """
        S = self.S
        has_rule = self.nu in HSGP_KERNEL_CONSTANTS

        # Number of basis functions per axis: grid-type default or user value.
        if self.M is None:
            M_ = list(GRID_DEFAULTS[self.grid_type]["M"])
        elif isinstance(self.M, int):
            M_ = [self.M, self.M]
        else:
            M_ = list(self.M)

        # Boundary factor per axis and the upper lengthscale bound.
        if isinstance(self.C, str):  # 'auto'
            lmax = self._lenscale_max_user
            if lmax is None:
                lmax = halfnormal_quantile(self.lenscale_scale, 0.95)
            C_ = hsgp_boundary_factor(lmax, S, self.nu)
        else:
            C_ = np.broadcast_to(np.asarray(self.C, dtype=float), S.shape).copy()
            if np.any(C_ < MIN_BOUNDARY_FACTOR):
                warnings.warn(
                    f"Boundary factor C={C_.tolist()} is below the recommended minimum "
                    f"of {MIN_BOUNDARY_FACTOR}; the approximation may be inaccurate.",
                    UserWarning,
                    stacklevel=3,
                )
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
                        stacklevel=3,
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
                    stacklevel=3,
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

        self.C_ = [float(c) for c in C_]
        self.M_ = M_
        self.L = [float(l) for l in L]
        self.lenscale_min = float(lmin)
        self.lenscale_max = float(lmax)

    @property
    def lenscale_bounds_years(self) -> Dict[str, Tuple[float, float]]:
        """
        Lengthscale bounds converted to years along each grid axis.

        The lengthscale is isotropic on the standardised grid, so its value in years
        differs between axes with different spread. Available after ``set_age_bounds``.

        Returns
        -------
        dict
            ``{axis_name: (lower, upper)}``. Axis names are ``('age_1', 'age_2')``
            for the 'age-age' grid and ``('age_diff', 'age')`` for the 'diff-age' grid.
        """
        names = ("age_1", "age_2") if self.grid_type == "age-age" else ("age_diff", "age")
        return {
            name: (self.lenscale_min * sd, self.lenscale_max * sd)
            for name, sd in zip(names, self.X_sd)
        }

    def _lenscale_prior(self) -> dist.Distribution:
        """Half-normal lengthscale prior truncated to the representable range."""
        return dist.TruncatedNormal(
            0.0, self.lenscale_scale, low=self.lenscale_min, high=self.lenscale_max
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

        self.X_sd = X.std(axis=0)
        Xn = (X - X.mean(axis=0)) / self.X_sd
        self.S = np.abs(Xn).max(axis=0)  # half-range of each standardised axis

        if self.prior_type == "global":
            tril_idx = tril_ix_row(self.A)
            self.X_tril = Xn[tril_idx]
            self.symm_tril_ix = symm_from_tril_ix_row(self.A)
        if self.prior_type == "full":
            tril_idx = tril_ix_row(self.A)
            self.X = Xn
            self.X_tril = Xn[tril_idx]
            self.symm_tril_ix = symm_from_tril_ix_row(self.A)
        else:
            self.X = Xn

    def _set_eigenfunctions(self) -> None:
        """Build the eigenfunction matrix for the HSGP approximation (internal method)."""
        if self.prior_type == "global":
            self.PHI = eigenfunctions(
                x=self.X_tril, ell=self.L, m=self.M_
            )  # (N_tri, M1*M2)
        elif self.prior_type == "full":
            self.PHI_diag = eigenfunctions(
                x=self.X_tril, ell=self.L, m=self.M_
            )  # (N_tri, M1*M2)
            self.PHI_non_diag = eigenfunctions(
                x=self.X, ell=self.L, m=self.M_
            )  # (N, M1*M2)
        else:
            self.PHI = eigenfunctions(x=self.X, ell=self.L, m=self.M_)  # (N, M1*M2)

    def _compute_sqrt_diag_spd(self, alpha, length) -> ArrayLike:
        """
        Compute the square root of the diagonal spectral density for the Matern kernel,
        given the scale (alpha) and lengthscale parameters (internal method).

        Parameters
        ----------
        alpha : float
            The scale parameter of the Matern kernel.
        length : float
            The lengthscale parameter of the Matern kernel.

        Returns
        -------
        spd : ArrayLike
            The diagonal spectral density evaluated at the eigenvalues.
        """
        return jnp.sqrt(
            diag_spectral_density_matern(
                nu=self.nu, alpha=alpha, length=length, ell=self.L, m=self.M_, dim=2
            )
        )

    def sample_global(self) -> ArrayLike:
        # sample GP scale parameter
        sigma = numpyro.sample("gp_scale", dist.HalfNormal(1.0))
        # sample GP lengthscale parameter
        lenscale = numpyro.sample("gp_lenscale", self._lenscale_prior())

        # compute diagonal spectral density
        sqrt_spd = self._compute_sqrt_diag_spd(sigma, lenscale)

        # Sample eigenfunction coefficients
        beta = numpyro.sample(
            "gp_coefs", dist.Normal(0.0, 1.0), sample_shape=(self.PHI.shape[-1],)
        )

        f = self.PHI @ (sqrt_spd * beta)
        f = f[self.symm_tril_ix]
        return f.reshape((self.A, self.A))

    def sample_partial(self) -> ArrayLike:
        # Sample GP scale parameters for each event
        sigma = numpyro.sample(
            "gp_scale", dist.HalfNormal(1.0), sample_shape=(self.event_dim,)
        )
        lenscale = numpyro.sample(
            "gp_lenscale", self._lenscale_prior(), sample_shape=(self.event_dim,)
        )

        # Vectorized SPD (shape: (event_dim, num_basis))
        sqrt_spd = jnp.squeeze(vmap(self._compute_sqrt_diag_spd)(sigma, lenscale))

        # Sample eigenfunction coefficients
        beta = numpyro.sample(
            "gp_coefs",
            dist.Normal(0.0, 1.0),
            sample_shape=(self.event_dim, self.PHI.shape[-1]),
        )

        f = (sqrt_spd * beta) @ self.PHI.T  # shape: (event_dim, N)
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

        # Vectorized SPD
        sqrt_spd_diag = jnp.squeeze(
            vmap(self._compute_sqrt_diag_spd)(sigma_diag, lenscale_diag)
        )
        sqrt_spd_non_diag = jnp.squeeze(
            vmap(self._compute_sqrt_diag_spd)(sigma_non_diag, lenscale_non_diag)
        )

        # Sample eigenfunction coefficients
        beta_diag = numpyro.sample(
            "gp_coefs_diag",
            dist.Normal(0.0, 1.0),
            sample_shape=(self.event_dim_diag, self.PHI_diag.shape[-1]),
        )
        beta_non_diag = numpyro.sample(
            "gp_coefs_non_diag",
            dist.Normal(0.0, 1.0),
            sample_shape=(self.event_dim_non_diag_eff, self.PHI_non_diag.shape[-1]),
        )

        f_diag = (sqrt_spd_diag * beta_diag) @ self.PHI_diag.T
        f_diag = f_diag[:, self.symm_tril_ix]
        f_diag = f_diag.reshape((self.event_dim_diag, self.A, self.A))

        f_non_diag = (sqrt_spd_non_diag * beta_non_diag) @ self.PHI_non_diag.T
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
