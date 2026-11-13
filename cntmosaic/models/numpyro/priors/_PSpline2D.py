from __future__ import annotations

from typing import Union

import jax
import numpy as np
import numpyro
from numpyro import distributions as dist

from ..._utils import age_age_grid, diff_age_age_grid, symm_from_tril_ix_row
from ..distributions._IGMRF2D import IGMRF2D
from ._Spline2D import Spline2D


class PSpline2D(Spline2D):
    """
    2D Penalized B-spline (P-spline) prior for contact matrix estimation.

    This prior class extends Spline2D by adding a Gaussian Markov Random Field (GMRF)
    penalty on the B-spline coefficients, providing adaptive smoothing that balances
    flexibility with regularization. The penalty is controlled by precision parameters
    (tau) that are learned from the data, enabling automatic smoothness selection.

    P-splines combine the flexibility of B-splines with the regularization of
    difference penalties, making them particularly well-suited for modeling smooth
    but potentially complex age-structured contact patterns.

    Mathematical Background
    -----------------------
    The model has two levels:

    1. **Data level**: Contact intensity f(x, y) is represented using tensor product B-splines:

           f(x, y) = ∑ᵢ ∑ⱼ βᵢⱼ Bᵢ(x) Bⱼ(y) = Φ β

    2. **Prior level**: Coefficients β have an IGMRF prior penalizing roughness:

           β ~ N(0, Q⁻¹)
           Q = τ ⊗ (D₁ᵀD₁ ⊗ I₂ + I₁ ⊗ D₂ᵀD₂)

       where D₁ and D₂ are difference operators of specified order, and τ is a single
       precision parameter sampled from a Gamma prior: τ ~ Gamma(shape, rate)

    The difference operators penalize adjacent coefficients:
    - order=1: First differences Δβᵢ = βᵢ - βᵢ₋₁ (penalize roughness)
    - order=2: Second differences Δ²βᵢ = βᵢ - 2βᵢ₋₁ + βᵢ₋₂ (penalize curvature)

    Higher order penalties produce progressively smoother estimates while maintaining
    flexibility for local features.

    Advantages over Standard Splines
    ---------------------------------
    - **Adaptive smoothing**: Precision τ learned from data, not fixed
    - **Regularization**: GMRF prior prevents overfitting with many basis functions
    - **Interpretability**: Penalty order controls smoothness type
    - **Computational efficiency**: Sparse precision matrices for large M
    - **Flexibility**: Can model both smooth trends and local features

    Parameters
    ----------
    prior_type : {'global', 'partial', 'full'}
        Structure of the prior:
        - 'global': Symmetric contact matrix with shared coefficients
        - 'partial': Asymmetric with dimension-specific coefficients
        - 'full': Separate priors for diagonal and off-diagonal elements
    M : int or list of int, default=30
        Number of B-spline basis functions per dimension. More basis functions
        allow greater flexibility but require stronger regularization.
    degree : int or list of int, default=3
        Degree of B-spline polynomials. Typically 3 (cubic) for smooth functions.
    order : int, default=1
        Order of the difference penalty in the GMRF:
        - order=1: Penalize first differences (random walk prior)
        - order=2: Penalize second differences (smoother, more common)
        Higher orders produce smoother surfaces.
    tau_shape : float, default=2.0
        Shape parameter for Gamma prior on precision τ. Larger values concentrate
        mass at higher precision (more smoothing). Common choices: 1.0-3.0.
    tau_rate : float, default=0.01
        Rate parameter for Gamma prior on precision τ. Smaller values allow
        more flexibility. Together with tau_shape, controls prior mean = shape/rate.
    grid_type : {'age-age', 'diff-age'}, default='age-age'
        Grid structure for contact matrix:
        - 'age-age': Standard age-by-age contact matrix
        - 'diff-age': Age difference representation

    Attributes
    ----------
    order : int
        Difference penalty order
    tau_shape, tau_rate : float
        Hyperparameters for Gamma prior on precision
    M, degree, A : int
        Inherited from Spline2D
    PHI, PHI_diag, PHI_non_diag : ndarray
        Basis matrices inherited from Spline2D

    Methods
    -------
    sample()
        Main sampling interface, dispatches to appropriate method
    sample_global()
        Sample symmetric contact matrix
    sample_partial()
        Sample asymmetric matrix with shared precision structure
    sample_full()
        Sample with separate diagonal/off-diagonal priors
    apply_inverse_transform(f)
        Apply inverse compositional transformation

    Examples
    --------
    >>> from cntmosaic.models.numpyro.priors._PSpline2D import PSpline2D
    >>> import numpyro
    >>> from jax import random
    >>> import jax.numpy as jnp
    >>>
    >>> # Global prior with second-order penalty
    >>> prior_global = PSpline2D(
    ...     prior_type='global',
    ...     M=25,
    ...     degree=3,
    ...     order=2,
    ...     tau_shape=2.0,
    ...     tau_rate=0.01
    ... )
    >>> prior_global.set_age_bounds(0, 80)
    >>>
    >>> def model():
    ...     f = prior_global.sample()
    ...     return f
    >>>
    >>> with numpyro.handlers.seed(rng_seed=42):
    ...     sample = model()
    >>> print(sample.shape)  # (17, 17) symmetric
    >>>
    >>> # Partial prior with ILR transformation
    >>> prior_partial = PSpline2D(
    ...     prior_type='partial',
    ...     M=30,
    ...     degree=3,
    ...     order=2
    ... )
    >>> prior_partial.set_age_bounds(0, 75)
    >>> prior_partial.set_event_dim(4)
    >>> prior_partial.set_loc(0.0)
    >>>
    >>> # Full prior with different penalties for diagonal/off-diagonal
    >>> prior_full = PSpline2D(
    ...     prior_type='full',
    ...     M=20,
    ...     degree=3,
    ...     order=2,
    ...     tau_shape=2.0,
    ...     tau_rate=0.01
    ... )

    Notes
    -----
    - The number of basis functions M should be chosen based on data size and
      expected complexity. Common choices: M = 20-40 for typical age ranges.
    - Second-order penalties (order=2) are most common as they penalize curvature
      and produce visually smooth surfaces.
    - Precision parameters τ are automatically learned, eliminating need for
      manual smoothing parameter tuning.
    - Computational cost: O(M²) for coefficient sampling due to sparse precision

    Comparison with Spline2D
    -------------------------
    - **Spline2D**: Fixed smoothness via basis choice, fast, simple
    - **PSpline2D**: Adaptive smoothness, better for unknown complexity, slightly slower

    See Also
    --------
    Spline2D : Base class with standard B-splines
    IGMRF2D : 2D Intrinsic Gaussian Markov Random Field distribution
    HSGP2D : Hilbert space Gaussian process alternative

    References
    ----------
    - Eilers, P. H., & Marx, B. D. (1996). Flexible smoothing with B-splines and
      penalties. Statistical Science, 11(2), 89-121.
    - Rue, H., & Held, L. (2005). Gaussian Markov Random Fields: Theory and Applications.
      Chapman & Hall/CRC.
    - Lang, S., & Brezger, A. (2004). Bayesian P-splines. Journal of Computational and
      Graphical Statistics, 13(1), 183-212.
    """

    pytree_aux_fields = (
        "self.PHI",
        "order",
    )

    def __init__(
        self,
        prior_type: str,
        M: Union[int, list[int]] = 30,
        degree: Union[int, list[int]] = 3,
        order: int = 1,
        tau_shape: float = 2.0,
        tau_rate: float = 0.01,
        grid_type: str = "age-age",
        bound_ext: float = 0.05,
    ):
        super().__init__(prior_type, M, degree, grid_type, bound_ext)
        self.order = order
        self.tau_shape = tau_shape
        self.tau_rate = tau_rate

    def set_age_bounds(self, min_age: int, max_age: int) -> None:
        return super().set_age_bounds(min_age, max_age)

    def _set_grid(self):
        if self.grid_type == "age-age":
            X = age_age_grid(self.A)
        elif self.grid_type == "diff-age":
            X = diff_age_age_grid(self.A)
        else:
            raise ValueError("grid_type must be 'age-age' or 'diff-age'")

        x = np.sort(np.unique(X[:, 0]))
        y = np.sort(np.unique(X[:, 1]))

        # Scale x and y to [0, 1]
        self.x = (x - self.min_age) / (self.max_age - self.min_age)
        self.y = (y - self.min_age) / (self.max_age - self.min_age)

        self.symm_tril_idx = symm_from_tril_ix_row(self.A)

    def sample_global(self) -> jax.Array:
        """
        Sample symmetric contact matrix using penalized B-splines.

        Generates a symmetric contact matrix by sampling B-spline coefficients from
        an IGMRF prior with precision parameter τ sampled from a Gamma distribution.
        The resulting matrix satisfies n(i,j) = n(j,i).

        Returns
        -------
        f : array, shape (A, A)
            Symmetric contact matrix. Each element represents contact intensity
            between age groups.

        Notes
        -----
        Sampling process:
        1. Sample precision: τ ~ Gamma(tau_shape, tau_rate)
        2. Sample coefficients: β ~ IGMRF2D(num_nodes=(M, M), order, cond_prec=τ)
        3. Compute latent field: f = Φ β
        4. Symmetrize using lower triangular indices

        No compositional transformation is applied for global priors.

        Examples
        --------
        >>> import numpyro
        >>> prior = PSpline2D(prior_type='global', M=20, order=2)
        >>> prior.set_age_bounds(0, 50)
        >>>
        >>> with numpyro.handlers.seed(rng_seed=42):
        ...     f = prior.sample_global()
        >>> print(f.shape)  # (11, 11)
        >>> print(jnp.allclose(f, f.T))  # True
        """
        num_nodes = (self.M, self.M)

        tau = numpyro.sample("spline_tau", dist.Gamma(self.tau_shape, self.tau_rate))

        beta = numpyro.sample(
            "spline_coefs",
            IGMRF2D(
                num_nodes, self.order, cond_prec=tau, rescale_marginal_variance=True
            ),
        )  # (M*M,)

        f = (self.PHI @ beta)[self.symm_tril_idx].reshape((self.A, self.A))
        return f

    def sample_partial(self) -> jax.Array:
        """
        Sample asymmetric contact matrix with dimension-specific precision parameters.

        Generates contact matrices allowing n(i,j) ≠ n(j,i) by sampling separate
        B-spline coefficients for each effective dimension, each with its own
        precision parameter τ. Suitable for modeling non-reciprocal contact patterns.

        Returns
        -------
        f : array, shape (event_dim, A, A)
            Contact matrix after inverse compositional transformation. Shape depends
            on the transformation type.

        Notes
        -----
        Sampling process:
        1. Sample precisions: τ ~ Gamma(tau_shape, tau_rate), shape (event_dim,)
        2. Sample coefficients: β ~ IGMRF2D with precision τ, shape (event_dim, M²)
        3. Compute latent field: f = Φ β
        4. Add location parameter: f += loc

        Each dimension gets its own precision τᵢ, enabling adaptive smoothness
        that can vary across contact settings or age groups.

        Examples
        --------
        >>> import numpyro
        >>> import jax.numpy as jnp
        >>> prior = PSpline2D(
        ...     prior_type='partial',
        ...     M=25,
        ...     order=2,
        ...     transform='alr'
        ... )
        >>> prior.set_age_bounds(0, 60)
        >>> prior.set_event_dim(3)
        >>> prior.set_loc(0.0)
        >>>
        >>> with numpyro.handlers.seed(rng_seed=42):
        ...     f = prior.sample_partial()
        >>> print(f.shape)  # (3, 13, 13)

        See Also
        --------
        sample_global : Symmetric contact matrix
        sample_full : Separate diagonal/off-diagonal priors
        """
        num_nodes = (self.M, self.M)

        tau = numpyro.sample(
            "spline_tau",
            dist.Gamma(self.tau_shape, self.tau_rate),
            sample_shape=(self.event_dim,),
        )

        beta = numpyro.sample(
            "spline_coefs",
            IGMRF2D(
                num_nodes, self.order, cond_prec=tau, rescale_marginal_variance=True
            ),
        )  # (event_dim_eff, M*M)

        beta = beta.swapaxes(0, 1)  # (M*M, event_dim_eff)
        f = self.PHI @ beta  # (A*A, event_dim_eff)
        f = f.swapaxes(0, 1)
        f = f.reshape((self.event_dim, self.A, self.A))

        return self.loc + f

    def sample_full(self) -> jax.Array:
        """
        Sample contact matrix with separate priors for diagonal and off-diagonal elements.

        Provides maximum flexibility by using separate penalized B-spline priors with
        independent precision parameters for diagonal elements (within-group contacts)
        and off-diagonal elements (between-group contacts). Each type can have different
        smoothness characteristics.

        Returns
        -------
        f : array, shape (event_dim, A, A)
            Contact matrix after inverse compositional transformation, with separately
            modeled diagonal and off-diagonal structure.

        Notes
        -----
        Sampling process:
        1. Sample diagonal precisions: τ_diag ~ Gamma(shape, rate), shape (event_dim_diag,)
        2. Sample off-diagonal precisions: τ_non_diag ~ Gamma(shape, rate), shape (event_dim_non_diag_eff,)
        3. Sample diagonal coefficients: β_diag ~ IGMRF2D with τ_diag
        4. Sample off-diagonal coefficients: β_non_diag ~ IGMRF2D with τ_non_diag
        5. Compute diagonal field: f_diag = Φ_diag β_diag (symmetrized)
        6. Compute off-diagonal field: f_non_diag = Φ_non_diag β_non_diag
        7. Combine: merge diagonal and off-diagonal elements
        8. Add location parameter and apply inverse transformation

        This allows, for example, within-age-group contacts to be smooth while
        between-age-group contacts have more complex patterns, or vice versa.

        For ALR/ILR transformations, the last diagonal element is excluded to
        maintain proper dimensionality.

        Examples
        --------
        >>> import numpyro
        >>> import jax.numpy as jnp
        >>> prior = PSpline2D(
        ...     prior_type='full',
        ...     M=20,
        ...     order=2,
        ...     tau_shape=2.0,
        ...     tau_rate=0.01,
        ...     transform='alr'
        ... )
        >>> prior.set_age_bounds(0, 45)
        >>> prior.set_event_dim(4)
        >>> prior.set_loc(0.0)
        >>>
        >>> with numpyro.handlers.seed(rng_seed=42):
        ...     f = prior.sample_full()
        >>> print(f.shape)  # (4, 10, 10)

        See Also
        --------
        sample_global : Symmetric contact matrix
        sample_partial : Shared precision structure
        """
        num_nodes = (self.M, self.M)

        tau_diag = numpyro.sample(
            "spline_tau_diag",
            dist.Gamma(self.tau_shape, self.tau_rate),
            sample_shape=(self.event_dim_diag,),
        )
        tau_non_diag = numpyro.sample(
            "spline_tau_non_diag",
            dist.Gamma(self.tau_shape, self.tau_rate),
            sample_shape=(self.event_dim_non_diag_eff,),
        )

        beta_diag = numpyro.sample(
            "spline_coefs_diag",
            IGMRF2D(
                num_nodes,
                self.order,
                cond_prec=tau_diag,
                rescale_marginal_variance=True,
            ),
        )  # (event_dim_diag, M*M)
        beta_non_diag = numpyro.sample(
            "spline_coefs_non_diag",
            IGMRF2D(
                num_nodes,
                self.order,
                cond_prec=tau_non_diag,
                rescale_marginal_variance=True,
            ),
        )  # (event_dim_non_diag, M*M)

        beta_diag = beta_diag.swapaxes(0, 1)  # (M*M, event_dim_diag)
        f_diag = self.PHI_diag @ beta_diag
        f_diag = f_diag[self.symm_tril_idx, :].swapaxes(0, 1)  # Must be symmetric
        f_diag = f_diag.reshape((self.event_dim_diag, self.A, self.A))

        beta_non_diag = beta_non_diag.swapaxes(0, 1)  # (M*M, event_dim_non_diag)
        f_non_diag = (self.PHI_non_diag @ beta_non_diag).swapaxes(0, 1)
        f_non_diag = f_non_diag.reshape((self.event_dim_non_diag_eff, self.A, self.A))

        # Assemble diagonal and off-diagonal blocks into full event grid
        f = self._assemble_full_prior_blocks(f_diag, f_non_diag)

        return self.loc + f

    def sample(self) -> jax.Array:
        """
        Sample contact matrix using penalized B-spline prior.

        Main sampling interface that dispatches to the appropriate method based on
        the configured prior_type. This provides a unified interface for all sampling
        strategies.

        Returns
        -------
        f : array
            Sampled contact matrix:
            - shape (A, A) for prior_type='global'
            - shape (event_dim, A, A) for prior_type='partial' or 'full'

        Notes
        -----
        Dispatching logic:
        - 'global': calls sample_global() → symmetric matrix
        - 'partial': calls sample_partial() → asymmetric with shared structure
        - 'full': calls sample_full() → separate diagonal/off-diagonal priors

        See Also
        --------
        sample_global : Symmetric contact matrix
        sample_partial : Asymmetric with dimension-specific precision
        sample_full : Separate diagonal and off-diagonal priors
        """

        if self.prior_type == "global":
            return self.sample_global()

        elif self.prior_type == "partial":
            return self.sample_partial()

        elif self.prior_type == "full":
            return self.sample_full()
