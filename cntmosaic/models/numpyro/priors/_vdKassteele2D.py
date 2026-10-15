import jax
import jax.numpy as jnp
import numpyro
from jax.typing import ArrayLike
from numpyro import distributions as dist

from ....utils import symm_from_tril_ix_col
from ..distributions import IGMRF2D, SymIGMRF2D
from ._Prior2D import Prior2D


class vdKassteele2D(Prior2D):
    """
    2D Intrinsic Gaussian Markov Random Field (IGMRF) prior for contact matrix estimation.

    This prior implements the van de Kassteele et al. approach for modeling social contact
    matrices using a 2D intrinsic Gaussian Markov Random Field.

    The IGMRF provides spatial smoothing through neighborhood-based penalties, with precision
    parameter τ controlling the strength of smoothing. Higher order penalties (order=2)
    produce smoother surfaces by penalizing curvature rather than just roughness.

    Mathematical Background
    -----------------------
    The contact intensity f(i,j) at age grid points (i,j) follows an IGMRF prior:

        f ~ IGMRF(order, τ)
        τ ~ Gamma(τ_shape, τ_rate)

    The precision matrix Q has structure:

        Q = τ * (D_x^T D_x ⊗ I_y + I_x ⊗ D_y^T D_y)

    where D_x and D_y are difference operators of specified order:
    - order=1: First differences penalize |f(i,j) - f(i-1,j)| (roughness)
    - order=2: Second differences penalize curvature in the surface

    When to Use
    -----------
    - Contact patterns expected to be smooth
    - Want to avoid basis function selection

    Parameters
    ----------
    prior_type : {'global', 'partial', 'full'}, default='global'
        Structure of the prior:
        - 'global': Symmetric contact matrix with shared coefficients
        - 'partial': Asymmetric with dimension-specific precision parameters
        - 'full': Separate priors for diagonal and off-diagonal elements
    order : int, default=2
        Order of the difference penalty:
        - order=1: First differences (random walk prior)
        - order=2: Second differences
    tau_shape : float, default=2.0
        Shape parameter for Gamma prior on precision τ. Controls concentration
        around prior mean. Typical values: 1.0-3.0.
    tau_rate : float, default=0.01
        Rate parameter for Gamma prior on precision τ. Smaller values allow
        more flexibility. Prior mean = tau_shape / tau_rate.

    Attributes
    ----------
    min_age, max_age : int
        Age bounds for the contact matrix
    A : int
        Number of age groups (max_age - min_age + 1)
    sym_idx : ndarray
        Indices for symmetrizing matrices from lower triangular
    order : int
        Difference penalty order
    tau_shape, tau_rate : float
        Hyperparameters for Gamma prior on precision
    event_dim : int
        Number of effective dimensions (inherited from Prior2D)

    Methods
    -------
    set_age_bounds(min_age, max_age)
        Configure age range and initialize grid structure
    sample()
        Main sampling interface, dispatches based on prior_type
    sample_single()
        Sample symmetric contact matrix (prior_type='global')
    sample_partial()
        Sample asymmetric matrix with shared structure (prior_type='partial')
    sample_full()
        Sample with separate diagonal/off-diagonal priors (prior_type='full')

    Examples
    --------
    >>> from cntmosaic.models.numpyro.priors import vdKassteele
    >>> import numpyro
    >>> from jax import random
    >>> import jax.numpy as jnp
    >>>
    >>> # Global symmetric prior
    >>> prior_global = vdKassteele(
    ...     transform=None,
    ...     prior_type='global',
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
    >>> print(sample.shape)  # (17, 17) symmetric matrix
    >>> print(jnp.allclose(sample, sample.T))  # True
    >>>
    >>> # Partial prior with multiple dimensions
    >>> prior_partial = vdKassteele(prior_type='partial', order=2)
    >>> prior_partial.set_age_bounds(0, 75)
    >>> prior_partial.set_event_dim(4)
    >>>
    >>> with numpyro.handlers.seed(rng_seed=42):
    ...     f = prior_partial.sample()
    >>> print(f.shape)  # (4, 16, 16) - note: event_dim, not event_dim_eff
    >>>
    >>> # Full prior with separate diagonal/off-diagonal
    >>> prior_full = vdKassteele(prior_type='full', order=2)
    >>> prior_full.set_age_bounds(0, 60)
    >>> prior_full.set_event_dim(9)  # 3x3 stratification
    >>>
    >>> with numpyro.handlers.seed(rng_seed=42):
    ...     f = prior_full.sample()
    >>> print(f.shape)  # (9, 13, 13)

    See Also
    --------
    Prior2D : Base class for 2D priors
    PSpline2D : Penalized B-spline alternative with transformations
    HSGP2D : Hilbert space Gaussian process prior
    SymIGMRF2D : Symmetric IGMRF distribution
    IGMRF2D : General IGMRF distribution

    References
    ----------
    - van de Kassteele, J., van Eijkeren, J., & Wallinga, J. (2017). Efficient
      estimation of age-specific social contact rates between men and women.
      Annals of Applied Statistics, 11(1), 320-339.
    - Rue, H., & Held, L. (2005). Gaussian Markov Random Fields: Theory and
      Applications. Chapman & Hall/CRC.
    """

    pytree_aux_fields = ("order", "tau_shape", "tau_rate")

    def __init__(
        self,
        prior_type: str = "global",
        order: int = 2,
        tau_shape: float = 2.0,
        tau_rate: float = 0.1,
    ):
        """
        Initialize vdKassteele IGMRF prior.

        Parameters
        ----------
        prior_type : {'global', 'partial', 'full'}, default='global'
            Prior structure type
        order : int, default=2
            Order of difference penalty (1 or 2)
        tau_shape : float, default=2.0
            Shape parameter for Gamma prior on precision
        tau_rate : float, default=0.1
            Rate parameter for Gamma prior on precision

        Raises
        ------
        ValueError
            If order is not 1 or 2
        """
        if order not in [1, 2]:
            raise ValueError(f"order must be 1 or 2, got {order}")

        super().__init__("age-age", prior_type)
        self.order = order
        self.tau_shape = tau_shape
        self.tau_rate = tau_rate

    def set_age_bounds(self, min_age: int, max_age: int) -> None:
        """
        Set age bounds and initialize grid structure.

        Configures the age range for the contact matrix and computes necessary
        indices for symmetrization operations.

        Parameters
        ----------
        min_age : int
            Minimum age (inclusive)
        max_age : int
            Maximum age (inclusive)

        Raises
        ------
        ValueError
            If min_age < 0 or max_age <= min_age

        Examples
        --------
        >>> prior = vdKassteele(prior_type='global')
        >>> prior.set_age_bounds(0, 80)
        >>> print(prior.A)  # 17 (5-year age groups)
        """
        if min_age < 0:
            raise ValueError(f"min_age must be non-negative, got {min_age}")
        if max_age <= min_age:
            raise ValueError(
                f"max_age must be greater than min_age, got max_age={max_age}, min_age={min_age}"
            )

        self.min_age = min_age
        self.max_age = max_age
        self.A = int(max_age - min_age + 1)

        self._set_grid()

    def _set_grid(self) -> None:
        """
        Initialize grid-specific structures.

        Computes symmetrization indices for converting lower triangular
        matrices to full symmetric matrices.

        Notes
        -----
        Called automatically by set_age_bounds(). Uses column-major ordering
        for compatibility with JAX operations.
        """
        self.symm_tril_ix = symm_from_tril_ix_col(self.A)

    def sample_single(self) -> jax.Array:
        """
        Sample symmetric contact matrix using IGMRF prior.

        Generates a symmetric contact matrix by sampling from a symmetric IGMRF
        distribution. Suitable for modeling reciprocal contact patterns where
        n(i,j) = n(j,i).

        Returns
        -------
        f : array, shape (A, A)
            Symmetric contact matrix. Each element f[i,j] represents the log
            contact intensity between age groups i and j.

        Notes
        -----
        Sampling process:
        1. Sample precision: τ ~ Gamma(tau_shape, tau_rate)
        2. Sample field: f ~ SymIGMRF2D((A, A), order=order, cond_prec=τ)
        3. Reshape to (A, A) matrix

        The SymIGMRF2D distribution ensures f is symmetric by construction,
        imposing f[i,j] = f[j,i] through the precision matrix structure.

        Examples
        --------
        >>> import numpyro
        >>> prior = vdKassteele(prior_type='global', order=2)
        >>> prior.set_age_bounds(0, 75)
        >>>
        >>> with numpyro.handlers.seed(rng_seed=42):
        ...     f = prior.sample_single()
        >>> print(f.shape)  # (16, 16)
        >>> print(jnp.allclose(f, f.T))  # True (symmetric)

        See Also
        --------
        SymIGMRF2D : Symmetric 2D IGMRF distribution
        sample_partial : Asymmetric variant
        """
        tau = jnp.asarray(
            numpyro.sample("tau", dist.Gamma(self.tau_shape, self.tau_rate))
        )
        f = jnp.asarray(
            numpyro.sample(
                "f",
                SymIGMRF2D(self.A, order=2, cond_prec=tau, rescale_marginal_variance=True),
            )
        )[self.symm_tril_ix]
        f = f.reshape((self.A, self.A))  # Column-major order to match SymIGMRF2D output

        return f

    def sample_partial(self) -> jax.Array:
        """
        Sample asymmetric contact matrix with dimension-specific precision.

        Generates contact matrices allowing n(i,j) ≠ n(j,i) by sampling separate
        IGMRF realizations for each dimension, each with its own precision parameter.
        Suitable for modeling non-reciprocal contact patterns across different
        settings or demographics.

        Returns
        -------
        f : array, shape (event_dim, A, A)
            Contact matrix stack. f[k, i, j] represents log contact intensity
            for dimension k between age groups i and j. Note: first dimension
            is event_dim, not event_dim_eff (no transformation applied).

        Notes
        -----
        Sampling process:
        1. Sample precisions: τ ~ Gamma(shape, rate), shape (event_dim,)
        2. Sample fields: f ~ IGMRF((A,A), order, cond_prec=τ), shape (event_dim, A²)
        3. Reshape to (event_dim, A, A)

        Each dimension gets independent precision τ_k, enabling adaptive smoothness
        that varies across contact settings, genders, or other stratifications.

        **Important**: Unlike PSpline2D, this returns event_dim dimensions (not
        event_dim_eff) because no simplex transformation is applied.

        Examples
        --------
        >>> import numpyro
        >>> import jax.numpy as jnp
        >>> prior = vdKassteele(prior_type='partial', order=2)
        >>> prior.set_age_bounds(0, 60)
        >>> prior.set_event_dim(4)  # e.g., 2x2 gender stratification
        >>>
        >>> with numpyro.handlers.seed(rng_seed=42):
        ...     f = prior.sample_partial()
        >>> print(f.shape)  # (4, 13, 13) - note: event_dim, not event_dim_eff

        See Also
        --------
        sample_single : Symmetric variant
        sample_full : Separate diagonal/off-diagonal structure
        """
        tau = jnp.asarray(
            numpyro.sample("tau", dist.Gamma(self.tau_shape, self.tau_rate))
        )
        f = jnp.asarray(
            numpyro.sample(
                "f",
                IGMRF2D(
                    (self.A, self.A),
                    order=self.order,
                    cond_prec=tau,
                    rescale_marginal_variance=True,
                ),
                sample_shape=(self.event_dim,),
            )
        )  # IGMRF2D returns C-order (row-major) flattened arrays, so reshape with default C-order
        f = f.reshape((self.event_dim, self.A, self.A))

        return f

    def sample_full(self) -> jax.Array:
        """
        Sample contact matrix with separate diagonal and off-diagonal priors.

        Provides maximum flexibility by using separate IGMRF priors with independent
        precision parameters for diagonal elements (within-group contacts) and
        off-diagonal elements (between-group contacts). This allows different
        smoothness for assortative vs cross-mixing patterns.

        Returns
        -------
        f : array, shape (event_dim, A, A)
            Contact matrix with separately modeled diagonal and off-diagonal structure.
            Note: first dimension is event_dim (no transformation applied).

        Notes
        -----
        Sampling process:
        1. Compute diagonal/off-diagonal dimensions from event_dim
        2. Sample diagonal precisions: τ_diag ~ Gamma(shape, rate), shape (event_dim_diag+1,)
        3. Sample off-diagonal precisions: τ_non_diag ~ Gamma(shape, rate), shape (event_dim_non_diag,)
        4. Sample diagonal field: f_diag ~ SymIGMRF2D with τ_diag (symmetric)
        5. Sample off-diagonal field: f_non_diag ~ IGMRF with τ_non_diag
        6. Merge diagonal and off-diagonal elements into full matrix

        The diagonal uses SymIGMRF2D to ensure symmetry, while off-diagonal uses
        standard IGMRF allowing asymmetry.

        For a D×D stratification (event_dim = D²):
        - Diagonal: D elements use SymIGMRF2D (symmetric by construction)
        - Off-diagonal: D²-D elements use IGMRF (potentially asymmetric)

        Examples
        --------
        >>> import numpyro
        >>> import jax.numpy as jnp
        >>> prior = vdKassteele(prior_type='full', order=2)
        >>> prior.set_age_bounds(0, 45)
        >>> prior.set_event_dim(9)  # 3x3 stratification
        >>>
        >>> with numpyro.handlers.seed(rng_seed=42):
        ...     f = prior.sample_full()
        >>> print(f.shape)  # (9, 10, 10)
        >>>
        >>> # Check diagonal elements (indices 0, 4, 8 for 3x3)
        >>> for k in [0, 4, 8]:
        ...     assert jnp.allclose(f[k], f[k].T)  # Diagonal blocks symmetric

        See Also
        --------
        sample_single : Fully symmetric variant
        sample_partial : Shared precision structure
        SymIGMRF2D : Symmetric IGMRF for diagonal
        IGMRF : General IGMRF for off-diagonal
        """
        tau = jnp.asarray(
            numpyro.sample("tau", dist.Gamma(self.tau_shape, self.tau_rate))
        )

        # Sample diagonal elements (symmetric)
        f_diag = jnp.asarray(
            numpyro.sample(
                "f_diag",
                SymIGMRF2D(
                    self.A, order=self.order, cond_prec=tau, rescale_marginal_variance=True
                ),
                sample_shape=(self.event_dim_diag,),
            )
        )[..., self.symm_tril_ix].reshape((self.event_dim_diag, self.A, self.A))

        # Sample off-diagonal elements (asymmetric allowed)
        f_non_diag = jnp.asarray(
            numpyro.sample(
                "f_non_diag",
                IGMRF2D((self.A, self.A), order=self.order, cond_prec=tau),
                sample_shape=(self.event_dim_non_diag_eff,),
            )
        ).reshape((self.event_dim_non_diag_eff, self.A, self.A))

        # Assemble diagonal and off-diagonal blocks using parent class method
        f = jnp.asarray(self._assemble_full_prior_blocks(f_diag, f_non_diag))

        return f

    def sample(self) -> jax.Array:
        """
        Sample contact matrix using IGMRF prior.

        Main sampling interface that dispatches to the appropriate method based on
        the configured prior_type. Provides a unified interface for all sampling
        strategies.

        Returns
        -------
        f : array
            Sampled contact matrix:
            - shape (A, A) for prior_type='global'
            - shape (event_dim, A, A) for prior_type='partial' or 'full'

            Note: For partial/full types, first dimension is event_dim (not
            event_dim_eff) since no simplex transformation is applied.

        Raises
        ------
        ValueError
            If prior_type is not one of 'global', 'partial', or 'full'

        Notes
        -----
        Dispatching logic:
        - 'global': calls sample_single() → symmetric (A, A) matrix
        - 'partial': calls sample_partial() → (event_dim, A, A) with shared structure
        - 'full': calls sample_full() → (event_dim, A, A) with diagonal/off-diagonal

        Examples
        --------
        >>> import numpyro
        >>> from jax import random
        >>>
        >>> # Global prior
        >>> prior_global = vdKassteele(prior_type='global')
        >>> prior_global.set_age_bounds(0, 75)
        >>> with numpyro.handlers.seed(rng_seed=42):
        ...     f = prior_global.sample()
        >>> print(f.shape)  # (16, 16)
        >>>
        >>> # Partial prior
        >>> prior_partial = vdKassteele(prior_type='partial')
        >>> prior_partial.set_age_bounds(0, 60)
        >>> prior_partial.set_event_dim(4)
        >>> with numpyro.handlers.seed(rng_seed=42):
        ...     f = prior_partial.sample()
        >>> print(f.shape)  # (4, 13, 13)

        See Also
        --------
        sample_single : Symmetric contact matrix
        sample_partial : Dimension-specific precision
        sample_full : Separate diagonal and off-diagonal priors
        """
        if self.prior_type == "global":
            return self.sample_single()

        elif self.prior_type == "partial":
            return self.sample_partial()

        elif self.prior_type == "full":
            return self.sample_full()
        else:
            raise ValueError(
                f"Unknown prior_type '{self.prior_type}'. "
                "Must be one of 'global', 'partial', or 'full'."
            )
