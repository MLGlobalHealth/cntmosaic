from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import factorial

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike

from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.util import is_prng_key
from numpyro.distributions.util import validate_sample

from ....utils import symm_from_tril_ix_col, tril_ix_col


def diff_matrix_np(num_nodes: int, order: int) -> NDArray:
    """
    Construct a finite difference matrix of given order using NumPy.

    This function creates a difference operator matrix D that computes finite
    differences of a specified order. The resulting matrix has shape
    (num_nodes - order, num_nodes) and is used to construct Laplacian matrices
    for Gaussian Markov Random Fields.

    Parameters
    ----------
    num_nodes : int
        The number of nodes (grid points) in one dimension.
        Must satisfy: num_nodes > order.
    order : int
        The order of the finite difference operator.
        - order=1: First differences (adjacent point differences)
        - order=2: Second differences (curvature/smoothness penalty)
        - order=k: k-th order differences

    Returns
    -------
    D : NDArray
        Finite difference matrix of shape (num_nodes - order, num_nodes).
        Each row represents a finite difference stencil of the specified order.

    Notes
    -----
    The coefficients are computed using binomial coefficients with alternating signs:

    .. math::
        D_{i,j} = \\binom{k}{j-i} (-1)^{k-(j-i)}

    where k is the order and j ranges from i to i+order.

    This implementation uses NumPy to avoid JAX tracing issues during
    matrix construction, which is done once at initialization.

    Examples
    --------
    >>> D1 = diff_matrix_np(5, 1)  # First differences
    >>> D1.shape
    (4, 5)
    >>> D1[0]  # First row: [1, -1, 0, 0, 0]
    array([ 1., -1.,  0.,  0.,  0.])

    >>> D2 = diff_matrix_np(5, 2)  # Second differences
    >>> D2.shape
    (3, 5)
    >>> D2[0]  # First row: [1, -2, 1, 0, 0]
    array([ 1., -2.,  1.,  0.,  0.])

    See Also
    --------
    scipy.sparse.diags : For constructing sparse difference matrices
    """
    D = np.zeros((num_nodes - order, num_nodes))
    i_vals = np.arange(order + 1)
    coeff = (factorial(order) / (factorial(i_vals) * factorial(order - i_vals))) * (
        -1
    ) ** (order - i_vals)
    for i in range(num_nodes - order):
        D[i, i : i + order + 1] = coeff

    return D


def tril_igmrf_indices(
    n: int, order: int | tuple[int, int] = 2
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Compute indices for lower-triangular IGMRF structure with boundary-aware differencing.

    This function determines which indices to keep in the horizontal (D1) and vertical (D2)
    difference operators when constructing a k-th-order random walk prior on a symmetric
    (lower-triangular) n×n grid. This is essential for symmetric contact matrix estimation
    where only the lower triangle (including diagonal) contains unique information.

    The function accounts for boundary constraints: difference operators near the edges
    of the triangular region must not reference nodes outside the valid triangular domain.

    Parameters
    ----------
    n : int
        Number of age groups (matrix dimension). The grid has n(n+1)/2 unique nodes
        in the lower triangle including the diagonal.
    order : int or tuple[int, int], default=2
        Order of the random walk prior in each dimension.
        - If int: same order k for both dimensions (1=RW1, 2=RW2, etc.)
        - If tuple: (order_horizontal, order_vertical) for anisotropic smoothing
        Must satisfy: 1 ≤ order < n for valid finite differences.

    Returns
    -------
    cix : NDArray
        1-D array of length n(n+1)/2 containing column indices for the unique
        lower-triangular nodes (i ≥ j). These map the full n² grid to the
        reduced triangular representation.
    rix1 : NDArray
        1-D array of row indices to keep in D1 (horizontal difference operator).
        These correspond to valid horizontal differences within the triangular domain.
    rix2 : NDArray
        1-D array of row indices to keep in D2 (vertical difference operator).
        These correspond to valid vertical differences within the triangular domain.

    Raises
    ------
    ValueError
        If order_h or order_v is not in the range [1, n).

    Notes
    -----
    The indices are computed using Fortran (column-major) ordering to match
    the convention used in matrix vectorization for contact matrices.

    For a symmetric matrix constraint, the lower triangle contains all unique
    information. The precision structure must respect this constraint by only
    penalizing differences between nodes within the valid triangular region.

    Examples
    --------
    >>> # 3x3 symmetric matrix, first-order differences
    >>> cix, rix1, rix2 = tril_igmrf_indices(3, order=1)
    >>> len(cix)  # 3*(3+1)/2 = 6 unique elements
    6
    >>>
    >>> # Different orders in each direction
    >>> cix, rix1, rix2 = tril_igmrf_indices(5, order=(1, 2))
    >>> len(cix)  # 5*(5+1)/2 = 15 unique elements
    15

    See Also
    --------
    symm_from_tril_ix_col : Expand lower triangle to full symmetric matrix
    tril_ix_col : Extract lower triangular indices

    References
    ----------
    - Rue, H., & Held, L. (2005). Gaussian Markov Random Fields: Theory and Applications.
      Chapman & Hall/CRC. Chapter 3: Intrinsic Gaussian Markov random fields.
    """
    # Parse order specification
    if isinstance(order, int):
        order_h, order_v = order, order
    else:
        order_h, order_v = order

    # Validate orders
    if not (1 <= order_h < n):
        raise ValueError(f"`order_h` must satisfy 1 ≤ order_h < n, got {order_h}")
    if not (1 <= order_v < n):
        raise ValueError(f"`order_v` must satisfy 1 ≤ order_v < n, got {order_v}")

    # --- columns: keep lower triangle incl. diagonal ------------------------
    ci_mat = np.arange(n**2, dtype=int).reshape(n, n, order="F")
    mask_ci = np.greater_equal.outer(np.arange(n), np.arange(n))
    cix = np.sort(ci_mat[mask_ci])

    # --- rows for D1: horizontal RWk ----------------------------------------
    # rix1.mat has (n-order_h) rows and n columns
    ri1_mat = np.arange((n - order_h) * n, dtype=int).reshape(n - order_h, n, order="F")
    mask_ri1 = np.greater_equal.outer(np.arange(n - order_h), np.arange(n))
    rix1 = ri1_mat[mask_ri1]

    # --- rows for D2: vertical RWk ------------------------------------------
    # rix2.mat has n rows and (n-order_v) columns
    ri2_mat = np.arange(n * (n - order_v), dtype=int).reshape(n, n - order_v, order="F")
    mask_ri2 = np.arange(n)[:, None] >= np.arange(n - order_v)[None, :] + order_v
    rix2 = ri2_mat[mask_ri2]

    return cix, rix1, rix2


class SymIGMRF2D(Distribution):
    """
    Symmetric 2-dimensional Intrinsic Gaussian Markov Random Field (IGMRF) distribution.

    This distribution represents a symmetric Gaussian Markov Random Field over a 2D lattice,
    constrained to be symmetric across the main diagonal. This symmetry constraint is
    particularly useful for modeling symmetric contact matrices in epidemiological applications,
    where contact rates between age groups i and j should equal those between j and i.

    The precision matrix Q has the form:

        Q = cond_prec × (L₁ + L₂)

    where L₁ and L₂ are Laplacian matrices constructed from finite difference operators,
    and the structure is applied only to the lower-triangular elements (including diagonal)
    with appropriate boundary handling.

    Mathematical Background
    -----------------------
    Unlike the general IGMRF2D which allows separate precision parameters in each dimension,
    SymIGMRF2D enforces:

    1. **Isotropic smoothing**: Same precision parameter for both dimensions
    2. **Symmetric structure**: Matrix[i,j] = Matrix[j,i] by construction
    3. **Reduced parameterization**: Only n(n+1)/2 unique elements for n×n matrix

    The distribution is intrinsic (improper) because the Laplacian has zero eigenvalues
    corresponding to constant functions on the grid. For sampling and likelihood computation,
    these zero eigenvalues are filtered out using eigendecomposition with a tolerance threshold.

    The symmetric constraint is enforced by:
    - Constructing the precision structure on the lower-triangular domain
    - Using specialized index mappings (sym_ix, tril_ix) to expand/contract between
      full n² and reduced n(n+1)/2 representations

    Key Differences from IGMRF2D
    -----------------------------
    - **Single precision parameter**: cond_prec (vs. cond_prec1, cond_prec2)
    - **Symmetric constraint**: Automatically enforced in sampling and likelihood
    - **Efficient storage**: Works with n(n+1)/2 parameters instead of n²
    - **Boundary-aware**: Custom index handling for triangular domain

    Parameters
    ----------
    num_nodes : int
        Number of nodes (age groups) in each dimension. The resulting matrix is
        num_nodes × num_nodes but only n(n+1)/2 elements are unique due to symmetry.
    order : int
        Order of the finite difference approximation (same for both dimensions).
        - order=1: First-order differences (random walk prior)
        - order=2: Second-order differences (smooth prior)
        Must satisfy: 1 ≤ order < num_nodes.
    loc : ArrayLike, default=0.0
        Location parameter (prior mean) of the distribution.
        Shape options:
        - Scalar (): broadcasted to (batch_shape, num_nodes²)
        - (num_nodes²,): no batch dimension
        - (batch_shape, num_nodes²): batched means
        Note: Even though only n(n+1)/2 elements are unique, loc is specified
        for the full n² vectorized matrix for consistency.
    cond_prec : ArrayLike, default=1.0
        Conditional precision (inverse variance) parameter controlling smoothness.
        Higher values → smoother surfaces. Shape options:
        - Scalar (): no batch dimension
        - (batch_shape,): batched precision parameters
        Must be positive.
    tol : float, default=1e-10
        Tolerance for filtering near-zero eigenvalues. Eigenvalues below this
        threshold are excluded from sampling and likelihood computations.
    rescale_marginal_variance : bool, default=False
        If True, rescale `L` (and its eigenvalues) so that the geometric mean
        of the diagonal of `L`'s generalized inverse
        `Sigma = U_sub @ diag(1 / lam_sub) @ U_sub.T` (the marginal variances
        at `cond_prec=1`) equals 1, following Sørbye & Rue (2014). This makes
        `cond_prec` comparable across different `num_nodes`/`order` choices.
    validate_args : bool, optional
        Whether to validate input arguments. Default is None.

    Attributes
    ----------
    batch_shape : tuple
        Shape of parameter batches, inferred from loc and cond_prec.
    event_shape : tuple
        Shape of a single sample, always (num_nodes²,).
    L : NDArray
        Combined Laplacian matrix (L₁ + L₂) for the triangular domain,
        shape (n(n+1)/2, n(n+1)/2).
    lam_sub : NDArray
        Non-zero eigenvalues of L (after filtering with tol).
    U_sub : NDArray
        Eigenvectors corresponding to non-zero eigenvalues.
    sym_ix : NDArray
        Indices mapping from lower triangle to full symmetric matrix.
    tril_ix : NDArray
        Indices extracting lower triangle from full matrix.

    Notes
    -----
    **Computational Considerations:**
    - Matrix construction uses NumPy (not JAX) to avoid tracing overhead
    - Eigendecomposition performed once at initialization
    - Symmetry enforced through indexing rather than constraints

    **Limitations:**
    - Currently does not support anisotropic precision (different scales per dimension)
    - Order must be the same for both dimensions
    - Cannot use different smoothness in horizontal vs. vertical directions

    Examples
    --------
    >>> from cntmosaic.models.numpyro.distributions import SymIGMRF2D
    >>> import jax.numpy as jnp
    >>> from jax import random
    >>>
    >>> # Create a symmetric 2D IGMRF with second-order differences
    >>> dist = SymIGMRF2D(num_nodes=10, order=2, cond_prec=1.0)
    >>> dist.event_shape
    (100,)
    >>>
    >>> # Sample from the distribution
    >>> key = random.PRNGKey(0)
    >>> samples = dist.sample(key, sample_shape=(5,))  # 5 samples
    >>> samples.shape
    (5, 100)
    >>>
    >>> # Verify symmetry: reshape and check
    >>> sample_matrix = samples[0].reshape(10, 10)
    >>> jnp.allclose(sample_matrix, sample_matrix.T)
    True
    >>>
    >>> # Batched precision parameters
    >>> dist_batch = SymIGMRF2D(
    ...     num_nodes=5,
    ...     order=1,
    ...     cond_prec=jnp.array([0.5, 1.0, 2.0])
    ... )
    >>> dist_batch.batch_shape
    (3,)
    >>> dist_batch.event_shape
    (25,)

    See Also
    --------
    IGMRF2D : Non-symmetric 2D version with separate precision parameters
    IGMRF : 1-dimensional version

    References
    ----------
    - Rue, H., & Held, L. (2005). Gaussian Markov Random Fields: Theory and Applications.
      Chapman & Hall/CRC. Chapter 3: Intrinsic GMRFs.
    - Prem, K., et al. (2017). Projecting social contact matrices in 152 countries using
      contact surveys and demographic data. PLOS Computational Biology, 13(9).
    - Sørbye, S. H., & Rue, H. (2014). Scaling intrinsic Gaussian Markov random field
      priors in spatial modelling. Spatial Statistics, 8, 39-51.
    """

    arg_constraints = {
        "num_nodes": constraints.positive_integer,
        "order": constraints.positive_integer,
        "cond_prec": constraints.positive,
    }
    support = constraints.real_vector
    reparametrized_params = ["loc", "cond_prec"]
    pytree_data_fields = ("loc", "cond_prec", "lam_sub", "U_sub", "L")
    pytree_aux_fields = ("num_nodes", "order", "sym_ix", "tril_ix")

    def __init__(
        self,
        num_nodes: int,
        order: int,
        loc: ArrayLike = 0.0,
        cond_prec: ArrayLike = 1.0,
        tol: float = 1e-10,
        rescale_marginal_variance: bool = False,
        *,
        validate_args: Optional[bool] = None,
    ):
        """
        Initialize a symmetric 2D Intrinsic Gaussian Markov Random Field distribution.

        This constructor builds the precision structure for a symmetric IGMRF, including
        the Laplacian matrix, eigendecomposition, and index mappings for the triangular
        representation.

        Parameters
        ----------
        num_nodes : int
            Number of nodes (age groups) in each dimension of the square matrix.
            Must be greater than the order parameter.
        order : int
            Order of finite difference approximation for both dimensions.
            Controls the smoothness penalty:
            - order=1: Penalizes first differences (adjacent values)
            - order=2: Penalizes second differences (curvature)
            Must satisfy: 1 ≤ order < num_nodes.
        loc : ArrayLike, default=0.0
            Location parameter (prior mean). Shape options:
            - Scalar: broadcasted to (batch_shape, num_nodes²)
            - (num_nodes²,): no batch dimension
            - (batch_shape, num_nodes²): batched
        cond_prec : ArrayLike, default=1.0
            Conditional precision parameter. Higher values enforce stronger smoothing.
            Shape options:
            - Scalar: no batch dimension
            - (batch_shape,): batched precision
            Must be positive.
        tol : float, default=1e-10
            Tolerance for eigenvalue filtering. Eigenvalues smaller than this are
            treated as zero and excluded from computations. This handles the
            intrinsic (improper) nature of the IGMRF.
        rescale_marginal_variance : bool, default=False
            If True, rescale `L` (and its eigenvalues) so that the geometric mean
            of the diagonal of `L`'s generalized inverse
            `Sigma = U_sub @ diag(1 / lam_sub) @ U_sub.T` (the marginal variances
            at `cond_prec=1`) equals 1, following Sørbye & Rue (2014). This makes
            `cond_prec` comparable across different `num_nodes`/`order` choices.
        validate_args : bool, optional
            Whether to enable input validation. Default is None.

        Raises
        ------
        ValueError
            If order is not in the valid range [1, num_nodes).

        Notes
        -----
        The initialization performs the following steps:

        1. **Build difference operators** (D1, D2) using finite differences
        2. **Apply triangular indexing** to respect symmetry constraints
        3. **Construct Laplacian** L = D₁ᵀD₁ + D₂ᵀD₂ on triangular domain
        4. **Eigendecomposition** to identify and remove zero eigenvalues
        5. **Optionally rescale** L and its eigenvalues (see `rescale_marginal_variance`)
        6. **Setup index mappings** between full and triangular representations
        7. **Broadcast parameters** to handle batched computations

        The use of NumPy (rather than JAX) for matrix construction avoids
        JAX tracing overhead during initialization.
        """

        self.num_nodes = num_nodes
        self.order = order

        # Note: The matrices are build in NumPy to avoid JAX tracing issues
        # Build the full Kronecker sum structure matrix
        D1 = np.kron(np.eye(self.num_nodes), diff_matrix_np(self.num_nodes, self.order))
        D2 = np.kron(diff_matrix_np(self.num_nodes, self.order), np.eye(self.num_nodes))

        # Select the lower-triangular indices while respecting the boundaries
        cxi, rix1, rix2 = tril_igmrf_indices(self.num_nodes, self.order)
        D1_red = D1[rix1][:, cxi]
        D2_red = D2[rix2][:, cxi]

        # Construct the laplacian matrices
        L1 = D1_red.T @ D1_red
        L2 = D2_red.T @ D2_red
        L_np = L1 + L2

        # Eigendecomposition on NumPy array (before JAX conversion) to avoid
        # TracerArrayConversionError when called inside JIT-compiled SVI body.
        lam, U = np.linalg.eigh(L_np)

        # Boolean filtering in Numpy (not traced by JAX)
        nonzero_mask = lam > tol
        lam_sub_raw = lam[nonzero_mask]
        U_sub = U[:, nonzero_mask]

        if rescale_marginal_variance:
            # Sørbye & Rue (2014): rescale L so the geometric mean of the diagonal
            # of its generalized inverse Sigma = U_sub @ diag(1/lam_sub) @ U_sub.T
            # (the marginal variances at cond_prec=1) is 1, making cond_prec
            # comparable across different num_nodes/order choices. diag(Sigma)_k =
            # sum_j U_sub[k,j]^2 / lam_sub[j], computed directly rather than
            # forming the full triangular-domain matrix.
            marginal_var = np.sum((U_sub**2) / lam_sub_raw, axis=1)
            scale = np.exp(np.mean(np.log(marginal_var)))
        else:
            scale = 1.0

        self.lam_sub = scale * lam_sub_raw
        self.U_sub = U_sub
        self.L = jnp.asarray(scale * L_np)

        self.sym_ix = symm_from_tril_ix_col(self.num_nodes)
        self.tril_ix = tril_ix_col(self.num_nodes)

        # ===== Determine batch shape from inputs =====
        if jnp.ndim(loc) == 0:  # Scalar loc: no batch dimension
            loc_batch_shape = ()
        elif jnp.ndim(loc) == 1:  # loc has shape (n1*n2,)
            loc_batch_shape = ()
        else:  # loc has shape (batch_shape, n1*n2)
            loc_batch_shape = jnp.shape(loc)[:-1]

        if jnp.ndim(cond_prec) == 0:  # Scalar cond_prec: no batch dimension
            cond_prec_batch_shape = ()
        else:  # cond_prec has shape (batch_shape,)
            cond_prec_batch_shape = jnp.shape(cond_prec)

        batch_shape = lax.broadcast_shapes(loc_batch_shape, cond_prec_batch_shape)

        # ===== Broadcast adjustments =====
        if jnp.ndim(loc) == 0:
            self.loc = jnp.broadcast_to(
                loc, batch_shape + (self.num_nodes * (self.num_nodes + 1) // 2,)
            )
        else:
            self.loc = jnp.broadcast_to(
                loc, batch_shape + (self.num_nodes * (self.num_nodes + 1) // 2,)
            )

        # Broadcast conditional precisions to (batch_shape,)
        self.cond_prec = jnp.broadcast_to(cond_prec, batch_shape)

        # Broadcast U_sub if there's a batch dimension
        if batch_shape:
            self.U_sub = jnp.broadcast_to(self.U_sub, batch_shape + self.U_sub.shape)
        else:
            self.U_sub = self.U_sub

        event_shape = (self.num_nodes * (self.num_nodes + 1) // 2,)

        super(SymIGMRF2D, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        """
        Sample from the symmetric IGMRF2D distribution.

        Sampling is performed by:
        1. Generating standard normal samples in the reduced (non-zero) eigenspace
        2. Scaling by the inverse square root of eigenvalues
        3. Transforming back to the triangular space via eigenvectors
        4. Expanding from triangular (n(n+1)/2) to full symmetric matrix (n²)
        5. Adding the location parameter

        The symmetry constraint is automatically enforced through the index mapping
        in step 4, which copies lower-triangular values to the upper triangle.

        Parameters
        ----------
        key : jax.Array
            PRNG key for random number generation.
        sample_shape : tuple[int, ...], default=()
            Shape of the sample batch to generate.
            Final sample shape will be: sample_shape + batch_shape + event_shape.

        Returns
        -------
        Array
            Samples from the lower-triangular (including diagonal) part of the
            symmetric IGMRF2D distribution, with shape:
            sample_shape + batch_shape + (num_nodes x (num_nodes + 1) // 2,)

        Notes
        -----
        The sampling process avoids explicit matrix inversion by leveraging the
        eigendecomposition computed during initialization. This is both numerically
        stable and computationally efficient.
        """
        assert is_prng_key(key)
        sub_event_shape = (self.lam_sub.shape[0],)
        eps_shape = sample_shape + self.batch_shape + sub_event_shape
        eps = jax.random.normal(key, shape=eps_shape)[..., jnp.newaxis]

        # Reshape cond_prec for proper broadcasting
        cond_prec_reshaped = self.cond_prec[..., jnp.newaxis]

        lam_sub = cond_prec_reshaped * self.lam_sub
        lam_sub = lam_sub.reshape(self.batch_shape + (-1,))
        scale = jnp.sqrt(1 / lam_sub)[..., jnp.newaxis]
        result = self.loc + jnp.squeeze(jnp.matmul(self.U_sub, scale * eps), axis=-1)

        return result

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        """
        Compute the log probability density of the symmetric IGMRF2D distribution.

        The log probability is computed using the reduced-rank formulation that
        excludes zero eigenvalues. For a symmetric IGMRF with precision Q = τL,
        the log density (up to normalization) is:

            log p(x) = -1/2 × [log|Q₊| + (x - μ)ᵀ Q (x - μ)]

        where Q₊ denotes the pseudo-inverse (determinant over non-zero eigenvalues).

        The computation uses only the lower-triangular elements since the matrix
        is constrained to be symmetric.

        Parameters
        ----------
        value : ArrayLike
            Sample value at which to evaluate log probability.
            Shape must be batch_shape + event_shape + (num_nodes + (num_nodes + 1) // 2,).

        Returns
        -------
        Array
            Log probability density with shape batch_shape.

        Notes
        -----
        **Mathematical Details:**

        The normalization constant accounts for the reduced dimension after
        removing zero eigenvalues. The quadratic form xᵀQx is computed using
        the Laplacian L on the triangular representation.

        **Numerical Stability:**

        - Log-determinant computed in log-space to avoid overflow
        - Uses eigendecomposition rather than direct inversion
        - Only processes non-zero eigenvalues (filtered during initialization)
        """
        diff = value - self.loc

        # Compute log determinant using non-zero eigenvalues only
        cond_prec_reshaped = self.cond_prec[..., jnp.newaxis]

        lam_sub = cond_prec_reshaped * self.lam_sub
        lam_sub = lam_sub.reshape(self.batch_shape + (-1,))
        log_det = jnp.sum(jnp.log(lam_sub), axis=-1)

        # Compute quadratic form using the Laplacian
        quad = self.cond_prec * jnp.sum((diff @ self.L) * diff, axis=-1)

        # Number of non-zero eigenvalues determines the effective dimension
        reduced_dim = len(self.lam_sub)

        return jnp.squeeze(-0.5 * (reduced_dim * jnp.log(2 * jnp.pi) - log_det + quad))

    @staticmethod
    def infer_shapes(
        num_nodes=(),
        order=(),
        loc=(),
        cond_prec=(),
        tol=(),
        rescale_marginal_variance=(),
    ) -> None:
        """
        Infer batch and event shapes from parameter shapes.

        Note: This implementation cannot determine event_shape because it depends
        on the VALUE of num_nodes (not its shape). We raise NotImplementedError
        to be consistent with IGMRF and IGMRF2D implementations.

        Parameters
        ----------
        num_nodes : tuple, optional
            Shape of num_nodes parameter (not the value).
        order : tuple, optional
            Shape of order parameter (not the value).
        loc : tuple, optional
            Shape of loc parameter.
        cond_prec : tuple, optional
            Shape of cond_prec parameter.
        tol : tuple, optional
            Shape of tol parameter.

        Raises
        ------
        NotImplementedError
            Always raised because event_shape depends on parameter values.
        """
        raise NotImplementedError(
            "SymIGMRF2D.infer_shapes() cannot be implemented because event_shape "
            "depends on the value of num_nodes parameter, not its shape."
        )
