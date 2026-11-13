from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.typing import ArrayLike
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import validate_sample
from numpyro.util import is_prng_key

from ._SymIGMRF2D import diff_matrix_np


class IGMRF2D(Distribution):
    """
    2-dimensional Intrinsic Gaussian Markov Random Field (IGMRF) distribution.

    This distribution represents an isotropic Gaussian Markov Random Field over a 2D
    lattice. The precision matrix Q has the form:

        Q = cond_prec * (L₁ ⊗ I₂ + I₁ ⊗ L₂)

    where L₁ and L₂ are Laplacian matrices constructed from finite difference operators
    of the specified order, I₁ and I₂ are identity matrices, and ⊗ denotes the Kronecker
    product.

    The distribution is intrinsic (improper) because the combined Laplacian L₁⊗I₂ + I₁⊗L₂
    has zero eigenvalues corresponding to polynomial trends of degree less than the order
    (for order=1, just the constant/overall-mean direction). `sample()` draws directly
    from the reduced eigenspace orthogonal to these trends, so its output automatically
    satisfies the corresponding constraint (a sum-to-zero constraint when order=1).
    `log_prob()` assumes `value` already satisfies it too -- it does not check or project
    for this, it simply computes the proper density on that reduced subspace, so any
    component of `value` along the null space is silently ignored (see `log_prob`'s
    docstring for details).

    Mathematical Background
    -----------------------
    An IGMRF models spatial correlation through finite differences. For a 2D grid,
    the precision matrix penalizes roughness in both dimensions with a single shared
    conditional precision:

    - order = 1: penalizes first differences (random walk prior)
    - order = 2: penalizes second differences (smooth prior)

    Because L₁⊗I₂ and I₁⊗L₂ share the eigenvectors U₁⊗U₂ (where U₁, U₂ are the
    eigenvectors of L₁, L₂), the combined eigenvalue for eigenvector U₁[:,i]⊗U₂[:,j] is
    λ₁[i] + λ₂[j]. This eigenvalue is zero only when *both* λ₁[i] = 0 and λ₂[j] = 0, so the
    null space has dimension order² (not simply the product of independently-truncated
    subspaces). The reduced eigenspace used for sampling/likelihood is therefore built by
    filtering the combined eigenvalues by a small tolerance, rather than slicing each
    dimension's eigenvectors independently.

    If `rescale_marginal_variance=True`, `L` is additionally rescaled so that the
    geometric mean of the diagonal of its generalized inverse (the marginal variances
    implied by `L` at `cond_prec=1`) is 1, following Sørbye & Rue (2014). This makes
    `cond_prec` comparable across different `num_nodes`/`order` choices, since without
    it the same `cond_prec` value implies different amounts of smoothing depending on
    grid size and order.

    References
    ----------
    - Rue, H., & Held, L. (2005). Gaussian Markov Random Fields: Theory and Applications.
      Chapman & Hall/CRC.
    - Sørbye, S. H., & Rue, H. (2014). Scaling intrinsic Gaussian Markov random field
      priors in spatial modelling. Spatial Statistics, 8, 39-51.

    Examples
    --------
    >>> from cntmosaic.models.numpyro.distributions import IGMRF2D
    >>> import jax.numpy as jnp
    >>> from jax import random
    >>>
    >>> # Create a 2D IGMRF with first-order differences
    >>> dist = IGMRF2D(num_nodes=(10, 10), order=1, cond_prec=1.0)
    >>>
    >>> # Sample from the distribution
    >>> key = random.PRNGKey(0)
    >>> samples = dist.sample(key, sample_shape=(5,))  # 5 samples of shape (100,)
    >>>
    >>> # Batched precision parameter
    >>> dist_batch = IGMRF2D(
    ...     num_nodes=(5, 5),
    ...     order=2,
    ...     cond_prec=jnp.array([1.0, 2.0, 3.0]),
    ... )  # batch_shape=(3,), event_shape=(25,)

    See Also
    --------
    IGMRF : 1-dimensional version
    SymIGMRF2D : Symmetric 2D version, same isotropic construction restricted to the
        lower-triangular (symmetric) domain
    """

    support = constraints.real_vector
    reparametrized_params = ["loc", "cond_prec"]
    pytree_data_fields = ("loc", "cond_prec", "lam_sub", "U_sub", "L")
    pytree_aux_fields = ("num_nodes", "order")

    def __init__(
        self,
        num_nodes: tuple[int, int],
        order: int,
        loc: ArrayLike = 0.0,
        cond_prec: ArrayLike = 1.0,
        tol: float = 1e-10,
        rescale_marginal_variance: bool = False,
        *,
        validate_args: Optional[bool] = None,
    ):
        """
        Initialize a 2D Intrinsic Gaussian Markov Random Field distribution.

        Parameters
        ----------
        num_nodes : tuple[int, int]
            Number of nodes in the grid for each dimension (n₁, n₂).
            Must be greater than order.
        order : int
            Order of the finite difference approximation, shared by both dimensions.
            Must satisfy: 0 < order < min(num_nodes).
        loc : ArrayLike, optional
            Location parameter (mean) of the distribution. Default is 0.0.
            Shape options:
            - Scalar (): broadcasted to (batch_shape, n₁ * n₂)
            - (n₁ * n₂,): no batch dimension
            - (batch_shape, n₁ * n₂): batched means
            The flattened grid uses row-major (C) order.
        cond_prec : ArrayLike, optional
            Conditional precision (inverse variance) parameter controlling smoothness.
            Default is 1.0. Shape options:
            - Scalar (): no batch dimension
            - (batch_shape,): batched precision parameters
            Must be positive.
        tol : float, optional
            Tolerance for filtering near-zero eigenvalues of the combined Laplacian.
            Eigenvalues below this threshold are treated as null and excluded from
            sampling and likelihood computations. Default is 1e-10.
        rescale_marginal_variance : bool, optional
            If True, rescale `L` (and its eigenvalues) so that the geometric mean of
            the diagonal of `L`'s generalized inverse
            `Sigma = U_sub @ diag(1 / lam_sub) @ U_sub.T` (the marginal variances at
            `cond_prec=1`) equals 1, following Sørbye & Rue (2014). This makes
            `cond_prec` comparable across different `num_nodes`/`order` choices.
            Default is False.
        validate_args : bool, optional
            Whether to validate input arguments. Default is None.
            When True, checks parameter constraints (e.g., positive precision).

        Attributes
        ----------
        batch_shape : tuple[int, ...]
            Shape of parameter batches, inferred from loc, cond_prec.
        event_shape : tuple[int, ...]
            Shape of a single sample, always (n₁ * n₂,).
        L : Array
            Combined Laplacian matrix L₁⊗I₂ + I₁⊗L₂, shape (n₁*n₂, n₁*n₂).
        lam_sub : Array
            Non-zero (structural, precision-independent) eigenvalues of L, shape
            (n₁*n₂ - order²,).
        U_sub : Array
            Eigenvectors corresponding to lam_sub, shape (..., n₁*n₂, n₁*n₂ - order²).

        Raises
        ------
        ValueError
            If order >= min(num_nodes).
            If cond_prec is non-positive (when validate_args=True).
        """

        self.num_nodes = num_nodes
        self.order = order
        n1, n2 = num_nodes

        # Build Laplacians and their eigendecomposition in NumPy (not JAX) to avoid
        # TracerArrayConversionError when this is called inside a JIT-compiled body:
        # the reduced eigenspace below is selected via a data-dependent boolean mask,
        # which requires concrete (untraced) arrays to produce a static output shape.
        D1 = diff_matrix_np(n1, order)
        D2 = diff_matrix_np(n2, order)
        L1 = D1.T @ D1
        L2 = D2.T @ D2

        lam1, U1 = np.linalg.eigh(L1)
        lam2, U2 = np.linalg.eigh(L2)

        # L1⊗I2 and I1⊗L2 share the eigenvectors U1⊗U2; the combined eigenvalue for
        # U1[:,i]⊗U2[:,j] is lam1[i] + lam2[j], which is zero only when both terms are.
        lam_full = (lam1[:, None] + lam2[None, :]).reshape(-1)
        U_full = np.kron(U1, U2)

        nonzero_mask = lam_full > tol
        lam_sub_raw = lam_full[nonzero_mask]
        U_sub = U_full[:, nonzero_mask]

        if rescale_marginal_variance:
            # Sørbye & Rue (2014): rescale L so the geometric mean of the diagonal of
            # its generalized inverse Sigma = U_sub @ diag(1/lam_sub) @ U_sub.T (the
            # marginal variances at cond_prec=1) is 1, making cond_prec comparable
            # across grid sizes/orders. diag(Sigma)_k = sum_j U_sub[k,j]^2 / lam_sub[j],
            # computed directly rather than forming the full n1*n2 x n1*n2 matrix.
            marginal_var = np.sum((U_sub**2) / lam_sub_raw, axis=1)
            scale = np.exp(np.mean(np.log(marginal_var)))
        else:
            scale = 1.0

        self.lam_sub = jnp.asarray(scale * lam_sub_raw)
        self.L = jnp.asarray(
            scale * (np.kron(L1, np.eye(n2)) + np.kron(np.eye(n1), L2))
        )

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
        self.loc = jnp.broadcast_to(loc, batch_shape + (n1 * n2,))
        self.cond_prec = jnp.broadcast_to(cond_prec, batch_shape)

        # Broadcast U_sub if there's a batch dimension
        if batch_shape:
            self.U_sub = jnp.broadcast_to(jnp.asarray(U_sub), batch_shape + U_sub.shape)
        else:
            self.U_sub = jnp.asarray(U_sub)

        event_shape = (n1 * n2,)

        super(IGMRF2D, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(
        self, key: jax.Array, sample_shape: tuple[int, ...] = ()
    ) -> jax.Array:
        """
        Sample from the IGMRF2D distribution.

        Sampling is performed by:
        1. Generating standard normal samples in the reduced eigenspace
        2. Scaling by the inverse square root of eigenvalues and cond_prec
        3. Transforming back to the original space via eigenvectors
        4. Adding the location parameter

        This avoids explicit matrix inversion and leverages the eigendecomposition.

        Parameters
        ----------
        key : jax.Array
            PRNG key for random number generation.
        sample_shape : tuple[int, ...], optional
            Shape of the sample batch to generate. Default is ().
            Final sample shape will be sample_shape + batch_shape + event_shape.

        Returns
        -------
        Array
            Samples from the distribution with shape:
            sample_shape + batch_shape + (num_nodes[0] * num_nodes[1],)
        """
        assert is_prng_key(key)
        sub_event_shape = (self.lam_sub.shape[-1],)
        eps_shape = sample_shape + self.batch_shape + sub_event_shape
        eps = jax.random.normal(key, shape=eps_shape)[..., jnp.newaxis]

        cond_prec_reshaped = self.cond_prec[..., jnp.newaxis, jnp.newaxis]

        scale = (1.0 / jnp.sqrt(self.lam_sub))[..., jnp.newaxis] / jnp.sqrt(
            cond_prec_reshaped
        )
        result = self.loc + jnp.squeeze(jnp.matmul(self.U_sub, scale * eps), axis=-1)

        return result

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        """
        Compute the log probability density of the IGMRF2D distribution.

        `Q = cond_prec * L` is rank-deficient: `L` has an `order²`-dimensional null
        space spanned by polynomial trends of degree less than `order` in each
        dimension (e.g. just the constant vector when `order=1`), which is what makes
        this an intrinsic (improper) GMRF. The density below is therefore the proper
        Gaussian density on the `reduced_dim = n1*n2 - order²`-dimensional subspace
        orthogonal to that null space, i.e. it implicitly assumes `value` has already
        been constrained to that subspace (a sum-to-zero constraint for `order=1`;
        higher-order polynomial-trend constraints for `order>1`). This is *not*
        checked or enforced here: because the null space of `L` is exactly the kernel
        of `Q`, any component of `value - loc` lying in it contributes nothing to the
        quadratic form and is silently ignored.

        Using the reduced-rank formulation over the `reduced_dim` non-zero eigenvalues
        `lam_sub` of `L`, the log density is:

        log p(x) = -1/2 * [reduced_dim * log(2π / cond_prec) - log|L₊| + cond_prec * (x - μ)ᵀ L (x - μ)]

        where `log|L₊| = sum(log(lam_sub))` is the precision-independent pseudo
        log-determinant of `L`; `cond_prec`'s contribution to the full pseudo
        log-determinant `log|Q₊| = reduced_dim * log(cond_prec) + log|L₊|` is carried
        through the `2π / cond_prec` term instead of through `log|L₊|`.

        Parameters
        ----------
        value : ArrayLike
            Sample value at which to evaluate log probability. Assumed to already lie
            in the subspace orthogonal to `L`'s null space (see above).
            Shape must be batch_shape + event_shape.

        Returns
        -------
        Array
            Log probability density, shape is batch_shape.
        """
        diff = value - self.loc

        # Structural (precision-independent) log determinant of the non-zero
        # eigenvalues. cond_prec's contribution to log|Q| is carried through the
        # `2 * pi / cond_prec` term below instead of being folded in here.
        log_det = jnp.sum(jnp.log(self.lam_sub))

        # Compute quadratic form: xᵀ Q x using the precomputed combined Laplacian
        quad = self.cond_prec * jnp.sum((diff @ self.L) * diff, axis=-1)

        # Number of non-zero eigenvalues (dynamic: n1*n2 - order² for the standard
        # finite-difference construction, derived from the tolerance-filtered mask)
        reduced_dim = self.lam_sub.shape[-1]

        return jnp.squeeze(
            -0.5 * (reduced_dim * jnp.log(2 * jnp.pi / self.cond_prec) - log_det + quad)
        )

    @staticmethod
    def infer_shapes(
        num_nodes=(),
        order=(),
        loc=(),
        cond_prec=(),
        tol=(),
    ) -> None:
        """
        Infer batch and event shapes from parameter shapes.

        Note: This implementation cannot determine event_shape because it depends
        on the VALUES of num_nodes (not its shape). We raise NotImplementedError
        to be consistent with the 1D IGMRF implementation.

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
            "IGMRF2D.infer_shapes() cannot be implemented because event_shape "
            "depends on the value of num_nodes parameter, not its shape."
        )
