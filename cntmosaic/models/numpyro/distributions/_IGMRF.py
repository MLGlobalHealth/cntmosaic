from __future__ import annotations

from typing import Optional

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.typing import ArrayLike
from numpyro.distributions import constraints
from numpyro.distributions.distribution import Distribution
from numpyro.distributions.util import validate_sample
from numpyro.util import is_prng_key

from ._SymIGMRF2D import diff_matrix_np


class IGMRF(Distribution):
    """
    1-dimensional Intrinsic Gaussian Markov Random Field (IGMRF) distribution.

    Models a chain of `num_nodes` nodes with precision matrix `Q = cond_prec * L`,
    where `L = D^T @ D` is a Laplacian built from an `order`-th order finite
    difference operator `D`. `L` penalizes roughness in the sequence:

    - order = 1: penalizes first differences (random walk prior)
    - order = 2: penalizes second differences (smooth/trend prior)

    `L` is rank-deficient (`Q` is intrinsic/improper) because it has an
    `order`-dimensional null space spanned by polynomial trends of degree less
    than `order` (for order=1, just the constant vector). `sample()` draws
    directly from the reduced eigenspace orthogonal to this null space, so its
    output automatically satisfies the corresponding constraint (sum-to-zero for
    order=1). `log_prob()` assumes `value` already satisfies it too -- see its
    docstring for details.

    If `rescale_marginal_variance=True`, `L` is additionally rescaled so that the
    geometric mean of the diagonal of its generalized inverse (the marginal
    variances implied by `L` at `cond_prec=1`) is 1, following Sørbye & Rue
    (2014). This makes `cond_prec` comparable across different
    `num_nodes`/`order` choices, since without it the same `cond_prec` value
    implies different amounts of smoothing depending on chain length and order.

    References
    ----------
    - Rue, H., & Held, L. (2005). Gaussian Markov Random Fields: Theory and
      Applications. Chapman & Hall/CRC.
    - Sørbye, S. H., & Rue, H. (2014). Scaling intrinsic Gaussian Markov random
      field priors in spatial modelling. Spatial Statistics, 8, 39-51.

    See Also
    --------
    IGMRF2D : 2-dimensional (isotropic lattice) version
    """

    arg_constraints = {
        "num_nodes": constraints.positive_integer,
        "order": constraints.positive_integer,
        "cond_prec": constraints.positive,
    }
    support = constraints.real_vector
    reparametrized_params = ["loc", "cond_prec"]
    pytree_data_fields = ("loc", "cond_prec", "lam_sub", "U_sub", "L")
    pytree_aux_fields = ("num_nodes", "order")

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
        Initialize a 1D Intrinsic Gaussian Markov Random Field distribution.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the chain. Must be greater than order.
        order : int
            Order of the finite difference operator. Must satisfy
            0 < order < num_nodes.
        loc : ArrayLike, optional
            Location parameter (mean) of the distribution. Default is 0.0.
            Shape options:
            - Scalar (): broadcasted to (batch_shape, num_nodes)
            - (num_nodes,): no batch dimension
            - (batch_shape, num_nodes): batched means
        cond_prec : ArrayLike, optional
            Conditional precision (inverse variance) parameter controlling
            smoothness. Default is 1.0. Shape options:
            - Scalar (): no batch dimension
            - (batch_shape,): batched precision parameters
            Must be positive.
        tol : float, optional
            Tolerance for filtering near-zero eigenvalues of `L`. Eigenvalues
            below this threshold are treated as null and excluded from sampling
            and likelihood computations. Default is 1e-10.
        rescale_marginal_variance : bool, optional
            If True, rescale `L` (and its eigenvalues) so that the geometric mean
            of the diagonal of `L`'s generalized inverse
            `Sigma = U_sub @ diag(1 / lam_sub) @ U_sub.T` (the marginal variances
            at `cond_prec=1`) equals 1, following Sørbye & Rue (2014). Default is
            False.
        validate_args : bool, optional
            Whether to validate input arguments. Default is None.

        Attributes
        ----------
        batch_shape : tuple[int, ...]
            Shape of parameter batches, inferred from loc, cond_prec.
        event_shape : tuple[int, ...]
            Shape of a single sample, always (num_nodes,).
        L : Array
            Laplacian matrix D^T @ D, shape (num_nodes, num_nodes).
        lam_sub : Array
            Non-zero (structural, precision-independent) eigenvalues of L, shape
            (num_nodes - order,).
        U_sub : Array
            Eigenvectors corresponding to lam_sub, shape
            (..., num_nodes, num_nodes - order).
        """

        self.num_nodes = num_nodes
        self.order = order

        # Build the Laplacian and its eigendecomposition in NumPy (not JAX) to
        # avoid TracerArrayConversionError when this is called inside a
        # JIT-compiled body: the reduced eigenspace below is selected via a
        # data-dependent boolean mask, which requires concrete (untraced) arrays
        # to produce a static output shape.
        D = diff_matrix_np(num_nodes, order)
        L = D.T @ D
        lam, U = np.linalg.eigh(L)

        nonzero_mask = lam > tol
        lam_sub_raw = lam[nonzero_mask]
        U_sub = U[:, nonzero_mask]

        if rescale_marginal_variance:
            # Sørbye & Rue (2014): rescale L so the geometric mean of the diagonal
            # of its generalized inverse Sigma = U_sub @ diag(1/lam_sub) @ U_sub.T
            # (the marginal variances at cond_prec=1) is 1, making cond_prec
            # comparable across chain lengths/orders. diag(Sigma)_k = sum_j
            # U_sub[k,j]^2 / lam_sub[j], computed directly rather than forming the
            # full num_nodes x num_nodes matrix.
            marginal_var = np.sum((U_sub**2) / lam_sub_raw, axis=1)
            scale = np.exp(np.mean(np.log(marginal_var)))
        else:
            scale = 1.0

        self.lam_sub = jnp.asarray(scale * lam_sub_raw)
        self.L = jnp.asarray(scale * L)

        # Convert to jax arrays
        loc = jnp.asarray(loc)
        cond_prec = jnp.asarray(cond_prec)

        # Determine batch shape from inputs
        if jnp.ndim(loc) == 0:  # Scalar loc: no batch dimension
            loc_batch_shape = ()
        else:  # loc has shape (batch_shape, num_nodes)
            loc_batch_shape = jnp.shape(loc)[:-1]

        if jnp.ndim(cond_prec) == 0:  # Scalar cond_prec: no batch dimension
            cond_prec_batch_shape = ()
        else:  # cond_prec has shape (batch_shape,)
            cond_prec_batch_shape = jnp.shape(cond_prec)

        batch_shape = lax.broadcast_shapes(loc_batch_shape, cond_prec_batch_shape)

        # Broadcast loc to (batch_shape, num_nodes)
        self.loc = jnp.broadcast_to(loc, batch_shape + (num_nodes,))

        # Broadcast cond_prec to (batch_shape,)
        self.cond_prec = jnp.broadcast_to(cond_prec, batch_shape)

        # Broadcast U_sub if there's a batch dimension
        if batch_shape:
            self.U_sub = jnp.broadcast_to(jnp.asarray(U_sub), batch_shape + U_sub.shape)
        else:
            self.U_sub = jnp.asarray(U_sub)

        event_shape = (num_nodes,)

        super(IGMRF, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(
        self, key: jax.Array, sample_shape: tuple[int, ...] = ()
    ) -> jax.Array:
        """
        Sample from the IGMRF distribution.

        Draws standard normal samples in the reduced (non-null) eigenspace,
        scales by the inverse square root of `lam_sub * cond_prec`, and
        transforms back to the original space via `U_sub`, avoiding explicit
        matrix inversion.

        Parameters
        ----------
        key : jax.Array
            PRNG key for random number generation.
        sample_shape : tuple[int, ...], optional
            Shape of the sample batch to generate. Default is ().

        Returns
        -------
        Array
            Samples with shape sample_shape + batch_shape + (num_nodes,).
        """
        assert is_prng_key(key)
        # Sample from the reduced space corresponding to non-zero eigenvalues
        eps_shape = sample_shape + self.batch_shape + self.lam_sub.shape[-1:]
        eps = random.normal(key, shape=eps_shape)[..., jnp.newaxis]

        # Add two trailing axes to cond_prec for broadcasting
        cond_prec_reshaped = self.cond_prec[..., jnp.newaxis, jnp.newaxis]

        scale = 1 / self.lam_sub[..., jnp.newaxis]
        scale = scale / cond_prec_reshaped
        scale = jnp.sqrt(scale)

        result = self.loc + jnp.squeeze(jnp.matmul(self.U_sub, scale * eps), axis=-1)

        return result

    @validate_sample
    def log_prob(self, value: ArrayLike) -> ArrayLike:
        """
        Compute the log probability density of the IGMRF distribution.

        `Q = cond_prec * L` is rank-deficient, so this is the proper Gaussian
        density on the `reduced_dim = num_nodes - order`-dimensional subspace
        orthogonal to `L`'s null space, i.e. it implicitly assumes `value` has
        already been constrained to that subspace (a sum-to-zero constraint for
        `order=1`). This is *not* checked or enforced here: any component of
        `value - loc` lying in the null space contributes nothing to the
        quadratic form and is silently ignored.

        log p(x) = -1/2 * [reduced_dim * log(2π / cond_prec) - log|L₊| + cond_prec * (x - μ)ᵀ L (x - μ)]

        where `log|L₊| = sum(log(lam_sub))` is the precision-independent pseudo
        log-determinant of `L`; `cond_prec`'s contribution to the full pseudo
        log-determinant is carried through the `2π / cond_prec` term instead.

        Parameters
        ----------
        value : ArrayLike
            Sample value at which to evaluate log probability. Assumed to
            already lie in the subspace orthogonal to `L`'s null space (see
            above). Shape must be batch_shape + event_shape.

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

        # Compute quadratic form: xᵀ Q x using the precomputed Laplacian
        quad = self.cond_prec * jnp.sum((diff @ self.L) * diff, axis=-1)

        # Number of non-zero eigenvalues (dynamic: num_nodes - order for the
        # standard finite-difference construction, derived from the
        # tolerance-filtered mask)
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

        Note: This implementation cannot determine event_shape because it
        depends on the VALUE of num_nodes (not its shape).

        Raises
        ------
        NotImplementedError
            Always raised because event_shape depends on parameter values.
        """
        raise NotImplementedError(
            "IGMRF.infer_shapes() cannot be implemented because event_shape "
            "depends on the value of num_nodes parameter, not its shape."
        )
