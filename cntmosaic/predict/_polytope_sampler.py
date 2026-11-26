"""
Hit-and-run sampler for the uniform distribution on a transportation polytope.

The transportation polytope ``T(r, c)`` is the set of non-negative tables with
row sums ``r`` and column sums ``c``.  Moves are made along isotropic directions
in the space of tables with zero row and column sums, obtained by double
centering a standard Gaussian matrix (``D = C Z C``), and the step length is
drawn uniformly on the chord that keeps every cell non-negative.

Optionally, some diagonal cells ``x_kk`` are fixed.  In the contact setting
these are the within-stratum contacts, so fixing them states how assortative
the mixing is.  Directions are then projected onto the tables that also have
a zero diagonal on the fixed cells, and the chain starts at the
quasi-independence table of the remaining cells (supplement, section A.5).

References
----------
Smith, R. L. (1984). Efficient Monte Carlo procedures for generating points
uniformly from bounded regions. Operations Research 32, 1296-1311.

Deming, W. E. and Stephan, F. F. (1940). On a least squares adjustment of a
sampled frequency table when the expected marginal totals are known. The
Annals of Mathematical Statistics 11, 427-444.
"""

from __future__ import annotations

from functools import partial
from math import prod
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike


def check_margins(
    row_sums: ArrayLike,
    col_sums: ArrayLike,
    rtol: float = 1e-6,
) -> Tuple[Array, Array]:
    """
    Check that row and column sums define a non-empty transportation polytope.

    Parameters
    ----------
    row_sums : ArrayLike
        Row sums ``r`` of shape ``(..., R)``.
    col_sums : ArrayLike
        Column sums ``c`` of shape ``(..., C)``.  Leading (batch) dimensions
        must match those of ``row_sums``.
    rtol : float, optional
        Relative tolerance when comparing the totals ``sum(r)`` and ``sum(c)``.

    Returns
    -------
    Tuple[Array, Array]
        The margins as floating point arrays.

    Raises
    ------
    ValueError
        If the margins are not finite and non-negative, if their batch shapes
        differ, or if their totals are zero or differ from one another.
    """
    dtype = jnp.result_type(float)
    r = jnp.asarray(row_sums, dtype=dtype)
    c = jnp.asarray(col_sums, dtype=dtype)

    if r.ndim == 0 or c.ndim == 0:
        raise ValueError("Row and column sums must be at least one-dimensional.")
    if r.shape[:-1] != c.shape[:-1]:
        raise ValueError(
            f"Batch shapes of row sums {r.shape[:-1]} and column sums "
            f"{c.shape[:-1]} do not match."
        )
    for name, m in (("Row", r), ("Column", c)):
        if not jnp.all(jnp.isfinite(m)):
            raise ValueError(f"{name} sums must be finite.")
        if jnp.any(m < 0):
            raise ValueError(f"{name} sums must be non-negative.")

    r_total, c_total = r.sum(-1), c.sum(-1)
    if jnp.any(r_total <= 0):
        raise ValueError("Total of the margins must be positive.")
    if not jnp.allclose(r_total, c_total, rtol=rtol, atol=0.0):
        raise ValueError(
            "Row and column sums must have the same total, got "
            f"{r_total} and {c_total}."
        )
    return r, c


def independence_table(row_sums: Array, col_sums: Array) -> Array:
    """
    Independence table ``r c^T / N``, which always lies in ``T(r, c)``.

    Parameters
    ----------
    row_sums : Array
        Row sums of shape ``(R,)``.
    col_sums : Array
        Column sums of shape ``(C,)``.

    Returns
    -------
    Array
        Table of shape ``(R, C)``.
    """
    return jnp.outer(row_sums, col_sums) / row_sums.sum()


def diagonal_is_feasible(
    row_sums: ArrayLike,
    col_sums: ArrayLike,
    diagonal: ArrayLike,
) -> Array:
    """
    Check whether fixed diagonal values leave room for a strictly positive table.

    Let ``S`` be the strata whose diagonal is fixed, ``x_kk = delta_k`` for
    ``k`` in ``S``, and let ``r' = r - delta``, ``c' = c - delta`` and
    ``N' = N - sum(delta)`` be the residual margins and mass (``delta_k = 0``
    outside ``S``).  A table in the constrained polytope with every free cell
    strictly positive exists if and only if ``delta >= 0``, ``r' > 0``,
    ``c' > 0`` and ``r'_k + c'_k < N'`` for every ``k`` in ``S``
    (Proposition 4 of the supplement).  Hit-and-run must start at such a
    table.

    Parameters
    ----------
    row_sums, col_sums : ArrayLike
        Margins of shape ``(..., K)``.
    diagonal : ArrayLike
        Fixed diagonal values of shape ``(..., K)``; ``NaN`` marks a free
        diagonal cell.

    Returns
    -------
    Array
        Boolean array of shape ``(...)``.
    """
    r, c = jnp.asarray(row_sums), jnp.asarray(col_sums)
    diagonal = jnp.asarray(diagonal, dtype=r.dtype)
    fixed = ~jnp.isnan(diagonal)
    delta = jnp.where(fixed, diagonal, 0.0)
    r_res, c_res = r - delta, c - delta
    n_res = r_res.sum(-1, keepdims=True)
    return (
        jnp.all(delta >= 0, -1)
        & jnp.all(r_res > 0, -1)
        & jnp.all(c_res > 0, -1)
        & jnp.all(~fixed | (r_res + c_res < n_res), -1)
    )


def _double_center(z: Array, mask: Array) -> Array:
    """
    Remove the row and column means of ``z`` over the cells in ``mask``.

    With every cell in ``mask`` this is ``C Z C`` with ``C = I - J / K``.
    """
    n_rows, n_cols = mask.any(1).sum(), mask.any(0).sum()
    z = z * mask
    row_mean = z.sum(1, keepdims=True) / n_cols
    col_mean = z.sum(0, keepdims=True) / n_rows
    grand_mean = z.sum() / (n_rows * n_cols)
    return (z - row_mean - col_mean + grand_mean) * mask


def _fixed_gram_inverse(fixed: Tuple[bool, ...]) -> Array:
    """
    Inverse of the Gram matrix ``G_S`` of the fixed diagonal constraints.

    ``G_S`` has entries ``C_jk^2`` for ``j, k`` in ``S`` (equation A.14).  It is
    padded with the identity outside ``S`` so that it is ``K x K`` and maps
    vectors that vanish outside ``S`` to vectors that vanish outside ``S``.
    """
    K = len(fixed)
    f = np.array(fixed)
    C = np.eye(K) - 1.0 / K
    G = np.where(np.outer(f, f), C**2, np.eye(K))
    return jnp.asarray(np.linalg.inv(G))


def _project_fixed_diagonal(w: Array, fixed: Array, G_inv: Array) -> Array:
    """
    Project ``W = C Z C`` onto the tables with zero row and column sums and a
    zero diagonal on ``S`` (equation A.17).

    ``D = W - C diag(lambda) C`` with ``lambda = G_S^{-1} (w_kk)_{k in S}``.
    The fixed diagonal cells of ``D`` are zero up to rounding and are set to
    zero exactly.
    """
    lam = G_inv @ jnp.where(fixed, jnp.diag(w), 0.0)
    d = w - _double_center(jnp.diag(lam), jnp.ones_like(w, dtype=bool))
    return jnp.where(jnp.diag(fixed), 0.0, d)


def _direction(
    key: Array,
    mask: Array,
    symmetric: bool,
    fixed: Tuple[bool, ...],
) -> Array:
    """
    Draw a unit-Frobenius-norm direction with zero row and column sums.

    Only cells in ``mask`` (rows and columns with positive margins) may move,
    so rows and columns with zero margins stay at zero.  When every margin is
    positive this is exactly ``C Z C / ||C Z C||_F``.  Fixed diagonal cells
    are projected out (equation A.17).  If ``symmetric``, the direction is
    replaced by its symmetric part ``(D + D^T) / 2``, which keeps zero row
    and column sums and a zero fixed diagonal.  If the polytope is a single
    point, the direction is zero.

    The projection is applied again after normalising.  When the polytope
    has few dimensions, the projected draw can be close to zero, and
    normalising it would magnify its rounding error, whose row and column
    sums are not zero, into a unit direction that moves the margins.
    Projecting again removes that error and leaves the exact direction
    unchanged.
    """
    G_inv = _fixed_gram_inverse(fixed) if any(fixed) else None

    def project(d):
        d = _double_center(d, mask)
        if any(fixed):
            d = _project_fixed_diagonal(d, jnp.array(fixed), G_inv)
        return (d + d.T) / 2 if symmetric else d

    d = project(jax.random.normal(key, mask.shape))
    d = project(d / jnp.linalg.norm(d))

    n_rows, n_cols = mask.any(1).sum(), mask.any(0).sum()
    if symmetric:
        dim = n_rows * (n_rows - 1) // 2 - sum(fixed)
    else:
        dim = (n_rows - 1) * (n_cols - 1) - sum(fixed)
    return jnp.where(dim <= 0, 0.0, d / jnp.linalg.norm(d))


def _chord(x: Array, d: Array) -> Tuple[Array, Array]:
    """Endpoints ``(t_-, t_+)`` of ``{t : x + t d >= 0}``."""
    ratio = -x / jnp.where(d == 0, 1.0, d)
    t_lo = jnp.max(jnp.where(d > 0, ratio, -jnp.inf))
    t_hi = jnp.min(jnp.where(d < 0, ratio, jnp.inf))
    return t_lo, t_hi


def _step(
    x: Array,
    key: Array,
    mask: Array,
    symmetric: bool,
    fixed: Tuple[bool, ...],
) -> Array:
    """One hit-and-run step from ``x``."""
    key_dir, key_t = jax.random.split(key)
    d = _direction(key_dir, mask, symmetric, fixed)
    t_lo, t_hi = _chord(x, d)
    t = jax.random.uniform(key_t, dtype=x.dtype, minval=t_lo, maxval=t_hi)
    # If the polytope is a single point then d = 0 and the chord is unbounded.
    x_new = x + jnp.where(jnp.isfinite(t), t, 0.0) * d
    if symmetric:
        # The compiled update may round a cell and its mirror differently; restore
        # exact symmetry.  This keeps the margins, since the row and column sums
        # are equal, and keeps every cell non-negative.
        x_new = (x_new + x_new.T) / 2
    if any(fixed):
        # Reset the fixed diagonal to prevent drift from rounding (section A.5.6).
        x_new = jnp.where(jnp.diag(jnp.array(fixed)), x, x_new)
    return x_new


def _quasi_independence(
    row_res: Array,
    col_res: Array,
    free: Array,
    symmetric: bool,
    max_sweeps: int = 1000,
) -> Array:
    """
    Quasi-independence table on the free cells, by iterative proportional fitting.

    Returns the non-negative table with margins ``(row_res, col_res)``, zeros
    outside ``free`` and every free cell strictly positive (equation A.18).
    Fitting stops once every row sum is within rounding error of ``row_res``,
    relative to that row sum.
    """
    rtol = 100 * jnp.finfo(row_res.dtype).eps

    def sweep(state):
        y, i = state
        y = y * (row_res / y.sum(1))[:, None]
        y = y * (col_res / y.sum(0))[None, :]
        return y, i + 1

    def not_converged(state):
        y, i = state
        row_error = jnp.abs(y.sum(1) / row_res - 1).max()
        return (row_error > rtol) & (i < max_sweeps)

    y, _ = jax.lax.while_loop(
        not_converged, sweep, sweep((free.astype(row_res.dtype), 0))
    )
    return (y + y.T) / 2 if symmetric else y


def _start(
    row_sums: Array,
    col_sums: Array,
    delta: Array,
    symmetric: bool,
    fixed: Tuple[bool, ...],
) -> Array:
    """
    Starting table: the independence table ``r c^T / N`` if no diagonal is
    fixed, else ``diag(delta)`` plus the quasi-independence table on the free
    cells with the residual margins.
    """
    if not any(fixed):
        return independence_table(row_sums, col_sums)
    fixed_cells = jnp.diag(jnp.array(fixed))
    y = _quasi_independence(row_sums - delta, col_sums - delta, ~fixed_cells, symmetric)
    return jnp.diag(delta) + y


def _run_chain(
    key: Array,
    row_sums: Array,
    col_sums: Array,
    delta: Array,
    num_samples: int,
    num_warmup: int,
    thin: int,
    symmetric: bool,
    fixed: Tuple[bool, ...],
) -> Array:
    """Run one chain; returns tables of shape ``(num_samples, R, C)``."""
    mask = jnp.outer(row_sums > 0, col_sums > 0)

    def advance(x, key, num_steps):
        def body(i, x):
            return _step(x, jax.random.fold_in(key, i), mask, symmetric, fixed)

        return jax.lax.fori_loop(0, num_steps, body, x)

    def sample(x, key):
        x = advance(x, key, thin)
        return x, x

    key_warmup, key_sample = jax.random.split(key)
    x = _start(row_sums, col_sums, delta, symmetric, fixed)
    x = advance(x, key_warmup, num_warmup)
    _, samples = jax.lax.scan(sample, x, jax.random.split(key_sample, num_samples))
    return samples


@partial(
    jax.jit,
    static_argnames=("num_samples", "num_warmup", "thin", "symmetric", "fixed"),
)
def _run_chains(
    keys, row_sums, col_sums, delta, num_samples, num_warmup, thin, symmetric, fixed
):
    run = partial(
        _run_chain,
        num_samples=num_samples,
        num_warmup=num_warmup,
        thin=thin,
        symmetric=symmetric,
        fixed=fixed,
    )
    return jax.vmap(run)(keys, row_sums, col_sums, delta)


def _fixed_diagonal(r: Array, c: Array, diagonal: ArrayLike) -> Tuple[Array, tuple]:
    """
    Validate fixed diagonal values against the margins.

    Returns the diagonal with zeros for free cells, and the set of fixed
    strata as a tuple of booleans (static for ``jax.jit``).
    """
    if r.shape != c.shape:
        raise ValueError("A fixed diagonal requires square tables.")
    diagonal = jnp.broadcast_to(jnp.asarray(diagonal, dtype=r.dtype), r.shape)
    is_fixed = ~jnp.isnan(diagonal).reshape(-1, r.shape[-1])
    if not jnp.all(is_fixed == is_fixed[0]):
        raise ValueError(
            "The same diagonal cells must be fixed in every table of the batch."
        )
    feasible = diagonal_is_feasible(r, c, diagonal)
    if not jnp.all(feasible):
        raise ValueError(
            f"{int((~feasible).sum())} of {feasible.size} tables have no strictly "
            "positive table with the fixed diagonal: each must satisfy "
            "delta >= 0, r - delta > 0, c - delta > 0 and "
            "(r_k - delta_k) + (c_k - delta_k) < N - sum(delta) for fixed k."
        )
    fixed = tuple(bool(f) for f in is_fixed[0])
    return jnp.where(jnp.isnan(diagonal), 0.0, diagonal), fixed


def sample_hit_run(
    key: Array,
    row_sums: ArrayLike,
    col_sums: ArrayLike,
    num_samples: int,
    num_warmup: int = 1600,
    thin: int = 1,
    symmetric: bool = False,
    diagonal: ArrayLike | None = None,
) -> Array:
    """
    Sample tables uniformly from the transportation polytope ``T(r, c)``.

    Runs the hit-and-run Markov chain started at the independence table
    ``r c^T / N``.  Each step draws an isotropic direction ``D`` with zero
    row and column sums and moves to a uniform point on the chord
    ``{X + t D : X + t D >= 0}``.  Rows and columns with zero margins are
    held at zero.  The chain runs on the margins divided by their total ``N``
    and the sampled tables are multiplied by ``N``, so the result does not
    depend on the scale of the margins.

    With ``symmetric=True`` the chain samples uniformly from the symmetric
    tables in ``T(r, r)``.  Directions are replaced by their symmetric part,
    and the start ``r r^T / N`` is symmetric, so every table stays symmetric.
    Hit-and-run leaves the uniform distribution invariant for any fixed
    direction distribution that is symmetric under ``D -> -D``, so the target
    is still exactly uniform.

    With ``diagonal`` the chain samples uniformly from the tables whose
    diagonal cells ``x_kk`` equal the given values for every ``k`` in a set
    ``S`` (Algorithm 2 of the supplement).  Directions are projected onto the
    tables with zero row and column sums and a zero diagonal on ``S``, and the
    chain starts at ``diag(delta)`` plus the quasi-independence table of the
    remaining cells.  Fixing ``m`` diagonal cells lowers the dimension of the
    polytope from ``(K - 1)^2`` to ``(K - 1)^2 - m``.

    Parameters
    ----------
    key : Array
        JAX PRNG key.
    row_sums : ArrayLike
        Row sums ``r`` of shape ``(..., R)``.
    col_sums : ArrayLike
        Column sums ``c`` of shape ``(..., C)``.  Leading dimensions are
        treated as a batch of independent polytopes, each with its own chain.
    num_samples : int
        Number of tables kept per chain.
    num_warmup : int, optional
        Number of burn-in steps discarded at the start of each chain
        (default 1600).  In a burn-in study with ``K = 10`` and equal margins
        (``docs/dev_docs/hit_and_run_sampler.ipynb``), tables drawn from 1000
        independent chains after about 1600 steps could not be told apart
        from the uniform distribution: the largest cell bias and
        Kolmogorov-Smirnov distance had reached their noise floors.  Once the
        chain is that far from the start, a single table per chain is enough,
        e.g. ``num_samples=1``.  Strongly unequal margins mix more slowly (in
        the same study, margins with ratios of about 10 needed more than
        25 600 steps), so pass a larger value for such margins.  For
        ``2 x 2`` tables the polytope is a line segment and a single step is
        already an exact draw.
    thin : int, optional
        Number of hit-and-run steps between kept tables.
    symmetric : bool, optional
        If ``True``, sample symmetric tables.  Requires square tables with
        ``row_sums == col_sums``.
    diagonal : ArrayLike or None, optional
        Fixed diagonal values ``delta`` of shape ``(..., K)``, or ``(K,)`` for
        the whole batch; ``NaN`` marks a free diagonal cell.  The same cells
        must be fixed in every table, and every table must pass
        :func:`diagonal_is_feasible`.  Requires square tables.

    Returns
    -------
    Array
        Sampled tables of shape ``(..., num_samples, R, C)``.

    Raises
    ------
    ValueError
        If the margins are invalid (see :func:`check_margins`), if
        ``symmetric=True`` with unequal margins, or if the fixed diagonal is
        infeasible for some table.

    Examples
    --------
    >>> import jax
    >>> tables = sample_hit_run(
    ...     jax.random.key(0), [3.0, 1.0], [2.0, 2.0], num_samples=100
    ... )
    >>> tables.shape
    (100, 2, 2)
    """
    r, c = check_margins(row_sums, col_sums)
    if symmetric and (r.shape != c.shape or not jnp.allclose(r, c)):
        raise ValueError("Symmetric tables require equal row and column sums.")
    if diagonal is None:
        delta, fixed = jnp.zeros_like(r), ()
    else:
        delta, fixed = _fixed_diagonal(r, c, diagonal)

    batch_shape, (R, C) = r.shape[:-1], (r.shape[-1], c.shape[-1])
    num_chains = prod(batch_shape)

    # Run on margins with total 1 so the chain is independent of the scale of r, c.
    total = r.sum(-1, keepdims=True)
    samples = _run_chains(
        jax.random.split(key, num_chains),
        (r / total).reshape(num_chains, R),
        (c / total).reshape(num_chains, C),
        (delta / total).reshape(num_chains, R),
        num_samples=num_samples,
        num_warmup=num_warmup,
        thin=thin,
        symmetric=symmetric,
        fixed=fixed,
    )
    return samples.reshape(*batch_shape, num_samples, R, C) * total[..., None, None]
