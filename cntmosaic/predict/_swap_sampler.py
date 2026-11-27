"""
Swap-move sampler for the uniform distribution on a transportation polytope.

The transportation polytope ``T(r, c)`` is the set of non-negative tables with
row sums ``r`` and column sums ``c``.  Each step of the sampler picks two
distinct rows ``i, i'`` and two distinct columns ``j, j'`` at random and adds
``t`` to cells ``(i, j)`` and ``(i', j')`` and ``-t`` to cells ``(i, j')`` and
``(i', j)``, which leaves every row and column sum unchanged.  The step ``t``
is drawn uniformly from the interval that keeps the four cells non-negative.
This is a continuous analogue of the swap moves of Diaconis and Sturmfels
(1998), and the uniform distribution on ``T(r, c)`` is its stationary
distribution.

For ``2 x 2`` tables a single cell determines the whole table, and the tables
are drawn exactly and independently instead.

Symmetric tables use each swap together with its mirror image.  When the
whole diagonal is fixed (section B of the supplement, Algorithm 2), each move
is either a basic swap on four distinct strata or a three-cycle on three
distinct strata; neither touches the diagonal.  Every move reads and writes
only the cells it changes: four for a swap, six for a three-cycle, and eight
for a mirrored swap.

References
----------
Diaconis, P. and Sturmfels, B. (1998). Algebraic algorithms for sampling from
conditional distributions. The Annals of Statistics 26, 363-397.
"""

from __future__ import annotations

from functools import partial
from math import prod
from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ._polytope_sampler import _fixed_diagonal, _start, check_margins


def _sample_two_by_two(
    key: Array, row_sums: Array, col_sums: Array, num_samples: int
) -> Array:
    """
    Draw ``2 x 2`` tables exactly from the uniform distribution on ``T(r, c)``.

    The cell ``x = x_11`` determines the table,
    ``[[x, r_1 - x], [c_1 - x, r_2 - c_1 + x]]``, and non-negativity holds for
    ``max(0, r_1 + c_1 - N) <= x <= min(r_1, c_1)`` (the Frechet bounds).  The
    uniform distribution on ``T(r, c)`` is the uniform distribution of ``x``
    on this interval.

    Parameters
    ----------
    key : Array
        JAX PRNG key.
    row_sums, col_sums : Array
        Margins of shape ``(..., 2)``.
    num_samples : int
        Number of tables per pair of margins.

    Returns
    -------
    Array
        Independent tables of shape ``(..., num_samples, 2, 2)``.
    """
    r1, r2 = row_sums[..., :1], row_sums[..., 1:]  # (..., 1)
    c1 = col_sums[..., :1]
    lower = jnp.maximum(0.0, r1 + c1 - row_sums.sum(-1, keepdims=True))
    upper = jnp.minimum(r1, c1)

    batch_shape = row_sums.shape[:-1]
    u = jax.random.uniform(key, (*batch_shape, num_samples), dtype=row_sums.dtype)
    x = lower + u * (upper - lower)  # (..., num_samples)

    first_row = jnp.stack([x, r1 - x], axis=-1)
    second_row = jnp.stack([c1 - x, r2 - c1 + x], axis=-1)
    return jnp.stack([first_row, second_row], axis=-2)


def _swap_step(
    x: Array, rows: Tuple[Array, Array], cols: Tuple[Array, Array], u: Array
) -> Array:
    """
    Apply one swap move to ``x`` (Algorithm 1, lines 4-6).

    The move adds ``t`` to cells ``(i, j)`` and ``(i', j')`` and subtracts
    ``t`` from cells ``(i, j')`` and ``(i', j)``.  The four cells stay
    non-negative for ``t`` in ``[t_-, t_+]`` with
    ``t_- = -min(x_ij, x_i'j')`` and ``t_+ = min(x_ij', x_i'j)``, and
    ``t = t_- + u (t_+ - t_-)``.

    Parameters
    ----------
    x : Array
        Table of shape ``(R, C)``.
    rows, cols : Tuple[Array, Array]
        Distinct row indices ``(i, i')`` and distinct column indices
        ``(j, j')``.
    u : Array
        A number in ``[0, 1]`` that sets the position of ``t`` in its interval.

    Returns
    -------
    Array
        The updated table.
    """
    (i, i2), (j, j2) = rows, cols
    t_lo = -jnp.minimum(x[i, j], x[i2, j2])
    t_hi = jnp.minimum(x[i, j2], x[i2, j])
    # Clipping guards against rounding, so the four cells stay non-negative.
    t = jnp.clip(t_lo + u * (t_hi - t_lo), t_lo, t_hi)
    return x.at[i, j].add(t).at[i2, j2].add(t).at[i, j2].add(-t).at[i2, j].add(-t)


def _distinct_pairs(key: Array, n: int, size: int) -> Tuple[Array, Array]:
    """
    Draw ``size`` uniformly random pairs of distinct indices from ``0, ..., n - 1``.

    The second index is the first shifted by a uniform offset in
    ``1, ..., n - 1``, modulo ``n``.
    """
    key_first, key_offset = jax.random.split(key)
    first = jax.random.randint(key_first, (size,), 0, n)
    offset = jax.random.randint(key_offset, (size,), 1, n)
    return first, (first + offset) % n


def _sweep(x: Array, key: Array) -> Array:
    """
    Apply ``R * C`` swap moves to the ``R x C`` table ``x``.

    The random rows, columns and step positions of all moves in the sweep are
    drawn at once, and the moves are then applied one after another.
    """
    R, C = x.shape
    num_moves = R * C
    key_rows, key_cols, key_u = jax.random.split(key, 3)
    i, i2 = _distinct_pairs(key_rows, R, num_moves)
    j, j2 = _distinct_pairs(key_cols, C, num_moves)
    u = jax.random.uniform(key_u, (num_moves,), dtype=x.dtype)

    def move(s, x):
        return _swap_step(x, (i[s], i2[s]), (j[s], j2[s]), u[s])

    return jax.lax.fori_loop(0, num_moves, move, x)


# ----------------------------------------------------------------------
# Moves on a few cells: symmetric tables and a fixed diagonal (section B)
# ----------------------------------------------------------------------


def _sparse_step(x: Array, rows: Array, cols: Array, coefs: Array, u: Array) -> Array:
    """
    Move ``x`` along a direction given by a few cells, reading only those cells.

    The direction adds ``coefs[e]`` to cell ``(rows[e], cols[e])``.  A cell may
    appear in several entries (a symmetric swap whose rows and columns
    overlap), so each entry first takes the total coefficient of its cell.
    The step ``t`` is uniform on the interval that keeps the cells
    non-negative, and every entry's cell is set to ``x + t * total``.  The
    totals are 0, +-1 or +-2, so a step to an end of the interval sets that
    cell to zero exactly.

    Parameters
    ----------
    x : Array
        Table of shape ``(R, C)``.
    rows, cols, coefs : Array
        Cells and coefficients of the direction, each of shape ``(m,)``.
    u : Array
        A number in ``[0, 1]`` that sets the position of ``t`` in its interval.

    Returns
    -------
    Array
        The updated table.
    """
    same_cell = (rows[:, None] == rows[None, :]) & (cols[:, None] == cols[None, :])
    total = (same_cell * coefs[None, :]).sum(axis=1)
    values = x[rows, cols]

    ratio = -values / jnp.where(total == 0, 1.0, total)
    t_lo = jnp.max(jnp.where(total > 0, ratio, -jnp.inf))
    t_hi = jnp.min(jnp.where(total < 0, ratio, jnp.inf))
    t = jnp.clip(t_lo + u * (t_hi - t_lo), t_lo, t_hi)
    # If every coefficient cancels, the interval is unbounded and nothing moves.
    t = jnp.where(jnp.isfinite(t), t, 0.0)
    return x.at[rows, cols].set(values + t * total)


def _swap_cells(i, i2, j, j2, dtype, pad: int = 0) -> Tuple[Array, Array, Array]:
    """
    Cells of the basic swap ``D^(i,i';j,j')`` (A.5): ``+1`` on ``(i, j)`` and
    ``(i', j')``, ``-1`` on ``(i, j')`` and ``(i', j)``.

    ``pad`` extra entries with coefficient 0 on cell ``(i, j)`` give the swap
    the same length as a three-cycle; they change nothing.
    """
    rows = jnp.stack([i, i2, i, i2] + [i] * pad)
    cols = jnp.stack([j, j2, j2, j] + [j] * pad)
    coefs = jnp.array([1, 1, -1, -1] + [0] * pad, dtype=dtype)
    return rows, cols, coefs


def _cycle_cells(i, j, k, dtype) -> Tuple[Array, Array, Array]:
    """
    Cells of the three-cycle ``D^(i,j,k)`` (B.5): ``+1`` on ``(i, j)``,
    ``(j, k)`` and ``(k, i)``, ``-1`` on ``(i, k)``, ``(j, i)`` and ``(k, j)``.
    It moves mass around ``i -> j -> k -> i`` without touching the diagonal.
    """
    rows = jnp.stack([i, j, k, i, j, k])
    cols = jnp.stack([j, k, i, k, i, j])
    coefs = jnp.array([1, 1, 1, -1, -1, -1], dtype=dtype)
    return rows, cols, coefs


def _mirrored(rows: Array, cols: Array, coefs: Array) -> Tuple[Array, Array, Array]:
    """Cells of ``D + D^T``: each cell together with its mirror image."""
    return (
        jnp.concatenate([rows, cols]),
        jnp.concatenate([cols, rows]),
        jnp.concatenate([coefs, coefs]),
    )


def _symmetric_sweep(x: Array, key: Array) -> Array:
    """
    Apply ``K * K`` symmetric swap moves ``D + D^T`` to the symmetric table ``x``.

    Each move picks rows ``i != i'`` and columns ``j != j'`` uniformly, as a
    basic swap does, and changes at most eight cells.  A cell and its mirror
    image receive the same value, so ``x`` stays exactly symmetric.
    """
    K = x.shape[0]
    num_moves = K * K
    key_rows, key_cols, key_u = jax.random.split(key, 3)
    i, i2 = _distinct_pairs(key_rows, K, num_moves)
    j, j2 = _distinct_pairs(key_cols, K, num_moves)
    u = jax.random.uniform(key_u, (num_moves,), dtype=x.dtype)

    def move(s, x):
        cells = _mirrored(*_swap_cells(i[s], i2[s], j[s], j2[s], x.dtype))
        return _sparse_step(x, *cells, u[s])

    return jax.lax.fori_loop(0, num_moves, move, x)


def _fixed_diagonal_sweep(x: Array, key: Array, symmetric: bool) -> Array:
    """
    Apply ``K * K`` moves of Algorithm 2 to ``x``, whose whole diagonal is fixed.

    Each move draws distinct strata uniformly.  With probability ``p`` it is a
    three-cycle on three of them, otherwise a basic swap with two of them as
    rows and the other two as columns.  Neither touches the diagonal, so both
    are always admissible and the diagonal keeps its exact values.  ``p = 1``
    for ``K = 3``, where no basic swap exists, and ``p = 1/2`` otherwise.

    For symmetric tables each swap is applied with its mirror image, and
    three-cycles are not used because ``D + D^T`` vanishes for them.  With
    ``K = 3`` no symmetric move exists: the only table is the start.
    """
    K = x.shape[0]
    if symmetric and K == 3:
        return x

    num_moves = K * K
    num_strata = 3 if K == 3 else 4
    key_strata, key_cycle, key_u = jax.random.split(key, 3)
    strata = jax.vmap(lambda k: jax.random.choice(k, K, (num_strata,), replace=False))(
        jax.random.split(key_strata, num_moves)
    )
    u = jax.random.uniform(key_u, (num_moves,), dtype=x.dtype)
    p = 0.0 if symmetric else (1.0 if K == 3 else 0.5)
    use_cycle = jax.random.uniform(key_cycle, (num_moves,)) < p

    def cells(s):
        a = strata[s]
        cycle = _cycle_cells(a[0], a[1], a[2], x.dtype)
        if K == 3:
            return cycle
        swap = _swap_cells(a[0], a[1], a[2], a[3], x.dtype, pad=2)
        if symmetric:
            return _mirrored(*_swap_cells(a[0], a[1], a[2], a[3], x.dtype))
        # Select the move entry by entry, so that only six cells are touched.
        return tuple(jnp.where(use_cycle[s], c, w) for c, w in zip(cycle, swap))

    def move(s, x):
        return _sparse_step(x, *cells(s), u[s])

    return jax.lax.fori_loop(0, num_moves, move, x)


def _sweep_function(symmetric: bool, fixed: Tuple[bool, ...]):
    """The sweep of the plain, symmetric or fixed-diagonal sampler."""
    if any(fixed):
        return partial(_fixed_diagonal_sweep, symmetric=symmetric)
    if symmetric:
        return _symmetric_sweep
    return _sweep


# ----------------------------------------------------------------------
# Chains
# ----------------------------------------------------------------------


def _run_chain(
    key: Array,
    row_sums: Array,
    col_sums: Array,
    delta: Array,
    num_samples: int,
    num_warmup_sweeps: int,
    thin_sweeps: int,
    symmetric: bool,
    fixed: Tuple[bool, ...],
) -> Array:
    """
    Run one chain; returns ``(num_samples, R, C)``.

    Without symmetry or a fixed diagonal the chain uses basic swaps from the
    independence table (Algorithm 1).  With a fixed diagonal it uses the
    moves of Algorithm 2 from ``diag(delta)`` plus the quasi-independence
    table of the other cells.
    """
    x0 = _start(row_sums, col_sums, delta, symmetric, fixed)
    sweep = _sweep_function(symmetric, fixed)

    def advance(x, key, num_sweeps):
        def body(s, x):
            return sweep(x, jax.random.fold_in(key, s))

        return jax.lax.fori_loop(0, num_sweeps, body, x)

    def sample(x, key):
        x = advance(x, key, thin_sweeps)
        return x, x

    key_warmup, key_sample = jax.random.split(key)
    x = advance(x0, key_warmup, num_warmup_sweeps)
    _, samples = jax.lax.scan(sample, x, jax.random.split(key_sample, num_samples))
    return samples


@partial(
    jax.jit,
    static_argnames=(
        "num_samples", "num_warmup_sweeps", "thin_sweeps", "symmetric", "fixed"
    ),
)
def _run_chains(
    keys, row_sums, col_sums, delta, num_samples, num_warmup_sweeps, thin_sweeps,
    symmetric, fixed,
):
    run = partial(
        _run_chain,
        num_samples=num_samples,
        num_warmup_sweeps=num_warmup_sweeps,
        thin_sweeps=thin_sweeps,
        symmetric=symmetric,
        fixed=fixed,
    )
    return jax.vmap(run)(keys, row_sums, col_sums, delta)


def _run_path(
    key: Array, init: Array, delta: Array, gaps: Array, fixed: Tuple[bool, ...]
) -> Array:
    """
    Run one chain from ``init``, recording the table after each gap of sweeps.

    Uses basic swaps (Algorithm 1), or the moves of Algorithm 2 when some
    diagonal cells are fixed.  Returns tables of shape ``(len(gaps), R, C)``;
    entry ``k`` is the table after ``gaps[0] + ... + gaps[k]`` sweeps.
    """
    sweep = _sweep_function(False, fixed)

    def record(x, inputs):
        key, gap = inputs

        def body(s, x):
            return sweep(x, jax.random.fold_in(key, s))

        x = jax.lax.fori_loop(0, gap, body, x)
        return x, x

    _, path = jax.lax.scan(record, init, (jax.random.split(key, gaps.size), gaps))
    return path


@partial(jax.jit, static_argnames=("fixed",))
def _run_paths(keys, init, delta, gaps, fixed):
    run = partial(_run_path, fixed=fixed)
    return jax.vmap(run, in_axes=(0, 0, 0, None))(keys, init, delta, gaps)


def _check_fully_fixed(fixed: Tuple[bool, ...]) -> None:
    """The swap sampler with a fixed diagonal (Algorithm 2) needs every cell fixed."""
    if not all(fixed):
        raise ValueError(
            "The swap sampler requires every diagonal cell to be fixed; "
            "use sample_hit_run for a partly fixed diagonal."
        )


def _independence_tables(row_sums: Array, col_sums: Array) -> Array:
    """Independence tables ``r c^T / N`` for margins of shape ``(..., R)``."""
    total = row_sums.sum(-1)[..., None, None]
    return row_sums[..., :, None] * col_sums[..., None, :] / total


def sample_swap(
    key: Array,
    row_sums: ArrayLike,
    col_sums: ArrayLike,
    num_samples: int,
    num_warmup_sweeps: int = 10,
    thin_sweeps: int = 1,
    symmetric: bool = False,
    diagonal: ArrayLike | None = None,
) -> Array:
    """
    Sample tables uniformly from the transportation polytope ``T(r, c)`` with
    swap moves.

    Runs the swap-move Markov chain (Algorithm 1 of the supplement) started at
    the independence table ``r c^T / N``.  Each move picks two distinct rows
    and two distinct columns uniformly at random and shifts mass around the
    ``2 x 2`` sub-table they form, by an amount drawn uniformly from the
    interval that keeps its cells non-negative.  A move touches only four
    cells, so the chain is run in sweeps of ``R * C`` moves.

    For ``2 x 2`` tables the whole table is determined by a single cell, and
    tables are drawn exactly and independently without a Markov chain.  If the
    table has a single row or column, the independence table is the only
    table with the given margins and is returned.

    With ``symmetric=True`` the chain samples uniformly from the symmetric
    tables in ``T(r, r)``: each move is a swap plus its mirror image.  With
    ``diagonal`` the chain samples uniformly from the tables whose diagonal
    cells ``x_kk`` all equal the given values (Algorithm 2 of the
    supplement).  Each move is then a three-cycle on three distinct strata
    (with probability 1/2, or always when ``K = 3``) or a basic swap on four
    distinct strata, and the chain starts at ``diag(delta)`` plus the
    quasi-independence table of the off-diagonal cells.  Both options can be
    combined; symmetric tables then use mirrored swaps only.

    The chain runs on the margins divided by their total ``N`` and the sampled
    tables are multiplied by ``N``, so the result does not depend on the
    scale of the margins.  Rows and columns with zero margins stay at zero.

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
    num_warmup_sweeps : int, optional
        Number of sweeps discarded at the start of each chain (default 10).
        In the burn-in study of the supplement (section A.3, Table A.1),
        chains started at the independence table were as far from it as
        uniform draws, on average, within at most 16 sweeps for every
        configuration.  Ignored for ``2 x 2`` tables.
    thin_sweeps : int, optional
        Number of sweeps between kept tables (default 1).  Ignored for
        ``2 x 2`` tables.
    symmetric : bool, optional
        If ``True``, sample symmetric tables.  Requires square tables with
        ``row_sums == col_sums``.
    diagonal : ArrayLike or None, optional
        Fixed diagonal values ``delta`` of shape ``(..., K)``, or ``(K,)`` for
        the whole batch.  Every diagonal cell must be fixed (no ``NaN``; use
        :func:`sample_hit_run` for a partly fixed diagonal), and every table
        must pass :func:`diagonal_is_feasible`.  Requires square tables.

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
    >>> tables = sample_swap(
    ...     jax.random.key(0), [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], num_samples=100
    ... )
    >>> tables.shape
    (100, 3, 3)
    """
    r, c = check_margins(row_sums, col_sums)
    if symmetric and (r.shape != c.shape or not jnp.allclose(r, c)):
        raise ValueError("Symmetric tables require equal row and column sums.")
    if diagonal is None:
        delta, fixed = jnp.zeros_like(r), ()
    else:
        delta, fixed = _fixed_diagonal(r, c, diagonal)
        _check_fully_fixed(fixed)

    batch_shape, (R, C) = r.shape[:-1], (r.shape[-1], c.shape[-1])
    out_shape = (*batch_shape, num_samples, R, C)

    if R == 1 or C == 1:
        x0 = _independence_tables(r, c)
        return jnp.broadcast_to(x0[..., None, :, :], out_shape)
    if R == C == 2:
        # With r = c every 2 x 2 table is symmetric, so this also covers symmetric.
        return _sample_two_by_two(key, r, c, num_samples)

    # Run on margins with total 1 so the chain is independent of the scale of r, c.
    num_chains = prod(batch_shape)
    total = r.sum(-1, keepdims=True)
    samples = _run_chains(
        jax.random.split(key, num_chains),
        (r / total).reshape(num_chains, R),
        (c / total).reshape(num_chains, C),
        (delta / total).reshape(num_chains, R),
        num_samples=num_samples,
        num_warmup_sweeps=num_warmup_sweeps,
        thin_sweeps=thin_sweeps,
        symmetric=symmetric,
        fixed=fixed,
    )
    return samples.reshape(out_shape) * total[..., None, None]


def sample_swap_path(
    key: Array,
    row_sums: ArrayLike,
    col_sums: ArrayLike,
    sweeps: Sequence[int],
    init: ArrayLike | None = None,
    diagonal: ArrayLike | None = None,
) -> Array:
    """
    Record swap-move chains after given numbers of sweeps.

    Runs the chains of :func:`sample_swap` and keeps the table after each
    number of sweeps in ``sweeps``, e.g. ``(0, 1, 2, 4, ..., 2**14)``, which
    makes it possible to follow how far a chain has moved from its start.
    Passing the last tables back as ``init`` continues the chains.

    For ``2 x 2`` tables a single step is already an exact draw (see
    :func:`sample_swap`), so the entry after ``0`` sweeps is ``init`` and
    every later entry is an independent exact draw.

    Parameters
    ----------
    key : Array
        JAX PRNG key.
    row_sums : ArrayLike
        Row sums ``r`` of shape ``(..., R)``.
    col_sums : ArrayLike
        Column sums ``c`` of shape ``(..., C)``.  Leading dimensions are
        treated as a batch of independent chains.
    sweeps : Sequence[int]
        Non-decreasing numbers of sweeps (``R * C`` moves each) at which to
        record the tables.
    init : ArrayLike or None, optional
        Starting tables of shape ``(..., R, C)`` with the given margins (and
        the fixed diagonal, if any).  By default the chains start at the
        independence table ``r c^T / N``, or with a fixed diagonal at
        ``diag(delta)`` plus the quasi-independence table of the other cells.
    diagonal : ArrayLike or None, optional
        Fixed diagonal values of shape ``(..., K)``, or ``(K,)`` for the whole
        batch.  Every diagonal cell must be fixed (see :func:`sample_swap`).

    Returns
    -------
    Array
        Tables of shape ``(..., len(sweeps), R, C)``.

    Raises
    ------
    ValueError
        If the margins are invalid (see :func:`check_margins`), if ``sweeps``
        is not a non-decreasing sequence of non-negative integers, if the
        fixed diagonal is infeasible, or if ``init`` does not have the given
        margins and fixed diagonal.

    Examples
    --------
    >>> import jax
    >>> path = sample_swap_path(
    ...     jax.random.key(0), [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], sweeps=[0, 1, 2, 4]
    ... )
    >>> path.shape
    (4, 3, 3)
    """
    r, c = check_margins(row_sums, col_sums)
    sweeps = np.asarray(sweeps)
    if sweeps.ndim != 1 or np.any(sweeps < 0) or np.any(np.diff(sweeps) < 0):
        raise ValueError(
            "sweeps must be a non-decreasing sequence of non-negative integers."
        )
    if diagonal is None:
        delta, fixed = jnp.zeros_like(r), ()
    else:
        delta, fixed = _fixed_diagonal(r, c, diagonal)
        _check_fully_fixed(fixed)
    batch_shape, (R, C) = r.shape[:-1], (r.shape[-1], c.shape[-1])
    out_shape = (*batch_shape, sweeps.size, R, C)
    fixed_cells = jnp.diag(jnp.array(fixed)) if any(fixed) else None

    if init is None and any(fixed):
        start = jax.vmap(lambda r, c, d: _start(r, c, d, False, fixed))
        x0 = start(r.reshape(-1, R), c.reshape(-1, C), delta.reshape(-1, R))
        x0 = x0.reshape(*batch_shape, R, C)
    elif init is None:
        x0 = _independence_tables(r, c)
    else:
        x0 = jnp.broadcast_to(jnp.asarray(init, dtype=r.dtype), (*batch_shape, R, C))
        rtol = float(jnp.sqrt(jnp.finfo(r.dtype).eps))  # allow rounding error
        keeps_diagonal = fixed_cells is None or jnp.allclose(
            jnp.where(fixed_cells, x0, 0.0),
            jnp.where(fixed_cells, delta[..., None, :] * jnp.eye(R), 0.0),
            rtol=rtol,
        )
        if not (
            jnp.all(x0 >= 0)
            and jnp.allclose(x0.sum(-1), r, rtol=rtol)
            and jnp.allclose(x0.sum(-2), c, rtol=rtol)
            and keeps_diagonal
        ):
            raise ValueError(
                "init must be non-negative tables with the given margins and "
                "fixed diagonal."
            )

    if R == 1 or C == 1:
        return jnp.broadcast_to(x0[..., None, :, :], out_shape)
    if R == C == 2:
        draws = _sample_two_by_two(key, r, c, sweeps.size)
        not_moved = jnp.asarray(sweeps == 0)[:, None, None]
        return jnp.where(not_moved, x0[..., None, :, :], draws)

    # Run on margins with total 1 so the chain is independent of the scale of r, c.
    num_chains = prod(batch_shape)
    total = r.sum(-1)[..., None, None]
    path = _run_paths(
        jax.random.split(key, num_chains),
        (x0 / total).reshape(num_chains, R, C),
        (delta / total[..., 0]).reshape(num_chains, R),
        jnp.asarray(np.diff(sweeps, prepend=0)),
        fixed=fixed,
    )
    return path.reshape(out_shape) * total[..., None, :, :]
