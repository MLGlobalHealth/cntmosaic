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
are drawn exactly and independently instead (section A.2 of the supplement).

Symmetric tables (same-age pairs, section A.3) use the symmetric swap move,
which shifts mass between an off-diagonal pair ``(i, j), (j, i)`` and the
diagonal cells ``(i, i), (j, j)``.  When the whole diagonal is fixed (section
B, Algorithm 2), each move is a basic swap on four distinct strata, or for
symmetric tables a swap plus its transpose on four distinct strata (B.6);
neither touches the diagonal.  With three strata the fixed-diagonal tables
form a segment, which is sampled directly (B.4), or a single table when they
are also symmetric (B.5).  Every move reads and writes only the cells it
changes.

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

from ._polytope_sampler import _fixed_diagonal, _mirror_upper, _start, check_margins


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
# Symmetric tables and a fixed diagonal
# ----------------------------------------------------------------------


def _distinct_strata(key: Array, n: int, size: int, count: int) -> Array:
    """
    Draw ``size`` ordered tuples of ``count`` distinct indices from ``0, ..., n - 1``.

    The ``m``-th index is drawn uniformly from the ``n - m`` indices not yet
    chosen: a uniform value in ``0, ..., n - m - 1`` is shifted past the
    indices already chosen, in increasing order.  Every ordered tuple of
    distinct indices is equally likely, and each draw costs ``O(count^2)``.

    Returns
    -------
    Array
        Integer array of shape ``(size, count)``.
    """
    chosen = []
    for m, key_m in enumerate(jax.random.split(key, count)):
        value = jax.random.randint(key_m, (size,), 0, n - m)
        if chosen:
            for taken in jnp.sort(jnp.stack(chosen), axis=0):
                value = value + (value >= taken)
        chosen.append(value)
    return jnp.stack(chosen, axis=1)


def _symmetric_swap_step(x: Array, i: Array, j: Array, u: Array) -> Array:
    """
    Apply the symmetric swap move ``D^(i,j) = E_ii + E_jj - E_ij - E_ji`` (A.9).

    The move shifts ``t`` from the off-diagonal pair ``(i, j), (j, i)`` to the
    diagonal cells ``(i, i)`` and ``(j, j)``, or back if ``t < 0``, with
    ``t_- = -min(x_ii, x_jj)`` and ``t_+ = x_ij`` (A.10).  The new value of the
    pair is computed once and written to both cells, so the table stays
    exactly symmetric.
    """
    t_lo = -jnp.minimum(x[i, i], x[j, j])
    t_hi = x[i, j]
    t = jnp.clip(t_lo + u * (t_hi - t_lo), t_lo, t_hi)
    pair = x[i, j] - t
    return x.at[i, i].add(t).at[j, j].add(t).at[i, j].set(pair).at[j, i].set(pair)


def _symmetric_fixed_step(
    x: Array, i: Array, j: Array, k: Array, l: Array, u: Array
) -> Array:
    """
    Apply ``D^(i,j,k,l)`` (B.6) for four distinct strata:
    ``(E_ij + E_ji) + (E_kl + E_lk) - (E_il + E_li) - (E_kj + E_jk)``.

    The move shifts ``t`` from the pairs ``{i, l}`` and ``{k, j}`` to the pairs
    ``{i, j}`` and ``{k, l}``, with ``t_- = -min(x_ij, x_kl)`` and
    ``t_+ = min(x_il, x_kj)`` (B.7).  It leaves the diagonal unchanged, and the
    new value of each pair is computed once and written to both of its cells.
    """
    t_lo = -jnp.minimum(x[i, j], x[k, l])
    t_hi = jnp.minimum(x[i, l], x[k, j])
    t = jnp.clip(t_lo + u * (t_hi - t_lo), t_lo, t_hi)
    gain_ij, gain_kl = x[i, j] + t, x[k, l] + t
    lose_il, lose_kj = x[i, l] - t, x[k, j] - t
    x = x.at[i, j].set(gain_ij).at[j, i].set(gain_ij)
    x = x.at[k, l].set(gain_kl).at[l, k].set(gain_kl)
    x = x.at[i, l].set(lose_il).at[l, i].set(lose_il)
    return x.at[k, j].set(lose_kj).at[j, k].set(lose_kj)


def _symmetric_sweep(x: Array, key: Array) -> Array:
    """Apply ``K * K`` symmetric swap moves (A.9) on uniform pairs ``i != j``."""
    K = x.shape[0]
    num_moves = K * K
    key_pairs, key_u = jax.random.split(key)
    i, j = _distinct_pairs(key_pairs, K, num_moves)
    u = jax.random.uniform(key_u, (num_moves,), dtype=x.dtype)

    def move(s, x):
        return _symmetric_swap_step(x, i[s], j[s], u[s])

    return jax.lax.fori_loop(0, num_moves, move, x)


def _fixed_diagonal_sweep(x: Array, key: Array) -> Array:
    """
    Apply ``K * K`` moves of Algorithm 2 (``K >= 4``): basic swaps whose rows
    ``i, i'`` and columns ``j, j'`` are four distinct strata, so that no move
    touches the diagonal (B.2).
    """
    K = x.shape[0]
    num_moves = K * K
    key_strata, key_u = jax.random.split(key)
    strata = _distinct_strata(key_strata, K, num_moves, 4)
    u = jax.random.uniform(key_u, (num_moves,), dtype=x.dtype)

    def move(s, x):
        i, i2, j, j2 = strata[s]
        return _swap_step(x, (i, i2), (j, j2), u[s])

    return jax.lax.fori_loop(0, num_moves, move, x)


def _symmetric_fixed_sweep(x: Array, key: Array) -> Array:
    """Apply ``K * K`` moves ``D^(i,j,k,l)`` (B.6) on four distinct strata, K >= 4."""
    K = x.shape[0]
    num_moves = K * K
    key_strata, key_u = jax.random.split(key)
    strata = _distinct_strata(key_strata, K, num_moves, 4)
    u = jax.random.uniform(key_u, (num_moves,), dtype=x.dtype)

    def move(s, x):
        i, j, k, l = strata[s]
        return _symmetric_fixed_step(x, i, j, k, l, u[s])

    return jax.lax.fori_loop(0, num_moves, move, x)


def _sweep_function(symmetric: bool, fixed: Tuple[bool, ...]):
    """The sweep of the plain, symmetric or fixed-diagonal sampler."""
    if any(fixed):
        return _symmetric_fixed_sweep if symmetric else _fixed_diagonal_sweep
    return _symmetric_sweep if symmetric else _sweep


# ----------------------------------------------------------------------
# Fixed diagonal with three strata: no chain needed (B.4, B.5)
# ----------------------------------------------------------------------

# The three-cycle D^(1,2,3) (B.4), the only direction with zero row and column
# sums and a zero diagonal when K = 3.
_THREE_CYCLE = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])


def _starting_tables(row_sums: Array, col_sums: Array, delta: Array, fixed) -> Array:
    """``diag(delta)`` plus the quasi-independence table (B.3), for a batch."""
    R = row_sums.shape[-1]
    start = jax.vmap(lambda r, c, d: _start(r, c, d, False, fixed))
    x0 = start(row_sums.reshape(-1, R), col_sums.reshape(-1, R), delta.reshape(-1, R))
    return x0.reshape(*row_sums.shape[:-1], R, R)


def _sample_three_cycle_segment(key: Array, x0: Array, num_samples: int) -> Array:
    """
    Draw tables uniformly from the segment ``{x0 + t D^(1,2,3)}`` (B.4).

    With three strata and a fixed diagonal every three-cycle equals
    ``+-D^(1,2,3)``, so the tables form a segment through ``x0`` with
    ``t_- = -min(x_12, x_23, x_31)`` and ``t_+ = min(x_13, x_21, x_32)`` (B.5).
    The draws are independent.

    Returns
    -------
    Array
        Tables of shape ``(..., num_samples, 3, 3)`` for ``x0`` of shape
        ``(..., 3, 3)``.
    """
    t_lo = -jnp.minimum(jnp.minimum(x0[..., 0, 1], x0[..., 1, 2]), x0[..., 2, 0])
    t_hi = jnp.minimum(jnp.minimum(x0[..., 0, 2], x0[..., 1, 0]), x0[..., 2, 1])
    u = jax.random.uniform(key, (*x0.shape[:-2], num_samples), dtype=x0.dtype)
    t = t_lo[..., None] + u * (t_hi - t_lo)[..., None]
    t = jnp.clip(t, t_lo[..., None], t_hi[..., None])
    cycle = jnp.asarray(_THREE_CYCLE, dtype=x0.dtype)
    return x0[..., None, :, :] + t[..., None, None] * cycle


def _symmetric_fixed_three_by_three(row_sums: Array, delta: Array) -> Array:
    """
    The single symmetric ``3 x 3`` table with margins ``r`` and diagonal ``delta``
    (B.5): ``x_ij = x_ji = (r'_i + r'_j - r'_k) / 2`` with ``r' = r - delta``.
    """
    rr = row_sums - delta
    x01 = (rr[..., 0] + rr[..., 1] - rr[..., 2]) / 2
    x02 = (rr[..., 0] + rr[..., 2] - rr[..., 1]) / 2
    x12 = (rr[..., 1] + rr[..., 2] - rr[..., 0]) / 2
    d = delta
    rows = [
        jnp.stack([d[..., 0], x01, x02], -1),
        jnp.stack([x01, d[..., 1], x12], -1),
        jnp.stack([x02, x12, d[..., 2]], -1),
    ]
    return jnp.stack(rows, -2)


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
    independence table (Algorithm 1), and symmetric swaps (A.9) for symmetric
    tables.  With a fixed diagonal it uses the moves of Algorithm 2, or
    ``D^(i,j,k,l)`` (B.6) for symmetric tables, from ``diag(delta)`` plus the
    quasi-independence table of the off-diagonal cells (B.3).
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

    Uses basic swaps (Algorithm 1), or the moves of Algorithm 2 when the
    diagonal is fixed.  Returns tables of shape ``(len(gaps), R, C)``;
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


def _check_not_two_by_two(shape: Tuple[int, ...]) -> None:
    """A fixed diagonal is not supported for ``2 x 2`` tables (B.4)."""
    if shape[-1] == 2:
        raise ValueError(
            "For 2 x 2 tables only one diagonal cell can be specified: fixing "
            "it determines the whole table, so the other diagonal cell cannot "
            "be chosen as well. The swap sampler does not fix the diagonal of "
            "2 x 2 tables."
        )


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
    tables in ``T(r, r)`` with symmetric swap moves (section A.3 of the
    supplement).  With ``diagonal`` it samples uniformly from the tables whose
    diagonal cells ``x_kk`` all equal the given values (section B): for
    ``K >= 4`` with basic swaps on four distinct strata (Algorithm 2), or
    with ``D^(i,j,k,l)`` (B.6) for symmetric tables, started at
    ``diag(delta)`` plus the quasi-independence table of the off-diagonal
    cells.  For ``K = 3`` the tables form a segment, which is sampled
    directly, or a single table if they are also symmetric (B.4, B.5).  A
    ``2 x 2`` table's diagonal cannot be fixed.

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
        must pass :func:`diagonal_is_feasible`.  Requires square tables with
        ``K >= 3``.

    Returns
    -------
    Array
        Sampled tables of shape ``(..., num_samples, R, C)``.

    Raises
    ------
    ValueError
        If the margins are invalid (see :func:`check_margins`), if
        ``symmetric=True`` with unequal margins, if the diagonal is partly
        fixed or infeasible for some table, or if it is given for ``2 x 2``
        tables.

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
        _check_not_two_by_two(r.shape)
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
    if any(fixed) and R == 3:
        if symmetric:
            x = _symmetric_fixed_three_by_three(r, delta)
            return jnp.broadcast_to(x[..., None, :, :], out_shape)
        x0 = _starting_tables(r, c, delta, fixed)
        return _sample_three_cycle_segment(key, x0, num_samples)

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
    samples = samples.reshape(out_shape) * total[..., None, None]
    # Exactly symmetric output (see _mirror_upper)
    return _mirror_upper(samples) if symmetric else samples


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

    For ``2 x 2`` tables, and for ``3 x 3`` tables with a fixed diagonal, the
    tables are drawn exactly (see :func:`sample_swap`), so the entry after
    ``0`` sweeps is ``init`` and every later entry is an independent exact
    draw.

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
        _check_not_two_by_two(r.shape)
        delta, fixed = _fixed_diagonal(r, c, diagonal)
        _check_fully_fixed(fixed)
    batch_shape, (R, C) = r.shape[:-1], (r.shape[-1], c.shape[-1])
    out_shape = (*batch_shape, sweeps.size, R, C)
    fixed_cells = jnp.diag(jnp.array(fixed)) if any(fixed) else None

    if init is None and any(fixed):
        x0 = _starting_tables(r, c, delta, fixed)
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
    if R == C == 2 or (any(fixed) and R == 3):
        if R == 2:
            draws = _sample_two_by_two(key, r, c, sweeps.size)
        else:
            draws = _sample_three_cycle_segment(key, x0, sweeps.size)
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
