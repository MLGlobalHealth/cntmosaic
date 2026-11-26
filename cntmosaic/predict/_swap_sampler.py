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

Symmetric tables use each swap together with its mirror image.  When some
diagonal cells are fixed (section B of the supplement), moves that would
change them are skipped, and three-cycle moves, which leave the diagonal
unchanged, are proposed as well.

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

from ._polytope_sampler import (
    _chord,
    _fixed_diagonal,
    _start,
    check_margins,
    independence_table,
)


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


def _two_by_two_fixed(row_sums: Array, col_sums: Array, delta: Array, fixed) -> Array:
    """
    The single ``2 x 2`` table with one fixed diagonal cell.

    Fixing ``x_11`` or ``x_22`` determines the whole table:
    ``x_11 = delta_1``, or ``x_11 = r_1 - c_2 + delta_2``.
    """
    r1, r2, c1 = row_sums[..., 0], row_sums[..., 1], col_sums[..., 0]
    x = delta[..., 0] if fixed[0] else r1 - col_sums[..., 1] + delta[..., 1]
    return jnp.stack(
        [jnp.stack([x, r1 - x], -1), jnp.stack([c1 - x, r2 - c1 + x], -1)], -2
    )


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
# General moves: symmetric tables and fixed diagonal cells (section B)
# ----------------------------------------------------------------------


def _move_direction(
    shape: Tuple[int, int],
    rows: Tuple[Array, Array],
    cols: Tuple[Array, Array],
    triple: Array,
    use_cycle: Array,
    symmetric: bool,
) -> Array:
    """
    Direction of one move as a dense ``K x K`` array.

    Either the basic swap ``D^(i,i';j,j')`` (A.5) or, if ``use_cycle``, the
    three-cycle ``D^(i,j,k)`` (B.5), which moves mass around ``i -> j -> k -> i``
    without touching the diagonal.  For symmetric tables the direction is
    ``D + D^T``, which keeps zero row and column sums and keeps the table
    symmetric.
    """
    (i, i2), (j, j2) = rows, cols
    a, b, c = triple
    zeros = jnp.zeros(shape)
    swap = zeros.at[i, j].add(1).at[i2, j2].add(1).at[i, j2].add(-1).at[i2, j].add(-1)
    cycle = (
        zeros.at[a, b].add(1).at[b, c].add(1).at[c, a].add(1)
        .at[a, c].add(-1).at[b, a].add(-1).at[c, b].add(-1)
    )
    d = jnp.where(use_cycle, cycle, swap)
    return d + d.T if symmetric else d


def _move_step(
    x: Array, d: Array, u: Array, delta: Array, fixed_cells: Array, symmetric: bool
) -> Array:
    """
    Move ``x`` along ``d`` by a uniform step on the chord ``{t : x + t d >= 0}``.

    A move that would change a fixed diagonal cell is not admissible (B.4) and
    is skipped.  The entries of ``d`` are 0, +-1 or +-2, so a step to an end
    of the chord sets that cell to zero exactly.  After the step, symmetric
    tables are symmetrised and fixed diagonal cells are reset to ``delta``
    against rounding.
    """
    d = jnp.where(jnp.any((d != 0) & fixed_cells), 0.0, d)
    t_lo, t_hi = _chord(x, d)
    t = jnp.clip(t_lo + u * (t_hi - t_lo), t_lo, t_hi)
    # If the move is skipped then d = 0 and the chord is unbounded.
    x = x + jnp.where(jnp.isfinite(t), t, 0.0) * d
    if symmetric:
        x = (x + x.T) / 2
    return jnp.where(fixed_cells, jnp.diag(delta), x)


def _cycle_probability(K: int, symmetric: bool, fixed: Tuple[bool, ...]) -> float:
    """
    Probability of proposing a three-cycle instead of a basic swap.

    Symmetrised three-cycles are zero, so symmetric tables use swaps only.
    With the whole diagonal fixed and ``K = 3``, no basic swap is admissible.
    Otherwise both move types are proposed when some diagonal cell is fixed.
    """
    if symmetric or not any(fixed):
        return 0.0
    if K == 3 and all(fixed):
        return 1.0
    return 0.5


def _general_sweep(
    x: Array, key: Array, delta: Array, symmetric: bool, fixed: Tuple[bool, ...]
) -> Array:
    """Apply ``K * K`` general moves (Algorithm 2 of the supplement) to ``x``."""
    K = x.shape[0]
    num_moves = K * K
    p = _cycle_probability(K, symmetric, fixed)
    fixed_cells = jnp.diag(jnp.array(fixed)) if any(fixed) else jnp.zeros((K, K), bool)

    key_rows, key_cols, key_triples, key_cycle, key_u = jax.random.split(key, 5)
    i, i2 = _distinct_pairs(key_rows, K, num_moves)
    j, j2 = _distinct_pairs(key_cols, K, num_moves)
    triples = jax.vmap(lambda k: jax.random.choice(k, K, (3,), replace=False))(
        jax.random.split(key_triples, num_moves)
    )
    use_cycle = jax.random.uniform(key_cycle, (num_moves,)) < p
    u = jax.random.uniform(key_u, (num_moves,), dtype=x.dtype)

    def move(s, x):
        d = _move_direction(
            x.shape, (i[s], i2[s]), (j[s], j2[s]), triples[s], use_cycle[s], symmetric
        )
        return _move_step(x, d, u[s], delta, fixed_cells, symmetric)

    return jax.lax.fori_loop(0, num_moves, move, x)


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

    Without symmetry or fixed cells the chain uses basic swaps from the
    independence table (Algorithm 1).  Otherwise it uses the general moves
    from the quasi-independence table (Algorithm 2).
    """
    if symmetric or any(fixed):
        x0 = _start(row_sums, col_sums, delta, symmetric, fixed)

        def sweep(x, key):
            return _general_sweep(x, key, delta, symmetric, fixed)

    else:
        x0, sweep = independence_table(row_sums, col_sums), _sweep

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
    if any(fixed):

        def sweep(x, key):
            return _general_sweep(x, key, delta, False, fixed)

    else:
        sweep = _sweep

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
    cells ``x_kk`` equal the given values for every ``k`` in a set ``S``
    (Algorithm 2 of the supplement).  It then proposes basic swaps and
    three-cycle moves, skips any move that would change a fixed cell, and
    starts at ``diag(delta)`` plus the quasi-independence table of the
    remaining cells.  Both options can be combined.

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

    batch_shape, (R, C) = r.shape[:-1], (r.shape[-1], c.shape[-1])
    out_shape = (*batch_shape, num_samples, R, C)

    if R == 1 or C == 1:
        x0 = _independence_tables(r, c)
        return jnp.broadcast_to(x0[..., None, :, :], out_shape)
    if R == C == 2:
        if any(fixed):
            x = _two_by_two_fixed(r, c, delta, fixed)
            return jnp.broadcast_to(x[..., None, :, :], out_shape)
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
        batch; ``NaN`` marks a free diagonal cell (see :func:`sample_swap`).

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
        if any(fixed):
            # The fixed cell determines the table.
            x = _two_by_two_fixed(r, c, delta, fixed)
            return jnp.broadcast_to(x[..., None, :, :], out_shape)
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
