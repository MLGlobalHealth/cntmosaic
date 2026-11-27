import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from cntmosaic.predict import sample_swap, sample_swap_path
from cntmosaic.predict._swap_sampler import (
    _distinct_strata,
    _swap_step,
    _symmetric_fixed_step,
    _symmetric_swap_step,
)


class TestSwapStep:
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    rows, cols = (jnp.array(0), jnp.array(2)), (jnp.array(1), jnp.array(0))

    @pytest.mark.parametrize("u, emptied", [(0.0, (0, 1)), (1.0, (0, 0))])
    def test_ends_of_the_interval(self, u, emptied):
        # Cells (0, 1) and (2, 0) gain t; cells (0, 0) and (2, 1) lose t.
        # t_- = -min(2, 7) = -2 empties cell (0, 1); t_+ = min(1, 8) = 1 empties (0, 0).
        x = _swap_step(self.x, self.rows, self.cols, jnp.array(u))
        assert x[emptied] == 0.0
        assert jnp.all(x >= 0)

    def test_touches_four_cells_and_keeps_margins(self):
        x = _swap_step(self.x, self.rows, self.cols, jnp.array(0.3))
        changed = np.argwhere(np.asarray(x != self.x))
        assert {tuple(cell) for cell in changed} == {(0, 1), (2, 0), (0, 0), (2, 1)}
        np.testing.assert_allclose(x.sum(1), self.x.sum(1), rtol=1e-6)
        np.testing.assert_allclose(x.sum(0), self.x.sum(0), rtol=1e-6)


class TestTwoByTwo:
    def test_exact_uniform(self):
        # For r = (3, 1), c = (2, 2) the exact law of x_11 is Uniform(1, 2).
        x = sample_swap(jax.random.key(0), [3.0, 1.0], [2.0, 2.0], num_samples=4000)
        x11 = np.asarray(x[:, 0, 0])
        assert stats.kstest(x11, stats.uniform(1.0, 1.0).cdf).pvalue > 0.01
        np.testing.assert_allclose(x.sum(-1), np.broadcast_to([3.0, 1.0], (4000, 2)), rtol=1e-5)
        np.testing.assert_allclose(x.sum(-2), np.broadcast_to([2.0, 2.0], (4000, 2)), rtol=1e-5)
        assert jnp.all(x >= 0)

    def test_single_table(self):
        # r_1 + c_1 - N = min(r_1, c_1) = 1, so only one table has these margins.
        x = sample_swap(jax.random.key(1), [1.0, 1.0], [2.0, 0.0], num_samples=5)
        np.testing.assert_allclose(x, np.broadcast_to([[1.0, 0.0], [1.0, 0.0]], (5, 2, 2)))

    def test_batch_shape(self):
        x = sample_swap(jax.random.key(2), jnp.ones((3, 4, 2)), jnp.ones((3, 4, 2)), num_samples=6)
        assert x.shape == (3, 4, 6, 2, 2)


class TestSampleSwap:
    def test_samples_lie_in_polytope(self):
        r = jnp.arange(1.0, 6.0)
        c = 15.0 * 0.75 ** jnp.arange(4.0) / (0.75 ** jnp.arange(4.0)).sum()
        x = sample_swap(jax.random.key(3), r, c, num_samples=200, num_warmup_sweeps=20)
        assert x.shape == (200, 5, 4)
        np.testing.assert_allclose(x.sum(-1), jnp.broadcast_to(r, (200, 5)), rtol=1e-4)
        np.testing.assert_allclose(x.sum(-2), jnp.broadcast_to(c, (200, 4)), rtol=1e-4)
        assert jnp.all(x >= 0)

    def test_equal_margins_mean(self):
        # With r = c = 1 every cell has mean 1/K by symmetry.
        K = 4
        x = sample_swap(
            jax.random.key(4), jnp.ones((64, K)), jnp.ones((64, K)),
            num_samples=200, num_warmup_sweeps=20, thin_sweeps=5,
        )
        np.testing.assert_allclose(x.mean((0, 1)), 1 / K, atol=0.02)

    def test_zero_margins_stay_zero(self):
        x = sample_swap(
            jax.random.key(5), [2.0, 0.0, 1.0], [0.0, 1.5, 1.5], num_samples=50,
            num_warmup_sweeps=5,
        )
        assert jnp.all(x[:, 1, :] == 0.0) and jnp.all(x[:, :, 0] == 0.0)

    def test_single_row(self):
        x = sample_swap(jax.random.key(6), [3.0], [1.0, 2.0], num_samples=4)
        np.testing.assert_allclose(x, np.broadcast_to([[1.0, 2.0]], (4, 1, 2)))

    def test_scale_equivariance(self):
        # Scaling by a power of 2 is exact, so the chain follows the same path.
        r, c, s = jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 1.0, 1.0]), 2.0**20
        x = sample_swap(jax.random.key(7), r, c, num_samples=20, num_warmup_sweeps=5)
        x_big = sample_swap(jax.random.key(7), s * r, s * c, num_samples=20, num_warmup_sweeps=5)
        np.testing.assert_allclose(x_big, s * x, rtol=1e-5)

    def test_batch_shape(self):
        x = sample_swap(
            jax.random.key(8), jnp.ones((2, 3, 4)), jnp.full((2, 3, 3), 4 / 3),
            num_samples=5, num_warmup_sweeps=2,
        )
        assert x.shape == (2, 3, 5, 4, 3)


class TestSampleSwapPath:
    r = jnp.array([1.0, 2.0, 3.0])
    c = jnp.array([3.0, 2.0, 1.0])

    def test_first_entry_is_the_start(self):
        path = sample_swap_path(jax.random.key(9), self.r, self.c, sweeps=[0, 1, 4])
        np.testing.assert_allclose(path[0], jnp.outer(self.r, self.c) / 6.0, rtol=1e-6)
        assert path.shape == (3, 3, 3)

    def test_recordings_keep_margins_and_move(self):
        key = jax.random.key(10)
        path = sample_swap_path(key, self.r, self.c, sweeps=[0, 2, 5])
        np.testing.assert_allclose(path.sum(-1), jnp.broadcast_to(self.r, (3, 3)), rtol=1e-5)
        np.testing.assert_allclose(path.sum(-2), jnp.broadcast_to(self.c, (3, 3)), rtol=1e-5)
        assert jnp.all(path >= 0)
        assert not jnp.allclose(path[1], path[2])  # the chain moved between recordings

    def test_init_continues_chains(self):
        start = sample_swap_path(jax.random.key(11), self.r, self.c, sweeps=[3])[0]
        path = sample_swap_path(jax.random.key(12), self.r, self.c, sweeps=[0, 3], init=start)
        np.testing.assert_allclose(path[0], start)
        np.testing.assert_allclose(path[1].sum(-1), self.r, rtol=1e-5)

    def test_two_by_two(self):
        path = sample_swap_path(
            jax.random.key(13), jnp.tile(jnp.array([3.0, 1.0]), (2000, 1)),
            jnp.tile(jnp.array([2.0, 2.0]), (2000, 1)), sweeps=[0, 1, 2],
        )
        np.testing.assert_allclose(path[:, 0, 0, 0], 1.5)  # independence table
        for k in (1, 2):
            x11 = np.asarray(path[:, k, 0, 0])
            assert stats.kstest(x11, stats.uniform(1.0, 1.0).cdf).pvalue > 0.01

    @pytest.mark.parametrize("sweeps", [[2, 1], [-1, 0], [[0, 1]]])
    def test_invalid_sweeps_raise(self, sweeps):
        with pytest.raises(ValueError, match="non-decreasing"):
            sample_swap_path(jax.random.key(14), self.r, self.c, sweeps=sweeps)

    def test_invalid_init_raises(self):
        with pytest.raises(ValueError, match="given margins"):
            sample_swap_path(jax.random.key(15), self.r, self.c, sweeps=[0], init=jnp.ones((3, 3)))


class TestMoves:
    x = jnp.arange(1.0, 17.0).reshape(4, 4)
    sym = x + x.T

    @staticmethod
    def changed(x_new, x_old):
        return {tuple(cell) for cell in np.argwhere(np.asarray(x_new != x_old))}

    def test_symmetric_swap_changes_four_cells(self):
        x = _symmetric_swap_step(self.sym, jnp.array(0), jnp.array(2), jnp.array(0.3))
        assert self.changed(x, self.sym) == {(0, 0), (2, 2), (0, 2), (2, 0)}
        np.testing.assert_array_equal(x, x.T)
        np.testing.assert_allclose(x.sum(1), self.sym.sum(1), rtol=1e-6)

    @pytest.mark.parametrize("u, emptied", [(0.0, (0, 0)), (1.0, (0, 2))])
    def test_symmetric_swap_ends_empty_a_cell(self, u, emptied):
        # t_- = -min(x_00, x_22) = -x_00 empties (0, 0); t_+ = x_02 empties (0, 2).
        x = _symmetric_swap_step(self.sym, jnp.array(0), jnp.array(2), jnp.array(u))
        assert x[emptied] == 0.0 and jnp.all(x >= 0)

    def test_symmetric_fixed_step_changes_eight_cells(self):
        i, j, k, l = (jnp.array(v) for v in (0, 1, 2, 3))
        x = _symmetric_fixed_step(self.sym, i, j, k, l, jnp.array(0.4))
        pairs = {(0, 1), (2, 3), (0, 3), (2, 1)}
        assert self.changed(x, self.sym) == pairs | {(b, a) for a, b in pairs}
        np.testing.assert_array_equal(jnp.diagonal(x), jnp.diagonal(self.sym))
        np.testing.assert_array_equal(x, x.T)
        np.testing.assert_allclose(x.sum(1), self.sym.sum(1), rtol=1e-6)

    def test_distinct_strata(self):
        strata = np.asarray(_distinct_strata(jax.random.key(0), 5, 2000, 4))
        assert strata.min() >= 0 and strata.max() < 5
        assert all(len(set(row)) == 4 for row in strata)
        # Each stratum appears in each position about equally often.
        counts = np.stack([np.bincount(strata[:, m], minlength=5) for m in range(4)])
        np.testing.assert_allclose(counts / 2000, 0.2, atol=0.04)


class TestSymmetricAndFixedDiagonal:
    r = jnp.array([1.0, 2.0, 3.0, 4.0])
    c = jnp.array([4.0, 3.0, 2.0, 1.0])

    def test_symmetric_tables(self):
        x = sample_swap(jax.random.key(20), self.r, self.r, num_samples=50, symmetric=True)
        np.testing.assert_array_equal(x, x.swapaxes(-1, -2))
        np.testing.assert_allclose(x.sum(-1), jnp.broadcast_to(self.r, (50, 4)), rtol=1e-5)
        assert jnp.all(x >= 0)
        assert not jnp.allclose(x[0], x[-1])

    def test_symmetric_requires_equal_margins(self):
        with pytest.raises(ValueError, match="equal row and column sums"):
            sample_swap(jax.random.key(21), self.r, self.c, num_samples=1, symmetric=True)

    @pytest.mark.parametrize("K", [3, 4, 5])
    @pytest.mark.parametrize("symmetric", [False, True])
    def test_fixed_diagonal(self, K, symmetric):
        r = jnp.arange(1.0, K + 1)
        c = r if symmetric else r[::-1]
        delta = 1.5 * r * c / r.sum()  # assortative, strictly feasible
        x = sample_swap(
            jax.random.key(22), r, c, num_samples=50, symmetric=symmetric, diagonal=delta
        )
        diag = jnp.diagonal(x, axis1=-2, axis2=-1)
        np.testing.assert_array_equal(diag, jnp.broadcast_to(diag[0], diag.shape))
        np.testing.assert_allclose(diag[0], delta, rtol=1e-5)
        np.testing.assert_allclose(x.sum(-1), jnp.broadcast_to(r, (50, K)), rtol=1e-5)
        np.testing.assert_allclose(x.sum(-2), jnp.broadcast_to(c, (50, K)), rtol=1e-5)
        assert jnp.all(x >= 0)
        if symmetric:
            np.testing.assert_array_equal(x, x.swapaxes(-1, -2))
        if not (symmetric and K == 3):  # a symmetric 3 x 3 table is determined
            assert not jnp.allclose(x[0], x[-1])

    def test_three_strata_lie_on_a_segment(self):
        r = jnp.array([1.0, 2.0, 3.0])
        c = jnp.array([3.0, 2.0, 1.0])
        x = sample_swap(jax.random.key(23), r, c, num_samples=200, diagonal=0.8 * r * c / 6)
        cycle = jnp.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
        t = x[:, 0, 1] - x[0, 0, 1]
        np.testing.assert_allclose(x, x[0] + t[:, None, None] * cycle, atol=1e-5)

    def test_symmetric_three_strata_closed_form(self):
        r = jnp.array([1.0, 2.0, 3.0])
        delta = 0.8 * r * r / r.sum()
        x = sample_swap(
            jax.random.key(24), r, r, num_samples=3, symmetric=True, diagonal=delta
        )
        rr = r - delta
        np.testing.assert_allclose(x[0, 0, 1], (rr[0] + rr[1] - rr[2]) / 2, rtol=1e-5)
        np.testing.assert_allclose(x[0, 1, 2], (rr[1] + rr[2] - rr[0]) / 2, rtol=1e-5)
        np.testing.assert_array_equal(x, jnp.broadcast_to(x[0], x.shape))

    @pytest.mark.parametrize("symmetric", [False, True])
    def test_two_by_two_diagonal_raises(self, symmetric):
        r = jnp.array([3.0, 1.0])
        c = r if symmetric else jnp.array([2.0, 2.0])
        with pytest.raises(ValueError, match="only one diagonal cell"):
            sample_swap(
                jax.random.key(25), r, c, num_samples=1, symmetric=symmetric,
                diagonal=jnp.array([1.0, 0.5]),
            )

    def test_partly_fixed_diagonal_raises(self):
        with pytest.raises(ValueError, match="every diagonal cell to be fixed"):
            sample_swap(
                jax.random.key(26), self.r, self.c, num_samples=1,
                diagonal=jnp.array([0.4, jnp.nan, 0.6, jnp.nan]),
            )

    def test_infeasible_diagonal_raises(self):
        r = jnp.ones(3)
        with pytest.raises(ValueError, match="no strictly positive table"):
            sample_swap(
                jax.random.key(27), r, r, num_samples=1, diagonal=jnp.array([1.0, 1.0, 0.0])
            )


class TestSampleSwapPathFixedDiagonal:
    r = jnp.array([1.0, 2.0, 3.0, 4.0])
    c = jnp.array([4.0, 3.0, 2.0, 1.0])
    delta = 1.5 * r * c / r.sum()

    def test_diagonal_fixed_along_the_path(self):
        path = sample_swap_path(
            jax.random.key(30), self.r, self.c, sweeps=[0, 1, 4, 16], diagonal=self.delta
        )
        diag = jnp.diagonal(path, axis1=-2, axis2=-1)
        np.testing.assert_array_equal(diag, jnp.broadcast_to(diag[0], diag.shape))
        np.testing.assert_allclose(diag[0], self.delta, rtol=1e-6)
        np.testing.assert_allclose(path.sum(-1), jnp.broadcast_to(self.r, (4, 4)), rtol=1e-5)
        np.testing.assert_allclose(path.sum(-2), jnp.broadcast_to(self.c, (4, 4)), rtol=1e-5)
        assert jnp.all(path >= 0)
        assert not jnp.allclose(path[0], path[-1])  # the chain moved

    def test_init_continues_with_the_diagonal(self):
        start = sample_swap_path(
            jax.random.key(31), self.r, self.c, sweeps=[3], diagonal=self.delta
        )[0]
        path = sample_swap_path(
            jax.random.key(32), self.r, self.c, sweeps=[0, 3], init=start,
            diagonal=self.delta,
        )
        np.testing.assert_allclose(path[0], start)
        np.testing.assert_allclose(jnp.diagonal(path[1]), self.delta, rtol=1e-6)

    def test_init_without_the_diagonal_raises(self):
        init = jnp.outer(self.r, self.c) / self.r.sum()  # diagonal r c / N, not delta
        with pytest.raises(ValueError, match="fixed diagonal"):
            sample_swap_path(
                jax.random.key(33), self.r, self.c, sweeps=[0], init=init,
                diagonal=self.delta,
            )
