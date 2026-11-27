import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scipy import stats

from cntmosaic.predict import (
    check_margins,
    diagonal_is_feasible,
    sample_hit_run,
)
from cntmosaic.predict._polytope_sampler import (
    _chord,
    _double_center,
    _fixed_gram_inverse,
    _project_fixed_diagonal,
)


class TestCheckMargins:
    def test_valid_margins_returned_as_float(self):
        r, c = check_margins([3, 1], [2, 2])
        assert jnp.issubdtype(r.dtype, jnp.floating)
        np.testing.assert_allclose(r, [3.0, 1.0])
        np.testing.assert_allclose(c, [2.0, 2.0])

    @pytest.mark.parametrize(
        "r, c, match",
        [
            ([3.0, -1.0], [1.0, 1.0], "non-negative"),
            ([1.0, 1.0], [-1.0, 3.0], "non-negative"),
            ([0.0, 0.0], [0.0, 0.0], "positive"),
            ([3.0, 1.0], [2.0, 3.0], "same total"),
            ([np.nan, 1.0], [1.0, 1.0], "finite"),
            ([[1.0, 1.0]], [1.0, 1.0], "Batch shapes"),
            (1.0, 1.0, "one-dimensional"),
        ],
    )
    def test_invalid_margins_raise(self, r, c, match):
        with pytest.raises(ValueError, match=match):
            check_margins(r, c)


class TestSampleTransportationPolytope:
    def test_samples_lie_in_polytope(self):
        r = jnp.arange(1.0, 6.0)
        c = 15.0 * 0.75 ** jnp.arange(4.0) / (0.75 ** jnp.arange(4.0)).sum()
        x = sample_hit_run(jax.random.key(0), r, c, num_samples=200)
        assert x.shape == (200, 5, 4)
        np.testing.assert_allclose(x.sum(-1), jnp.broadcast_to(r, (200, 5)), rtol=1e-4)
        np.testing.assert_allclose(x.sum(-2), jnp.broadcast_to(c, (200, 4)), rtol=1e-4)
        assert jnp.all(x >= -1e-5)

    def test_two_by_two_is_uniform(self):
        # For r = (3, 1), c = (2, 2) the exact law of x_11 is Uniform(1, 2).
        x11 = sample_hit_run(
            jax.random.key(1), [3.0, 1.0], [2.0, 2.0], num_samples=4000
        )[:, 0, 0]
        assert jnp.all((x11 >= 1.0 - 1e-5) & (x11 <= 2.0 + 1e-5))
        assert abs(float(x11.mean()) - 1.5) < 0.03
        assert abs(float(x11.var()) - 1 / 12) < 0.01

    def test_equal_margins_mean(self):
        # With r = c = 1 every cell has mean 1/K by symmetry.
        K = 4
        x = sample_hit_run(
            jax.random.key(2), jnp.ones((64, K)), jnp.ones((64, K)),
            num_samples=500, thin=10,
        )
        np.testing.assert_allclose(x.mean((0, 1)), 1 / K, atol=0.02)

    def test_zero_margins_stay_zero(self):
        x = sample_hit_run(
            jax.random.key(3), [2.0, 0.0, 1.0], [0.0, 1.5, 1.5], num_samples=100
        )
        assert jnp.all(jnp.isfinite(x))
        assert jnp.all(x[:, 1, :] == 0.0) and jnp.all(x[:, :, 0] == 0.0)
        np.testing.assert_allclose(x.sum(-1)[:, [0, 2]], [[2.0, 1.0]] * 100, rtol=1e-4)

    def test_single_point_polytope(self):
        # A single row fixes the table to the column sums.
        x = sample_hit_run(
            jax.random.key(4), [3.0], [1.0, 2.0], num_samples=10
        )
        np.testing.assert_allclose(x, jnp.broadcast_to(jnp.array([[1.0, 2.0]]), (10, 1, 2)))

    def test_scale_equivariance(self):
        r = jnp.array([1.0, 2.0, 3.0])
        c = jnp.array([4.0, 1.0, 1.0])
        key = jax.random.key(6)
        x = sample_hit_run(key, r, c, num_samples=50)
        # Scaling by a power of 2 is exact, so the chain follows the same path.
        s = 2.0**20
        x_big = sample_hit_run(key, s * r, s * c, num_samples=50)
        np.testing.assert_allclose(x_big, s * x, rtol=1e-5)
        np.testing.assert_allclose(x_big.sum(-1), jnp.broadcast_to(s * r, (50, 3)), rtol=1e-4)

    def test_batch_shape(self):
        r = jnp.ones((2, 3, 4))
        c = jnp.full((2, 3, 2), 2.0)
        x = sample_hit_run(jax.random.key(5), r, c, num_samples=7)
        assert x.shape == (2, 3, 7, 4, 2)

    def test_symmetric_tables(self):
        r = jnp.array([1.0, 2.0, 3.0, 4.0])
        x = sample_hit_run(
            jax.random.key(7), r, r, num_samples=100, symmetric=True
        )
        np.testing.assert_array_equal(x, x.swapaxes(-1, -2))
        np.testing.assert_allclose(x.sum(-1), jnp.broadcast_to(r, (100, 4)), rtol=1e-4)
        assert jnp.all(x >= -1e-5)

    def test_symmetric_two_by_two_is_uniform(self):
        # Symmetric tables with r = c = (2, 1) are [[x, 2 - x], [2 - x, x - 1]],
        # so x_11 ~ Uniform(1, 2).
        x11 = sample_hit_run(
            jax.random.key(8), [2.0, 1.0], [2.0, 1.0], num_samples=4000, symmetric=True
        )[:, 0, 0]
        assert jnp.all((x11 >= 1.0 - 1e-5) & (x11 <= 2.0 + 1e-5))
        assert abs(float(x11.mean()) - 1.5) < 0.03

    def test_symmetric_requires_equal_margins(self):
        with pytest.raises(ValueError, match="equal row and column sums"):
            sample_hit_run(
                jax.random.key(9), [3.0, 1.0], [2.0, 2.0], num_samples=1, symmetric=True
            )


def explicit_projector(K, fixed):
    """Orthogonal projector onto ker(A_S), built from the constraint matrix (A.13)."""
    A = np.vstack([np.kron(np.ones(K), np.eye(K)), np.kron(np.eye(K), np.ones(K))])
    B = [np.kron(np.eye(K)[k], np.eye(K)[k]) for k in range(K) if fixed[k]]
    A_S = np.vstack([A, *B])
    return np.eye(K * K) - np.linalg.pinv(A_S) @ A_S


class TestFixedDiagonal:
    @pytest.mark.parametrize(
        "fixed",
        [(True, False), (True, True, True), (True, False, True, False, False), (True,) * 5],
    )
    def test_projection_matches_explicit_projector(self, fixed):
        K = len(fixed)
        Z = np.random.default_rng(0).normal(size=(K, K))
        W = _double_center(jnp.asarray(Z), jnp.ones((K, K), dtype=bool))
        D = _project_fixed_diagonal(W, jnp.array(fixed), _fixed_gram_inverse(fixed))
        # Column-major vec, as in the supplement
        expected = (explicit_projector(K, fixed) @ Z.ravel(order="F")).reshape(
            K, K, order="F"
        )
        np.testing.assert_allclose(D, expected, atol=1e-5)

    @pytest.mark.parametrize("symmetric", [False, True])
    def test_samples_keep_margins_and_diagonal(self, symmetric):
        r = jnp.array([1.0, 2.0, 3.0, 4.0])
        c = r if symmetric else jnp.array([4.0, 3.0, 2.0, 1.0])
        diagonal = jnp.array([0.5, jnp.nan, 1.0, jnp.nan])
        x = sample_hit_run(
            jax.random.key(10), r, c, num_samples=100, symmetric=symmetric,
            diagonal=diagonal,
        )
        np.testing.assert_array_equal(x[:, 0, 0], 0.5)
        np.testing.assert_array_equal(x[:, 2, 2], 1.0)
        np.testing.assert_allclose(x.sum(-1), jnp.broadcast_to(r, (100, 4)), rtol=1e-4)
        np.testing.assert_allclose(x.sum(-2), jnp.broadcast_to(c, (100, 4)), rtol=1e-4)
        assert jnp.all(x >= -1e-5)
        if symmetric:
            np.testing.assert_array_equal(x, x.swapaxes(-1, -2))

    def test_full_diagonal_k3_is_uniform_on_segment(self):
        # With K = 3 and the whole diagonal fixed, the tables form a segment, so
        # every hit-and-run step is an independent uniform draw (Remark 2).
        r = jnp.ones(3)
        x = sample_hit_run(
            jax.random.key(11), r, r, num_samples=2000, num_warmup=0,
            diagonal=jnp.array([0.2, 0.3, 0.4]),
        )
        cycle = jnp.roll(jnp.eye(3), 1, axis=1)
        t_lo, t_hi = _chord(x[0], cycle - cycle.T)
        lo, hi = float(x[0, 0, 1] + t_lo), float(x[0, 0, 1] + t_hi)
        ks = stats.kstest(np.asarray(x[:, 0, 1]), stats.uniform(lo, hi - lo).cdf)
        assert ks.pvalue > 0.01

    def test_feasibility(self):
        r = jnp.ones(3)
        # Remark 1: each value lies within its Frechet bounds, yet no table exists.
        assert not diagonal_is_feasible(r, r, jnp.array([1.0, 1.0, 0.0]))
        # The diagonal of the independence table is always strictly feasible.
        assert diagonal_is_feasible(r, r, jnp.array([1 / 3, 1 / 3, jnp.nan]))

    def test_infeasible_diagonal_raises(self):
        r = jnp.ones(3)
        with pytest.raises(ValueError, match="no strictly positive table"):
            sample_hit_run(
                jax.random.key(12), r, r, num_samples=1,
                diagonal=jnp.array([1.0, 1.0, jnp.nan]),
            )

    def test_two_by_two_with_both_cells_fixed_raises(self):
        with pytest.raises(ValueError, match="no strictly positive table"):
            sample_hit_run(
                jax.random.key(13), [3.0, 1.0], [2.0, 2.0], num_samples=1,
                diagonal=jnp.array([1.5, 0.5]),
            )

    def test_fixed_cells_must_match_across_batch(self):
        r = jnp.ones((2, 3))
        diagonal = jnp.array([[0.2, jnp.nan, jnp.nan], [jnp.nan, 0.2, jnp.nan]])
        with pytest.raises(ValueError, match="same diagonal cells"):
            sample_hit_run(
                jax.random.key(14), r, r, num_samples=1, diagonal=diagonal
            )
