import warnings

import jax
import numpy as np
import pytest

from cntmosaic.predict import counts_to_intensities, sample_full_counts

STRATA = ["a", "b", "c"]
S, A, K = 3, 5, len(STRATA)


def make_z_marginals(seed=0, concentration=1.0):
    """Reciprocal margins: a symmetric total T_ab split across strata at random."""
    rng = np.random.default_rng(seed)
    U = rng.uniform(1.0, 2.0, size=(S, A, A))
    T = U + U.swapaxes(1, 2)
    shares = rng.dirichlet(np.full(K, concentration), size=(S, A, A))  # (S, A, A, K)
    return {k: T * shares[..., i] for i, k in enumerate(STRATA)}


def stack_counts(counts):
    """Stack a counts dictionary into X[s, k, l, a, b]."""
    return np.stack([np.stack([counts[f"{k}->{l}"] for l in STRATA], 1) for k in STRATA], 1)


@pytest.fixture(scope="module", params=["swap", "hit_and_run"])
def sampler(request):
    """Run the tests with both samplers."""
    return request.param


@pytest.fixture(scope="module")
def z_marg():
    return make_z_marginals()


@pytest.fixture(scope="module")
def X(z_marg, sampler):
    return stack_counts(
        sample_full_counts(z_marg, jax.random.key(0), num_warmup=50, sampler=sampler)
    )


def test_shapes(z_marg, sampler):
    counts = sample_full_counts(z_marg, jax.random.key(0), num_warmup=5, sampler=sampler)
    assert set(counts) == {f"{k}->{l}" for k in STRATA for l in STRATA}
    assert all(v.shape == (S, A, A) for v in counts.values())


def test_margins(X, z_marg):
    Z = np.stack([z_marg[k] for k in STRATA], 1)  # (S, K, A, A)
    # Row sums over target strata give z^k_ab; column sums over source strata give z^l_ba.
    np.testing.assert_allclose(X.sum(2), Z, rtol=1e-4)
    np.testing.assert_allclose(X.sum(1), Z.swapaxes(2, 3), rtol=1e-4)


def test_reciprocity(X):
    # X^{kl}_ab = X^{lk}_ba, including the diagonal a = b.
    np.testing.assert_allclose(X, X.transpose(0, 2, 1, 4, 3), rtol=1e-6)


def test_non_negative(X):
    assert np.all(X >= -1e-6 * X.max())


def test_unequal_totals_are_reconciled(z_marg, sampler):
    tilt = np.triu(np.full((A, A), 1.1)) + np.tril(np.ones((A, A)), -1)
    z_tilted = {k: v * tilt for k, v in z_marg.items()}
    with pytest.warns(UserWarning, match="rescaled to their average"):
        counts = sample_full_counts(
            z_tilted, jax.random.key(1), num_warmup=5, sampler=sampler
        )
    X = stack_counts(counts)
    np.testing.assert_allclose(X, X.transpose(0, 2, 1, 4, 3), rtol=1e-6)
    assert np.all(X >= -1e-6 * X.max())


def test_counts_to_intensities():
    counts = {"a->b": np.ones((2, 3, 3))}
    P_sa = {"a": np.array([2.0, 0.0, 4.0])}
    m = counts_to_intensities(counts, P_sa)["a->b"]
    np.testing.assert_allclose(m[:, 0], 0.5)
    np.testing.assert_allclose(m[:, 1], 0.0)
    np.testing.assert_allclose(m[:, 2], 0.25)


class TestAssortativity:
    # Age-specific fractions for stratum "a", one common fraction for stratum "c"
    eta = {
        "a": np.random.default_rng(1).uniform(0.3, 0.5, size=(A, A)),
        "b": 0.35,
        "c": 0.3,
    }

    def eta_of(self, k):
        return np.broadcast_to(self.eta[k], (S, A, A))

    @pytest.fixture(scope="class")
    def z_even(self):
        # Shares close to 1/3, so the fractions above are feasible for every table.
        return make_z_marginals(concentration=50.0)

    @pytest.fixture(scope="class")
    def X_eta(self, z_even, sampler):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no table may need shrinking
            counts = sample_full_counts(
                z_even, jax.random.key(2), num_warmup=50, assortativity=self.eta,
                sampler=sampler,
            )
        return stack_counts(counts)

    def test_fixed_diagonal_on_sampled_side(self, X_eta, z_even):
        a, b = np.tril_indices(A)  # respondent age a >= contact age b
        for i, k in enumerate(STRATA):
            if k in self.eta:
                expected = self.eta_of(k)[:, a, b] * z_even[k][:, a, b]
                np.testing.assert_allclose(X_eta[:, i, i][:, a, b], expected, rtol=1e-5)

    def test_other_side_follows_by_reciprocity(self, X_eta, z_even):
        a, b = np.triu_indices(A, k=1)  # a < b: the fraction is implied
        z, eta = z_even["a"], self.eta_of("a")
        implied = eta[:, b, a] * z[:, b, a] / z[:, a, b]
        np.testing.assert_allclose(
            X_eta[:, 0, 0][:, a, b] / z[:, a, b], implied, rtol=1e-5
        )

    def test_margins_reciprocity_non_negative(self, X_eta, z_even):
        Z = np.stack([z_even[k] for k in STRATA], 1)
        np.testing.assert_allclose(X_eta.sum(2), Z, rtol=1e-4)
        np.testing.assert_allclose(X_eta.sum(1), Z.swapaxes(2, 3), rtol=1e-4)
        np.testing.assert_allclose(X_eta, X_eta.transpose(0, 2, 1, 4, 3), rtol=1e-6)
        assert np.all(X_eta >= -1e-6 * X_eta.max())

    def test_per_draw_fractions(self, z_even, sampler):
        eta = np.random.default_rng(2).uniform(0.3, 0.5, size=(S, A, A))
        X = stack_counts(
            sample_full_counts(
                z_even, jax.random.key(6), num_warmup=5,
                assortativity={"a": 0.3, "b": eta, "c": 0.3},
                sampler=sampler,
            )
        )
        a, b = np.tril_indices(A)
        np.testing.assert_allclose(
            X[:, 1, 1][:, a, b], eta[:, a, b] * z_even["b"][:, a, b], rtol=1e-5
        )

    def test_infeasible_fractions_are_shrunk(self, z_marg, sampler):
        with pytest.warns(UserWarning, match="infeasible for"):
            counts = sample_full_counts(
                z_marg, jax.random.key(3), num_warmup=5,
                assortativity={"a": 0.99, "b": 0.3, "c": 0.3},
                sampler=sampler,
            )
        X = stack_counts(counts)
        Z = np.stack([z_marg[k] for k in STRATA], 1)
        np.testing.assert_allclose(X.sum(2), Z, rtol=1e-4)
        assert np.all(X >= -1e-6 * X.max())
        # Shrinking only lowers the fraction on the sampled side
        a, b = np.tril_indices(A)
        assert np.all(X[:, 0, 0][:, a, b] <= 0.99 * z_marg["a"][:, a, b] * (1 + 1e-5))

    @pytest.mark.parametrize(
        "assortativity, match",
        [
            ({"x": 0.5}, "Unknown strata"),
            ({"a": 1.5}, r"\[0, 1\]"),
            ({"a": np.full((A + 1, A), 0.3)}, "broadcastable"),
        ],
    )
    def test_invalid_assortativity_raises(self, z_marg, assortativity, match):
        with pytest.raises(ValueError, match=match):
            sample_full_counts(
                z_marg, jax.random.key(4), num_warmup=1, assortativity=assortativity
            )

    def test_partial_assortativity(self, z_even):
        # Hit-and-run supports a partly fixed diagonal; swap moves need every stratum.
        counts = sample_full_counts(
            z_even, jax.random.key(8), num_warmup=5, assortativity={"a": 0.4},
            sampler="hit_and_run",
        )
        a, b = np.tril_indices(A)
        np.testing.assert_allclose(
            counts["a->a"][:, a, b], 0.4 * z_even["a"][:, a, b], rtol=1e-5
        )
        with pytest.raises(ValueError, match="missing"):
            sample_full_counts(
                z_even, jax.random.key(8), num_warmup=5, assortativity={"a": 0.4},
                sampler="swap",
            )

    def test_two_strata_allow_one_fraction(self):
        z = make_z_marginals()
        z2 = {"a": z["a"], "b": z["b"] + z["c"]}
        with pytest.raises(ValueError, match="at most one stratum"):
            sample_full_counts(
                z2, jax.random.key(5), num_warmup=1, assortativity={"a": 0.3, "b": 0.3}
            )


def test_unknown_sampler_raises(z_marg):
    with pytest.raises(ValueError, match="sampler must be one of"):
        sample_full_counts(z_marg, jax.random.key(7), sampler="gibbs")
