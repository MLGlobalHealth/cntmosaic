"""Unit tests for the predict module's sampling and bound utilities.

Statistical tests use fixed seeds and compare means within 4 standard errors,
so they are deterministic.
"""

import numpy as np
import pytest
from scipy.stats import beta

from .. import _predict
from ...models.tests.fixtures import partial_large_sample  # noqa: F401
from .._predict import (
    feasible_mixing_bounds,
    rtruncated_beta,
    rtruncated_dirichlet,
    sample_eta,
    z_marginals,
)


# ============================================================================
# Helpers
# ============================================================================


def _rejection_reference(alpha, lb, ub, n, rng):
    """Exact truncated Dirichlet draws via rejection from the untruncated one."""
    d = rng.dirichlet(alpha, size=n)
    keep = np.all((d >= lb) & (d <= ub), axis=1)
    return d[keep]


def _feasible_bounds(w, lo=0.5, hi=2.0):
    """Bounds around simplex weights w with sum(lb) = lo < 1 <= sum(ub)."""
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    return lo * w, np.minimum(1.0, hi * w)


def _sample(alpha, lb, ub, n, seed, **kwargs):
    K = len(alpha)
    return rtruncated_dirichlet(
        np.broadcast_to(alpha, (n, K)),
        np.broadcast_to(lb, (n, K)),
        np.broadcast_to(ub, (n, K)),
        rng=np.random.default_rng(seed),
        **kwargs,
    )


def _assert_on_simplex_within(x, lb, ub):
    assert np.all(x >= lb - 1e-10)
    assert np.all(x <= ub + 1e-10)
    np.testing.assert_allclose(x.sum(axis=-1), 1.0, atol=1e-10)


def _assert_means_close(x, ref, n_se=4.0):
    se = np.sqrt(x.var(axis=0) / len(x) + ref.var(axis=0) / len(ref))
    diff = np.abs(x.mean(axis=0) - ref.mean(axis=0))
    assert np.all(diff < n_se * se), f"max z = {np.max(diff / se):.1f}"


# ============================================================================
# rtruncated_beta
# ============================================================================


class TestRTruncatedBeta:

    def test_within_bounds_and_broadcast_shape(self):
        a = np.full((4, 1), 0.5)
        lb = np.array([0.1, 0.2, 0.3])
        x = rtruncated_beta(a, 2.0, lb=lb, ub=0.6, rng=np.random.default_rng(0))

        assert x.shape == (4, 3)
        assert np.all(x >= lb) and np.all(x <= 0.6)

    @pytest.mark.parametrize("a, b, lb, ub", [(0.5, 0.5, 0.2, 0.7), (2.0, 5.0, 0.3, 0.9)])
    def test_mean_matches_theory(self, a, b, lb, ub):
        n = 100_000
        x = rtruncated_beta(
            np.full(n, a), b, lb=lb, ub=ub, rng=np.random.default_rng(1)
        )
        expected = beta(a, b).expect(lb=lb, ub=ub, conditional=True)

        assert abs(x.mean() - expected) < 4 * x.std() / np.sqrt(n)

    def test_negligible_mass_interval(self):
        # CDF(ub) underflows to 0: draws must still lie in [lb, ub], at the
        # bound nearest the mode
        x = rtruncated_beta(1e4, 1.0, lb=0.1, ub=0.89, rng=np.random.default_rng(2))
        assert x == pytest.approx(0.89)

    def test_upper_tail_interval(self):
        lb = 1 - 1e-6
        x = rtruncated_beta(
            np.full(1_000, 2.0), 2.0, lb=lb, ub=1.0, rng=np.random.default_rng(3)
        )
        assert np.all((x >= lb) & (x <= 1.0))
        assert np.unique(x).size > 1


# ============================================================================
# rtruncated_dirichlet
# ============================================================================


class TestRTruncatedDirichlet:

    def test_3d_bounds_and_mean(self):
        alpha = np.full(3, 0.5)
        lb = np.array([0.1, 0.2, 0.3])
        ub = np.array([0.9, 0.8, 0.7])

        x = _sample(alpha, lb, ub, n=50_000, seed=0)
        ref = _rejection_reference(alpha, lb, ub, 2_000_000, np.random.default_rng(1))

        _assert_on_simplex_within(x, lb, ub)
        _assert_means_close(x, ref)

    def test_10d_bounds_and_mean(self):
        alpha = np.linspace(1.0, 3.0, 10)
        # Wide enough for a usable rejection reference (~13% acceptance);
        # both lb and ub still bind in ~70% of untruncated draws
        lb, ub = _feasible_bounds(np.linspace(1.0, 3.0, 10), lo=0.25, hi=2.0)

        x = _sample(alpha, lb, ub, n=20_000, seed=2)
        ref = _rejection_reference(alpha, lb, ub, 2_000_000, np.random.default_rng(3))

        assert len(ref) > 10_000, "rejection reference too small; widen bounds"
        _assert_on_simplex_within(x, lb, ub)
        _assert_means_close(x, ref)

    def test_100d_symmetric_mean_is_one_over_k(self):
        K = 100
        alpha = np.full(K, 0.5)
        lb, ub = np.full(K, 0.5 / K), np.full(K, 2.0 / K)

        x = _sample(alpha, lb, ub, n=5_000, seed=4)

        _assert_on_simplex_within(x, lb, ub)
        # Exchangeability: every component has mean exactly 1/K
        se = x.std(axis=0) / np.sqrt(len(x))
        assert np.all(np.abs(x.mean(axis=0) - 1 / K) < 4 * se)

    def test_100d_permutation_equivariance(self):
        K = 100
        rng = np.random.default_rng(5)
        alpha = rng.uniform(0.5, 2.0, K)
        lb, ub = _feasible_bounds(rng.uniform(1.0, 3.0, K))
        perm = rng.permutation(K)

        x = _sample(alpha, lb, ub, n=2_000, seed=6)
        x_perm = _sample(alpha[perm], lb[perm], ub[perm], n=2_000, seed=7)

        _assert_on_simplex_within(x, lb, ub)
        _assert_on_simplex_within(x_perm, lb[perm], ub[perm])
        # Permuting the inputs must permute the means, with no order effects
        _assert_means_close(x_perm, x[:, perm])

    @pytest.mark.parametrize("K", [3, 10])
    def test_untruncated_mean(self, K):
        alpha = np.linspace(0.5, 3.0, K)
        x = _sample(alpha, 0.0, 1.0, n=20_000, seed=8)

        se = x.std(axis=0) / np.sqrt(len(x))
        assert np.all(np.abs(x.mean(axis=0) - alpha / alpha.sum()) < 4 * se)

    def test_k2_matches_truncated_beta(self):
        alpha = np.array([0.5, 1.5])
        lb, ub = np.array([0.2, 0.3]), np.array([0.6, 0.9])
        x = _sample(alpha, lb, ub, n=100_000, seed=9)

        # x_0 ~ Beta(alpha_0, alpha_1) on [max(lb0, 1-ub1), min(ub0, 1-lb1)]
        expected = beta(*alpha).expect(lb=0.2, ub=0.6, conditional=True)

        _assert_on_simplex_within(x, lb, ub)
        assert abs(x[:, 0].mean() - expected) < 4 * x[:, 0].std() / np.sqrt(len(x))

    def test_batch_shape(self):
        S, A, K = 3, 4, 5
        lb, ub = _feasible_bounds(np.arange(1, K + 1))
        x = rtruncated_dirichlet(
            np.ones((S, A, A, K)), lb, ub, rng=np.random.default_rng(10)
        )

        assert x.shape == (S, A, A, K)
        _assert_on_simplex_within(x, lb, ub)

    @pytest.mark.parametrize(
        "lb, ub",
        [
            ([0.5, 0.4, 0.3], [0.9, 0.9, 0.9]),  # sum(lb) > 1
            ([0.1, 0.1, 0.1], [0.3, 0.3, 0.3]),  # sum(ub) < 1
            ([0.5, 0.1, 0.1], [0.4, 0.9, 0.9]),  # lb > ub
        ],
    )
    def test_infeasible_bounds_raise(self, lb, ub):
        with pytest.raises(ValueError, match="Infeasible bounds"):
            rtruncated_dirichlet(np.ones(3), lb, ub, rng=np.random.default_rng(0))

    def test_reproducible(self):
        alpha = np.full(3, 0.5)
        lb, ub = np.array([0.1, 0.2, 0.3]), np.array([0.9, 0.8, 0.7])

        assert np.array_equal(
            _sample(alpha, lb, ub, n=100, seed=11),
            _sample(alpha, lb, ub, n=100, seed=11),
        )


# ============================================================================
# Synthetic stratified contact counts
# ============================================================================

S, A, LABELS = 4, 5, ["M", "F"]


@pytest.fixture
def reciprocal_truth():
    """Fully stratified counts with Z^{st}_ab = Z^{ts}_ba, and their marginals."""
    rng = np.random.default_rng(0)

    def symmetric():
        X = rng.gamma(2.0, 50.0, (S, A, A))
        return (X + X.transpose(0, 2, 1)) / 2

    MF = rng.gamma(2.0, 50.0, (S, A, A))
    Z = {
        "M->M": symmetric(),
        "F->F": symmetric(),
        "M->F": MF,
        "F->M": MF.transpose(0, 2, 1),
    }
    z = {s: sum(Z[f"{s}->{t}"] for t in LABELS) for s in LABELS}
    return Z, z


@pytest.fixture
def P_sa(monkeypatch):
    """Patch _compute_P_sa with age-varying stratified population sizes."""
    P = {"M": np.linspace(50.0, 150.0, A), "F": np.linspace(120.0, 60.0, A)}
    monkeypatch.setattr(_predict, "_compute_P_sa", lambda dataloader: P)
    return P


# ============================================================================
# z_marginals
# ============================================================================


class _StubSummariser:
    def __init__(self, cint):
        self._cint = cint

    def get_posterior_samples(self, quantity):
        assert quantity == "cint"
        return self._cint


def test_z_marginals(P_sa):
    rng = np.random.default_rng(0)
    cint = {f"{s}->All": rng.uniform(0.1, 2.0, (S, A, A)) for s in LABELS}

    z = z_marginals(_StubSummariser(cint), dataloader=None)

    assert list(z) == LABELS
    for s in LABELS:
        assert z[s].shape == (S, A, A)
        # Scaled by the participant (row) age population, not the contact age
        np.testing.assert_allclose(z[s], cint[f"{s}->All"] * P_sa[s][None, :, None])


# ============================================================================
# feasible_mixing_bounds
# ============================================================================


class TestFeasibleMixingBounds:

    def test_truth_within_count_bounds(self, reciprocal_truth):
        Z, z = reciprocal_truth
        lb, ub = feasible_mixing_bounds(z, return_eta=False)

        for key, truth in Z.items():
            assert np.all(lb[key] <= truth + 1e-9)
            assert np.all(truth <= ub[key] + 1e-9)

    def test_eta_bounds(self, reciprocal_truth):
        _, z = reciprocal_truth
        lb, ub = feasible_mixing_bounds(z, return_eta=True)
        count_lb, count_ub = feasible_mixing_bounds(z, return_eta=False)

        assert set(lb) == {f"{s}->{t}" for s in LABELS for t in LABELS}
        for s in LABELS:
            for t in LABELS:
                key = f"{s}->{t}"
                assert lb[key].shape == (S, A, A)
                assert np.all((0 <= lb[key]) & (lb[key] <= ub[key]) & (ub[key] <= 1))
                np.testing.assert_allclose(lb[key], count_lb[key] / z[s])
                np.testing.assert_allclose(ub[key], np.minimum(1, count_ub[key] / z[s]))

    def test_bounds_admit_a_simplex(self, reciprocal_truth):
        _, z = reciprocal_truth
        lb, ub = feasible_mixing_bounds(z, return_eta=True)

        for s in LABELS:
            assert np.all(sum(lb[f"{s}->{t}"] for t in LABELS) <= 1 + 1e-9)
            assert np.all(sum(ub[f"{s}->{t}"] for t in LABELS) >= 1 - 1e-9)

    def test_hand_computed(self):
        z = {"M": np.array([[[3.0]]]), "F": np.array([[[5.0]]])}
        lb, ub = feasible_mixing_bounds(z, return_eta=True)

        expected = {  # key: (lb, ub)
            "M->M": (0.0, 1.0),
            "M->F": (0.0, 1.0),
            "F->M": (0.0, 0.6),
            "F->F": (0.4, 1.0),
        }
        for key, (e_lb, e_ub) in expected.items():
            assert lb[key].item() == pytest.approx(e_lb)
            assert ub[key].item() == pytest.approx(e_ub)

    def test_zero_marginal(self):
        z = {"M": np.array([[[0.0]]]), "F": np.array([[[5.0]]])}
        lb, ub = feasible_mixing_bounds(z, return_eta=True)

        for t in LABELS:
            assert lb[f"M->{t}"].item() == 0.0
            assert ub[f"M->{t}"].item() == 1.0
        assert not any(np.isnan(v).any() for v in [*lb.values(), *ub.values()])


# ============================================================================
# sample_eta
# ============================================================================


class TestSampleEta:

    @pytest.fixture
    def setup(self, reciprocal_truth, P_sa):
        _, z = reciprocal_truth
        lb, ub = feasible_mixing_bounds(z, return_eta=True)
        return z, lb, ub

    def _eta(self, setup, seed=1, **kwargs):
        z, lb, ub = setup
        return sample_eta(z, lb, ub, None, rng=np.random.default_rng(seed), **kwargs)

    def test_keys_and_shapes(self, setup):
        eta = self._eta(setup)

        assert set(eta) == {f"{s}->{t}" for s in LABELS for t in LABELS}
        assert all(v.shape == (S, A, A) for v in eta.values())

    def test_within_bounds(self, setup):
        _, lb, ub = setup
        eta = self._eta(setup)

        for key, e in eta.items():
            assert np.all(e >= lb[key] - 1e-9)
            assert np.all(e <= ub[key] + 1e-9)

    def test_reciprocity(self, setup):
        z, _, _ = setup
        eta = self._eta(setup)

        for s in LABELS:
            for t in LABELS:
                np.testing.assert_allclose(
                    z[s] * eta[f"{s}->{t}"],
                    (z[t] * eta[f"{t}->{s}"]).transpose(0, 2, 1),
                )

    def test_sums_to_one_in_every_cell(self, setup):
        """Fractions of each source sum to 1 over targets in every (a, b)."""
        eta = self._eta(setup)

        for s in LABELS:
            total = sum(eta[f"{s}->{t}"] for t in LABELS)
            np.testing.assert_allclose(total, 1.0, atol=1e-8)

    def test_improper_prior_raises(self, setup):
        with pytest.raises(ValueError, match="improper.*requires alpha >"):
            self._eta(setup, alpha=0.5)

    def test_non_reciprocal_totals_warn(self, setup):
        z, lb, ub = setup
        z_bad = {s: v.copy() for s, v in z.items()}
        z_bad["M"][:, 1, 0] *= 1.5

        with pytest.warns(UserWarning, match="not reciprocal"):
            sample_eta(z_bad, lb, ub, None, rng=np.random.default_rng(0))

    def test_reproducible(self, setup):
        eta1, eta2 = self._eta(setup, seed=3), self._eta(setup, seed=3)

        assert all(np.array_equal(eta1[k], eta2[k]) for k in eta1)

    def test_external_counts_shift_towards_target(self, setup):
        _, _, ub = setup
        tril = np.tril(np.ones((A, A), dtype=bool))
        ext = {f"{s}->{t}": np.zeros((A, A)) for s in LABELS for t in LABELS}
        ext["M->M"][:] = 1e4

        base = self._eta(setup)["M->M"][:, tril]
        shifted = self._eta(setup, external_counts=ext)["M->M"][:, tril]

        assert shifted.mean() > base.mean()
        np.testing.assert_allclose(shifted, ub["M->M"][:, tril], atol=0.05)


# ============================================================================
# End-to-end: fitted partially stratified model
# ============================================================================


def test_z_marginals_totals_are_reciprocal(partial_large_sample):
    """Total contact counts sum_s z^s_ab must be symmetric in (a, b).

    Regression test: partially stratified models used the first stratum's
    contact population as the offset instead of the total population.
    """
    from jax.random import PRNGKey

    from ...analysis import ModelSummariser
    from ...dataloader import ContactSurveyLoader, StratificationData
    from ...models._GenMixFF import GenMixFF
    from ...models.numpyro.priors import PSpline2D

    part_data, cnt_data, pop_data = partial_large_sample
    strat_vars = pop_data.get_strat_vars()
    strat_data = StratificationData.from_counts(
        pop_data.data, age_col="age", strat_var_cols=strat_vars, count_col="P"
    )
    dataloader = ContactSurveyLoader.from_containers(
        part_data, cnt_data, pop_data, strat_data
    )
    priors = {"rate": PSpline2D(prior_type="global", M=5)}
    priors.update({v: PSpline2D(prior_type="partial", M=5) for v in strat_vars})
    model = GenMixFF(dataloader, priors, "poisson")
    model.run_inference_mcmc(PRNGKey(0), num_warmup=10, num_samples=10, num_chains=1)

    z = z_marginals(ModelSummariser(model), dataloader)
    total = sum(z.values())

    np.testing.assert_allclose(total, total.transpose(0, 2, 1), rtol=1e-4)


def test_sample_eta_k2_matches_quadrature(monkeypatch):
    """Mean of one cell's free value matches its exact 1-D prior density.

    For K = 2 and a > b, x = Z^{MF}_ab fixes the whole 2 x 2 table:
    X = [[r0 - x, x], [c0 - r0 + x, c1 - x]], and the mirror cell holds X^T.
    """
    from scipy import integrate

    alpha, n = 4.0, 4_000
    P = {"M": np.array([80.0, 120.0]), "F": np.array([110.0, 90.0])}
    monkeypatch.setattr(_predict, "_compute_P_sa", lambda dataloader: P)
    zM = np.array([[150.0, 60.0], [90.0, 140.0]])
    zF = np.array([[170.0, 40.0], [10.0, 160.0]])  # zM + zF is symmetric
    z = {"M": np.broadcast_to(zM, (n, 2, 2)), "F": np.broadcast_to(zF, (n, 2, 2))}

    eta = sample_eta(z, None, None, None, alpha=alpha, rng=np.random.default_rng(0))
    x = eta["M->F"][:, 1, 0] * zM[1, 0]

    Q = np.stack([P["M"], P["F"]], axis=-1)
    Q = alpha * Q / Q.sum(axis=-1, keepdims=True)  # conc[age, target]
    r0, r1, c0, c1 = zM[1, 0], zF[1, 0], zM[0, 1], zF[0, 1]

    def density(v):
        X00, X01, X10, X11 = r0 - v, v, c0 - r0 + v, c1 - v
        ab, ba = Q[0], Q[1]  # contact age b = 0 at (1, 0); a = 1 at (0, 1)
        return (
            X00 ** (ab[0] - 1) * X01 ** (ab[1] - 1) * X10 ** (ab[0] - 1) * X11 ** (ab[1] - 1)
            * X00 ** (ba[0] - 1) * X10 ** (ba[1] - 1) * X01 ** (ba[0] - 1) * X11 ** (ba[1] - 1)
        )

    lo, hi = max(0.0, r0 - c0), min(r0, c1)
    norm = integrate.quad(density, lo, hi)[0]
    expected = integrate.quad(lambda v: v * density(v), lo, hi)[0] / norm

    assert abs(x.mean() - expected) < 4 * x.std() / np.sqrt(n)


def test_sample_eta_k3_constraints(monkeypatch):
    rng = np.random.default_rng(0)
    labels, S3, A3 = ["a", "b", "c"], 3, 4
    Z = {}
    for i, s in enumerate(labels):
        for t in labels[i:]:
            X = rng.gamma(2.0, 50.0, (S3, A3, A3))
            if s == t:
                Z[f"{s}->{t}"] = (X + X.transpose(0, 2, 1)) / 2
            else:
                Z[f"{s}->{t}"], Z[f"{t}->{s}"] = X, X.transpose(0, 2, 1)
    z = {s: sum(Z[f"{s}->{t}"] for t in labels) for s in labels}
    P = {s: np.linspace(50.0, 150.0, A3) * (1 + 0.2 * i) for i, s in enumerate(labels)}
    monkeypatch.setattr(_predict, "_compute_P_sa", lambda dataloader: P)
    lb, ub = feasible_mixing_bounds(z, return_eta=True)

    eta = sample_eta(z, lb, ub, None, alpha=3.0, rng=np.random.default_rng(1))

    for s in labels:
        np.testing.assert_allclose(sum(eta[f"{s}->{t}"] for t in labels), 1.0, atol=1e-10)
        for t in labels:
            key = f"{s}->{t}"
            np.testing.assert_allclose(
                z[s] * eta[key], (z[t] * eta[f"{t}->{s}"]).transpose(0, 2, 1), rtol=1e-10
            )
            assert np.all(eta[key] >= lb[key] - 1e-9)
            assert np.all(eta[key] <= ub[key] + 1e-9)
