"""
Tests for SocialMix.prop_mix_factorisation (socialmixr's split_matrix):
m[i, j] = c * q * d[i] * a[i, j] * p[j].
"""

import numpy as np
import pytest

from ...utils import AgeGroupSpecs
from ...weights import WeightSpecs
from ...weights.tests.factors import LinearAgeFactor
from ..classical._socialmix_utils import PropMixFactorisation, prop_mix_factorise
from ..classical._SocialMix import SocialMix
from .test_SocialMix_weights import SPECS, _quiet, _survey


def test_reconstruction_and_identities():
    part, cnt, pop, *_ = _survey()
    sm = _quiet(SocialMix, part, cnt, SPECS, pop)
    fac = sm.prop_mix_factorisation()

    assert isinstance(fac, PropMixFactorisation)
    np.testing.assert_allclose(fac.matrix, sm.cint()["All->All"].central)
    np.testing.assert_allclose(fac.reconstruct(), fac.matrix)
    assert fac.spectral_radius == pytest.approx(fac.mean_contacts * fac.normalisation)
    assert fac.spectral_radius == pytest.approx(np.max(np.abs(np.linalg.eigvals(fac.matrix))))
    np.testing.assert_allclose(fac.assortativity @ fac.population_share, 1.0)
    np.testing.assert_allclose(fac.contacts, fac.matrix.sum(axis=1))


def test_hand_computed_example_matches_socialmixr_formulas():
    m = np.array([[3.0, 1.0], [2.0, 4.0]])  # eigenvalues 5 and 2
    P = np.array([100.0, 300.0])
    fac = prop_mix_factorise(m, P, AgeGroupSpecs(0, 19, 10))

    np.testing.assert_allclose(fac.population_share, [0.25, 0.75])
    assert fac.mean_contacts == pytest.approx((100 * 4 + 300 * 6) / 400)  # 5.5
    assert fac.spectral_radius == pytest.approx(5.0)
    assert fac.normalisation == pytest.approx(5.0 / 5.5)
    np.testing.assert_allclose(fac.relative_contacts, [0.8, 1.2])
    np.testing.assert_allclose(fac.assortativity, [[3.0, 1 / 3], [4 / 3, 8 / 9]])

    # Mixing in proportion to population size gives a == 1 everywhere.
    r = np.array([4.0, 6.0])
    proportionate = prop_mix_factorise(np.outer(r, fac.population_share), P, AgeGroupSpecs(0, 19, 10))
    np.testing.assert_allclose(proportionate.assortativity, 1.0)


def test_preconditions():
    part, cnt, pop, *_ = _survey()
    with pytest.raises(ValueError, match="population sizes"):
        _quiet(SocialMix, part, cnt, SPECS).prop_mix_factorisation()

    part_s, cnt_s, pop_s, *_ = _survey(sex=True)
    with pytest.raises(NotImplementedError, match="age-only"):
        _quiet(SocialMix, part_s, cnt_s, SPECS, pop_s).prop_mix_factorisation()

    m = np.array([[0.0, 0.0], [2.0, 4.0]])
    with pytest.warns(UserWarning, match=r"\[0, 10\)"):
        fac = prop_mix_factorise(m, np.array([1.0, 1.0]), AgeGroupSpecs(0, 19, 10))
    assert np.isnan(fac.assortativity[0]).all()
    assert np.isfinite(fac.assortativity[1]).all()


def test_weighted_and_unweighted():
    part, cnt, pop, *_ = _survey()
    sm = _quiet(SocialMix, part, cnt, SPECS, pop, weight_specs=WeightSpecs(LinearAgeFactor()))
    fac_w = sm.prop_mix_factorisation()
    fac_u = sm.prop_mix_factorisation(weighted=False)

    assert fac_w.weighted and not fac_u.weighted
    np.testing.assert_allclose(fac_w.matrix, sm.cint()["All->All"].central)
    np.testing.assert_allclose(fac_u.matrix, sm.cint(weighted=False)["All->All"].central)
    assert not np.allclose(fac_w.assortativity, fac_u.assortativity)
