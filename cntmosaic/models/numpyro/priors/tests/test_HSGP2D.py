import warnings

import numpy as np
import pytest
from numpyro import handlers
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_matern

from .._HSGP2D import HSGP2D
from .._hsgp_config import (
    GRID_DEFAULTS,
    HSGP_KERNEL_CONSTANTS,
    halfnormal_quantile,
    hsgp_boundary_factor,
    hsgp_max_lenscale,
    hsgp_min_lenscale,
    hsgp_min_num_basis,
    kernel_constants,
)


def test_initialisation():

  prior = HSGP2D(
    C=[1.5, 1.5],
    M=[30, 30],
    grid_type='age-age',
    prior_type='global'
  )

  assert prior.C == [1.5, 1.5]
  assert prior.M == [30, 30]
  assert prior.grid_type == 'age-age'
  assert prior.prior_type == 'global'


def test_default_initialisation():
  prior = HSGP2D(prior_type='global')
  assert prior.C == 'auto'
  assert prior.M is None
  assert prior.lenscale_scale == 0.5


# ---------------------------------------------------------------------------
# Configuration rules (_hsgp_config)
# ---------------------------------------------------------------------------

def test_kernel_constants():
  assert kernel_constants(float('inf')) == (3.2, 1.75)
  assert kernel_constants(3 / 2) == (4.5, 3.42)
  assert kernel_constants(5 / 2) == (4.1, 2.65)
  assert kernel_constants(7 / 2) == kernel_constants(5 / 2)
  assert 1 / 2 not in HSGP_KERNEL_CONSTANTS
  with pytest.raises(ValueError, match='nu=0.5'):
    kernel_constants(1 / 2)


def test_boundary_factor_floor():
  # A small lengthscale relative to the half-range hits the 1.2 floor.
  np.testing.assert_allclose(hsgp_boundary_factor(0.01, [1.0, 2.0], 5 / 2), [1.2, 1.2])
  np.testing.assert_allclose(hsgp_boundary_factor(1.0, 2.0, 5 / 2), [4.1 / 2.0])


def test_matches_pymc_heuristic():
  # pymc.gp.hsgp_approx.approx_hsgp_hyperparams(
  #   x_range=[-5, 95], lengthscale_range=[1, 50], cov_func='expquad') -> (280, 3.2)
  S = (95 - (-5)) / 2
  c = hsgp_boundary_factor(50.0, S, float('inf'))
  np.testing.assert_allclose(c, [3.2])
  assert hsgp_min_num_basis(c * S, 1.0, float('inf')) == [280]


def test_min_max_lenscale_are_inverse_rules():
  S = np.array([1.7, 2.4])
  lmax = 0.9
  C = hsgp_boundary_factor(lmax, S, 5 / 2)
  # Without the floor binding, C supports exactly lmax.
  assert hsgp_max_lenscale(C, S, 5 / 2) == pytest.approx(lmax)
  L = C * S
  M = hsgp_min_num_basis(L, 0.2, 5 / 2)
  assert hsgp_min_lenscale(L, M, 5 / 2) <= 0.2


def test_halfnormal_quantile():
  assert halfnormal_quantile(0.5, 0.95) == pytest.approx(0.5 * 1.959964, rel=1e-5)


# ---------------------------------------------------------------------------
# Auto configuration per grid type
# ---------------------------------------------------------------------------

def test_auto_config_age_age():
  prior = HSGP2D(prior_type='global', grid_type='age-age')
  prior.set_age_bounds(0, 84)

  assert prior.M_ == GRID_DEFAULTS['age-age']['M']
  assert prior.C_[0] == pytest.approx(prior.C_[1])
  np.testing.assert_allclose(prior.C_, [2.347, 2.347], atol=1e-3)
  assert prior.lenscale_max == pytest.approx(halfnormal_quantile(0.5, 0.95))
  assert prior.lenscale_min == pytest.approx(0.266, abs=1e-3)
  assert prior.lenscale_min < prior.lenscale_max
  np.testing.assert_allclose(prior.L, np.array(prior.C_) * prior.S)

  years = prior.lenscale_bounds_years
  assert set(years) == {'age_1', 'age_2'}
  np.testing.assert_allclose(years['age_1'], (6.5, 24.0), atol=0.1)


def test_auto_config_diff_age():
  prior = HSGP2D(prior_type='global', grid_type='diff-age')
  prior.set_age_bounds(0, 84)

  assert prior.M_ == GRID_DEFAULTS['diff-age']['M']
  # The age-difference axis has a larger standardised half-range, so a smaller C.
  assert prior.S[0] > prior.S[1]
  assert prior.C_[0] < prior.C_[1]
  np.testing.assert_allclose(prior.C_, [1.660, 2.347], atol=1e-3)
  assert prior.lenscale_min == pytest.approx(0.266, abs=1e-3)
  assert prior.lenscale_min < prior.lenscale_max
  assert set(prior.lenscale_bounds_years) == {'age_diff', 'age'}


def test_auto_config_respects_M_and_scale():
  coarse = HSGP2D(prior_type='global', M=20)
  coarse.set_age_bounds(0, 84)
  fine = HSGP2D(prior_type='global', M=60)
  fine.set_age_bounds(0, 84)
  assert coarse.M_ == [20, 20]
  assert fine.lenscale_min < coarse.lenscale_min

  wide = HSGP2D(prior_type='global', lenscale_scale=1.0)
  wide.set_age_bounds(0, 84)
  assert wide.lenscale_max == pytest.approx(halfnormal_quantile(1.0, 0.95))
  assert wide.C_[0] > coarse.C_[0]


# ---------------------------------------------------------------------------
# Explicit overrides
# ---------------------------------------------------------------------------

def test_explicit_C_derives_lenscale_max():
  prior = HSGP2D(prior_type='global', C=[1.5, 1.5], M=[20, 20])
  prior.set_age_bounds(0, 84)
  assert prior.C_ == [1.5, 1.5]
  assert prior.lenscale_max == pytest.approx(hsgp_max_lenscale(1.5, prior.S, 5 / 2))
  assert prior.lenscale_min == pytest.approx(0.340, abs=1e-3)
  assert prior.lenscale_max == pytest.approx(0.626, abs=1e-3)


def test_explicit_lenscale_max_beyond_C_warns():
  prior = HSGP2D(prior_type='global', C=1.5, lenscale_max=0.9)
  with pytest.warns(UserWarning, match='exceeds the largest lengthscale'):
    prior.set_age_bounds(0, 84)
  assert prior.lenscale_max == 0.9


def test_C_below_floor_warns():
  prior = HSGP2D(prior_type='global', C=1.1)
  with pytest.warns(UserWarning, match='below the recommended minimum'):
    prior.set_age_bounds(0, 84)


def test_M_too_small_raises():
  prior = HSGP2D(prior_type='global', M=5)
  with pytest.raises(ValueError, match='lengthscale range is empty'):
    prior.set_age_bounds(0, 84)


def test_lenscale_min_below_derived_warns():
  prior = HSGP2D(prior_type='global', lenscale_min=0.1)
  with pytest.warns(UserWarning, match='Increase M'):
    prior.set_age_bounds(0, 84)
  assert prior.lenscale_min == 0.1


def test_lenscale_min_above_derived_no_warning():
  prior = HSGP2D(prior_type='global', lenscale_min=0.4)
  with warnings.catch_warnings():
    warnings.simplefilter('error')
    prior.set_age_bounds(0, 84)
  assert prior.lenscale_min == 0.4


def test_nu_half_requires_explicit_settings():
  with pytest.raises(ValueError, match='nu=1/2'):
    HSGP2D(prior_type='global', nu=1 / 2)
  with pytest.raises(ValueError, match='nu=1/2'):
    HSGP2D(prior_type='global', nu=1 / 2, C=2.0)

  prior = HSGP2D(prior_type='global', nu=1 / 2, C=2.0, lenscale_min=0.3)
  prior.set_age_bounds(0, 20)
  assert prior.lenscale_min == 0.3
  assert prior.lenscale_max == pytest.approx(halfnormal_quantile(0.5, 0.95))


@pytest.mark.parametrize(
  'kwargs',
  [
    {'C': 'wide'},
    {'C': -1.0},
    {'C': [1.5]},
    {'M': 0},
    {'M': [10]},
    {'lenscale_scale': 0.0},
    {'lenscale_max': -1.0},
    {'lenscale_min': 0.5, 'lenscale_max': 0.4},
  ],
)
def test_invalid_params(kwargs):
  with pytest.raises((ValueError, TypeError)):
    HSGP2D(prior_type='global', **kwargs)


# ---------------------------------------------------------------------------
# Sampling: lengthscale stays within the representable range
# ---------------------------------------------------------------------------

def _lenscale_sites(trace):
  return {k: v['value'] for k, v in trace.items() if k.startswith('gp_lenscale')}


@pytest.mark.parametrize('grid_type', ['age-age', 'diff-age'])
@pytest.mark.parametrize('prior_type', ['global', 'partial', 'full'])
def test_lenscale_samples_within_bounds(prior_type, grid_type):
  A = 15
  prior = HSGP2D(prior_type=prior_type, grid_type=grid_type, M=20)
  prior.set_age_bounds(0, A - 1)
  if prior_type != 'global':
    prior.set_event_dim(2)
    prior.set_loc(0.0)

  for seed in range(5):
    with handlers.seed(rng_seed=seed):
      tr = handlers.trace(prior.sample).get_trace()
    sites = _lenscale_sites(tr)
    assert sites
    for value in sites.values():
      value = np.asarray(value)
      assert np.all(value >= prior.lenscale_min)
      assert np.all(value <= prior.lenscale_max)

  with handlers.seed(rng_seed=0):
    f = prior.sample()
  expected = {'global': (A, A), 'partial': (2, A, A), 'full': (4, A, A)}[prior_type]
  assert f.shape == expected


# ---------------------------------------------------------------------------
# Accuracy of the defaults against the exact Matern 5/2 covariance
# ---------------------------------------------------------------------------

def _matern52(r, ell):
  z = np.sqrt(5.0) * r / ell
  return (1.0 + z + z ** 2 / 3.0) * np.exp(-z)


def _max_cov_error(prior, ell, n=300, seed=0):
  rng = np.random.default_rng(seed)
  X = prior.X[rng.choice(prior.X.shape[0], n, replace=False)]
  PHI = np.asarray(eigenfunctions(x=X, ell=prior.L, m=prior.M_))
  spd = np.asarray(
    diag_spectral_density_matern(nu=prior.nu, alpha=1.0, length=ell, ell=prior.L, m=prior.M_, dim=2)
  )
  K_hsgp = (PHI * spd) @ PHI.T
  D = np.linalg.norm(X[:, None] - X[None], axis=-1)
  return np.abs(K_hsgp - _matern52(D, ell)).max()


@pytest.mark.parametrize('grid_type', ['age-age', 'diff-age'])
def test_default_config_is_accurate_on_prior_support(grid_type):
  prior = HSGP2D(prior_type='partial', grid_type=grid_type)
  prior.set_age_bounds(0, 84)

  assert _max_cov_error(prior, prior.lenscale_min) < 0.05
  assert _max_cov_error(prior, prior.lenscale_max) < 0.05
  # Below the lower bound the approximation degrades noticeably.
  assert _max_cov_error(prior, 0.5 * prior.lenscale_min) > 0.1
