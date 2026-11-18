import warnings

import jax.numpy as jnp
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
    hsgp_max_lenscale_per_axis,
    hsgp_min_lenscale,
    hsgp_min_lenscale_per_axis,
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
  assert prior.kernel == 'product'


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


def test_per_axis_rules():
  S, C, M = np.array([1.7, 2.4]), np.array([2.0, 1.5]), [30, 50]
  a1, a2 = kernel_constants(5 / 2)
  np.testing.assert_allclose(hsgp_max_lenscale_per_axis(C, S, 5 / 2), C * S / a1)
  np.testing.assert_allclose(hsgp_min_lenscale_per_axis(C * S, M, 5 / 2), a2 * C * S / np.array(M))
  # The shared-lengthscale rules take the binding axis.
  assert hsgp_max_lenscale(C, S, 5 / 2) == pytest.approx(np.min(C * S / a1))
  assert hsgp_min_lenscale(C * S, M, 5 / 2) == pytest.approx(np.max(a2 * C * S / np.array(M)))
  # Per-axis lenscale_min in hsgp_min_num_basis.
  assert hsgp_min_num_basis(C * S, [0.2, 0.4], 5 / 2) == [
    int(np.ceil(a2 * C[0] * S[0] / 0.2)), int(np.ceil(a2 * C[1] * S[1] / 0.4))]


def test_halfnormal_quantile():
  assert halfnormal_quantile(0.5, 0.95) == pytest.approx(0.5 * 1.959964, rel=1e-5)


# ---------------------------------------------------------------------------
# Auto configuration per grid type
# ---------------------------------------------------------------------------

KERNELS = ['product', 'isotropic']


@pytest.mark.parametrize('kernel', KERNELS)
def test_auto_config_age_age(kernel):
  prior = HSGP2D(prior_type='global', grid_type='age-age', kernel=kernel)
  prior.set_age_bounds(0, 84)

  assert prior.M_ == GRID_DEFAULTS['age-age']['M']
  assert prior.C_[0] == pytest.approx(prior.C_[1])
  np.testing.assert_allclose(prior.C_, [2.347, 2.347], atol=1e-3)
  np.testing.assert_allclose(prior.lenscale_max, [halfnormal_quantile(0.5, 0.95)] * 2)
  np.testing.assert_allclose(prior.lenscale_min, [0.266, 0.266], atol=1e-3)
  assert np.all(np.array(prior.lenscale_min) < np.array(prior.lenscale_max))
  np.testing.assert_allclose(prior.L, np.array(prior.C_) * prior.S)

  years = prior.lenscale_bounds_years
  assert set(years) == {'age_1', 'age_2'}
  np.testing.assert_allclose(years['age_1'], (6.5, 24.0), atol=0.1)


@pytest.mark.parametrize('kernel', KERNELS)
def test_auto_config_diff_age(kernel):
  prior = HSGP2D(prior_type='global', grid_type='diff-age', kernel=kernel)
  prior.set_age_bounds(0, 84)

  assert prior.M_ == GRID_DEFAULTS['diff-age']['M']
  # The age-difference axis has a larger standardised half-range, so a smaller C.
  assert prior.S[0] > prior.S[1]
  assert prior.C_[0] < prior.C_[1]
  np.testing.assert_allclose(prior.C_, [1.660, 2.347], atol=1e-3)
  np.testing.assert_allclose(prior.lenscale_min, [0.266, 0.266], atol=1e-3)
  assert np.all(np.array(prior.lenscale_min) < np.array(prior.lenscale_max))
  assert set(prior.lenscale_bounds_years) == {'age_diff', 'age'}


def test_auto_config_respects_M_and_scale():
  coarse = HSGP2D(prior_type='global', M=20)
  coarse.set_age_bounds(0, 84)
  fine = HSGP2D(prior_type='global', M=60)
  fine.set_age_bounds(0, 84)
  assert coarse.M_ == [20, 20]
  assert np.all(np.array(fine.lenscale_min) < np.array(coarse.lenscale_min))

  wide = HSGP2D(prior_type='global', lenscale_scale=1.0)
  wide.set_age_bounds(0, 84)
  np.testing.assert_allclose(wide.lenscale_max, [halfnormal_quantile(1.0, 0.95)] * 2)
  assert wide.C_[0] > coarse.C_[0]


def test_product_per_axis_settings():
  # Different scales and basis sizes per axis give different per-axis bounds.
  prior = HSGP2D(prior_type='global', grid_type='diff-age', M=[80, 30],
                 lenscale_scale=[0.3, 1.0])
  prior.set_age_bounds(0, 84)
  np.testing.assert_allclose(
    prior.lenscale_max, [halfnormal_quantile(0.3, 0.95), halfnormal_quantile(1.0, 0.95)])
  np.testing.assert_allclose(prior.C_, hsgp_boundary_factor(np.array(prior.lenscale_max), prior.S, 5 / 2))
  np.testing.assert_allclose(prior.lenscale_min, hsgp_min_lenscale_per_axis(prior.L, prior.M_, 5 / 2))
  years = prior.lenscale_bounds_years
  assert years['age_diff'][0] == pytest.approx(prior.lenscale_min[0] * prior.X_sd[0])
  assert years['age'][1] == pytest.approx(prior.lenscale_max[1] * prior.X_sd[1])


def test_product_empty_range_names_axis():
  # Too few basis functions along the age-difference axis only.
  prior = HSGP2D(prior_type='global', grid_type='diff-age', M=[5, 40])
  with pytest.raises(ValueError, match="lengthscale range is empty \\(age_diff"):
    prior.set_age_bounds(0, 84)


# ---------------------------------------------------------------------------
# Explicit overrides
# ---------------------------------------------------------------------------

def test_explicit_C_derives_lenscale_max():
  prior = HSGP2D(prior_type='global', C=[1.5, 1.5], M=[20, 20])
  prior.set_age_bounds(0, 84)
  assert prior.C_ == [1.5, 1.5]
  np.testing.assert_allclose(prior.lenscale_max, hsgp_max_lenscale_per_axis(1.5, prior.S, 5 / 2))
  np.testing.assert_allclose(prior.lenscale_min, [0.340, 0.340], atol=1e-3)
  np.testing.assert_allclose(prior.lenscale_max, [0.626, 0.626], atol=1e-3)


def test_explicit_C_isotropic_uses_binding_axis():
  prior = HSGP2D(prior_type='global', grid_type='diff-age', C=[1.5, 1.5], M=[20, 20],
                 kernel='isotropic')
  prior.set_age_bounds(0, 84)
  np.testing.assert_allclose(prior.lenscale_max, [hsgp_max_lenscale(1.5, prior.S, 5 / 2)] * 2)
  np.testing.assert_allclose(prior.lenscale_min, [hsgp_min_lenscale(prior.L, prior.M_, 5 / 2)] * 2)


def test_explicit_lenscale_max_beyond_C_warns():
  prior = HSGP2D(prior_type='global', C=1.5, lenscale_max=0.9)
  with pytest.warns(UserWarning, match='exceeds the largest lengthscale'):
    prior.set_age_bounds(0, 84)
  assert prior.lenscale_max == [0.9, 0.9]


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
  assert prior.lenscale_min == [0.1, 0.1]


def test_lenscale_min_above_derived_no_warning():
  prior = HSGP2D(prior_type='global', lenscale_min=0.4)
  with warnings.catch_warnings():
    warnings.simplefilter('error')
    prior.set_age_bounds(0, 84)
  assert prior.lenscale_min == [0.4, 0.4]


def test_nu_half_requires_explicit_settings():
  with pytest.raises(ValueError, match='nu=1/2'):
    HSGP2D(prior_type='global', nu=1 / 2)
  with pytest.raises(ValueError, match='nu=1/2'):
    HSGP2D(prior_type='global', nu=1 / 2, C=2.0)

  prior = HSGP2D(prior_type='global', nu=1 / 2, C=2.0, lenscale_min=0.3)
  prior.set_age_bounds(0, 20)
  assert prior.lenscale_min == [0.3, 0.3]
  np.testing.assert_allclose(prior.lenscale_max, [halfnormal_quantile(0.5, 0.95)] * 2)


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
    {'lenscale_min': [0.1, 0.5], 'lenscale_max': [0.4, 0.4]},
    {'lenscale_scale': [0.5]},
    {'lenscale_scale': [0.5, -1.0]},
    {'kernel': 'rbf'},
    {'kernel': 'isotropic', 'lenscale_scale': [0.5, 1.0]},
    {'kernel': 'isotropic', 'lenscale_min': [0.1, 0.2]},
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


@pytest.mark.parametrize('kernel', KERNELS)
@pytest.mark.parametrize('grid_type', ['age-age', 'diff-age'])
@pytest.mark.parametrize('prior_type', ['global', 'partial', 'full'])
def test_lenscale_samples_within_bounds(prior_type, grid_type, kernel):
  A = 15
  prior = HSGP2D(prior_type=prior_type, grid_type=grid_type, M=20, kernel=kernel)
  prior.set_age_bounds(0, A - 1)
  if prior_type != 'global':
    prior.set_event_dim(2)
    prior.set_loc(0.0)

  lmin, lmax = np.array(prior.lenscale_min), np.array(prior.lenscale_max)
  if kernel == 'isotropic':
    lmin, lmax = lmin[0], lmax[0]
  for seed in range(5):
    with handlers.seed(rng_seed=seed):
      tr = handlers.trace(prior.sample).get_trace()
    sites = _lenscale_sites(tr)
    assert sites
    for value in sites.values():
      value = np.asarray(value)
      if kernel == 'product':
        assert value.shape[-1] == 2
      assert np.all(value >= lmin)
      assert np.all(value <= lmax)

  with handlers.seed(rng_seed=0):
    f = prior.sample()
  expected = {'global': (A, A), 'partial': (2, A, A), 'full': (4, A, A)}[prior_type]
  assert f.shape == expected


def _site_shapes(prior):
  with handlers.seed(rng_seed=0):
    tr = handlers.trace(prior.sample).get_trace()
  return {k: tuple(np.shape(v['value'])) for k, v in tr.items() if k.startswith('gp_')}


def test_site_shapes_product_vs_isotropic():
  M = [12, 16]
  for kernel, ls in [('product', (2,)), ('isotropic', ())]:
    g = HSGP2D(prior_type='global', M=M, kernel=kernel)
    g.set_age_bounds(0, 14)
    assert _site_shapes(g) == {'gp_scale': (), 'gp_lenscale': ls, 'gp_coefs': (192,)}

    fu = HSGP2D(prior_type='full', M=M, kernel=kernel)
    fu.set_age_bounds(0, 14)
    fu.set_event_dim(3)
    fu.set_loc(0.0)
    shapes = _site_shapes(fu)
    assert shapes['gp_lenscale_diag'] == (3,) + ls
    assert shapes['gp_lenscale_non_diag'] == (3,) + ls
    assert shapes['gp_coefs_diag'] == (3, 192)


@pytest.mark.parametrize('grid_type', ['age-age', 'diff-age'])
def test_symmetry_and_reciprocity(grid_type):
  g = HSGP2D(prior_type='global', grid_type=grid_type, M=20)
  g.set_age_bounds(0, 14)
  with handlers.seed(rng_seed=1):
    f = np.asarray(g.sample())
  np.testing.assert_allclose(f, f.T, atol=1e-6)

  fu = HSGP2D(prior_type='full', grid_type=grid_type, M=20)
  fu.set_age_bounds(0, 14)
  fu.set_event_dim(2)
  fu.set_loc(0.0)
  with handlers.seed(rng_seed=1):
    f = np.asarray(fu.sample())
  np.testing.assert_allclose(f[0], f[0].T, atol=1e-6)    # within-group block symmetric
  np.testing.assert_allclose(f[1], f[2].T, atol=1e-6)    # reciprocity between groups


# ---------------------------------------------------------------------------
# Kronecker evaluation equals the explicit 2D basis
# ---------------------------------------------------------------------------

def _reference(prior, tr):
  """Explicit-basis evaluation (the pre-0.6 implementation) from traced values."""
  def sqrt_w(alpha, length):
    if prior.kernel == 'isotropic':
      spd = diag_spectral_density_matern(
        nu=prior.nu, alpha=alpha, length=length, ell=prior.L, m=prior.M_, dim=2)
      return np.sqrt(np.asarray(spd))
    s = [np.asarray(diag_spectral_density_matern(
      nu=prior.nu, alpha=1.0, length=length[d], ell=prior.L[d], m=prior.M_[d], dim=1))
      for d in range(2)]
    return np.sqrt(alpha * np.outer(s[0], s[1]).ravel())

  v = lambda k: np.asarray(tr[k]['value'])
  A = prior.A
  PHI_all = np.asarray(eigenfunctions(x=prior.X, ell=prior.L, m=prior.M_))
  PHI_tril = np.asarray(eigenfunctions(x=prior.X_tril, ell=prior.L, m=prior.M_))
  sym = lambda a, l, b: (PHI_tril @ (sqrt_w(a, l) * b))[prior.symm_tril_ix].reshape(A, A)
  asym = lambda a, l, b: (PHI_all @ (sqrt_w(a, l) * b)).reshape(A, A)
  if prior.prior_type == 'global':
    return sym(v('gp_scale'), v('gp_lenscale'), v('gp_coefs'))
  if prior.prior_type == 'partial':
    return np.stack([asym(*t) for t in zip(v('gp_scale'), v('gp_lenscale'), v('gp_coefs'))])
  fd = np.stack([sym(*t) for t in zip(
    v('gp_scale_diag'), v('gp_lenscale_diag'), v('gp_coefs_diag'))])
  fn = np.stack([asym(*t) for t in zip(
    v('gp_scale_non_diag'), v('gp_lenscale_non_diag'), v('gp_coefs_non_diag'))])
  return np.asarray(prior._assemble_full_prior_blocks(jnp.asarray(fd), jnp.asarray(fn)))


@pytest.mark.parametrize('kernel', KERNELS)
@pytest.mark.parametrize('grid_type', ['age-age', 'diff-age'])
@pytest.mark.parametrize('prior_type', ['global', 'partial', 'full'])
def test_kronecker_matches_explicit_basis(prior_type, grid_type, kernel):
  prior = HSGP2D(prior_type=prior_type, grid_type=grid_type, M=[18, 22], kernel=kernel)
  prior.set_age_bounds(0, 20)
  if prior_type != 'global':
    prior.set_event_dim(2)
    prior.set_loc(0.0)
  with handlers.seed(rng_seed=3):
    tr = handlers.trace(prior.sample).get_trace()
  with handlers.seed(rng_seed=3):
    f = np.asarray(prior.sample())
  np.testing.assert_allclose(f, _reference(prior, tr), atol=1e-5)


# ---------------------------------------------------------------------------
# Accuracy of the defaults against the exact Matern 5/2 covariance
# ---------------------------------------------------------------------------

def _matern52(r, ell):
  z = np.sqrt(5.0) * r / ell
  return (1.0 + z + z ** 2 / 3.0) * np.exp(-z)


def _max_cov_error(prior, ell, n=300, seed=0):
  """Max |HSGP - exact| covariance error at ``n`` random grid points, unit variance."""
  rng = np.random.default_rng(seed)
  sel = rng.choice(prior.X.shape[0], n, replace=False)
  X = prior.X[sel]
  W = np.asarray(prior._sqrt_weights(1.0, jnp.asarray(ell))) ** 2   # (M_1, M_2)
  # covariance between selected points via the per-axis bases
  idx0 = prior.idx_all[sel] // prior.PHI_2.shape[0]
  idx1 = prior.idx_all[sel] % prior.PHI_2.shape[0]
  P1, P2 = prior.PHI_1[idx0], prior.PHI_2[idx1]                      # (n, M_1), (n, M_2)
  K_hsgp = np.einsum('aj,ak,jk,bj,bk->ab', P1, P2, W, P1, P2)
  D = X[:, None] - X[None]
  if prior.kernel == 'product':
    ell = np.broadcast_to(ell, (2,))
    K = _matern52(np.abs(D[..., 0]), ell[0]) * _matern52(np.abs(D[..., 1]), ell[1])
  else:
    K = _matern52(np.linalg.norm(D, axis=-1), ell)
  return np.abs(K_hsgp - K).max()


@pytest.mark.parametrize('kernel', KERNELS)
@pytest.mark.parametrize('grid_type', ['age-age', 'diff-age'])
def test_default_config_is_accurate_on_prior_support(grid_type, kernel):
  prior = HSGP2D(prior_type='partial', grid_type=grid_type, kernel=kernel)
  prior.set_age_bounds(0, 84)
  lmin, lmax = np.array(prior.lenscale_min), np.array(prior.lenscale_max)
  if kernel == 'isotropic':
    lmin, lmax = lmin[0], lmax[0]

  assert _max_cov_error(prior, lmin) < 0.05
  assert _max_cov_error(prior, lmax) < 0.05
  # Below the lower bound the approximation degrades noticeably.
  assert _max_cov_error(prior, 0.5 * lmin) > 0.1


def test_product_accurate_with_unequal_lengthscales():
  # Short age-difference and long age lengthscales, each at its own bound. The
  # settings keep both boundary factors above the 1.2 floor: when the floor binds,
  # the rule does not guarantee this accuracy near the grid edges (either kernel).
  prior = HSGP2D(prior_type='partial', grid_type='diff-age', M=[60, 30],
                 lenscale_scale=[0.5, 1.0])
  prior.set_age_bounds(0, 84)
  for ell in [prior.lenscale_min, prior.lenscale_max,
              [prior.lenscale_min[0], prior.lenscale_max[1]]]:
    assert _max_cov_error(prior, np.array(ell)) < 0.05
