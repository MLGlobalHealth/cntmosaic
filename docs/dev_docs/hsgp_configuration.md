# Dev Docs - HSGP2D configuration

This note explains how `HSGP2D` chooses its approximation settings: the boundary factor $C$, the number of basis functions $M$, and the bounds of the lengthscale prior. It also lists the sources these rules come from. The code is in `cntmosaic/models/numpyro/priors/_hsgp_config.py` (the rules) and `_HSGP2D.py` (`_configure_hsgp`).

## Problem

A Hilbert space approximate GP (HSGP) represents a GP on $[-L, L]$ with $M$ Laplacian eigenfunctions. Each setting limits which lengthscales $\ell$ the approximation reproduces accurately:

- The boundary $L = C \cdot S$ must be wide enough for the **largest** lengthscale. Here $S$ is the half-range of the centred inputs.
- $M$ must be large enough for the **smallest** lengthscale.

Before this change `HSGP2D` used fixed values, `C=[2.0, 2.0]` and `M=[30, 30]`, with an untruncated `HalfNormal(1)` lengthscale prior. With these values about 50% of the prior mass lay below the smallest representable lengthscale, and the upper tail needed $C \approx 5$. The approximation therefore silently changed the prior.

## Rule

We use the rule of Riutort-Mayol et al. (2023, Sec. 4.3.1). PyMC (`approx_hsgp_hyperparams`) and EpiNow2 (`gp_opts()`) use the same rule. For a kernel with constants $(a_1, a_2)$, on each input axis $d$:

$$
C_d \ge \max\left(a_1 \frac{\ell}{S_d},\ 1.2\right), \qquad M_d \ge a_2 \frac{C_d S_d}{\ell} = a_2 \frac{L_d}{\ell}.
$$

| Kernel | $a_1$ | $a_2$ | Source |
|---|---|---|---|
| Squared exponential | 3.2 | 1.75 | Riutort-Mayol et al. (2023) |
| Matérn 3/2 | 4.5 | 3.42 | Riutort-Mayol et al. (2023) |
| Matérn 5/2 | 4.1 | 2.65 | Riutort-Mayol et al. (2023) |
| Matérn 7/2 | 4.1 | 2.65 | Reuses Matérn 5/2. This is conservative, because smoother kernels need fewer basis functions. |
| Matérn 1/2 | none | none | No published constants. |

The paper derived these constants for one-dimensional GPs. For more than one dimension it recommends applying them to each axis separately, which is what we do.

## How `HSGP2D` applies the rule

The grid is standardised per axis, so $S_d = \max_i |x_{id}|$ on that scale. One standardised unit is one standard deviation of the grid axis.

### Kernels

`HSGP2D(kernel=...)` selects the covariance structure. Both kernels use the same basis functions, products of 1D Laplacian eigenfunctions $\phi_j(x_1)\phi_k(x_2)$. They differ only in the spectral weight $W_{jk}$ on each basis function.

- **`kernel="product"` (default).** $k(x, x') = \sigma_f^2\, k_\nu(|x_1 - x_1'|/\ell_1)\, k_\nu(|x_2 - x_2'|/\ell_2)$, with one lengthscale per axis. The weights are the outer product of the two 1D Matérn spectral densities, $W_{jk} = \sigma_f^2 S_{\nu,1}(\sqrt{\lambda_j}; \ell_1)\, S_{\nu,1}(\sqrt{\lambda_k}; \ell_2)$, so the Gram matrix on a product grid is $K_1 \otimes K_2$. On the diff-age grid, $\ell_{\text{age}} > \ell_{\text{age diff}}$ smooths along cohorts while keeping the assortative diagonal sharp.
- **`kernel="isotropic"`.** $k(x, x') = \sigma_f^2\, k_\nu(\lVert x - x' \rVert/\ell)$, with one shared lengthscale. The weights are the two-dimensional Matérn spectral density evaluated at $(\sqrt{\lambda_j}, \sqrt{\lambda_k})$. This is what `numpyro.contrib.hsgp` implements and what cntmosaic used up to 0.5.0. Its weights do not factorise for any Matérn $\nu$, so it is **not** a product kernel; only the squared exponential is both.

Both kernels are evaluated in Kronecker form: $F = \Phi_1 (\sqrt{W} \circ Z)\, \Phi_2^\top$ on the rectangle spanned by the unique values of each axis, followed by a gather of the required cells. Both grids are subsets of such a rectangle. The diff-age grid's $A^2$ points lie in the $(2A-1) \times A$ rectangle.

`gp_scale` is the marginal variance $\sigma_f^2$ (it is passed to numpyro as `alpha`).

### Bounds

The 1D rules apply directly to each factor of the product kernel. The isotropic kernel's shared lengthscale must satisfy both axes at once, so the binding axis decides.

| | `kernel="product"` (per axis $d$) | `kernel="isotropic"` (shared) |
|---|---|---|
| Upper bound, `C='auto'` | $\ell_{\max,d}$ = 95% quantile of HalfNormal(`lenscale_scale`$_d$); $C_d = \max(a_1 \ell_{\max,d}/S_d, 1.2)$ | $\ell_{\max}$ = 95% quantile; $C_d = \max(a_1 \ell_{\max}/S_d, 1.2)$ |
| Upper bound, explicit `C` | $\ell_{\max,d} = C_d S_d / a_1$ | $\ell_{\max} = \min_d C_d S_d / a_1$ |
| Lower bound | $\ell_{\min,d} = a_2 L_d / M_d$ | $\ell_{\min} = a_2 \max_d (L_d / M_d)$ |

1. **User overrides.** `lenscale_scale`, `lenscale_min` and `lenscale_max` may be one value per axis under the product kernel. `HSGP2D` warns if a user `lenscale_max` exceeds what `C` supports, or if a user `lenscale_min` is below what `M` can represent; the product-kernel messages name the axis.
2. **Prior.** $\ell \sim \text{TruncatedNormal}(0, \sigma, \text{low}=\ell_{\min}, \text{high}=\ell_{\max})$, per axis for the product kernel. This is a half-normal with **hard** bounds, so every lengthscale the prior allows is approximated accurately. The truncation is two-sided so that $C$ can be computed from $\ell_{\max}$ directly, with no circular dependence between $C$, $\ell_{\min}$ and the prior's quantiles.
3. **Errors.** `HSGP2D` raises if $\ell_{\min} \ge \ell_{\max}$ (on any axis), and the message gives the $M$ needed. It warns if any $C_d < 1.2$, as PyMC does. For $\nu = 1/2$ it requires an explicit `C` and `lenscale_min`.

The resolved values are available after `set_age_bounds` as `C_`, `M_`, `L`, `lenscale_min` and `lenscale_max` (lists with one value per axis; equal values for the isotropic kernel). `lenscale_bounds_years` gives the bounds in years along each axis.

### Range of representable lengthscales

When $C$ is above its 1.2 floor, the upper and lower bounds give $\ell_{\max} / \ell_{\min} = M / (a_1 a_2)$. For Matérn 5/2 this is about $M / 10.9$. **Only $M$ controls how wide a range of lengthscales the approximation covers.** `lenscale_scale` moves the range up or down.

### Grid types

`GRID_DEFAULTS` in `_hsgp_config.py` holds the per-grid defaults; this is where the age-age and diff-age defaults can diverge. Both grids currently use $M = 40$ per axis. They already get different $C$ per axis, because their axes have different standardised half-ranges:

- **age-age:** both axes are ages, with $S \approx \sqrt{3} \approx 1.71$.
- **diff-age:** the age-difference axis has a triangular distribution with $S \approx \sqrt{6} \approx 2.42$, so it needs a smaller $C$. The age axis has $S \approx 1.71$.

## Defaults and worked numbers

Defaults: $\nu = 5/2$, `lenscale_scale` $= 0.5$, $M = 40$ per axis, which gives 1600 basis functions. The numbers below are for ages 0–84, computed with `_configure_hsgp`. With the defaults, the per-axis bounds of the product kernel coincide with the shared bounds of the isotropic kernel. Under `C='auto'` every axis gets $L_d = a_1 \ell_{\max}$, so $\ell_{\min,d} = a_2 L_d / M_d$ is the same on both axes.

| Grid | $C$ | $\ell$ range (standardised) | $\ell$ range (years) |
|---|---|---|---|
| age-age | [2.35, 2.35] | [0.266, 0.980] | age: 6.5–24.0 |
| diff-age | [1.66, 2.35] | [0.266, 0.980] | age difference: 9.2–34.0; age: 6.5–24.0 |

Other choices considered for ages 0–84 on the age-age grid:

| $\sigma$ | $M$ | $C$ | $\ell_{\min}$ (years) | $\ell_{\max}$ (years) | Basis functions |
|---|---|---|---|---|---|
| 0.25 | 30 | 1.2 | 4.5 | 12 | 900 |
| 0.5 | 30 | 2.35 | 8.7 | 24 | 900 |
| **0.5** | **40** | **2.35** | **6.5** | **24** | **1600** |
| 1.0 | 50 | 4.7 | 10.4 | 48 | 2500 |

### Accuracy check

`test_default_config_is_accurate_on_prior_support` compares the HSGP covariance $\Phi\,\mathrm{diag}(W)\,\Phi^\top$ with the exact Matérn 5/2 covariance, isotropic or product according to `kernel`, on 300 random grid points, for ages 0–84. Both kernels stay below 0.05 at both bounds. With unit variance, the maximum absolute error for the isotropic kernel is:

| Lengthscale | age-age | diff-age |
|---|---|---|
| $\ell_{\min}$ | 0.017 | 0.017 |
| $\ell_{\max}$ | 0.002 | 0.013 |
| $\ell_{\min} / 2$ | 0.165 | 0.165 |

So the lower bound matters: below it, the approximation underestimates the prior variance by about 16%.

**Caveat at the 1.2 floor.** When the floor binds ($a_1 \ell_{\max} / S_d < 1.2$), the rule does not guarantee this accuracy near the grid edges, for either kernel. Example: diff-age, `lenscale_scale=0.3`, $M = 80$. The age-difference axis gets $C = 1.2$, and the maximum covariance error at $\ell_{\max}$ is 0.16 (isotropic) or 0.23 (product, that axis alone). `test_product_accurate_with_unequal_lengthscales` therefore uses settings where the floor does not bind.

## Using non-default settings

For a sensitivity analysis, you can override any of the derived settings:

```python
HSGP2D(prior_type="global", M=60)                         # finer basis: lower lenscale_min
HSGP2D(prior_type="global", lenscale_scale=1.0)           # longer lengthscales: larger C
HSGP2D(prior_type="global", C=[1.5, 1.5], M=[20, 20])     # explicit C: lenscale_max derived from C
HSGP2D(prior_type="global", lenscale_min=0.4, lenscale_max=0.8)  # explicit prior bounds
HSGP2D(prior_type="global", grid_type="diff-age", M=[80, 40])     # finer basis on the age-difference axis
HSGP2D(prior_type="global", lenscale_scale=[0.5, 1.0])           # per-axis lengthscale scales (product kernel)
HSGP2D(prior_type="global", kernel="isotropic")                  # pre-0.6 isotropic kernel
```

## Next steps

- **Boundary conditions.** numpyro's eigenfunctions use Dirichlet conditions, which force the prior variance to zero at $\pm L$. Raising $M$ does not fix this; only a wider domain does. Options:
  - Neumann or mixed conditions, which PyMC's `HSGP(boundary=...)` offers;
  - Dirichlet–Neumann averaging (Kutri & Scheichl, 2024), which cancels the boundary terms.

  This is out of scope for this iteration.
- **Per-axis $M$ for diff-age.** With the default $M = [40, 40]$, the age-difference axis resolves lengthscales down to about 9 years and the age axis down to about 6.5 years. The product kernel has per-axis lengthscales, so a sharper assortative diagonal only needs a larger $M$ on the age-difference axis (e.g. `M=[80, 40]`). Changing `GRID_DEFAULTS['diff-age']` is open.
- **Floor of $C$.** See the caveat above: when the 1.2 floor binds, the maximum covariance error near the edges exceeds 0.05.
- **Post-fit check.** Riutort-Mayol et al. (Sec. 4.5) refit with larger $M$ and $C$ when the posterior lengthscale reaches the representable limit. Under hard truncation the equivalent signal is posterior mass piling up at $\ell_{\min}$ or $\ell_{\max}$. Not implemented.

## References

- Riutort-Mayol, G., Bürkner, P.-C., Andersen, M. R., Solin, A., & Vehtari, A. (2023). Practical Hilbert space approximate Bayesian Gaussian processes for probabilistic programming. *Statistics and Computing*, 33, 17. <https://link.springer.com/article/10.1007/s11222-022-10167-2> (arXiv: <https://arxiv.org/abs/2004.11408>). Sec. 4.3.1 gives the rules; Sec. 4.5 gives the diagnostics.
- Solin, A., & Särkkä, S. (2020). Hilbert space methods for reduced-rank Gaussian process regression. *Statistics and Computing*, 30, 419–446.
- PyMC, `pymc.gp.hsgp_approx.approx_hsgp_hyperparams` (source): <https://www.pymc.io/projects/docs/en/latest/_modules/pymc/gp/hsgp_approx.html>. Notebooks:
  - HSGP-Basic: <https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Basic.html>
  - HSGP-Advanced: <https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Advanced.html>
  - HSGP-Boundary-Conditions: <https://www.pymc.io/projects/examples/en/latest/gaussian_processes/HSGP-Boundary-Conditions.html>
- EpiNow2:
  - Issue #1557, "Choose HSGP boundary factor and number of basis functions from the lengthscale": <https://github.com/epiforecasts/EpiNow2/issues/1557>.
  - PR #1562: <https://github.com/epiforecasts/EpiNow2/pull/1562>. It sets $C$ from the 95% and $M$ from the 5% quantile of the lengthscale prior, and reports a maximum covariance error below 0.05 over the prior's 5–95% range.
- brms, `gp()` documentation: <https://www.rdocumentation.org/packages/brms/versions/2.22.0/topics/gp>. It uses default $c = 5/4$ and gives no rule for $k$.
- Zhu, H., & Li, C. (2026). Fast and provably accurate sequential designs using Hilbert space Gaussian processes. arXiv:2604.20414. Gives error bounds that split into a boundary term (exponential in $L$) and a truncation term (Matérn: $(L/m)^{2\nu}$).
- Kutri, R., & Scheichl, R. (2024). Dirichlet–Neumann averaging: the DNA of efficient Gaussian process simulation. arXiv:2412.07929.
- Hilbert space methods for approximating multi-output latent variable Gaussian processes (2025). arXiv:2505.16919. Derives the minimum $M$ from the lengthscale prior and reports bias and poor calibration when $M$ is too small.
- Van Soom, M., & De Boi, I. On basis function selection for sparse Gaussian process regression. arXiv:2609.26624. Selects basis functions instead of keeping the full tensor product; not reviewed in full.
