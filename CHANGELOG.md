# Changelog

## 0.6.0

### Added

- **`SocialMix.prop_mix_factorisation()`**: the proportionate-mixing factorisation of socialmixr's `split_matrix()`, $m_{ij} = c\,q\,d_i\,a_{ij}\,p_j$. It returns a `PropMixFactorisation` with the assortativity matrix (contacts relative to mixing in proportion to population size), contacts per person, population shares, the mean number of contacts, the spectral radius and the normalisation. Point estimate only; needs `pop_data`; age-only fits.
- **Participant survey weights for `SocialMix`** (new subpackage `cntmosaic.weights`).
  - Weighting is opt-in: pass `weight_specs=WeightSpecs(...)` to `SocialMix` (or to `SocialMix.from_dataframes`).
  - `cint()` and `rate()` then return the weighted estimate by default; `cint(weighted=False)` still gives the unweighted one.
  - Weight factors:
    - `DesignWeights`: weights supplied with the survey;
    - `CategoryWeights`: a fixed weight per category;
    - `PostStratWeights`: match target shares;
    - `DayOfWeekWeights`: weekdays and weekend days in the ratio 5:2, matched **within each age group** by default (`within="cell"`; `within="sample"` matches it over the whole sample only, as socialmixr). An age group that lacks weekday or weekend reports cannot meet the ratio and gives a `WeightWarning`. Day codes may be single labels, e.g. `weekday="weekday", weekend="weekend"`;
    - `AgePostStratWeights`: post-stratify single-year age against a reference population.
  - Post-stratification shares are taken among participants with a known value, so a missing value (`on_missing="average"`) receives the average factor. This differs from socialmixr, which counts missing values in the denominator and so slightly downweights them.
  - Optional truncation: `WeightSpecs(..., threshold=3)`.
  - Diagnostics in `sm.weights`: per-participant factors and weights, next to the participant columns they are computed from (e.g. `part_dayofweek`, `part_age`); per-cell effective sample size and design effect.
  - `WeightSpecs.preview(part_data, age_group_specs)` shows the weights before fitting.
  - `run_inference_bootstrap()` rebuilds the weights on every resample.
  - See tutorial `05_Tutorial_SocialMix_Weights.ipynb`.
- `ParticipantData(aux_cols=...)`: auxiliary columns kept for weighting (e.g. day of week, design weights). They are renamed `part_<name>`, are not stratification variables, and keep their missing values.
- `SocialMix.from_dataframes(aux_cols=..., weight_specs=...)`.
- `WeightWarning`, a `UserWarning` subclass for weighting messages.
- `sample_hit_run()` and `check_margins()` in `cntmosaic.predict`: a hit-and-run sampler for the uniform distribution on tables with given row and column sums (optionally symmetric tables). Written in JAX; batches of margins run as independent chains.
- `sample_full_counts()` and `counts_to_intensities()` in `cntmosaic.predict`.
- `sample_swap()` in `cntmosaic.predict`: a swap-move sampler for the same uniform distribution, which changes four cells per move and runs in sweeps of `R * C` moves (default burn-in: 10 sweeps). `2 x 2` tables are drawn exactly. It also samples symmetric tables (`symmetric=True`) and tables with fixed diagonal cells (`diagonal=...`, using three-cycle moves as well). `sample_swap_path()` records the chains after given numbers of sweeps, and can continue them.
- **Assortativity constraints in prediction.** `Predictor.predict_full_matrices(assortativity={...})` fixes the fraction of each stratum's contacts that stay within the stratum, for each age pair (or one value for all). The sampler then draws uniformly from the tables with that diagonal (`diagonal=...` in `sample_swap()` and `sample_hit_run()`, `diagonal_is_feasible()`). Fractions that are infeasible for a table are moved towards those of the independence table until feasible, with a warning.

### Changed (breaking)

- **`HSGP2D` now uses a tensor-product Matérn kernel by default** (`kernel="product"`), $k(x,x') = \sigma_f^2\,k_\nu(|x_1-x_1'|/\ell_1)\,k_\nu(|x_2-x_2'|/\ell_2)$, with **one lengthscale per grid axis** (`(age_1, age_2)` or `(age_diff, age)`). Previously it used numpyro's two-dimensional spectral density, i.e. an isotropic Matérn on the standardised grid with one shared lengthscale; this is still available as `kernel="isotropic"` and reproduces earlier fits exactly.
  - The sample sites `gp_lenscale`, `gp_lenscale_diag` and `gp_lenscale_non_diag` gain a trailing axis of size 2 under the product kernel. Other site names and shapes are unchanged.
  - `lenscale_scale`, `lenscale_min` and `lenscale_max` accept one value per axis (product kernel only). The configuration rules are applied per axis: $\ell_{\min,d} = a_2 L_d / M_d$ and, with an explicit `C`, $\ell_{\max,d} = C_d S_d / a_1$.
  - The resolved `lenscale_min` and `lenscale_max` are now lists with one value per axis for both kernels (equal values for the isotropic kernel).
  - Both kernels are evaluated in Kronecker form, $\Phi_1 (\sqrt{W} \circ Z) \Phi_2^\top$, followed by a gather of the grid cells, instead of forming the $N \times M_1 M_2$ basis matrix. This is 7–10× faster per gradient evaluation for 85 ages and $M = 40$ per axis. The attributes `PHI`, `PHI_diag` and `PHI_non_diag` are replaced by `PHI_1`, `PHI_2`, `idx_all` and `idx_sym`.
  - `gp_scale` is, as before, the marginal **variance** of the GP (it is passed to numpyro as `alpha`); the docstring now says so.

- **`SocialMix(apply_reciprocity=...)` now defaults to `False`** (also in `SocialMix.from_dataframes`). The reciprocity adjustment is a modelling choice and is not appropriate in every analysis, so it must be requested explicitly with `apply_reciprocity=True`. The summarisers already defaulted to `False`.
- **`load_polymod_uk()` returns the data close to the source**, so that age grouping is done by the user.
  - The precomputed `part_age_grp` and `cnt_age_grp` columns are removed.
  - The gender columns are plain strings, not categorical.
  - `population` has single-year ages, `age` (0 to 90, where 90 stands for 90+), with `gender` and `P`, instead of 5-year `age_grp` bands.
  - The participants table now includes `dayofweek` and the further POLYMOD participant variables.
  - Tutorial 01 shows the preprocessing.
- **`Predictor` samples the fully stratified tables directly.** For each posterior draw and age pair, the table of fully stratified contact counts is drawn uniformly from all non-negative tables whose row and column sums match the partially stratified counts, instead of sampling associativity fractions from a truncated Dirichlet distribution.
  - The tables are drawn with swap moves by default; `sampler="hit_and_run"` uses the hit-and-run sampler instead.
  - `predict_full_matrices(key, num_warmup=None, sampler="swap")` and `spectral_radius(key, method=..., num_warmup=None, sampler="swap")` take a JAX PRNG key. `num_warmup` counts sweeps for swap moves (default 10) and steps for hit-and-run (default 1600). The `alpha`, `rng`, `external_eta` and `external_counts` arguments are removed.
  - `get_cache_info()` reports `z_marginals_loaded` instead of `bounds_loaded`.
  - `sample_eta()`, `rtruncated_beta()`, `rtruncated_dirichlet()` and `feasible_mixing_bounds()` are deprecated and no longer used.

### Removed (breaking)

`ParticipantData(weight_col=...)` has been removed, together with `ParticipantData.normalize_weights()`, `ParticipantData.check_weights_normalized()`, `SocialMix(normalise_weights=...)` and the automatic weighting of contact counts in `ContactSurveyLoader` (Bayesian models no longer use participant weights).

- **Why:** with the default `normalise_weights=False`, the old path gave a wrongly scaled `SocialMix` estimate, and its bootstrap ignored the weights.
- **Migration:** keep the weight column as an auxiliary column and apply it explicitly:

```python
# Before (0.5)
part = ParticipantData(df, id_col="id", age_col="age", weight_col="w")
sm = SocialMix(part, cnt, specs, pop, normalise_weights=True)

# After (0.6)
from cntmosaic.weights import WeightSpecs, DesignWeights

part = ParticipantData(df, id_col="id", age_col="age", aux_cols="w")
sm = SocialMix(part, cnt, specs, pop, weight_specs=WeightSpecs(DesignWeights("w")))
```

The tutorial figure helpers `save_tutorial_figure()`, `list_tutorial_figures()`, `convert_notebook_to_html()` and `FIGURE_DIR` have been removed from `cntmosaic.utils`. The tutorials are now part of the documentation (`docs/source/tutorials/`) and are executed when the documentation is built, so their figures no longer need to be saved separately. Importing `cntmosaic.utils` also no longer creates a `docs/tutorials/figures/` folder.

### Changed

- **Participants whose age lies outside `age_group_specs` are dropped**, with one warning giving the count. Previously they were left in the data with a missing age group: excluded from the point estimate, but drawn by the bootstrap, where they made every replicate fail ("insufficient sample sizes").
- **`SocialMix` no longer modifies the containers you pass in.** It works on copies, and always re-bins single-year ages, so refitting the same `ParticipantData` with different age groups now gives correct results.
- **The bootstrap no longer silently drops replicates on unexpected errors.** Only degenerate resamples (`ValueError`, `ZeroDivisionError`, `FloatingPointError`) count as failed iterations; any other error is raised.
- **`SocialMix.from_dataframes` warns** when a floating-point `part_*` column would become a stratification variable (pass it in `aux_cols` instead).

### Fixed

- **Demographic multipliers out of order in `GenMix*` models.** With a FULL stratification variable followed by another variable (e.g. `strat_var_cols=["gender", "ethn"]` on both sides), `StratificationData.compute_demopty` returned the multipliers in (source strata, target strata) order, while `log_delta` and the full labels interleave source and target per variable. The model therefore subtracted the wrong population offset from most cross-strata blocks, and the fitted matrices were not reciprocal. The multipliers are now built from the full labels, row by row. **Fits with two or more stratification variables must be rerun.** Single-variable fits, and fits in which only the last variable is FULL, are unaffected. See `docs/dev_docs/bug_multiplier_ordering.md`.
- **Stratum order.** With two or more participant stratification variables, `SocialMix` (and `Prem2`) filed point estimates under the wrong stratum labels, and they disagreed with the bootstrap. Results for two or more participant strata change.
- **Unused contact categories.** A contact stratification variable with an unused category raised an `IndexError`.
- **Population strata order.** Stratified population data is now matched to contact strata by category value, so the order of variables and categories in the population data no longer matters.
- **Columns kept on merge.** Adaptive age-group merging (at fit time and at bootstrap time) kept only the id, age group and strata, deleting `part_age`, `part_repeat` and other columns. It now keeps every column.
