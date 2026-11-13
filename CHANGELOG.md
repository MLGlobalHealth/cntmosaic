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

### Changed (breaking)

- **`SocialMix(apply_reciprocity=...)` now defaults to `False`** (also in `SocialMix.from_dataframes`). The reciprocity adjustment is a modelling choice and is not appropriate in every analysis, so it must be requested explicitly with `apply_reciprocity=True`. The summarisers already defaulted to `False`.
- **`load_polymod_uk()` returns the data close to the source**, so that age grouping is done by the user.
  - The precomputed `part_age_grp` and `cnt_age_grp` columns are removed.
  - The gender columns are plain strings, not categorical.
  - `population` has single-year ages, `age` (0 to 90, where 90 stands for 90+), with `gender` and `P`, instead of 5-year `age_grp` bands.
  - The participants table now includes `dayofweek` and the further POLYMOD participant variables.
  - Tutorial 01 shows the preprocessing.

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

- **Stratum order.** With two or more participant stratification variables, `SocialMix` (and `Prem2`) filed point estimates under the wrong stratum labels, and they disagreed with the bootstrap. Results for two or more participant strata change.
- **Unused contact categories.** A contact stratification variable with an unused category raised an `IndexError`.
- **Population strata order.** Stratified population data is now matched to contact strata by category value, so the order of variables and categories in the population data no longer matters.
- **Columns kept on merge.** Adaptive age-group merging (at fit time and at bootstrap time) kept only the id, age group and strata, deleting `part_age`, `part_repeat` and other columns. It now keeps every column.
