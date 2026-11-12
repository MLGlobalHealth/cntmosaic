# socialmixr: issues found during the cntmosaic port

This file records problems in socialmixr that we found while porting its weighting to cntmosaic. It is meant for passing on to the socialmixr developers (<https://github.com/epiforecasts/socialmixr>). Nothing listed here has been reported upstream yet. When an issue is filed, add its link and status.

Versions checked:

- **Development 0.7.0.9000.** Local clone `/Users/shozen/Imperial/0_Research/socialmixr`, commit `89dd471` (2026-09-28), also installed into R. It follows the 0.7.0 release (tag `v0.7.0`).

Earlier checks were against 0.6.0.9000 (commit `b0d7134`). They were re-checked against 0.7.0.9000 on 2026-11-11. An issue about the legacy day-of-week weighting of release 0.5.1 (a missing day counted as a weekend day) was removed on 2026-11-12, because that code is no longer reachable in 0.7.0.9000. The remaining issues were renumbered.

Notation: N participants, of whom M have a missing value and n = N − M a known value. Group g has target t_g and n_g participants with a known value, and T = Σ_g t_g.

| # | Issue | Version | Severity | Status |
|---|---|---|---|---|
| S1 | The "neutral average" weight for missing values is not the average | 0.6.0.9000 – 0.7.0.9000 | Medium | Not reported |
| S2 | `weigh_by_age()` silently has no effect when the population's bands are the matrix's age groups | 0.7.0 – 0.7.0.9000 | Medium | Not reported |
| S3 | Day-of-week weights match 5:2 over the whole sample, not within the age groups the matrix is estimated in | 0.5.1 – 0.7.0.9000 | High | Not reported |

---

## S1. `weigh(..., groups = )` gives missing values less than the average weight

**Version:** 0.6.0.9000 to 0.7.0.9000 (still present at `89dd471`). It affects `weigh_by_dayofweek()`, `contact_matrix(weigh_dayofweek = TRUE)` and any grouped call to `weigh()`.

**Code** (`weigh_grouped`, `R/weigh.R:362-367` at `89dd471`):

```r
n_total <- length(col_vals)
weight_factor <- ifelse(
  is.na(group_idx),
  sum(target) / n_total,
  target[group_idx] / group_counts[group_idx]
)
```

**Problem.** The documentation of `weigh_by_dayofweek()` says that participants with an `NA` day "get the neutral average `7 / N`". However, N counts the missing participants too. The known participants' factors t_g / n_g sum to T over n known participants, so their average is T / n. A missing value therefore gets n / N times the average known factor, not the average itself. The more values are missing, the more the missing participants are downweighted. This contradicts the documented intent that the weight should be neutral.

The same applies to values that match no group: they get T / N, and the documentation describes this as "an average weight".

**Size on POLYMOD UK.** Verified in R with 0.7.0.9000 on `polymod[country == "United Kingdom"]` (710 weekday, 258 weekend, 44 missing):

| Group | n | Weight |
|---|---|---|
| Weekday | 710 | 0.0070423 (5/710) |
| Weekend | 258 | 0.0077519 (2/258) |
| Missing | 44 | 0.0069170 (7/1012) |
| Mean of known | 968 | 0.0072314 (7/968) |

The missing participants get 968/1012 = 0.957 of the average known factor.

**Suggested fix.** Use `sum(target) / sum(!is.na(group_idx))`, i.e. divide T by the number of participants with a matched value. This makes the factor for missing values equal to the mean known factor, as the documentation states.

**cntmosaic behaviour.** f_i = (t_g / T) / (n_g / n) with n counting known values only. The known factors therefore average exactly 1, and a missing value gets 1 (`on_missing="average"`). See `cntmosaic/weights/_poststrat.py` and decision D31 in `dev_plan_weights.md`.

---

## S2. `weigh_by_age()` silently has no effect when the population's bands are the matrix's age groups

**Version:** 0.7.0 to 0.7.0.9000 (still present at `89dd471`). Up to 0.6.0.9000, `weigh_by_age()` interpolated the population to single years, so the issue did not arise. It affects the pipeline `weigh_by_age(survey, pop)`. It does not affect `contact_matrix(weigh_age = TRUE)`, which requires a single-year population (see its `survey_pop` documentation).

**Code** (`weight_by_age`, `R/contact-matrix-utils.R:759` at `89dd471`):

```r
ref_limits <- sort(survey_pop_full$lower.age.limit)
participants[, age.band := reduce_age_groups(part_age, ref_limits)]
...
participants[weighted, weight := weight * population.proportion / age.proportion]
```

**Problem.** Since 0.7, `weigh_by_age()` post-stratifies at the bands of the population it is given, with no interpolation. The documentation presents this as a feature: participants are binned into the population's bands, "whatever resolution `pop` is supplied at". But the contact matrix is a ratio estimator within each participant age group. A factor that is constant within every age group therefore cancels between the numerator and the denominator. When `pop` is given in the matrix's own age groups (e.g. 5-year bands from WPP with 5-year `age_limits`), or in coarser bands, every participant in an age group gets the same factor. The weighting then has no effect on the matrix, and there is no warning. Only bands finer than the matrix age groups change anything.

**Reproduction** (R, 0.7.0.9000, POLYMOD UK, `age_limits = seq(0, 75, 5)`):

```r
uk   <- polymod[country == "United Kingdom"] |> assign_age_groups(age_limits = seq(0, 75, 5))
pop5 <- data.frame(age = limits_to_age_groups(seq(0, 75, 5), notation = "brackets"),
                   population = seq(4e6, 1e6, length.out = 16))
pop1 <- data.frame(age = limits_to_age_groups(0:99, notation = "brackets"),
                   population = round(seq(8e5, 2e5, length.out = 100)))
m0 <- compute_matrix(uk)$matrix
max(abs(compute_matrix(weigh_by_age(uk, pop5))$matrix - m0))  # 8.9e-16: no effect
max(abs(compute_matrix(weigh_by_age(uk, pop1))$matrix - m0))  # 0.32
```

Within each age group, the standard deviation of the weight under `pop5` is exactly 0.

**Suggested fix.** Warn (or error) when the reference bands are not finer than the participant age groups, since the weighting then cannot change the matrix. Alternatively, document prominently that `weigh_by_age()` only has an effect with a population finer than `age_limits`, ideally in single years.

**cntmosaic behaviour.** `AgePostStratWeights` always weights by single-year age, spreading a banded population uniformly over its single years. Any factor that is constant within every cell raises a `WeightWarning` ("… is constant within every cell …, so it has no effect on the contact matrix").

---

## S3. Day-of-week weights match 5:2 over the whole sample, not within the age groups the matrix is estimated in

**Version:** all checked versions, 0.5.1 to 0.7.0.9000 (still present at `89dd471`). In 0.5.1 the counts came from `by = (dayofweek %in% 1:5)` over the whole sample. It affects `weigh_by_dayofweek()` and `contact_matrix(weigh_dayofweek = TRUE)`. The same applies to any grouped `weigh()` whose targets are meant to hold within each age group.

**Code** (`weigh_by_dayofweek`, `R/weigh.R`, which calls `weigh_grouped`):

```r
weigh(survey, "dayofweek", target = c(5, 2), groups = list(1:5, c(0, 6)))
# weigh_grouped: group_counts <- tabulate(group_idx, ...)  # counts over the whole sample
```

**Problem.** The weekday and weekend counts n_wd and n_we are taken over the whole sample. The weights therefore give weekday and weekend reports the ratio 5:2 across all participants. But the contact matrix is a weighted mean computed separately in each participant age group, and the weights are normalised within each age group (`normalise_weights(participants, by = "age.group")`). What matters for row c of the matrix is the weighted weekday share *within* age group c:

λ_c = n_{c,wd} · 5/n_wd / (n_{c,wd} · 5/n_wd + n_{c,we} · 2/n_we)

This equals 5/7 only when age group c has the same weekday-to-weekend ratio as the whole sample. Otherwise, every age group's weekday share is shifted by the same odds ratio, (5/2)·(n_we/n_wd). An age group whose own day mix already deviates from 5:2 in the same direction as that shift is pushed further from 5:2, so its estimate moves away from the weekly average it is meant to estimate.

**Size on POLYMOD UK.** Verified in R with 0.7.0.9000 on `polymod[country == "United Kingdom"]`, `age_limits = seq(0, 75, 5)`, among participants with a known day. The whole sample over-represents weekdays, so every age group's share is shifted down, and the overall share is exactly 5/7 = 0.714:

| Age group | n | Weekday share, unweighted | Weekday share, weighted |
|---|---|---|---|
| [0,5) | 89 | 0.674 | **0.653** |
| [10,15) | 97 | 0.680 | **0.659** |
| [20,25) | 54 | 0.648 | **0.626** |
| [50,55) | 63 | 0.698 | **0.678** |
| [5,10) | 99 | 0.778 | 0.761 |
| [35,40) | 68 | 0.809 | 0.794 |

In the four age groups that already had too few weekday reports, the weighting moves the share *further* from 5/7. As a result, the total contact intensity of the under-5s falls from 8.87 to 8.79 with weighting. A weekly average should raise it, since young children have more contacts on weekdays.

**Suggested fix.** Take the counts within each participant age group (and stratum) when the weights are used for an age-structured matrix, so that 5:2 holds in every row. At least, offer this as an option and document that the current behaviour matches 5:2 only across the whole sample. Where an age group lacks one day type, the ratio cannot be met there, and a warning would help.

**cntmosaic behaviour.** `DayOfWeekWeights` defaults to `within="cell"`: counts are taken within each age group × participant stratum, so the weighted weekday share is exactly 5/7 in every cell (warning when a cell lacks a day type). `within="sample"` reproduces the sample-wide matching. See tutorial 05, Section 4.
