# Review: Tutorial 05 — Survey Weights in SocialMix

Review date: 2026-11-11
Reviewed file: `tutorials/05_Tutorial_SocialMix_Weights.ipynb` (branch `socialmix-survey-weights-ext`)

Three reviewers each read the tutorial from a different standpoint and answered the same five questions:

1. Is it easy to understand at first sight?
2. Could you use the package in your own work after reading it?
3. What remains unclear (concepts, interface usage)?
4. What details need refining? Is the terminology explained? Are there gaps?
5. Do the formatting and sectioning make it easy to read?

The reviewers:

- **Infectious disease / public health researcher.** An SEIR modeller who uses POLYMOD and CoMix through R's `socialmixr`.
- **Applied statistician.** Specialises in survey methodology. Checked the text against the implementation.
- **First-year undergraduate.** Has done intro Python and school statistics. Read only the tutorials, not the source.

Cell numbers refer to the notebook's cell indices, counting from 0.

**Ratings (1–5)**

| Reviewer | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| Epidemiologist | 3 | 3 | 3 | 4 | 3 |
| Statistician | 4 | 3 | 3 | 3 | 3 |
| Undergraduate | 2 | 2 | 2 | 2 | 3 |

---

## 1. Issues raised by at least two reviewers

| Issue | Raised by | Cells |
|---|---|---|
| Plot axis labels differ between plots ("Respondent age group" / "Age of contacting individual" / "Age of respondent"). The notation says c is a row, but participants are on the x-axis. | All three | 5, 13, 20, 36 |
| "a naive day-of-week weights can underestimate contact intensity in children" has a grammar slip and makes the method sound wrong. The actual point is choosing the wrong estimand. | Epi, Stat | 4 |
| The notation table comes too early and is incomplete: it omits τ, n_eff, deff and λ, and uses "factor" for two different things. | All three | 5 |
| "Cell" is not defined until cell 50, where it turns out to be age group × participant stratum. The "All, …" label prefix is never explained. | All three | 16, 28, 50 |
| Column naming is inconsistent. The tutorial uses `PostStratWeights("gender")` while the column is `part_gender`, and the factor columns are `dayofweek`, `age` and `poststrat:gender`. This contradicts "each factor column is named after its factor". | Epi, UG | 16, 29, 38 |
| Pandas plumbing makes the API look harder to use than it is. | All three | 17, 25, 34, 38, 41 |
| Section 3 (cells 14–26) is too long, and the derivation in cells 24–26 loses readers. | All three | 14–26 |
| Equation (1) appears twice, and Section 4 ("How does weighting work?") comes after the reader has already used it. | Stat, Epi | 18, 27 |
| The bootstrap silently changes the point estimate by merging 70–79, and the tutorial gives no advice on which estimate to report. | All three | 52–53 |
| Leftover `UserWarning` output looks like an error to a beginner, and suppressing it elsewhere is inconsistent. | Epi, UG | 11, 20, 22 |

## 2. Correctness issues

These come from the statistician. The epidemiologist found #2 independently. Items marked **verified** were checked against the code after the review.

1. **The target population in the sex example doesn't match the sample (cells 38–39). Verified.**
   - `sex_targets` sums `Male`/`Female` over all ages in `UK_population_2011.csv`, but the participants are aged 0–79 only (`AgeGroupSpecs(0, 79, 5)`).
   - The reviewer computed the female share for ages 0–79 as 50.3%, against the 50.9% used. Women should therefore be downweighted more than the tutorial shows.
2. **The socialmixr equivalence is overstated (cells 24 and 58). Verified.**
   - The docstring at `cntmosaic/weights/_poststrat.py:17-19` says that socialmixr counts missing values in n, which downweights them, while cntmosaic gives them factor 1.
   - POLYMOD UK has 43 missing days, and the tutorial doesn't mention the difference.
3. **The `run_inference_bootstrap` docstring contradicts the code. Verified; this is a docstring bug, not a tutorial bug.**
   - `cntmosaic/models/classical/_SocialMix.py:819-821` says small age groups are merged only when `adaptive_merge=True` and that a `ValueError` is raised otherwise.
   - The stability merge actually runs regardless (`cntmosaic/models/classical/_socialmix_validation.py:517-523`). That is what happens in cell 53 with the default `adaptive_merge=False`.
4. **The "total weight of a category = target share" column in cell 28** holds only for the composite weights before the per-cell renormalisation (`cntmosaic/weights/_specs.py:362-368`). In the sex example the female share after the final weights is 0.50865, against a target of 0.50862.
5. **Cell 14, "a missing day receives the average factor".** With `within="cell"` this is the cell average. With `within="sample"` it is the sample average, not the average of the participant's own cell.
6. **Overclaims:**
   - Cell 56 says the weights "barely change the intervals", but [70,80) widens from 0.561 to 0.628 (+12%).
   - "Design effects indicate where precision is lost" isn't borne out by the output. [45,50) has deff 1.19 and widens by 12%, while [30,35) has deff 1.34 and doesn't widen. Differences of about 1% are within Monte Carlo error at 500 replicates.
   - In cell 23, "the weighted weekly average lies between…" holds only for participants whose day is known.
   - In cell 27, "Population sizes do not enter m_cd" is contradicted by `AgePostStratWeights`, which uses P_a.
   - In cell 23, the R0 claim is not demonstrated. R0 depends on the dominant eigenvalue of the whole matrix (epidemiologist).

### Resolution log

| # | Issue | Decision | Status |
|---|---|---|---|
| C1 | Sex example targets cover all ages, but the sample covers 0–79 (cells 38–39) | Restrict the census to ages 0–79 in cell 38. Add a sentence to cell 37 saying that targets should describe the population the sample represents. Update the numbers in cell 39 (target 50.3%; the whole census would give 50.9%). New factors: F 0.971, M 1.031. | Done. Cells 0–38 re-executed. |
| C2 | socialmixr equivalence overstated for missing days (cells 24, 58) | Not a tutorial issue: socialmixr's missing-day handling is faulty and cntmosaic's is correct. No caveat, no recipe for matching socialmixr, and the cell 58 comment stays. Fixed the module docstring of `_poststrat.py` so it describes both socialmixr versions precisely. Recorded the faults in `socialmixr_issues.md` (as S1 and S2 at the time; see the renumbering note below), to pass to the socialmixr developers. | Done |
| — | Update socialmixr before continuing | Pulled the local clone to `89dd471` (0.7.0.9000, after the v0.7.0 release) and installed it into R. S2 confirmed in R on this version. Updated `socialmixr_issues.md` and the `_poststrat.py` docstring to match. | Done |
| C3 | The `run_inference_bootstrap` docstring says small groups are merged only with `adaptive_merge=True` | Option (a): make the docs match the code. Rewrote the `run_inference_bootstrap` docstring and its Raises section, the `adaptive_merge` parameter doc of `SocialMix`, and the strategy in `_validate_bootstrap_requirements`. They now describe two stages: empty groups (governed by the flag) and small groups (merged whenever Σ exp(−n_{s,c}) ≥ 5×10⁻⁵, whatever the flag). No code change. | Done |
| C4 | Cell 28's "Total weight of a category" holds for the factors, not the final weights | Renamed the column to "Total factor of a category". No explanatory sentence added. | Done |
| C5 | Cell 14's "missing day receives the average factor": cell vs sample average | No change. Cell 14 describes `within="cell"` only, and cell 24 says "factor of 1" without claiming it is the cell average. | No change |
| C6a | Overclaims in the bootstrap comparison (cells 54, 56) | Cell 54: "Weighting trades bias for variance" became "Weights usually make an estimate less precise." Cell 56 rewritten: the day weights leave the average width unchanged but move individual groups; the design effect is a guide, not a prediction ([30, 35) vs [45, 50)); a Monte Carlo caveat added, with no numerical backing requested. | Done |
| C6b | Cell 23: "In every age group, the weighted weekly average lies between…" | Verified false for [25, 30) (12.14 vs 11.81/11.50) and [50, 55) (9.61 vs 9.95/9.63), because of the missing-day reports. The sentence now says the average combines weekday and weekend reports 5:2 together with the missing-day reports, and therefore usually lies between the two, as in the age groups shown. | Done |
| C6c | Cell 27, consequence 3: "Population sizes do not enter m_cd" | Now refers to the population data given to `SocialMix`, and adds that a population can still enter through a weight factor (e.g. `AgePostStratWeights`), but only through the relative weights within each age group. | Done |
| C6d | Cell 23's R₀ claim is not demonstrated | No change. "May" already softens it enough. | No change |
| X1 | Inconsistent axis labels; notation says c is a row but the plots put participants on the x-axis (cells 5, 13, 20, 36) | Every panel in cells 13, 20 and 36 now uses `xlabel="Age of respondent"` and `ylabel="Age of contact"`. No clarification added to the notation and no "how to read" sentence: both belong in earlier material (e.g. tutorial 02), not here. | Done; notebook re-executed |
| X2 | Cell 4 frames the weights as faulty ("a naive day-of-week weights can underestimate…") | Rewritten: "weighting changes what is being estimated". Day-of-week weights target the weekly average, which understates children's contact intensity during the school term, and so may underestimate R₀ and final size for school-term transmission. | Done |
| X3 | The notation table (cell 5) is intimidating and incomplete | Option (c): removed the table. Added inline definitions where symbols are first used: 𝒩_c and the index i (cell 14), y_id and m_cd (cell 18), n_c = \|𝒩_c\| (cell 27), N (cell 33). Every other symbol was already defined inline. **This shifts every later cell index down by one; this log and the reviews keep the original numbering.** | Done |
| X4 | "Cell" undefined until cell 50; "All, " prefix unexplained | Cell 16's `cell` bullet now defines a cell as age group × participant strata. It notes that this fit has none, so cells are labelled `All` plus the age group and are the age groups throughout the tutorial. | Done |
| X5 | Column naming looks inconsistent (`aux_cols` names, `"gender"` vs `part_gender`, factor column names) | Cell 10: explained that `aux_cols` takes DataFrame names and stores them with the `part_` prefix, and that factors accept names with or without it. Cell 16: the factor column is labelled by the factor's `name` (`dayofweek`/`age` for the presets, `<kind>:<col>` for the others, `name=` overrides). Cell 29 item 1 unchanged. | Done |
| X6 | Pandas plumbing hides the API (cells 17, 25, 34, 38, 41) | **Package change.** `WeightResult.participants` now carries each factor's source column under its `part_*` name, once and in factor order, between `cell` and the factor columns. Implementation: new `WeightFactor.source_columns()`, which defaults to `required_columns()`; `AgePostStratWeights` returns `part_age`; dtypes are preserved; a factor name that clashes with a source column raises. Tests updated or added in `weights/tests/test_specs.py` and `models/tests/test_SocialMix_weights.py`, and those pass. Tutorial: the joins were removed from cells 34, 38 and 41; cells 17 and 25 got a `# Display only:` comment; cell 16 documents the `part_dayofweek` column. Notebook re-executed. | Done |
| X7 | Section 3 too long; the `within=` derivation (cells 24–26) loses readers | Section 3 split. "3. Day-of-week weights" keeps cells 14–18. "4. Matching 5:2 within age groups or across the sample?" opens with an "In short" takeaway (use the default `within="cell"`; consider `"sample"` only for very small day-type counts). "Should we weight or stratify?" moved to **Section 10**, after the bootstrap. Its first sentence now refers to "the matrix weighted by day of week (Section 3)", and `sm_dow`/`M_dow` moved into the `within=` code cell. Later sections renumbered (5–9, 11, 12), with the contents list and all cross-references updated. Cell 26's first paragraph now leads with the conclusion and follows the under-5s only. Notebook re-executed; the quoted numbers are unchanged. | Done |
| X8 | Equation (1) repeated; "How does weighting work?" comes after the worked example | The repetition is intentional: it helps non-statistical readers. The display stays, and step 3 now opens with "Recall the weighted mean (1)…". Section order unchanged (concrete example first). | Done |
| X9 | Bootstrap: which estimate to report; merge rule; resampling scheme and interval type; replicate weights | Section 9, paragraph 1: resampling is over the whole sample, not within age groups, so n_c varies; the intervals are percentile intervals (2.5%/97.5%). Paragraph 2: the merge happens when a resample could contain none of a group's participants (exact rule in the `run_inference_bootstrap` docstring); the merge persists, so take the point estimate after bootstrapping and report the merged groups; to control grouping, pre-merge with `AgeGroupSpecs(0, 79, cuts=list(range(5, 75, 5)))`. Limitation 3: replicate weights supplied with a survey cannot be used. | Done |
| X10 | `UserWarning` output looks like errors (cells 11, 20, 22) | `subset()` now silences the routine data notes, as `fit()` does. The data-loading cell keeps its two notes visible, with a sentence before it explaining them (`part_gender` stored as categorical; 3 contacts without an age dropped). The bootstrap `WeightWarning` stays (the text refers to it). Also fixed a pandas `FutureWarning` introduced by X6 (`groupby(..., observed=True)` in the sex example). Notebook re-executed: only those two intended warning outputs remain. | Done |
| S1 | Estimand, estimator name and assumptions never stated | Added a "What the weights estimate" paragraph at the end of Section 5. It covers the estimand (an average over the target composition); names the estimator (Hájek / post-stratified); states that weekend reporters in c are assumed to have representative weekend contact behaviour in c; and says that missing values get the average factor, so their own composition is not corrected. No MCAR/MAR jargon. | Done |
| E1 | No socialmixr mapping beyond `weigh.dow` | Added a "Coming from socialmixr" table to Section 12 for socialmixr 0.7 (`weigh_by_dayofweek`, `weigh_by_age`/`weigh_age`, `weigh()` with each target form, `weight_threshold`), with no missing-day caveats. One sentence notes that `AgePostStratWeights` always weights by single-year age, while `weigh_by_age()` weights at the population's bands and has no effect when they are the matrix's age groups. Verified in R; recorded in `socialmixr_issues.md` (S3 at the time, now S2); D19 in the dev plan annotated. | Done |
| E2 | No recommended default; never shows a "model-ready" matrix with reciprocity | (a) Section 12 now opens with a typical starting point: `DayOfWeekWeights()`, plus `AgePostStratWeights` if single years within age groups are unevenly sampled, then check design effects before deciding on truncation. (b) **Package change:** `apply_reciprocity` now defaults to `False` in `SocialMix`, `from_dataframes`, the validator and the bootstrap, because reciprocity is a choice and is wrong in some scenarios. Docstring, `test_SocialMix.py` (explicit `True` in the not-applicable warning test; a new default-is-off test) and CHANGELOG (Changed, breaking) updated. The four SocialMix test files pass. Section 2's text now says the adjustment is off by default and depends on the analysis. Tutorial 02 already passes `apply_reciprocity=True` explicitly. | Done |
| E3 | No example methods paragraph | Added a methods-section template after the Section 11 checklist, filled in for the tutorial's setup (5:2 within age groups, single-year age post-stratification, average weight for a missing day, normalisation and truncation at 3, percentile bootstrap with 500 resamples and the 70–79 merge). | Done |
| S3 | Unclear whether `run_inference_bootstrap(weighted=False)` reuses or drops the weights | Verified in the code: no bound weights are passed, so replicates are unweighted. Section 9 now says it "ignores the weights entirely, as if the model had been fitted without `weight_specs`". | Done |
| S4 | Section 1's conditions describe post-stratification only | Added a sentence after the two conditions: design weights supplied with the survey are a different case; they follow from how the sample was drawn, and the survey documentation usually states how to use them (Section 6). | Done |
| E6 | Raking unavailable, but no alternative given | Section 7 ("Combining factors") now says that raked weights can be computed with survey software (e.g. R `survey::rake()`) and passed with `DesignWeights` as the only factor. Limitation 2 points to the same route. | Done |
| E4 | Weighting with participant strata never shown | One sentence added after the Section 4 takeaway: with participant strata (e.g. sex), cells are age group × sex, `within="cell"` matches 5:2 in every age group of each sex, and the counts per cell are smaller. No extra fit. | Done |
| S6 / E7 | Difference plots (age post-stratification; weekday vs weekly) not interpreted | No change: the tutorial is already long, and interpretation is left to the reader. Largest cells for reference: age plot [65,70)×[60,65) +0.82 (the single 66-year-old); weekday plot diagonal [10,15) +1.94, [5,10) +1.43, off-diagonal [70,75)×[20,25) −0.71. | No change |
| S5 | Sections 6–7 read like API reference | Option (a): new end-of-notebook section "Appendix: factor arguments and data assumptions". It holds the `DayOfWeekWeights` argument table, its six data assumptions, the labelled-day error demo (two code cells), the `AgePostStratWeights` population-data rules and the `PostStratWeights` further arguments. Section 7 keeps the concepts and examples, with one-sentence pointers to the appendix. Section 6's decision table stays. The contents list and checklist references (day coding, `on_missing`) now point to the appendix. Notebook re-executed, no errors. | Done |
| U2 | Terms used without definition; no rules of thumb | Partly. Design weights, design effect and bootstrap were already defined after S4 and X9. *Post-stratification* is now defined at first use (Section 5: "one whose weights match the sample composition to known population shares"). No rules of thumb for deff or `threshold`: there is no accepted universal value. | Done |
| E8 | Section 5's five consequences could be a callout | Turned into a blockquote callout, "Consequences of normalisation", with the numbered list kept (renders in Jupyter, Colab and GitHub). Also fixed C6c's cross-reference in consequence 3: "(consequence 2)" became "(consequences 1 and 2)". | Done |
| U1, U3, U4, U5, E10, S7 | Toy example; `cint`/`.central` basics; plugged-in numbers; pipeline diagram; symmetric colour scales; `.head(4)` table | Skipped: length, basics belong in tutorial 02, or already covered. U5 was not answered explicitly and is treated as skipped. | Skipped |
| — | socialmixr issues log follow-up (2026-11-12) | Removed the 0.5.1 missing-day-as-weekend issue: the code (`weight_by_day_of_week`) is not exported, called or tested in 0.7.0.9000, and the clone is up to date with upstream. Renumbered: old S2 → **S1** (missing values below average weight), old S3 → **S2** (`weigh_by_age()` no-op). Added **S3**: socialmixr matches 5:2 over the whole sample (`within="sample"` behaviour) in every checked version. Verified in R: under-5s' weekday share 0.674 → 0.653, intensity 8.87 → 8.79. | Done |

## 3. Suggested priorities

1. Fix the correctness issues: sex targets restricted to 0–79, the socialmixr missing-day caveat, the cell 28 table, and the overclaims in cell 56.
2. State the estimand and the assumptions near the start. Mark the derivation in cells 24–26 as optional, with a one-line takeaway first.
3. Add a socialmixr mapping table, a recommended default with a model-ready matrix (reciprocity on, bootstrap), and an example methods paragraph.
4. Make the axis labels consistent, define "cell" and the column-naming rule early, and move the pandas plumbing into helpers.
5. Outside the tutorial: fix the `run_inference_bootstrap` docstring.

---

## 4. Full review: infectious disease / public health researcher

**Overall verdict.** I'm an SEIR modeller who uses POLYMOD through socialmixr. This tutorial is careful and correct, and it handles "should I weight at all?" better than anything socialmixr ships. The best content is the weight-versus-stratify argument (cells 19–23) and the explanation of `within="cell"` versus `"sample"` (cells 24–26). But the tutorial is written for someone who wants to understand the estimator, not for someone who has to swap their R pipeline for this one by Friday. Three things I need are never stated in one place: the default to use, the code that reproduces socialmixr, and the sentence to put in my methods section. Reciprocity is off for the whole notebook (cell 12), so I never see the matrix I would actually put into a next-generation matrix.

### Q1 – Easy to understand at first sight?
- The introduction (cells 0, 4) and the 5:2 example are easy to follow. But the notation table (cell 5) comes before any data and lists 11 symbols. I skipped it and had to come back.
- Cell 4 says "a naive day-of-week weights can **underestimate contact intensity in children**". That reads as if the weights were wrong. What cells 19–23 actually show is that the weekly average is the wrong *target* for school-term transmission questions. As framed, the opening makes me wary of the method when the real point is which estimand to choose.
- The first practical payoff arrives at cell 15. I'd like a three-line "here's a weighted fit" before the maths in cell 14.

### Q2 – Could I use it in my own work?
- **socialmixr parity is only half covered.**
  - Cell 24 maps `weigh.dow` to `within="sample"`. Nothing maps `weigh.age` to `AgePostStratWeights`, `weights=` to `DesignWeights`, or `weight.threshold` to `threshold=`.
  - I also checked the source. `_poststrat.py` says that with missing days the results *differ* from socialmixr: socialmixr downweights participants with no recorded day, while cntmosaic gives them factor 1. The tutorial never says this. If I try to reproduce published socialmixr numbers on POLYMOD (43 missing days), I won't match and won't know why.
  - Truncation is applied per age group here (cells 47, 27). The tutorial doesn't say whether socialmixr normalises over the whole sample instead.
  - I need a side-by-side table that includes the known differences.
- **No recommended default.** After 10 sections I still don't know what to use for a "typical country analysis". `DayOfWeekWeights()` alone? Day plus age? With `threshold=3`? Cell 58 lists everything and recommends nothing.
- **No model-ready matrix.**
  - Every fit uses `apply_reciprocity=False` (cell 13).
  - I'd like one closing cell that does a weighted fit with reciprocity on, plus a per-capita rate or an NGM/R0 comparison between the unweighted and weighted matrices.
  - Cell 23 talks about R0 but never computes it.
- **No guidance on reporting methods.** The checklist in cell 57 is good, but I need an example paragraph, e.g. "participants were weighted to a 5:2 weekday/weekend ratio within each 5-year age group; weights were truncated at 3× the age-group mean; 95% intervals from 500 weighted bootstrap replicates."
- Stratified fits (e.g. by sex) combined with weights are never shown, even though the labels say "All, [0, 5)" and the warning in cell 50 mentions "age group x participant strata".

### Q3 – What remains unclear
- **"Cell".** The text says age group, the API says `within="cell"`, the outputs say `cells`, and the labels carry an unexplained "All, " prefix. I only learn in cell 50 that a cell is age group × stratum.
- **Column naming.**
  - I pass `aux_cols=["part_gender"]` but then write `PostStratWeights("gender", ...)` (cell 38). The prefix rule is only explained under `DayOfWeekWeights` (cell 29, item 1).
  - The factor columns are named `dayofweek`, `age` and `poststrat:gender`, which contradicts cell 16's "each factor column is named after its factor".
- **The bootstrap changes the point estimate** (cells 52–53). This matters for reporting:
  - Which estimate do I report?
  - Should I merge 70–79 myself up front?
  
  The tutorial notes the change but doesn't advise.
- **What age post-stratification is for.** Cell 33 says it only rebalances single years *within* age groups. A practitioner may assume it corrects the population age structure. State explicitly that population age structure only enters through reciprocity and per-capita rates. At the moment this is buried in cell 27, item 3.

### Q4 – Terminology, gaps, accuracy
- Cell 4: "a naive day-of-week weights" is a grammar slip and also misleading (see Q1).
- Cell 23: "understate contact intensity ... underestimate R0" is true for this comparison, but R0 depends on the dominant eigenvalue of the whole matrix. Either show it or soften it.
- Cell 26 is dense, with about 10 numbers in one paragraph. Lead with the conclusion ("sample-wide weights can push an age group *further* from 5:2") and give two numbers.
- Cell 47 says "it gives up part of the correction (bias) in exchange for stability (variance)", which is good. Cell 54's "Weighting trades bias for variance" repeats the idea; say it once.
- Cell 44 says raking is "not available". Good, but also tell me what to do instead: rake externally and pass the result as `DesignWeights`.
- Cell 20's difference panel shows strong off-diagonal artefacts, e.g. [20,25)×[70,75). These are small-cell noise that readers may over-interpret; add a sentence saying so.
- The figure axis labels are inconsistent:
  - "Respondent age group" in cell 13.
  - "Age of contacting individual" in cell 20.
  - "Age of respondent" on one panel but not the other in cell 36.
  
  The colour scales in the difference panels are also asymmetric around 0.

### Q5 – Formatting and sectioning
- Section 3 (cells 14–26) is far too long. It holds the method, the weight-vs-stratify discussion and the socialmixr comparison. Split it into "3. Day-of-week weights", "4. Weight or stratify?" and "5. `within=` and socialmixr".
- Section 4 repeats equation (1) word for word (cell 27, step 3). Most of its five "consequences" could be a short callout.
- Cells 17, 25, 34, 38 and 41 are mostly pandas plumbing that joins ids back to covariates, and they make the API look harder to use than it is. Hide this in helpers, or add a convenience method that attaches auxiliary columns to `weights.participants`.
- `UserWarning` noise is left in cells 11, 20 and 22, although `fit()` suppresses it elsewhere. This is inconsistent.
- Keep: the summary code block (cell 58), the checklist (cell 57), the error-message demo (cells 30–32) and the design-effect table (cell 46).

### Top 5 changes I'd ask for
1. **Add a "Coming from socialmixr" table.**
   - Map `weigh.dow`, `weigh.age`, `weights=` and `weight.threshold` to their cntmosaic equivalents.
   - State the known differences: missing-day handling, and normalisation and truncation per age group.
   - Include one numeric check against socialmixr output on POLYMOD UK.
2. **Give a recommended default and an end-to-end "model-ready" cell.**
   - The cell should do a weighted fit with reciprocity on, a weighted bootstrap and per-capita rates.
   - Ideally, add an unweighted vs weighted R0 / dominant-eigenvalue comparison to back up cells 4 and 23.
3. **Add a template paragraph for the methods section** next to the checklist in cell 57. It should cover how to report the age-group merge that the bootstrap does (cell 52).
4. **Restructure Section 3** into three shorter sections. Reframe cell 4 as "choose the right target" rather than "weights underestimate children".
5. **Define "cell" and the column-naming rules early.** These are the `part_` prefix and the factor column names. Also cut the pandas plumbing (cells 17, 25, 34, 38, 41), and make the figure axis labels and warning suppression consistent.

Sources checked beyond the tutorial: `cntmosaic/weights/_poststrat.py` (module docstring on how missing days differ from socialmixr) and `cntmosaic/weights/_specs.py` (normalisation per age group, then truncation).

---

## 5. Full review: applied statistician

**Overall verdict.** This is a careful tutorial built on real data, and it is better than most package vignettes on weighting. The derivation of within-cell vs sample-wide matching (cell 24) and the checklist (cell 57) are the strongest parts and should stay. As a survey statistician, though, I could not find the estimand stated anywhere. Equation (1) is never named as a Hájek ratio estimator. The ignorability assumptions behind post-stratification and behind imputing missing days are never stated. A few claims are stronger than the code and output support. The most important are the "total weight of a category" column in cell 28 and the sex example in cells 38–39, whose target population doesn't match the sampling frame. Sections 5–6 are long API reference material that interrupts the statistical argument.

### Q1 – Ease of understanding at first sight
- The day-of-week example is well chosen and well paced (cells 14–18). The worked factor values check out: (5/7)/(75/97) = 0.924 and (2/7)/(28/87) = 0.888.
- Section 1 (cell 4) conflates two ideas. It says weighting "can underestimate contact intensity in children", but Section 3 shows that the weights estimate their own target correctly; the actual problem is choosing the wrong estimand. Say "estimand mismatch" in Section 1 rather than "undesired consequences". Also, "a naive day-of-week weights" is ungrammatical.
- The two conditions in Section 1 only describe post-stratification. Design weights (unequal selection probabilities) don't fit "average over a factor", yet they appear in cell 40.

### Q2 – Could I use it in my own work?
- **The estimand is never stated.** I would expect something like "the population mean, over persons in age group c (and over days of the week), of daily contacts with age group d". Without it, "representative" (cell 0) and "appropriate target" (cell 19) have nothing to refer to.
- **Equation (1) needs naming as the Hájek (ratio) estimator.** Say also that factors based on sample counts make it a post-stratified estimator, not a Horvitz–Thompson one. This one sentence would orient every survey reader.
- **Variance estimation:**
  - Cell 52 doesn't say that the bootstrap draws N participants i.i.d. from the whole sample rather than within age groups (`_socialmix_bootstrap.py:407`), so the n_c vary between replicates.
  - It doesn't say that the intervals are percentile intervals with alpha 0.05 (`_ModelSummariserSocialMix.py:235`).
  - Neither replicate weights supplied with a survey nor Rao–Wu are supported. Section 9 should list both under limitations.
- **The automatic merge rule in the bootstrap is undocumented.** Age groups are merged while sum_c exp(−n_c) ≥ 5×10⁻⁵ (`_socialmix_validation.py:504–523`). Readers need this rule to predict when `sm.weights` will change.
- **Diagnostics:**
  - Kish's n_eff and deff are well presented.
  - Missing: deff measures the inflation from weight variability alone, under the assumption that the weights are unrelated to y.
  - Post-stratifying on a predictive variable can *reduce* variance. Cell 55 shows "day" narrowing some intervals, e.g. [10,15) from 0.493 to 0.478.

### Q3 – What remains unclear
- **How missing days are treated.**
  - Giving participants with a missing day factor 1 amounts to mean imputation, under the assumption that the day is missing completely at random (MCAR) within each cell. State this as an assumption (cells 14, 24).
  - It also changes the "5:2" claim in cell 18. The ratio holds only among reports with a known day, and m_cd is actually a mixture of three groups.
- **Ignorability is never stated.** For post-stratification to remove bias, weekday reporters within cell c must represent weekday behaviour in c (MAR given cell × day type). The tutorial never says so.
- **Interface: does 5:2 apply within age groups or within cells?** Cell 28 says "within age group", but the code's cells are age group × participant strata (`_poststrat.py:292–300`). The tutorial doesn't say what `within="cell"` does once participant strata such as sex are used.
- **`run_inference_bootstrap(weighted=False)`.** Cell 52 doesn't say whether this is the same as an unweighted fit, or whether it reuses the full-sample weights.
- **The 5:2 convention is asserted without justification** in cell 0. Holidays only come up in cell 23, and the convention also assumes all weekdays are alike.

### Q4 – Terminology, gaps, misleading statements
- The notation table (cell 5) omits τ, n_eff, deff, λ, n_a and n_{c,wd}. Its entry "f^{(k)}_i: the factor that the k-th weight factor gives" uses "factor" for two different things.
- N is defined as "total number of participants", but in the formula in cell 33 the code's n counts only participants within the age range who have a known value (`_poststrat.py:75`).
- Cell 27, consequence 3: "Population sizes do not enter m_cd" is contradicted by `AgePostStratWeights`, which uses P_a. Change it to "population sizes used for reciprocity/rates".
- Cell 23: "In every age group, the weighted weekly average lies between…" holds only for participants whose day is known; the group with missing days can pull the average outside that range. The table also shows only 4 rows.
- Cell 56: "The day-of-week weights barely change the intervals", but [70,80) widens from 0.561 to 0.628 (+12%).
  - "Design effects indicate where precision is lost" is loose. [45,50) widens by 12% with deff 1.19, while [30,35) has deff 1.34 and doesn't widen.
  - With 500 replicates, differences of about 1% are within Monte Carlo error. Say so.
- Cell 47, "trades bias for variance": truncation does, but weighting in general need not (see Q2).

### Q5 – Formatting and sectioning
- Equation (1) appears twice (cells 18 and 27), and Section 4 ("How does weighting work?") comes after the reader has already used it. Either move Section 4 before Section 3, or present Section 3 as a worked instance that follows the general setup.
- Sections 5–6 (the argument tables, and the six data assumptions listed in cell 29) read like API docs. Move them to the reference page or a collapsible appendix.
- Cell 36's figure gets no interpretation. The large +0.8 / −0.3 cells at respondent age [65,70) are exactly the effect of the single 66-year-old and deserve a sentence.
- **The axis labels are inconsistent** across cells 13, 20 and 36 ("Respondent/Contact age group", "Age of contacting/contacted individual", "Age of respondent/contact"). The notation table treats rows c as participants, yet participants are on the x-axis of the plots.
- Code cells 17, 25 and 34 contain pandas plumbing (pivot, reindex, removeprefix) that obscures the point. A helper or a `WeightResult` method would help.

### Correctness issues (separate from style)
1. **The target population for sex doesn't match the sampling frame (cells 38–39).**
   - The targets are summed over all ages, including 80+ and 90+, but every participant is aged 0–79 (`AgeGroupSpecs(0, 79, 5)`, and the sample has no 80+).
   - The female share for ages 0–79 is 50.3%, not the 50.9% used, so women should be downweighted more.
   - This is a textbook post-stratification error in a teaching example.
2. **The column "total weight of a category = target share, whatever the sample count" (cell 28)** holds only for the composite weights with `within="sample"`.
   - The final weights are renormalised to n_c per cell (`_specs.py:362–368`), so their totals across the sample are not exactly the targets.
   - In the sex example the female share after the final weights is 0.50865, against a target of 0.50862.
   - The CategoryWeights column has the same problem.
3. **The socialmixr equivalence is overstated (cells 24, 58).** With missing days, socialmixr counts them in n and downweights them, whereas cntmosaic gives them factor 1; the code's own docstring says so (`_poststrat.py:17–19`). The tutorial says `within="sample"` "corresponds to" socialmixr without this caveat, and POLYMOD UK has 43 missing days.
4. **The bootstrap silently changes the point estimate (cells 52–53).**
   - An inference routine that mutates `sm.weights` and `cint()` is a design hazard. At minimum, the tutorial should tell readers to re-extract point estimates after bootstrapping.
   - Separately, the `SocialMix.run_inference_bootstrap` docstring (`_SocialMix.py:819–821`) says merging happens only with `adaptive_merge=True` and that a `ValueError` is raised otherwise. But the stability merge runs regardless (`_socialmix_validation.py:517–523`), which is what happened in cell 53 under the default `adaptive_merge=False`. The tutorial matches the code; the docstring does not.
5. **Cell 14, "participants with a missing day receive the average factor".** This is the cell average under `within="cell"` but the *sample* average under `within="sample"`, which is not the average of the participant's own cell.

### Top 5 changes I'd ask for
1. **In Section 4, state the estimand and name equation (1)** as a Hájek / post-stratified ratio estimator. List the assumptions explicitly: within-cell ignorability of the weighting variable, MCAR for missing days, and the 5:2 calendar convention.
2. **Fix the sex example** (cells 38–39) to use population targets for ages 0–79. **Correct the "total weight" column** in cell 28 so that it describes the composite weights before cell normalisation.
3. **Expand Section 8.**
   - Say that the bootstrap resamples participants i.i.d. without stratifying, and that the intervals are percentile intervals.
   - State the merge rule and its side effect on the point estimates.
   - Add replicate weights, Rao–Wu and household clustering (hh_id) to the limitations.
   - Temper the claims in cell 56 and mention Monte Carlo error.
4. **Qualify the socialmixr equivalence** (missing-day handling), and **clarify that `within="cell"`** means age group × participant strata.
5. **Restructure.**
   - Put the general setup (Section 4) before the worked example, or merge the two.
   - Move the argument and assumption tables of Sections 5–6 to the API reference.
   - Interpret the cell 36 figure.
   - Make the axis labels consistent across all plots.

---

## 6. Full review: first-year undergraduate

**Overall impression.** I worked through it and got the gist: you give survey respondents different "importance" numbers so that, for example, weekday and weekend reports count in a 5:2 ratio. Section 3 does a good job of this; the table in cell 17 and the paragraph in cell 18 were where it first really made sense. After that it gets harder fast. From cell 24 onward it reads as if written for someone who already knows survey statistics. I met "post-stratification", "odds ratio", "design weights", "inverse inclusion probabilities", "raking", "Kish", "bias/variance" and "bootstrap" without explanation. I could copy the summary in cell 58, but I couldn't say why I'd choose one factor over another.

### Where I got lost
- **Cell 0.** "A contact matrix is an *average*": of what, over whom? There's no small picture or tiny example matrix until cell 13.
- **Cell 4.** "R₀ and final size" are used without definition.
- **Cell 5.** The notation table comes before any example, so I forgot it immediately. I didn't know what "a single year of age" vs "age groups" meant for a vs c.
- **Cell 9.** `np.select`, `.isin` and `value_counts`. I've never seen `np.select`.
- **Cell 12.** "Reciprocity". I could follow "same in both directions at the population level", but not why that would hide the effect of the weights.
- **Cell 13.** `sm.cint()["All->All"]`: what is `cint`, and why "All->All"? The `warnings.catch_warnings` block looks scary.
- **Cell 14.** The set notation 𝒩_c ∩ 𝒲 and |·| was hard. "Receives the target share of their day type divided by its share..." only made sense after I saw the numbers in cell 18.
- **Cell 17.** `pivot_table`, `pd.concat({...}, axis=1)`, `reindex` and `removeprefix`: 14 lines of pandas just to make a table. I couldn't tell if *I* need to write this for my own data.
- **Cells 24–26.** I lost the thread here. "Odds ratio", λ^wd_c and the mixture derivation went over my head. I took away "use `within="cell"` unless groups are small" but couldn't say why.
- **Cell 28.** "Inverse inclusion probabilities, non-response adjustments", "inverse sampling fraction" and "post-stratification". The decision table assumes I know these.
- **Cell 33.** I didn't understand why age weights only rebalance *within* an age group, or why I'd want that.
- **Cell 41.** `rng.choice(..., p=[0.8, 0.2])`, and why a weight of 2.0 turns into 1.666667 and 0.793651.
- **Cell 45.** "Kish's effective sample size" and "design effect": a formula is given, but no intuition first.
- **Cell 47.** "Gives up part of the correction (bias) in exchange for stability (variance)". I've never met bias/variance.
- **Cell 52.** "Bootstrap", "resampling with replacement" and "95% intervals" are never explained.

### Q1 – Easy at first sight?
- The intro (cell 0) and the weekday example are nice, and the 5:2 idea is intuitive.
- The notation table (cell 5) is intimidating so early. I'd rather each symbol were introduced when it is first used.
- The figures (cells 13, 20, 36) aren't explained, and I couldn't tell which axis is the respondent.
  - Cell 13 says "Respondent age group" on the x-axis.
  - Cell 20 says "Age of contacting individual / contacted individual".
  - Cell 36 mixes the two.
  - Cell 5 says c is a *row*, but in the plot the respondent is on the *x* (column) axis.
- "Intensity" (the colour bar), "mean number of contacts" (m_cd) and `cint`: are these the same thing?

### Q2 – Could I apply it to new data?
Mostly by copying cell 58. I'd still need to ask my supervisor:
- Which variables does our survey have, and which ones *should* we weight by? Section 1 lists conditions, but I can't judge "influences contact behaviour" myself.
- How are our days coded: 0 = Sunday or 7 = Sunday? Cell 29 helps a lot here.
- Where do I get population targets, and in what format?
- Do we have design weights? I'm not sure what they are.
- What `threshold` should I use? Cell 47 uses 3, with no rule of thumb.
- What design effect is "too large"?
- Do I report the weighted, the unweighted, or the weekday-only stratified matrix?

### Q3 – What remains unclear
**Concepts:**
- post-stratification (never defined in plain words);
- the difference between design, category and post-stratification weights;
- why normalisation exists if it doesn't change the answer (cell 27);
- bootstrap and confidence intervals;
- bias vs variance;
- raking.

**Interface:**
- What `preview()` returns. A `WeightResult`? It's only named in cell 45.
- What `.central`, `.upper` and `.lower` are (cells 20, 55).
- What `cell` means. It looks like "All, [5, 10)", but what is "All"? Cell 50 mentions "participant strata", which were never introduced.
- Why the factor column is `dayofweek` in one place but `poststrat:gender` in cell 38.
- Why `PostStratWeights("gender", ...)` works when the column is `part_gender`.
- When `adaptive_merge` is needed.

### Q4 – Details to refine
- **Undefined terms:** reciprocity, R₀, final size, odds ratio, inclusion probability, sampling fraction, non-response, raking, Kish, design effect, bootstrap, 95% interval, truncation/capping, `NaN`/`pd.NA`, `Categorical`.
- **Undefined symbols:** ∩, |·|, ∏, w̃, γ_c, τ, λ, `\tag{1}`.
- **Unfamiliar code idioms:** `np.select`, `pivot_table`, `pd.concat` with a dict, `.loc[...].to_numpy()`, `.query` with backticks (cell 48), `lambda t:` in `.assign` (cell 34), `try/except ValueError`, `warnings.catch_warnings(record=True)`, a list comprehension with `issubclass`.
- **Worked examples needed:**
  - cell 14: plug in the under-5 numbers, (5/7)/(59/87) = 1.053;
  - cell 27: a four-person toy example of normalisation;
  - cell 41: show how 2.0 becomes 1.67 after normalisation;
  - cell 45: a three-weight example of n_eff.
- **A diagram would help in cell 27:** factors → multiply → normalise → (cap) → weighted mean.
- Cell 26 packs a lot of numbers into one paragraph. A short "so what" sentence first would help.

### Q5 – Formatting and sectioning
- The numbered contents and section headings are clear. The checklist (cell 57) and the summary code (cell 58) are the most useful parts for me.
- Cells 24–26 are the most intimidating part: a derivation plus a 16-row, 6-column table that wraps three times. They could be an optional "advanced" box.
- Several code cells (17, 25, 34, 41, 55) are mostly pandas for building tables, not cntmosaic usage, and I couldn't tell which lines matter. A comment such as `# display only` would help.
- The `UserWarning` outputs (cells 11, 20, 22) look like errors to a beginner. Say they are harmless.
- The equation in cell 18 is repeated word for word in cell 27. That's fine, but say "recall".

### Glossary I wish it had
contact matrix / contact intensity / `cint`; respondent vs contact; reciprocity; stratify vs weight; post-stratification; target share; design weight; inclusion probability; sampling fraction; non-response; normalisation; composite weight; odds ratio; effective sample size; design effect; truncation/capping and τ; bias vs variance; bootstrap; 95% interval; raking; strata/"All"; `adaptive_merge`; R₀; ∩, |·|, ∏, w̃.

### Top 5 changes that would help me most
1. Add a plain-language "what is a weight" paragraph in Section 1, with a four-person toy example (two weekday, two weekend, worked by hand), before any notation.
2. Define post-stratification, design weights, design effect and bootstrap in one sentence each where first used, or in a glossary box. Add rules of thumb for when the design effect is too large and for choosing `threshold`.
3. Mark cells 24–26 (sample vs cell derivation) as optional/advanced, and lead with a one-line takeaway: "use the default unless an age group has very few weekend reports".
4. Separate cntmosaic usage code from table-making display code, and label the display-only cells. Explain `cint()`, `["All->All"]`, `.central/.upper/.lower`, and what `preview()` returns.
5. Fix and explain the figures:
   - use consistent axis labels;
   - say which axis is the respondent, and reconcile this with "c is a row";
   - add one sentence on how to read the difference plots.
