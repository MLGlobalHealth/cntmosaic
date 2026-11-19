# Bug: demographic multipliers out of order with two or more stratification variables

**Status:** fixed on branch `fix/demopty-multiplier-order` (2026-11-18)
**Affects:** `GenMixCC`, `GenMixFC`, `GenMixFF` fits in which a FULL stratification variable is followed by another variable
**Consequence:** the fitted contact matrices are wrong, not just mislabelled, and are not reciprocal. Affected fits must be rerun.

## Symptom

The age × sex × race/ethnicity fit of the EpiStorm-Mix survey (`cntmosaic_pcbi`, `case_studies/03_Epistorm_Matrices.ipynb`, `GenMixCC` with `strat_var_cols=["gender", "ethn"]`) gives a contact rate matrix M that does not satisfy reciprocity.

Reciprocity means P_a[i] M_{a→b}[i, j] = P_b[j] M_{b→a}[j, i] for every pair of strata a, b and ages i, j, where P is the population the model was fitted with (ACS 2020–2024).

| Check (posterior median, `summarise_cint()`) | Result |
|---|---|
| ‖C − Cᵀ‖_F / ‖C‖_F for the joint 170 × 170 matrix, C = diag(P) M | **1.00** |
| Same-stratum blocks, e.g. Male_White → Male_White | 0 (exactly reciprocal) |
| Total contacts Male_White → Female_White ÷ Female_White → Male_White | **9.9** (should be 1) |

Most cross-strata blocks *are* exactly reciprocal, but with the wrong population. For example:
- `Male_White->Female_White` is exactly reciprocal with its transpose block when its rows are weighted by **Male_Asian**'s population.
- `Male_Asian->Male_White` is reciprocal when weighted by **Male_Black**'s population.

That pattern points to an index permutation in the population offset, rather than noise.

## Root cause

The model subtracts the demographic multipliers from the log contact-rate modifiers **by position**, in `models/numpyro/_GenMixCC.py` (same in `_GenMixFC.py`, `_GenMixFF.py`):

```python
log_delta = jnp.log(delta) - jnp.log(self.data.multipliers)
```

Two orders are in play:
- **`log_delta` and the labels:** both are ordered like the full stratification labels (`infer_full_strat_labels`, `make_flat_ix`, `kron_sum_mode_1`). These **interleave source and target per variable**, with the rightmost index varying fastest. For two FULL variables the order is (g, g′, e, e′): n = ((g·2 + g′)·5 + e)·5 + e′.
- **The multipliers:** `StratificationData.compute_demopty` built them as one outer product of all source strata with all target strata, and then flattened it:

  ```python
  Q_stab = Q_sa[:, None, :, None] * Q_tb[None, :, None, :]
  result = Q_stab.reshape(n_strata_source * n_strata_target, A, A)
  ```

  This is order (g, e, g′, e′): m = (g·5 + e)·10 + (g′·5 + e′).

The two orders coincide only when there is a single variable, or when every variable before the last one is PARTIAL. In every other case, block n receives the multiplier of a different stratum pair (source n // n_target, target n % n_target). This explains the table above: label index 25 (`Male_White->Female_White`) gets the multiplier for (Male_Asian, Female_White).

Because the offset enters the likelihood, **the posterior itself is distorted**. Relabelling the summaries afterwards cannot recover the correct matrix.

## Affected configurations

| Stratification (in `strat_var_cols` order) | Affected? |
|---|---|
| Single variable, PARTIAL or FULL | No |
| FULL, FULL (e.g. sex × ethnicity on both sides) | **Yes** |
| FULL, PARTIAL | **Yes** |
| PARTIAL, FULL | No (orders coincide) |
| PARTIAL only | No (no target axis) |

`compute_marginal_demopty` (used to centre the priors) works on one variable at a time and is unaffected.

## Fix

`compute_demopty` now builds the FULL multipliers **from the labels**:
1. It splits each `full_labels[n]` into its source and target strata.
2. It looks both up in Q.
3. It sets `result[n] = outer(Q_source[s_n], Q_target[t_n])`.

The row order is therefore the label order by construction, for any number and mix of FULL and PARTIAL variables. When no labels are passed, `_default_full_labels` generates them in the same order as `infer_full_strat_labels`.

`_handle_nans` now takes the per-row source and target labels, so its warning names the correct stratum pairs.

Tests are in `cntmosaic/dataloader/tests/test_stratification_demopty.py`:
- `test_full_multipliers_follow_full_labels` checks every row against the outer product of the population shares for the pair named by its label, for full-full (both orders), full-partial and partial-full.
- `test_full_prop_multi` and `test_mixed_prop_multi` in `test_StratificationData.py` asserted the old order and were corrected.
- The new tests fail on the previous code, and all dataloader tests pass with the fix.

### Minimal reproduction (previous code)

```python
import numpy as np, pandas as pd
from cntmosaic._types import StratMode
from cntmosaic.dataloader.containers._StratificationData import StratificationData
from cntmosaic.dataloader._stratification import infer_full_strat_labels

rng = np.random.default_rng(0)
df = pd.DataFrame([(a, s, e) for a in range(3) for s in "MF" for e in "WBH"],
                  columns=["age", "sex", "ethn"])
df["prop"] = rng.uniform(0.5, 1.5, len(df))
df["prop"] /= df.groupby("age")["prop"].transform("sum")

modes = {"sex": StratMode.FULL, "ethn": StratMode.FULL}
labels = infer_full_strat_labels(
    {"sex": 4, "ethn": 9},
    {"sex": [f"{a}->{b}" for a in "MF" for b in "MF"],
     "ethn": [f"{a}->{b}" for a in "WBH" for b in "WBH"]},
)
sd = StratificationData(data=df, age_col="age", strat_var_cols=["sex", "ethn"], prop_col="prop")
mult = sd.compute_demopty(modes, labels)

def share(stratum, vars_):
    mask = np.logical_and.reduce([df[v] == x for v, x in zip(vars_, stratum.split("_"))])
    return df[mask].groupby("age")["prop"].sum().to_numpy()

n = labels.index("M_W->F_W")
src, tgt = labels[n].split("->")
expected = np.outer(share(src, ["sex", "ethn"]), share(tgt, ["sex", "ethn"]))
print(np.allclose(mult[n], expected))  # False before the fix, True after
```

## Impact on existing outputs

Anything fitted with an affected configuration must be **refitted** with the fixed package:
- `cntmosaic_pcbi/case_studies/03_Epistorm_Matrices.ipynb`, age × sex × ethnicity section. Regenerate `data/outputs/age_gender_ethn-age_gender_ethn.pkl`, then rerun everything downstream of it (`03_Epistorm_Epydemix.ipynb` scenarios and figures, `04_Sensitivity_Analysis.ipynb`).
- `docs/source/tutorials/04_Tutorial_Feature_Selection.ipynb`, which fits `strat_var_cols=["sex", "educ"]`.
- Any other analysis using `GenMix*` with two or more stratification variables.

The age-only, age × sex and age × ethnicity fits in `03_Epistorm_Matrices.ipynb` are unaffected.

After refitting, check that ‖diag(P)M − (diag(P)M)ᵀ‖ / ‖diag(P)M‖ ≈ 0 for the joint matrix. Medians of reciprocal posterior draws are themselves reciprocal, so this should hold up to numerical precision.

## Related, harmless

When `compute_demopty` is called **without** labels for several PARTIAL variables, `_build_Q_and_labels` returns Q rows in the pivot's alphabetical order but labels in order of first appearance. The model always passes labels, so this does not affect fits. The FULL path no longer relies on it.
