import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from cntmosaic._types import StratMode
from cntmosaic.dataloader.containers._StratificationData import StratificationData


def test_single_full_variable_marginal_equals_full():
    # Simple synthetic population proportions for two ages and two strata
    df = pd.DataFrame(
        {
            "age": [0, 0, 1, 1],
            "sex": ["A", "B", "A", "B"],
            "prop": [0.6, 0.4, 0.7, 0.3],
        }
    )

    strat_data = StratificationData(
        data=df, age_col="age", strat_var_cols="sex", prop_col="prop"
    )

    strat_modes = {"sex": StratMode.FULL}

    marginal = strat_data.compute_marginal_demopty(strat_modes)
    demopty = strat_data.compute_demopty(strat_modes)

    assert "sex" in marginal
    marg_mat = marginal["sex"]

    # Shapes should match (n_strata^2, A, A)
    assert marg_mat.shape == demopty.shape

    # Values should be identical for the single FULL stratification variable case
    assert_allclose(marg_mat, demopty, atol=1e-12)


# ---------------------------------------------------------------------------
# Row order of the FULL multipliers
# ---------------------------------------------------------------------------
# The model subtracts log(multipliers) from log_delta by position, and log_delta
# is ordered like ``infer_full_strat_labels``. Each multiplier row must therefore
# be the outer product for the stratum pair named by the label at that position.

import pytest

from cntmosaic.dataloader._stratification import infer_full_strat_labels


def _two_var_population():
    """Population shares by age, sex (2) and ethnicity (3); distinct values."""
    rng = np.random.default_rng(0)
    ages, sexes, ethns = [0, 1, 2], ["M", "F"], ["W", "B", "H"]
    df = pd.DataFrame(
        [(a, s, e) for a in ages for s in sexes for e in ethns],
        columns=["age", "sex", "ethn"],
    )
    df["prop"] = rng.uniform(0.5, 1.5, len(df))
    df["prop"] /= df.groupby("age")["prop"].transform("sum")
    df["sex"] = pd.Categorical(df["sex"], categories=sexes)
    df["ethn"] = pd.Categorical(df["ethn"], categories=ethns)
    return df


def _share(df, stratum, vars_):
    """Population share by age of one stratum label, e.g. 'M_W' or 'M'."""
    mask = np.ones(len(df), dtype=bool)
    for var, value in zip(vars_, stratum.split("_")):
        mask &= (df[var] == value).to_numpy()
    return df[mask].groupby("age")["prop"].sum().to_numpy()


def _full_labels(df, strat_modes):
    """Full labels exactly as assemble_strat_kwargs builds them."""
    dims, labels = {}, {}
    for var, mode in strat_modes.items():
        cats = list(df[var].cat.categories)
        if mode == StratMode.FULL:
            labels[var] = [f"{a}->{b}" for a in cats for b in cats]
        else:
            labels[var] = [f"{a}->All" for a in cats]
        dims[var] = len(labels[var])
    return infer_full_strat_labels(dims, labels)


@pytest.mark.parametrize(
    "strat_modes",
    [
        {"sex": StratMode.FULL, "ethn": StratMode.FULL},
        {"ethn": StratMode.FULL, "sex": StratMode.FULL},
        {"sex": StratMode.FULL, "ethn": StratMode.PARTIAL},
        {"sex": StratMode.PARTIAL, "ethn": StratMode.FULL},
    ],
    ids=["full-full", "full-full-swapped", "full-partial", "partial-full"],
)
def test_full_multipliers_follow_full_labels(strat_modes):
    df = _two_var_population()
    strat_data = StratificationData(
        data=df, age_col="age", strat_var_cols=list(strat_modes), prop_col="prop"
    )
    source_vars = list(strat_modes)
    target_vars = [v for v, m in strat_modes.items() if m == StratMode.FULL]
    full_labels = _full_labels(df, strat_modes)

    multipliers = strat_data.compute_demopty(strat_modes, full_labels)

    assert multipliers.shape[0] == len(full_labels)
    for n, label in enumerate(full_labels):
        source, target = label.split("->")
        expected = np.outer(
            _share(df, source, source_vars), _share(df, target, target_vars)
        )
        assert_allclose(multipliers[n], expected, atol=1e-12, err_msg=label)

    # Without explicit labels the same order is inferred from the data.
    assert_allclose(strat_data.compute_demopty(strat_modes), multipliers)
