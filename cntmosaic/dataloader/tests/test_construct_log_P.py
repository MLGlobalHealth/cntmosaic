import numpy as np
import pandas as pd
import pytest

from ..._types import StratMode
from .._ColumnSpec import ColumnSpec
from .._observation import construct_log_P
from .._stratification import infer_strat_pixs

A = 4
SEX = ["M", "F"]
REGION = ["Urban", "Rural"]


@pytest.fixture
def pop_df():
    rng = np.random.default_rng(0)
    idx = pd.MultiIndex.from_product([SEX, REGION, range(A)], names=["sex", "region", "age"])
    df = idx.to_frame(index=False)
    df["sex"] = pd.Categorical(df["sex"], categories=SEX, ordered=True)
    df["region"] = pd.Categorical(df["region"], categories=REGION, ordered=True)
    df["P"] = rng.integers(100, 1000, len(df)).astype(float)
    return df


def _col_spec(pop_strat_vars):
    return ColumnSpec(part_age="age", pop_age="age", P="P", pop_strat_vars=pop_strat_vars)


def _P(pop_df, by):
    """Population summed over every variable not in `by`, as {key: (A,) array}."""
    grouped = pop_df.groupby(by + ["age"], observed=False)["P"].sum()
    return grouped.unstack("age").to_numpy()


@pytest.mark.parametrize("strat_modes", [{}, {"sex": StratMode.PARTIAL}])
def test_no_full_vars_gives_total_population(pop_df, strat_modes):
    log_P = construct_log_P(pop_df, _col_spec(["sex", "region"]), strat_modes)

    total = pop_df.groupby("age")["P"].sum().to_numpy()
    assert log_P.shape == (1, A)
    np.testing.assert_allclose(log_P[0], np.log(total))


def test_full_var_matches_legacy(pop_df):
    sex_only = pop_df.groupby(["sex", "age"], observed=False)["P"].sum().reset_index()
    col_spec = _col_spec(["sex"])

    log_P = construct_log_P(sex_only, col_spec, {"sex": StratMode.FULL})

    assert log_P.shape == (len(SEX), A)
    np.testing.assert_allclose(log_P, construct_log_P(sex_only, col_spec))


def test_mixed_modes_sum_over_partial_vars(pop_df):
    strat_modes = {"sex": StratMode.FULL, "region": StratMode.PARTIAL}
    log_P = construct_log_P(pop_df, _col_spec(["sex", "region"]), strat_modes)

    assert log_P.shape == (len(SEX), A)
    np.testing.assert_allclose(log_P, np.log(_P(pop_df, ["sex"])))


def test_rows_align_with_infer_strat_pixs(pop_df):
    strat_modes = {"sex": StratMode.FULL, "region": StratMode.FULL}
    log_P = construct_log_P(pop_df, _col_spec(["sex", "region"]), strat_modes)

    # One observation per contact (sex, region) combination
    combos = pd.MultiIndex.from_product([SEX, REGION]).to_frame(index=False)
    df_full = pd.DataFrame(
        {
            "cnt_sex": pd.Categorical(combos[0], categories=SEX, ordered=True),
            "cnt_region": pd.Categorical(combos[1], categories=REGION, ordered=True),
        }
    )
    flat_pixs = infer_strat_pixs(df_full, strat_modes, {"sex": 4, "region": 4})

    for (sex, region), pix in zip(combos.itertuples(index=False), flat_pixs):
        sel = (pop_df["sex"] == sex) & (pop_df["region"] == region)
        expected = pop_df[sel].sort_values("age")["P"].to_numpy()
        np.testing.assert_allclose(log_P[pix], np.log(expected))
