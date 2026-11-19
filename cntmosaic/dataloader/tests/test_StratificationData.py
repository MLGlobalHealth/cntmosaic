"""
Tests for StratificationData dataclass.

Test organisation
-----------------
1. TestCoreContracts  — init contracts, from_counts proportion computation,
                        original-df immutability.
2. TestInputValidation — ValueError on bad age specifications and invalid proportions.
3. TestAccessorMethods — get_strat_vars, get_strat_var_schema, _age_groupby_cols.
4. TestDemopty         — compute_demopty for PARTIAL/FULL/mixed modes, including
                         the age_min/max variant.
"""

import numpy as np
import pandas as pd
import pytest

from ..._types import StratMode
from ..containers._StratificationData import StratificationData
from .fixtures import df_strat_age_min_max, df_strat_count, df_strat_multi_var, df_strat_single_var


# =====================
# 1. Core Contracts
# =====================


class TestCoreContracts:
    """Test the fundamental data-pipeline guarantees of StratificationData."""

    def test_init_with_age_min_max(self, df_strat_age_min_max):
        """StratificationData works with age_min_col/age_max_col; age_col is None."""
        strat_data = StratificationData(
            data=df_strat_age_min_max,
            age_min_col="age_min",
            age_max_col="age_max",
            strat_var_cols="stratum",
            prop_col="prop",
        )
        assert strat_data.age_min_col == "age_min"
        assert strat_data.age_max_col == "age_max"
        assert strat_data.age_col is None

    def test_init_single_var(self, df_strat_single_var):
        """Basic init with age_col and single strat_var_col produces categorical column."""
        strat_data = StratificationData(
            data=df_strat_single_var,
            age_col="age",
            strat_var_cols="stratum",
            prop_col="prop",
        )
        assert strat_data.data["stratum"].dtype.name == "category"
        assert strat_data.data.shape == (6, 3)  # 3 ages × 2 strata, 3 columns

    def test_from_counts_computes_proportions(self, df_strat_count):
        """from_counts() normalises counts to proportions that sum to 1.0 per age group."""
        strat_data = StratificationData.from_counts(
            data=df_strat_count,
            age_col="age",
            strat_var_cols="stratum",
            count_col="count",
        )
        group_sums = strat_data.data.groupby("age")["prop"].sum()
        assert np.allclose(group_sums.values, 1.0)

    def test_original_dataframe_not_mutated(self, df_strat_single_var):
        """StratificationData works on a copy; the caller's DataFrame is unchanged."""
        original_cols = df_strat_single_var.columns.tolist()
        original_shape = df_strat_single_var.shape
        StratificationData(
            data=df_strat_single_var,
            age_col="age",
            strat_var_cols="stratum",
            prop_col="prop",
        )
        assert df_strat_single_var.columns.tolist() == original_cols
        assert df_strat_single_var.shape == original_shape


# =====================
# 2. Input Validation
# =====================


class TestInputValidation:
    """Test ValueError on bad inputs."""

    def test_missing_age_specification(self):
        """No age_col and no age_min/max raises ValueError."""
        df = pd.DataFrame({"stratum": ["A", "B"], "prop": [0.4, 0.6]})
        df["stratum"] = pd.Categorical(df["stratum"], categories=["A", "B"])
        with pytest.raises(ValueError, match="Must specify an age representation"):
            StratificationData(data=df, strat_var_cols="stratum", prop_col="prop")

    def test_age_col_and_age_min_max_raises(self, df_strat_age_min_max):
        """Providing age_col and age_min/max raises ValueError."""
        df = df_strat_age_min_max.copy()
        df["age"] = df["age_min"]
        with pytest.raises(ValueError, match="Age specification forms are mutually exclusive"):
            StratificationData(
                data=df,
                age_col="age",
                age_min_col="age_min",
                age_max_col="age_max",
                strat_var_cols="stratum",
                prop_col="prop",
            )

    def test_age_min_without_age_max_raises(self, df_strat_age_min_max):
        """age_min_col without age_max_col raises ValueError."""
        with pytest.raises(ValueError, match="Both 'age_min_col' and 'age_max_col' must be specified"):
            StratificationData(
                data=df_strat_age_min_max,
                age_min_col="age_min",
                strat_var_cols="stratum",
                prop_col="prop",
            )

    def test_proportions_not_summing_to_one_raises(self):
        """Proportions that don't sum to 1.0 per age group raise ValueError."""
        df = pd.DataFrame(
            {
                "age": [0, 0, 1, 1],
                "stratum": ["A", "B", "A", "B"],
                "prop": [0.4, 0.4, 0.5, 0.5],  # age=0 sums to 0.8, not 1.0
            }
        )
        df["stratum"] = pd.Categorical(df["stratum"], categories=["A", "B"])
        with pytest.raises(ValueError, match="must sum to 1.0 within each age group"):
            StratificationData(data=df, age_col="age", strat_var_cols="stratum", prop_col="prop")


# =====================
# 3. Accessor Methods
# =====================


class TestAccessorMethods:
    """Test properties and methods that expose processed data."""

    def test_get_strat_vars(self, df_strat_multi_var):
        """get_strat_vars() returns the list of stratification variable names."""
        strat_data = StratificationData(
            data=df_strat_multi_var,
            age_col="age",
            strat_var_cols=["stratum1", "stratum2"],
            prop_col="prop",
        )
        strat_vars = strat_data.get_strat_vars()
        assert strat_vars == ["stratum1", "stratum2"]

    def test_get_strat_var_schema(self, df_strat_multi_var):
        """get_strat_var_schema() returns correct categories and integer codes."""
        strat_data = StratificationData(
            data=df_strat_multi_var,
            age_col="age",
            strat_var_cols=["stratum1", "stratum2"],
            prop_col="prop",
        )
        schema = strat_data.get_strat_var_schema()
        expected_schema = {
            "stratum1": {"categories": ["A", "B"], "codes": [0, 1]},
            "stratum2": {"categories": ["X", "Y", "Z"], "codes": [0, 1, 2]},
        }
        assert schema == expected_schema

    def test_age_groupby_cols_property(self, df_strat_age_min_max):
        """_age_groupby_cols returns [age_min_col, age_max_col] for range form."""
        strat_data = StratificationData(
            data=df_strat_age_min_max,
            age_min_col="age_min",
            age_max_col="age_max",
            strat_var_cols="stratum",
            prop_col="prop",
        )
        assert strat_data._age_groupby_cols == ["age_min", "age_max"]


# =====================
# 4. Demopty
# =====================


class TestDemopty:
    """Test compute_demopty for PARTIAL, FULL, mixed modes, and age_min/max form."""

    def test_partial_prop_single(self, df_strat_single_var):
        strat_data = StratificationData(
            data=df_strat_single_var,
            age_col="age",
            strat_var_cols="stratum",
            prop_col="prop",
        )
        strat_modes = {"stratum": StratMode.PARTIAL}
        props = strat_data.compute_demopty(strat_modes=strat_modes)
        assert props.shape == (2, 3, 1)
        expected = np.array([[0.3, 0.4, 0.2], [0.7, 0.6, 0.8]])[:, :, np.newaxis]
        np.testing.assert_allclose(props, expected)

    def test_partial_prop_multi(self, df_strat_multi_var):
        strat_data = StratificationData(
            data=df_strat_multi_var,
            age_col="age",
            strat_var_cols=["stratum1", "stratum2"],
            prop_col="prop",
        )
        strat_modes = {"stratum1": StratMode.PARTIAL, "stratum2": StratMode.PARTIAL}
        props = strat_data.compute_demopty(strat_modes=strat_modes)
        assert props.shape == (6, 3, 1)

        expected = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3],
                [0.1, 0.2, 0.1],
                [0.1, 0.2, 0.1],
                [0.1, 0.1, 0.1],
                [0.5, 0.1, 0.1],
            ]
        )[:, :, np.newaxis]  # shape (6, 3, 1)
        np.testing.assert_allclose(props, expected)

    def test_full_prop_single(self, df_strat_single_var):
        strat_data = StratificationData(
            data=df_strat_single_var,
            age_col="age",
            strat_var_cols="stratum",
            prop_col="prop",
        )
        strat_modes = {"stratum": StratMode.FULL}
        props = strat_data.compute_demopty(strat_modes=strat_modes)
        assert props.shape == (4, 3, 3)

        expected = np.array([0.3, 0.7, 0.4, 0.6, 0.2, 0.8]).reshape((2, 3), order="F")
        expected = (expected[:, None, :, None] * expected[None, :, None, :]).reshape(
            4, 3, 3
        )
        np.testing.assert_allclose(props, expected)

    def test_full_prop_multi(self, df_strat_multi_var):
        strat_data = StratificationData(
            data=df_strat_multi_var,
            age_col="age",
            strat_var_cols=["stratum1", "stratum2"],
            prop_col="prop",
        )
        strat_modes = {"stratum1": StratMode.FULL, "stratum2": StratMode.FULL}
        props = strat_data.compute_demopty(strat_modes=strat_modes)
        assert props.shape == (36, 3, 3)

        prop_array = np.array(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],  # (age=0, A-X, A-Y, A-Z, B-X, B-Y, B-Z)
                [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],  # (age=1, A-X, A-Y, A-Z, B-X, B-Y, B-Z)
                [0.3, 0.3, 0.1, 0.1, 0.1, 0.1],  # (age=2, A-X, A-Y, A-Z, B-X, B-Y, B-Z)
            ]
        ).T  # shape (3, 6)

        # Rows follow the full labels, interleaved per variable:
        # (stratum1, stratum1', stratum2, stratum2').
        Q = prop_array.reshape(2, 3, 3)  # (stratum1, stratum2, age)
        expected = np.einsum("ika,jlb->ijklab", Q, Q).reshape(36, 3, 3)
        np.testing.assert_allclose(props, expected)

    def test_mixed_prop_multi(self, df_strat_multi_var):
        strat_data = StratificationData(
            data=df_strat_multi_var,
            age_col="age",
            strat_var_cols=["stratum1", "stratum2"],
            prop_col="prop",
        )
        strat_modes = {"stratum1": StratMode.FULL, "stratum2": StratMode.PARTIAL}
        props = strat_data.compute_demopty(strat_modes=strat_modes)
        assert props.shape == (12, 3, 3)

        prop_s1 = np.array(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],  # (age=0, A-X, A-Y, A-Z, B-X, B-Y, B-Z)
                [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],  # (age=1, A-X, A-Y, A-Z, B-X, B-Y, B-Z)
                [0.3, 0.3, 0.1, 0.1, 0.1, 0.1],  # (age=2, A-X, A-Y, A-Z, B-X, B-Y, B-Z)
            ]
        ).T  # shape (6, 3)

        prop_s2 = np.array(
            [
                [0.3, 0.7],  # (age=0, s1=A, B)
                [0.6, 0.4],  # (age=1, s1=A, B)
                [0.7, 0.3],  # (age=2, s1=A, B)
            ]
        ).T  # shape (2, 3)

        # Rows follow the full labels: (stratum1, stratum1', stratum2).
        Q_source = prop_s1.reshape(2, 3, 3)  # (stratum1, stratum2, age)
        expected = np.einsum("ika,jb->ijkab", Q_source, prop_s2).reshape(12, 3, 3)
        np.testing.assert_allclose(props, expected)

    def test_compute_demopty_partial_age_min_max(self, df_strat_age_min_max):
        """compute_demopty works for PARTIAL mode with age_min/max age representation."""
        strat_data = StratificationData(
            data=df_strat_age_min_max,
            age_min_col="age_min",
            age_max_col="age_max",
            strat_var_cols="stratum",
            prop_col="prop",
        )
        strat_modes = {"stratum": StratMode.PARTIAL}
        props = strat_data.compute_demopty(strat_modes=strat_modes)
        assert props.shape == (2, 3, 1)
