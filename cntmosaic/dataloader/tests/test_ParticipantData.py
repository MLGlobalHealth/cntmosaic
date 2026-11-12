"""
Tests for ParticipantData dataclass.

This module tests the validation, properties, and methods of the ParticipantData class,
ensuring robust handling of participant survey data.

Test organisation
-----------------
1. TestCoreContracts  — column renaming/standardisation, categorical conversion,
                        NaN dropping, schema correctness, original-df immutability.
2. TestInputValidation — TypeError/ValueError/KeyError on bad inputs.
3. TestAccessorMethods — properties and methods that expose processed data.
4. TestEdgeCases       — genuine boundary conditions (age=0, float ages, age groups).
"""

import numpy as np
import pandas as pd
import pytest

from ..containers._ParticipantData import ParticipantData
from .fixtures import df_part_age_grps, df_part_one_year, df_part_age_min_max

# =====================
# 1. Core Contracts
# =====================


class TestCoreContracts:
    """Test the fundamental data-pipeline guarantees of ParticipantData."""

    def test_column_renaming_with_age_col(self, df_part_one_year):
        """Columns are renamed: id_col→'id', age_col→'part_age'."""
        part_data = ParticipantData(df_part_one_year, id_col="id", age_col="age")
        assert part_data.data.columns.tolist() == ["id", "part_age"]

    def test_column_renaming_with_age_grp_col(self, df_part_age_grps):
        """Columns are renamed: id_col→'id', age_grp_col→'part_age_grp'."""
        part_data = ParticipantData(
            df_part_age_grps, id_col="id", age_grp_col="age_grp"
        )
        assert part_data.data.columns.tolist() == ["id", "part_age_grp"]

    def test_column_renaming_with_age_min_max_cols(self, df_part_age_min_max):
        """age_min/max_col are renamed and part_age_grp is synthesised as ordered categorical."""
        part_data = ParticipantData(
            df_part_age_min_max,
            id_col="id",
            age_min_col="age_min",
            age_max_col="age_max",
        )
        assert "id" in part_data.data.columns
        assert "part_age_min" in part_data.data.columns
        assert "part_age_max" in part_data.data.columns
        assert "part_age_grp" in part_data.data.columns
        import pandas as pd
        assert isinstance(part_data.data["part_age_grp"].dtype, pd.CategoricalDtype)
        assert part_data.data["part_age_grp"].cat.ordered

    def test_column_renaming_single_strat_var(self, df_part_one_year):
        """A string strat_var_col is normalised to a list and renamed with 'part_' prefix."""
        part_data = ParticipantData(
            df_part_one_year, id_col="id", age_col="age", strat_var_cols="sex"
        )
        assert part_data.strat_vars == ["sex"]
        assert part_data.data.columns.tolist() == ["id", "part_age", "part_sex"]

    def test_column_renaming_multiple_strat_vars(self, df_part_one_year):
        """Multiple strat_var_cols are all renamed with 'part_' prefix."""
        part_data = ParticipantData(
            df_part_one_year,
            id_col="id",
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
        )
        assert part_data.strat_vars == ["sex", "hhsize"]
        assert part_data.data.columns.tolist() == [
            "id",
            "part_age",
            "part_sex",
            "part_hhsize",
        ]

    def test_column_renaming_with_repeat_col(self, df_part_one_year):
        """repeat_col is renamed to 'part_repeat'."""
        part_data = ParticipantData(
            df_part_one_year, id_col="id", age_col="age", repeat_col="repeat"
        )
        assert part_data.data.columns.tolist() == ["id", "part_age", "part_repeat"]

    def test_categorical_conversion(self, df_part_one_year):
        """Object-type stratification columns are converted to categorical dtype."""
        part_data = ParticipantData(
            df_part_one_year,
            id_col="id",
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
        )
        assert part_data.data["part_sex"].dtype.name == "category"
        assert part_data.data["part_hhsize"].dtype.name == "category"

    @pytest.mark.parametrize(
        "nan_col, df_factory",
        [
            (
                "id",
                lambda: pd.DataFrame(
                    {"id": [1, 2, np.nan, 4], "age": [25, 34, 45, 52]}
                ),
            ),
            (
                "age",
                lambda: pd.DataFrame({"id": [1, 2, 3, 4], "age": [25, np.nan, 45, 52]}),
            ),
            (
                "sex",
                lambda: pd.DataFrame(
                    {
                        "id": [1, 2, 3, 4],
                        "age": [25, 34, 45, 52],
                        "sex": ["M", np.nan, "M", "F"],
                    }
                ),
            ),
        ],
    )
    def test_nan_rows_dropped_with_warning(self, nan_col, df_factory):
        """Rows with NaN in any required column are dropped with a UserWarning."""
        df = df_factory()
        kwargs = dict(id_col="id", age_col="age")
        if nan_col == "sex":
            kwargs["strat_var_cols"] = "sex"

        with pytest.warns(UserWarning, match="Dropped 1 row"):
            part_data = ParticipantData(data=df, **kwargs)
        assert part_data.n == 3

    def test_original_dataframe_not_mutated(self, df_part_one_year):
        """ParticipantData works on a copy; the caller's DataFrame is unchanged."""
        original_cols = df_part_one_year.columns.tolist()
        original_shape = df_part_one_year.shape
        ParticipantData(
            df_part_one_year,
            id_col="id",
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
        )
        assert df_part_one_year.columns.tolist() == original_cols
        assert df_part_one_year.shape == original_shape

    def test_amb_cnt_col_retained(self):
        """An explicitly provided amb_cnt_col is kept in the output DataFrame."""
        df = pd.DataFrame({"id": [1, 2, 3], "age": [25, 34, 45], "grp_cnt": [0, 1, 0]})
        part_data = ParticipantData(
            data=df, id_col="id", age_col="age", amb_cnt_col="grp_cnt"
        )
        assert part_data.amb_cnt_col == "grp_cnt"
        assert "grp_cnt" in part_data.data.columns
        assert part_data.data["grp_cnt"].tolist() == [0, 1, 0]


# =====================
# 2. Input Validation
# =====================


class TestInputValidation:
    """Test TypeError/ValueError/KeyError on bad inputs."""

    def test_invalid_dataframe_type(self):
        """Non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
            ParticipantData(data=[1, 2, 3], id_col="id", age_col="age")

    def test_missing_age_specification(self):
        """Neither age_col nor age_grp_col raises ValueError."""
        df = pd.DataFrame({"id": [1, 2, 3], "age": [25, 34, 45]})
        with pytest.raises(ValueError, match="Must specify exactly one"):
            ParticipantData(data=df, id_col="id")

    def test_non_exclusive_age_spec(self):
        """Both age_col and age_grp_col raises ValueError."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "age": [25, 34, 45],
                "age_group": pd.IntervalIndex.from_tuples(
                    [(20, 30), (30, 40), (40, 50)]
                ),
            }
        )
        with pytest.raises(
            ValueError, match="Age specification forms are mutually exclusive"
        ):
            ParticipantData(
                data=df, id_col="id", age_col="age", age_grp_col="age_group"
            )

    def test_missing_id_column(self):
        """Missing ID column raises KeyError with informative message."""
        df = pd.DataFrame({"participant_id": [1, 2, 3], "age": [25, 34, 45]})
        with pytest.raises(KeyError, match="Missing participant ID column 'id'"):
            ParticipantData(data=df, id_col="id", age_col="age")

    def test_missing_age_column(self):
        """Missing age column raises KeyError with informative message."""
        df = pd.DataFrame({"id": [1, 2, 3], "participant_age": [25, 34, 45]})
        with pytest.raises(
            KeyError, match=r"Missing participant age column\(s\) 'age'"
        ):
            ParticipantData(data=df, id_col="id", age_col="age")

    def test_missing_strat_col(self):
        """Missing stratification column raises KeyError with informative message."""
        df = pd.DataFrame(
            {"id": [1, 2, 3], "age": [25, 34, 45], "sex": ["M", "F", "M"]}
        )
        with pytest.raises(
            KeyError, match="strat_var_cols '\\['sex', 'hhsize'\\]' is specified"
        ):
            ParticipantData(
                data=df, id_col="id", age_col="age", strat_var_cols=["sex", "hhsize"]
            )

    def test_duplicate_participant_ids(self):
        """Duplicate IDs raise ValueError."""
        df = pd.DataFrame({"id": [1, 2, 2, 3], "age": [25, 34, 45, 52]})
        with pytest.raises(ValueError, match="duplicate participant ID"):
            ParticipantData(data=df, id_col="id", age_col="age")

    def test_negative_ages(self):
        """Negative ages raise ValueError."""
        df = pd.DataFrame({"id": [1, 2, 3, 4], "age": [25, -5, 45, 52]})
        with pytest.raises(ValueError, match="negative values"):
            ParticipantData(data=df, id_col="id", age_col="age")

    def test_non_numeric_ages(self):
        """Non-numeric ages raise ValueError."""
        df = pd.DataFrame({"id": [1, 2, 3, 4], "age": ["25", "34", "45", "52"]})
        with pytest.raises(ValueError, match="must contain numeric values"):
            ParticipantData(data=df, id_col="id", age_col="age")


# =====================
# 3. Accessor Methods
# =====================


class TestAccessorMethods:
    """Test properties and methods that expose processed data in specific formats."""

    def test_age_range_property(self):
        """age_range returns correct (min, max) tuple."""
        df = pd.DataFrame({"id": [1, 2, 3, 4], "age": [18, 25, 65, 42]})
        part_data = ParticipantData(data=df, id_col="id", age_col="age")
        assert part_data.age_range == (18, 65)

    def test_age_range_raises_with_age_groups(self):
        """age_range raises ValueError when age_grp_col is used."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "age_group": pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15)]),
            }
        )
        df["age_group"] = df["age_group"].astype("category")
        part_data = ParticipantData(data=df, id_col="id", age_grp_col="age_group")
        with pytest.raises(ValueError, match="only available when using 'age_col'"):
            _ = part_data.age_range

    def test_get_sample_sizes_by_age(self):
        """get_sample_sizes() groups by age and returns correct counts."""
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6], "age": [25, 25, 34, 34, 34, 45]})
        part_data = ParticipantData(data=df, id_col="id", age_col="age")
        sample_sizes = part_data.get_sample_sizes()
        assert isinstance(sample_sizes, pd.DataFrame)
        assert np.array_equal(sample_sizes["part_age"].values, np.array([25, 34, 45]))
        assert np.array_equal(sample_sizes["N"].values, np.array([2, 3, 1]))

    def test_get_sample_sizes_with_age_groups(self):
        """get_sample_sizes() works correctly with age group intervals."""
        intervals = pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (5, 10), (10, 15)])
        df = pd.DataFrame({"id": [1, 2, 3, 4], "age_group": intervals})
        df["age_group"] = df["age_group"].astype("category")
        part_data = ParticipantData(data=df, id_col="id", age_grp_col="age_group")
        sample_sizes = part_data.get_sample_sizes()
        assert isinstance(sample_sizes, pd.DataFrame)
        assert len(sample_sizes) == 3  # Three unique intervals

    def test_summary_with_age_col(self):
        """summary() returns all expected keys and correct values for age_col case."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "age": [18, 25, 45, 65],
                "gender": ["M", "F", "M", "F"],
            }
        )
        part_data = ParticipantData(
            data=df, id_col="id", age_col="age", strat_var_cols="gender"
        )
        summary = part_data.summary()
        assert isinstance(summary, dict)
        assert summary["n"] == 4
        assert summary["id_col"] == "id"
        assert summary["age_col"] == "age"
        assert summary["age_grp_col"] is None
        assert summary["age_range"] == (18, 65)
        assert summary["strat_vars"] == ["gender"]

    def test_summary_with_age_groups(self):
        """summary() omits age_range when age_grp_col is used."""
        df = pd.DataFrame(
            {
                "pid": [1, 2, 3],
                "age_group": pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15)]),
            }
        )
        df["age_group"] = df["age_group"].astype("category")
        part_data = ParticipantData(data=df, id_col="pid", age_grp_col="age_group")
        summary = part_data.summary()
        assert summary["n"] == 3
        assert summary["id_col"] == "pid"
        assert summary["age_col"] is None
        assert summary["age_grp_col"] == "age_group"
        assert "age_range" not in summary
        assert summary["strat_vars"] == []

    def test_get_strat_var_schema(self, df_part_one_year):
        """get_strat_var_schema() returns correct categories and integer codes."""
        part_data = ParticipantData(
            df_part_one_year,
            id_col="id",
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
        )
        schema = part_data.get_strat_var_schema()
        assert schema["sex"] == {"categories": ["F", "M"], "codes": [0, 1]}
        assert schema["hhsize"] == {
            "categories": ["1", "2", "3", "4", "5+"],
            "codes": [0, 1, 2, 3, 4],
        }

    def test_get_strat_vars_with_prefix(self, df_part_one_year):
        """get_strat_vars(prefix=True) returns names with 'part_' prepended."""
        part_data = ParticipantData(
            df_part_one_year,
            id_col="id",
            age_col="age",
            strat_var_cols=["sex", "hhsize"],
        )
        assert part_data.get_strat_vars(prefix=True) == ["part_sex", "part_hhsize"]
        assert part_data.get_strat_vars(prefix=False) == ["sex", "hhsize"]

    def test_get_strat_vars_empty(self):
        """get_strat_vars() returns empty list when no stratification variables specified."""
        df = pd.DataFrame({"id": [1, 2, 3], "age": [25, 34, 45]})
        part_data = ParticipantData(data=df, id_col="id", age_col="age")
        assert part_data.get_strat_vars() == []


# =====================
# 4. Edge Cases
# =====================


class TestEdgeCases:
    """Test genuine boundary conditions."""

    def test_single_participant(self):
        """Single-row DataFrame is valid; age_range is a degenerate tuple."""
        df = pd.DataFrame({"id": [1], "age": [25]})
        part_data = ParticipantData(data=df, id_col="id", age_col="age")
        assert part_data.n == 1
        assert part_data.age_range == (25, 25)

    def test_age_zero(self):
        """Age 0 is a valid non-negative age."""
        df = pd.DataFrame({"id": [1, 2, 3], "age": [0, 5, 10]})
        part_data = ParticipantData(data=df, id_col="id", age_col="age")
        assert part_data.age_range == (0, 10)

    def test_float_ages(self):
        """Float ages are accepted and preserved."""
        df = pd.DataFrame({"id": [1, 2, 3], "age": [25.5, 34.2, 45.8]})
        part_data = ParticipantData(data=df, id_col="id", age_col="age")
        assert part_data.age_range == (25.5, 45.8)


# =====================
# 5. Removed API (dev plan D15, D28)
# =====================


class TestRemovedWeightApi:
    """`weight_col` and its normalisation helpers were removed cleanly."""

    def test_weight_col_argument_is_gone(self):
        df = pd.DataFrame({"id": [1, 2], "age": [20, 30], "w": [1.0, 2.0]})
        with pytest.raises(TypeError, match="weight_col"):
            ParticipantData(data=df, id_col="id", age_col="age", weight_col="w")

    def test_normalisation_methods_are_gone(self):
        assert not hasattr(ParticipantData, "normalize_weights")
        assert not hasattr(ParticipantData, "check_weights_normalized")


# =====================
# 6. Auxiliary columns (dev plan: ParticipantData(aux_cols=...))
# =====================


class TestAuxCols:
    """Auxiliary columns are kept for weighting: prefixed, not strata, NaN kept."""

    def _df(self):
        return pd.DataFrame(
            {
                "pid": [1, 2, 3, 4],
                "age": [10, 20, np.nan, 40],
                "sex": ["F", "M", "F", "M"],
                "dayofweek": [1.0, np.nan, 6.0, 0.0],
                "country": ["UK", None, "DE", "UK"],
                "w": [1.5, 2.0, 0.5, 1.0],
            }
        )

    def test_kept_renamed_and_not_strata(self):
        part = ParticipantData(
            self._df(), id_col="pid", age_col="age", strat_var_cols="sex",
            aux_cols=["dayofweek", "country", "w"],
        )
        for col in ("part_dayofweek", "part_country", "part_w"):
            assert col in part.data.columns
        assert part.strat_vars == ["sex"]
        assert part.get_aux_cols() == ["part_dayofweek", "part_country", "part_w"]
        assert part.get_aux_cols(prefix=False) == ["dayofweek", "country", "w"]

    def test_missing_values_kept_and_rows_aligned(self):
        with pytest.warns(UserWarning, match="Dropped 1 row"):
            part = ParticipantData(
                self._df(), id_col="pid", age_col="age", aux_cols=["dayofweek", "w"]
            )
        # Only the row with a missing age (a required column) is dropped.
        assert part.data["id"].tolist() == [1, 2, 4]
        assert part.data["part_dayofweek"].isna().tolist() == [False, True, False]
        assert part.data["part_w"].tolist() == [1.5, 2.0, 1.0]

    def test_object_aux_becomes_categorical_with_nan(self):
        part = ParticipantData(self._df(), id_col="pid", age_col="age", aux_cols="country")
        assert isinstance(part.data["part_country"].dtype, pd.CategoricalDtype)
        assert part.data["part_country"].isna().sum() == 1

    def test_without_aux_cols_unchanged(self):
        part = ParticipantData(self._df(), id_col="pid", age_col="age")
        assert part.get_aux_cols() == []
        assert "part_dayofweek" not in part.data.columns

    def test_duplicate_index_is_aligned(self):
        df = self._df().dropna(subset=["age"])
        df.index = [0, 0, 1]
        part = ParticipantData(df, id_col="pid", age_col="age", aux_cols="w")
        assert part.data["part_w"].tolist() == [1.5, 2.0, 1.0]

    def test_missing_column(self):
        with pytest.raises(KeyError, match="aux_cols"):
            ParticipantData(self._df(), id_col="pid", age_col="age", aux_cols="nope")

    def test_duplicates(self):
        with pytest.raises(ValueError, match="more than once"):
            ParticipantData(self._df(), id_col="pid", age_col="age", aux_cols=["w", "w"])

    @pytest.mark.parametrize("aux", ["sex", "age", "pid"])
    def test_overlap_with_declared_columns(self, aux):
        with pytest.raises(ValueError, match="already declared"):
            ParticipantData(
                self._df(), id_col="pid", age_col="age", strat_var_cols="sex",
                aux_cols=aux,
            )

    def test_reserved_name(self):
        df = self._df().rename(columns={"w": "repeat"})
        with pytest.raises(ValueError, match="reserved"):
            ParticipantData(df, id_col="pid", age_col="age", aux_cols="repeat")

    def test_clash_after_prefixing(self):
        df = self._df()
        df["part_sex"] = df["sex"]
        with pytest.raises(ValueError, match="clash"):
            ParticipantData(
                df, id_col="pid", age_col="age", strat_var_cols="sex", aux_cols="part_sex"
            )
