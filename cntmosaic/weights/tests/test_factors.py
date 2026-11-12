"""
Tests for DesignWeights, table input, key matching and CategoryWeights
(dev plan: Factor specifications, Table input, Key matching rules).
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from .. import CategoryWeights, DesignWeights, WeightSpecs
from .._table import as_pairs, norm_key
from .helpers import factor_values, make_ctx


# ============================================================================
# DesignWeights
# ============================================================================


class TestDesignWeights:
    def test_values_are_the_factor(self):
        part = pd.DataFrame({"id": [1, 2, 3], "part_w": [0.5, 2.0, 1.5]})
        np.testing.assert_allclose(factor_values(DesignWeights("w"), part), [0.5, 2.0, 1.5])
        np.testing.assert_allclose(
            factor_values(DesignWeights("part_w"), part, rows=[2, 2, 0]), [1.5, 1.5, 0.5]
        )

    def test_default_name_and_override(self):
        assert DesignWeights("part_w").name == "design:w"
        assert DesignWeights("w", name="dw").name == "dw"
        assert DesignWeights.sample_dependent is False

    def test_missing_values_raise(self):
        part = pd.DataFrame({"id": [1, 2], "part_w": [1.0, np.nan]})
        with pytest.raises(ValueError, match="missing"):
            DesignWeights("w").bind(make_ctx(part))

    def test_non_numeric_raises(self):
        part = pd.DataFrame({"id": [1, 2], "part_w": ["a", "b"]})
        with pytest.raises(ValueError, match="numeric"):
            DesignWeights("w").bind(make_ctx(part))

    def test_non_positive_raises_via_specs(self):
        part = pd.DataFrame({"id": [1, 2], "part_w": [1.0, 0.0]})
        with pytest.raises(ValueError, match=r"strictly positive.*\[2\]"):
            WeightSpecs(DesignWeights("w"))._bind(make_ctx(part))

    def test_hash_eq_pickle(self):
        a, b = DesignWeights("w"), DesignWeights("w")
        assert a == b and hash(a) == hash(b)
        assert pickle.loads(pickle.dumps(a)) == a


# ============================================================================
# Table input
# ============================================================================


class TestTableInput:
    def test_dict_and_dataframe_agree(self):
        d = {"UK": 1.0, "DE": 2.0}
        df = pd.DataFrame({"country": ["DE", "UK"], "w": [2.0, 1.0]})
        assert as_pairs(d, default_key_col="country") == as_pairs(df, default_key_col="country")

    def test_named_columns(self):
        df = pd.DataFrame({"k": [1, 2], "a": [1.0, 2.0], "b": [3.0, 4.0]})
        with pytest.raises(ValueError, match="value_col"):
            as_pairs(df, default_key_col="k")
        assert as_pairs(df, default_key_col="x", key_col="k", value_col="b") == ((1, 3.0), (2, 4.0))

    @pytest.mark.parametrize(
        "table, match",
        [
            (pd.DataFrame({"x": [1], "v": [1.0]}), "key column"),
            (pd.DataFrame({"k": [1, 1], "v": [1.0, 2.0]}), "duplicate"),
            ({1: np.nan}, "missing"),
            ({1: "a"}, "not numeric"),
            ({1: -1.0}, "strictly positive"),
            ({1: 0.0}, "strictly positive"),
            ({}, "empty"),
        ],
    )
    def test_errors(self, table, match):
        with pytest.raises(ValueError, match=match):
            as_pairs(table, default_key_col="k")

    def test_zero_allowed_for_targets(self):
        assert as_pairs({1: 0.0, 2: 1.0}, default_key_col="k", allow_zero=True)[0] == (1, 0.0)

    def test_keys_normalised(self):
        assert norm_key(3.0) == 3 and isinstance(norm_key(np.float64(3.0)), int)
        assert norm_key(np.int64(4)) == 4 and norm_key(3.5) == 3.5
        assert norm_key("3") == "3"
        assert as_pairs({3.0: 1.0}, default_key_col="k") == as_pairs({3: 1.0}, default_key_col="k")


# ============================================================================
# CategoryWeights and key matching
# ============================================================================


def _part(values, dtype=None):
    s = pd.Series(values, dtype=dtype)
    return pd.DataFrame({"id": np.arange(len(s)) + 1, "part_x": s})


class TestCategoryWeights:
    def test_lookup(self):
        f = CategoryWeights("x", {"a": 1.0, "b": 2.5})
        np.testing.assert_allclose(factor_values(f, _part(["b", "a", "b"])), [2.5, 1.0, 2.5])
        assert f.name == "category:x"
        assert f.sample_dependent is False

    def test_dict_and_dataframe_equal(self):
        a = CategoryWeights("x", {"a": 1.0, "b": 2.5})
        b = CategoryWeights("x", pd.DataFrame({"x": ["b", "a"], "w": [2.5, 1.0]}))
        assert a == b and hash(a) == hash(b)

    def test_table_is_read_only_and_copied(self):
        d = {"a": 1.0}
        f = CategoryWeights("x", d)
        d["a"] = 99.0
        assert f.table["a"] == 1.0
        with pytest.raises(TypeError):
            f.table["a"] = 2.0
        assert pickle.loads(pickle.dumps(f)) == f

    def test_unknown_value_raises(self):
        f = CategoryWeights("x", {"a": 1.0})
        with pytest.raises(ValueError, match=r"'c' \(x2\)"):
            f.bind(make_ctx(_part(["a", "c", "c"])))

    def test_missing_value_raises(self):
        f = CategoryWeights("x", {"a": 1.0})
        with pytest.raises(ValueError, match="missing"):
            f.bind(make_ctx(_part(["a", None])))

    @pytest.mark.parametrize(
        "values, dtype",
        [
            ([1, 2, 1], "int64"),
            ([1.0, 2.0, 1.0], "float64"),
            ([1, 2, 1], "Int64"),
            ([1, 2, 1], "category"),
        ],
    )
    def test_key_matching_across_dtypes(self, values, dtype):
        f = CategoryWeights("x", {1: 1.0, 2: 3.0})
        np.testing.assert_allclose(factor_values(f, _part(values, dtype)), [1.0, 3.0, 1.0])

    def test_string_does_not_match_int(self):
        f = CategoryWeights("x", {1: 1.0})
        with pytest.raises(ValueError, match="'1'"):
            f.bind(make_ctx(_part(["1"])))

    def test_unused_categories_are_ignored(self):
        part = _part(pd.Categorical(["a", "a"], categories=["a", "zzz"]))
        f = CategoryWeights("x", {"a": 2.0})
        np.testing.assert_allclose(factor_values(f, part), [2.0, 2.0])

    def test_float_with_nan_matches_but_nan_raises(self):
        f = CategoryWeights("x", {1: 1.0, 2: 3.0})
        with pytest.raises(ValueError, match="missing"):
            f.bind(make_ctx(_part([1.0, np.nan, 2.0])))
