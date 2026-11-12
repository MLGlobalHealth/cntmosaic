"""
Tests for PostStratWeights, DayOfWeekWeights and AgePostStratWeights
(dev plan: Factor specifications; D5, D7, D11, D19).
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from ...dataloader import PopulationData
from .. import (
    AgePostStratWeights,
    CategoryWeights,
    DayOfWeekWeights,
    PostStratWeights,
    WeightSpecs,
    WeightWarning,
)
from .helpers import factor_values, make_ctx


def _part(values, **extra):
    df = pd.DataFrame({"id": np.arange(len(values)) + 1, "part_x": values})
    for k, v in extra.items():
        df[k] = v
    return df


# ============================================================================
# PostStratWeights
# ============================================================================


class TestPostStratWeights:
    def test_target_share_over_sample_share(self):
        f = PostStratWeights("x", {"a": 1.0, "b": 1.0})
        values = factor_values(f, _part(["a", "a", "a", "b"]))
        np.testing.assert_allclose(values, [2 / 3, 2 / 3, 2 / 3, 2.0])
        # The weighted mix equals the target mix.
        assert values[:3].sum() == pytest.approx(values[3])

    def test_matches_socialmixr_after_normalisation(self):
        x = ["a"] * 5 + ["b"] * 2 + ["c"] * 3
        targets = {"a": 2.0, "b": 5.0, "c": 3.0}
        part = _part(x)
        w = WeightSpecs(PostStratWeights("x", targets))._bind(make_ctx(part)).weights()
        n_g = pd.Series(x).value_counts()
        socialmixr = np.array([targets[v] / n_g[v] for v in x])
        np.testing.assert_allclose(w, socialmixr * len(x) / socialmixr.sum())

    def test_missing_average_and_raise(self):
        part = _part(["a", None, "b", "b"])
        f = PostStratWeights("x", {"a": 1.0, "b": 1.0}, on_missing="average")
        values = factor_values(f, part)
        assert values[1] == 1.0
        # Shares are taken among the 3 participants with a known value, so the
        # known factors average 1 and the missing value gets the average factor.
        np.testing.assert_allclose(values[[0, 2, 3]], [0.5 / (1 / 3), 0.5 / (2 / 3), 0.5 / (2 / 3)])
        assert values[[0, 2, 3]].mean() == pytest.approx(1.0)
        with pytest.raises(ValueError, match="missing value"):
            PostStratWeights("x", {"a": 1.0, "b": 1.0}).bind(make_ctx(part))

    def test_groups(self):
        f = PostStratWeights("x", {"lo": 1.0, "hi": 3.0}, groups={"lo": [1, 2], "hi": [3]})
        values = factor_values(f, _part([1, 2, 3, 3]))
        np.testing.assert_allclose(values, [0.5, 0.5, 1.5, 1.5])

    def test_overlapping_groups_raise(self):
        with pytest.raises(ValueError, match="overlap"):
            PostStratWeights("x", {"a": 1, "b": 1}, groups={"a": [1, 2], "b": [2, 3]})

    def test_group_labels_must_match_targets(self):
        with pytest.raises(ValueError, match="same labels"):
            PostStratWeights("x", {"a": 1, "c": 1}, groups={"a": [1], "b": [2]})

    def test_dataframe_targets_with_groups_need_key_col(self):
        df = pd.DataFrame({"label": ["lo", "hi"], "t": [1.0, 2.0]})
        with pytest.raises(ValueError, match="key_col"):
            PostStratWeights("x", df, groups={"lo": [1], "hi": [2]})
        f = PostStratWeights("x", df, groups={"lo": [1], "hi": [2]}, key_col="label")
        assert dict(f.table) == {"hi": 2.0, "lo": 1.0}

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError, match="no group"):
            PostStratWeights("x", {"a": 1.0}).bind(make_ctx(_part(["a", "z"])))

    def test_zero_target(self):
        f = PostStratWeights("x", {"a": 1.0, "b": 0.0})
        with pytest.raises(ValueError, match="target 0"):
            f.bind(make_ctx(_part(["a", "b"])))
        with pytest.warns(WeightWarning, match="no participants"):
            bound = f.bind(make_ctx(_part(["a", "a"])))
        np.testing.assert_allclose(bound(np.arange(2)), [1.0, 1.0])

    def test_empty_declared_group_warns_once(self):
        f = PostStratWeights("x", {"a": 1.0, "b": 1.0, "c": 1.0})
        with pytest.warns(WeightWarning) as record:
            bound = f.bind(make_ctx(_part(["a", "a", "b"])))
        assert sum("no participants" in str(w.message) for w in record) == 1
        # Empty group's target is left out of T: a and b share the weight.
        np.testing.assert_allclose(bound(np.arange(3)), [0.75, 0.75, 1.5])
        # Silent in replicates.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            bound(np.array([0, 0, 0]))

    def test_within_cell(self):
        part = _part(["a", "b", "a", "a", "b", "a"])
        cell = [0, 0, 1, 1, 1, 2]  # cell 2 lacks group b
        f = PostStratWeights("x", {"a": 1.0, "b": 1.0}, within="cell")
        values = factor_values(f, part, cell=cell)
        np.testing.assert_allclose(values, [1.0, 1.0, 0.75, 0.75, 1.5, 1.0])

    def test_rows_counted_with_multiplicity(self):
        f = PostStratWeights("x", {"a": 1.0, "b": 1.0})
        values = factor_values(f, _part(["a", "b"]), rows=[0, 0, 0, 1])
        np.testing.assert_allclose(values, [2 / 3, 2 / 3, 2 / 3, 2.0])

    def test_literals_validated(self):
        with pytest.raises(ValueError, match="within"):
            PostStratWeights("x", {"a": 1.0}, within="global")
        with pytest.raises(ValueError, match="on_missing"):
            PostStratWeights("x", {"a": 1.0}, on_missing="ignore")

    def test_value_semantics(self):
        a = PostStratWeights("x", {"a": 1.0}, groups={"a": [1, 2]})
        b = PostStratWeights("x", {"a": 1.0}, groups={"a": (2, 1)})
        assert a == b and hash(a) == hash(b)
        assert pickle.loads(pickle.dumps(a)) == a
        assert a.sample_dependent is True

    def test_differs_from_category_weights(self):
        """Same mapping, different meaning: shares vs fixed factors."""
        x = ["UK"] * 6 + ["DE"] * 2
        mapping = {"UK": 0.5, "DE": 0.5}
        part = _part(x)
        w_ps = WeightSpecs(PostStratWeights("x", mapping))._bind(make_ctx(part)).weights()
        w_cat = WeightSpecs(CategoryWeights("x", mapping))._bind(make_ctx(part)).weights()
        share = lambda w: w[:6].sum() / w.sum()
        assert share(w_ps) == pytest.approx(0.5)
        assert share(w_cat) == pytest.approx(0.75)


# ============================================================================
# DayOfWeekWeights
# ============================================================================


class TestDayOfWeekWeights:
    def _polymod_like(self):
        # 709 weekday, 258 weekend, 44 missing participants (POLYMOD UK counts).
        days = [1] * 709 + [0] * 258 + [np.nan] * 44
        return pd.DataFrame({"id": np.arange(len(days)), "part_dayofweek": days})

    def test_polymod_factors(self):
        # Shares are taken among the 967 participants with a known day.
        values = factor_values(DayOfWeekWeights(), self._polymod_like())
        assert values[0] == pytest.approx((5 / 7) / (709 / 967))
        assert values[0] == pytest.approx(0.9742, abs=1e-4)
        assert values[709] == pytest.approx((2 / 7) / (258 / 967))
        assert values[709] == pytest.approx(1.0709, abs=1e-4)
        assert values[-1] == 1.0

    def test_known_factors_average_one(self):
        # A missing day receives the average factor: the known factors average 1,
        # over the sample ("sample") and within every cell ("cell").
        rng = np.random.default_rng(3)
        days = rng.choice([0, 1, 2, 3, 6], size=400).astype(float)
        days[::9] = np.nan
        cell = rng.integers(0, 4, size=400)
        part = pd.DataFrame({"id": np.arange(400), "part_dayofweek": days})
        known = ~np.isnan(days)
        f_s = factor_values(DayOfWeekWeights(within="sample"), part)
        assert f_s[known].mean() == pytest.approx(1.0)
        f_c = DayOfWeekWeights(within="cell").bind(make_ctx(part, cell=cell))(np.arange(400))
        for c in range(4):
            assert f_c[known & (cell == c)].mean() == pytest.approx(1.0)
        assert np.all(f_s[~known] == 1.0) and np.all(f_c[~known] == 1.0)

    def test_proportional_to_socialmixr_without_missing(self):
        days = [1] * 709 + [0] * 258
        part = pd.DataFrame({"id": np.arange(len(days)), "part_dayofweek": days})
        w = WeightSpecs(DayOfWeekWeights(within="sample"))._bind(make_ctx(part)).weights()
        smr = np.r_[np.full(709, 5 / 709), np.full(258, 2 / 258)]
        np.testing.assert_allclose(w, smr * len(smr) / smr.sum())

    def test_missing_differs_from_socialmixr(self):
        # socialmixr gives a missing day 7 / N, below the mean known factor
        # 7 / (N - |M|); here a missing day gets the mean known factor.
        part = self._polymod_like()
        w = WeightSpecs(DayOfWeekWeights(within="sample"))._bind(make_ctx(part)).weights()
        assert w[-1] == pytest.approx(w[:967].mean())

    def test_cell_without_weekend_warns(self):
        part = pd.DataFrame({"id": np.arange(6), "part_dayofweek": [1, 2, 1, 0, 1, 6]})
        ctx = make_ctx(part, cell=[0, 0, 0, 1, 1, 1])
        with pytest.warns(WeightWarning, match=r"c0 \(no weekend\)"):
            DayOfWeekWeights().bind(ctx)
        with warnings.catch_warnings():
            warnings.simplefilter("error", WeightWarning)
            DayOfWeekWeights(within="sample").bind(ctx)

    def test_labelled_column_single_codes(self):
        labels = pd.DataFrame(
            {"id": np.arange(4), "part_dayofweek": ["weekday", "weekend", "weekday", None]}
        )
        codes = pd.DataFrame({"id": np.arange(4), "part_dayofweek": [1, 0, 1, np.nan]})
        f = DayOfWeekWeights(weekday="weekday", weekend="weekend")
        assert f.weekday == ("weekday",) and f.weekend == ("weekend",)
        np.testing.assert_allclose(
            factor_values(f, labels), factor_values(DayOfWeekWeights(), codes)
        )

    def test_categorical_and_float_codes(self):
        expected = factor_values(
            DayOfWeekWeights(),
            pd.DataFrame({"id": np.arange(4), "part_dayofweek": [1, 0, 2, np.nan]}),
        )
        cat = pd.Categorical([1, 0, 2, np.nan], categories=[0, 1, 2, 3, 4, 5, 6, 7])
        floats = [1.0, 0.0, 2.0, np.nan]
        for values in (cat, floats):
            part = pd.DataFrame({"id": np.arange(4), "part_dayofweek": values})
            np.testing.assert_allclose(factor_values(DayOfWeekWeights(), part), expected)

    @pytest.mark.parametrize("value", ["Weekday", "1", "missing"])
    def test_inexact_codes_raise_with_expected_values(self, value):
        part = pd.DataFrame({"id": [1, 2], "part_dayofweek": [1, value]})
        with pytest.raises(ValueError, match="expected values: 0, 1, 2, 3, 4, 5, 6"):
            DayOfWeekWeights().bind(make_ctx(part))

    def test_weekday_share(self):
        rng = np.random.default_rng(1)
        days = rng.choice([0, 1, 2, 3, 6], size=300, p=[0.1, 0.4, 0.3, 0.1, 0.1]).astype(float)
        days[::17] = np.nan
        cell = rng.integers(0, 3, size=300)
        part = pd.DataFrame({"id": np.arange(300), "part_dayofweek": days})
        known = ~np.isnan(days)
        weekday = np.isin(days, [1, 2, 3, 4, 5])
        for within in ("sample", "cell"):
            w = WeightSpecs(DayOfWeekWeights(within=within))._bind(
                make_ctx(part, cell=cell)
            ).composite()
            if within == "sample":
                share = w[known & weekday].sum() / w[known].sum()
                assert share == pytest.approx(5 / 7)
            else:
                for c in range(3):
                    m = known & (cell == c)
                    assert w[m & weekday].sum() / w[m].sum() == pytest.approx(5 / 7)

    def test_unknown_code_raises(self):
        part = pd.DataFrame({"id": [1, 2], "part_dayofweek": [1, 7]})
        with pytest.raises(ValueError, match="7"):
            DayOfWeekWeights().bind(make_ctx(part))

    def test_missing_raise_option(self):
        part = pd.DataFrame({"id": [1, 2], "part_dayofweek": [1, np.nan]})
        with pytest.raises(ValueError, match="missing"):
            DayOfWeekWeights(on_missing="raise").bind(make_ctx(part))

    def test_custom_coding_and_validation(self):
        part = pd.DataFrame({"id": [1, 2, 3], "part_dow": [1, 6, 7]})
        f = DayOfWeekWeights(col="dow", weekday=(1, 2, 3, 4, 5), weekend=(6, 7))
        values = factor_values(f, part)
        np.testing.assert_allclose(values, [(5 / 7) / (1 / 3), (2 / 7) / (2 / 3), (2 / 7) / (2 / 3)])
        with pytest.raises(ValueError, match="overlap"):
            DayOfWeekWeights(weekday=(1, 2), weekend=(2, 3))
        with pytest.raises(ValueError, match="ratio"):
            DayOfWeekWeights(ratio=(5, 0))


# ============================================================================
# AgePostStratWeights
# ============================================================================


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _single_year_pop(sizes, sex=False):
    ages = np.arange(len(sizes))
    if not sex:
        df = pd.DataFrame({"age": ages, "P": sizes})
        return _quiet(PopulationData, df, size_col="P", age_col="age")
    df = pd.DataFrame(
        {
            "age": np.r_[ages, ages],
            "sex": ["F"] * len(ages) + ["M"] * len(ages),
            "P": np.r_[sizes, sizes] / 2,
        }
    )
    return _quiet(PopulationData, df, size_col="P", age_col="age", strat_var_cols="sex")


def _ages_part(ages, **extra):
    df = pd.DataFrame({"id": np.arange(len(ages)) + 1, "part_age": ages})
    for k, v in extra.items():
        df[k] = v
    return df


class TestAgePostStratWeights:
    def test_formula(self):
        sizes = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        ages = [0, 0, 1, 3, 3, 3]  # age 2 and 4 not observed
        f = AgePostStratWeights(_single_year_pop(sizes))
        values = factor_values(f, _ages_part(ages))
        P = sizes[[0, 1, 3]].sum()  # population over observed ages
        n = len(ages)
        expected = [(sizes[a] / P) / (ages.count(a) / n) for a in ages]
        np.testing.assert_allclose(values, expected)

    def test_strata_summed_and_floor(self):
        sizes = np.array([10.0, 20.0, 30.0])
        a = AgePostStratWeights(_single_year_pop(sizes))
        b = AgePostStratWeights(_single_year_pop(sizes, sex=True))
        assert a == b and hash(a) == hash(b)
        np.testing.assert_allclose(
            factor_values(a, _ages_part([0.4, 1.9, 2.0])),
            factor_values(a, _ages_part([0, 1, 2])),
        )

    def test_banded_population_without_rounding(self):
        df = pd.DataFrame({"lo": [0, 3], "hi": [2, 5], "P": [10.0, 50.0]})
        pop = _quiet(PopulationData, df, size_col="P", age_min_col="lo", age_max_col="hi")
        f = AgePostStratWeights(pop)
        population = dict(f.population)
        assert population[0] == pytest.approx(10 / 3)  # not rounded
        assert population[5] == pytest.approx(50 / 3)
        assert sorted(population) == [0, 1, 2, 3, 4, 5]

    def test_interval_population(self):
        df = pd.DataFrame(
            {
                "grp": pd.Categorical(
                    [pd.Interval(0, 2, closed="left"), pd.Interval(2, 5, closed="left")]
                ),
                "P": [20.0, 30.0],
            }
        )
        pop = _quiet(PopulationData, df, size_col="P", age_grp_col="grp")
        assert AgePostStratWeights(pop).population == (
            (0, 10.0), (1, 10.0), (2, 10.0), (3, 10.0), (4, 10.0)
        )

    def test_open_band_raises(self):
        df = pd.DataFrame(
            {
                "grp": pd.Categorical(
                    [pd.Interval(0, 5, closed="left"), pd.Interval(5, np.inf, closed="left")]
                ),
                "P": [20.0, 30.0],
            }
        )
        try:
            pop = _quiet(PopulationData, df, size_col="P", age_grp_col="grp")
        except Exception:
            pytest.skip("PopulationData rejects open-ended bands itself")
        with pytest.raises(ValueError, match="open-ended"):
            AgePostStratWeights(pop)

    def test_uncovered_age_raises(self):
        f = AgePostStratWeights(_single_year_pop(np.array([10.0, 20.0])))
        with pytest.raises(ValueError, match=r"age\(s\) \[5\] have no population"):
            f.bind(make_ctx(_ages_part([0, 1, 5])))

    def test_zero_population_at_observed_age_raises(self):
        f = AgePostStratWeights(_single_year_pop(np.array([10.0, 0.0])))
        with pytest.raises(ValueError, match="target 0"):
            f.bind(make_ctx(_ages_part([0, 1])))

    def test_needs_single_year_ages(self):
        f = AgePostStratWeights(_single_year_pop(np.array([10.0])))
        part = pd.DataFrame({"id": [1], "part_age_grp": [pd.Interval(0, 5, closed="left")]})
        with pytest.raises(ValueError, match="age groups"):
            f.bind(make_ctx(part))
        part = pd.DataFrame({"id": [1], "part_age_min": [0], "part_age_max": [4]})
        with pytest.raises(ValueError, match="age ranges"):
            f.bind(make_ctx(part))

    def test_no_empty_group_warnings(self):
        f = AgePostStratWeights(_single_year_pop(np.arange(1, 101, dtype=float)))
        with warnings.catch_warnings():
            warnings.simplefilter("error", WeightWarning)
            f.bind(make_ctx(_ages_part([10, 20, 30])))

    def test_within_cell_equals_sample_without_strata(self):
        f_s = AgePostStratWeights(_single_year_pop(np.arange(1, 21, dtype=float)))
        f_c = AgePostStratWeights(_single_year_pop(np.arange(1, 21, dtype=float)), within="cell")
        ages = [1, 1, 2, 5, 5, 5, 12, 14, 14, 19]
        cell = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  # cells are age groups [0,10), [10,20)
        part = _ages_part(ages)
        w_s = WeightSpecs(f_s)._bind(make_ctx(part, cell=cell)).weights()
        with warnings.catch_warnings():
            # Every cell lacks most single years of age; that is not warned about.
            warnings.simplefilter("error", WeightWarning)
            w_c = WeightSpecs(f_c)._bind(make_ctx(part, cell=cell)).weights()
        np.testing.assert_allclose(w_s, w_c)

    def test_pickle(self):
        f = AgePostStratWeights(_single_year_pop(np.array([1.0, 2.0])))
        g = pickle.loads(pickle.dumps(f))
        assert g == f and g.population == f.population
