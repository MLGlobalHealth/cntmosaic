"""
SocialMix with weight_specs (dev plan, Core item; D1, D8, D10, D14, D27).

- Weighting is opt-in; with weight_specs, cint()/rate() default to the
  weighted ratio estimator M[c,d] = sum_i w_i y_id / sum_i w_i per cell.
- The unweighted estimate stays available (weighted=False), with one cache per
  value of ``weighted``.
- The bootstrap refuses weighted fits until the weighted bootstrap exists.
- Prem2 (which shares the SocialMix loader) is unaffected.
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from ...dataloader import ContactData, ParticipantData, PopulationData
from ...utils import AgeGroupSpecs
from ...weights import WeightResult, WeightSpecs
from ...weights.tests.factors import ConstantFactor, LinearAgeFactor
from ..classical._Prem2 import Prem2
from ..classical._SocialMix import SocialMix

SPECS = AgeGroupSpecs(0, 19, 10)  # [0, 10), [10, 20)


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _survey(sex=False):
    """Participants aged 0-19 (three per year); contacts vary with age so a
    weight that varies with age changes the estimate."""
    ages = np.tile(np.arange(20), 3)
    df_part = pd.DataFrame({"pid": np.arange(len(ages)), "age": ages})
    kwargs = {}
    if sex:
        df_part["sex"] = np.where(df_part["pid"] % 2 == 0, "F", "M")
        kwargs["strat_var_cols"] = "sex"
    rows = []
    for pid, age in zip(df_part["pid"], df_part["age"]):
        rows += [{"pid": pid, "cnt_age": 5}] * (1 + age % 3)  # contacts in [0, 10)
        rows += [{"pid": pid, "cnt_age": 15}] * (age // 4)  # contacts in [10, 20)
    df_cnt = pd.DataFrame(rows)
    part = _quiet(ParticipantData, df_part, id_col="pid", age_col="age", **kwargs)
    cnt = _quiet(ContactData, df_cnt, id_col="pid", age_col="cnt_age")
    df_pop = pd.DataFrame({"age": np.arange(20), "P": 100.0})
    pop = _quiet(PopulationData, df_pop, size_col="P", age_col="age")
    return part, cnt, pop, df_part, df_cnt


def _expected_weighted_cint(df_part, df_cnt, weight_of_age, group_cols=()):
    """Weighted ratio estimator computed directly from the raw tables."""
    part = df_part.copy()
    part["c"] = (part["age"] >= 10).astype(int)
    part["w"] = weight_of_age(part["age"].to_numpy(dtype=float))
    cnt = df_cnt.copy()
    cnt["d"] = (cnt["cnt_age"] >= 10).astype(int)
    y = cnt.groupby(["pid", "d"]).size().unstack(fill_value=0).reindex(
        part["pid"], fill_value=0
    )
    out = {}
    keys = [tuple()] if not group_cols else list(part.groupby(list(group_cols)).groups)
    for key in keys:
        sel = np.ones(len(part), bool) if not group_cols else (
            part[list(group_cols)].apply(tuple, axis=1) == (key if isinstance(key, tuple) else (key,))
        ).to_numpy()
        M = np.zeros((2, 2))
        for c in (0, 1):
            m = sel & (part["c"].to_numpy() == c)
            w = part.loc[m, "w"].to_numpy()
            M[c] = (w[:, None] * y.to_numpy()[m]).sum(0) / w.sum()
        out[key] = M
    return out


class TestOptIn:
    def test_unweighted_by_default(self):
        part, cnt, pop, *_ = _survey()
        sm = _quiet(SocialMix, part, cnt, SPECS, pop)
        assert sm.weight_specs is None
        assert sm.weights is None and sm.Y_w is None and sm.W is None
        with pytest.raises(ValueError, match="weighted=True requires weight_specs"):
            sm.cint(weighted=True)
        with pytest.raises(ValueError, match="weighted=True requires weight_specs"):
            sm.rate(weighted=True)

    def test_rejects_non_specs(self):
        part, cnt, pop, *_ = _survey()
        with pytest.raises(TypeError, match="WeightSpecs"):
            SocialMix(part, cnt, SPECS, pop, weight_specs=[LinearAgeFactor()])

    def test_weighted_false_matches_unweighted_fit(self):
        part, cnt, pop, *_ = _survey()
        plain = _quiet(SocialMix, part, cnt, SPECS, pop)
        weighted = _quiet(
            SocialMix, part, cnt, SPECS, pop, weight_specs=WeightSpecs(LinearAgeFactor())
        )
        for label, cs in plain.cint().items():
            np.testing.assert_allclose(weighted.cint(weighted=False)[label].central, cs.central)
        for label, cs in _quiet(plain.rate).items():
            np.testing.assert_allclose(
                _quiet(weighted.rate, weighted=False)[label].central, cs.central
            )


class TestWeightedEstimator:
    @pytest.mark.parametrize("apply_reciprocity", [False, True])
    def test_constant_factor_changes_nothing(self, apply_reciprocity):
        part, cnt, pop, *_ = _survey()
        sm = _quiet(
            SocialMix, part, cnt, SPECS, pop, apply_reciprocity=apply_reciprocity,
            weight_specs=WeightSpecs(ConstantFactor(value=5.0)),
        )
        np.testing.assert_allclose(sm.W, sm.N)
        np.testing.assert_allclose(sm.Y_w, sm.Y)
        np.testing.assert_allclose(
            sm.cint()["All->All"].central, sm.cint(weighted=False)["All->All"].central
        )

    def test_matches_direct_weighted_mean(self):
        part, cnt, pop, df_part, df_cnt = _survey()
        factor = LinearAgeFactor(intercept=1.0, slope=0.5)
        sm = _quiet(
            SocialMix, part, cnt, SPECS, pop, apply_reciprocity=False,
            weight_specs=WeightSpecs(factor),
        )
        expected = _expected_weighted_cint(df_part, df_cnt, lambda a: 1.0 + 0.5 * a)[()]
        np.testing.assert_allclose(sm.cint()["All->All"].central, expected)
        # The age-varying weight really changes the estimate.
        assert not np.allclose(expected, sm.cint(weighted=False)["All->All"].central)
        # Normalised weights sum to the cell sizes, so W equals N.
        np.testing.assert_allclose(sm.W, sm.N)

    def test_partial_stratification(self):
        part, cnt, pop, df_part, df_cnt = _survey(sex=True)
        sm = _quiet(
            SocialMix, part, cnt, SPECS, pop, apply_reciprocity=False,
            weight_specs=WeightSpecs(LinearAgeFactor(intercept=1.0, slope=0.5)),
        )
        expected = _expected_weighted_cint(
            df_part, df_cnt, lambda a: 1.0 + 0.5 * a, group_cols=("sex",)
        )
        for sex in ("F", "M"):
            np.testing.assert_allclose(sm.cint()[f"{sex}->All"].central, expected[sex])
        assert sm.W.shape == sm.N.shape == (2, 2)

    def test_rate_uses_weighted_cint(self):
        part, cnt, pop, *_ = _survey()
        sm = _quiet(
            SocialMix, part, cnt, SPECS, pop, apply_reciprocity=False,
            weight_specs=WeightSpecs(LinearAgeFactor()),
        )
        rate = _quiet(sm.rate)["All->All"].central
        np.testing.assert_allclose(rate, sm.cint()["All->All"].central / sm.P[None, :])


class TestCaching:
    def test_one_cache_per_weighted_value(self):
        part, cnt, pop, *_ = _survey()
        sm = _quiet(
            SocialMix, part, cnt, SPECS, pop, apply_reciprocity=False,
            weight_specs=WeightSpecs(LinearAgeFactor()),
        )
        w1 = sm.cint()
        u1 = sm.cint(weighted=False)
        w2 = sm.cint(weighted=True)
        u2 = sm.cint(weighted=False)
        assert w1 is w2 and u1 is u2 and w1 is not u1
        assert not np.allclose(w1["All->All"].central, u1["All->All"].central)
        # rate passes `weighted` through instead of reusing the default cint
        ru = _quiet(sm.rate, weighted=False)["All->All"].central
        np.testing.assert_allclose(ru, u1["All->All"].central / sm.P[None, :])


class TestDiagnostics:
    def test_weights_property(self):
        part, cnt, pop, *_ = _survey()
        specs = WeightSpecs(LinearAgeFactor())
        sm = _quiet(SocialMix, part, cnt, SPECS, pop, weight_specs=specs)
        result = sm.weights
        assert isinstance(result, WeightResult)
        assert result is sm.weights  # cached
        assert result.specs is specs
        table = result.participants
        assert len(table) == 60
        np.testing.assert_array_equal(table["id"].to_numpy(), sm.part_data.data["id"].to_numpy())
        np.testing.assert_allclose(table.groupby("cell")["weight"].sum().to_numpy(), [30, 30])


class TestReuseAndPickling:
    def test_one_specs_two_fits(self):
        part, cnt, pop, *_ = _survey()
        part2, cnt2, pop2, *_ = _survey(sex=True)
        specs = WeightSpecs(LinearAgeFactor())
        sm1 = _quiet(SocialMix, part, cnt, SPECS, pop, weight_specs=specs)
        before = sm1.cint()["All->All"].central.copy()
        sm2 = _quiet(SocialMix, part2, cnt2, SPECS, pop2, weight_specs=specs)
        assert sm1._bound is not sm2._bound
        sm1._cint.clear()
        np.testing.assert_allclose(sm1.cint()["All->All"].central, before)

    def test_fitted_weighted_socialmix_pickles(self):
        part, cnt, pop, *_ = _survey()
        sm = _quiet(
            SocialMix, part, cnt, SPECS, pop, weight_specs=WeightSpecs(LinearAgeFactor())
        )
        restored = pickle.loads(pickle.dumps(sm))
        np.testing.assert_allclose(restored._bound.weights(), sm._bound.weights())


class TestFromDataframes:
    def test_weight_specs_pass_through(self):
        part, cnt, pop, *_ = _survey()
        specs = WeightSpecs(LinearAgeFactor())
        sm = _quiet(
            SocialMix.from_dataframes, part.data, cnt.data, SPECS, weight_specs=specs
        )
        assert sm.weight_specs is specs and sm.weights is not None


class TestWeightedBootstrap:
    """Bootstrap item: weights rebuilt on every resample (D25, Problem 12)."""

    def test_neutral_weights_reproduce_unweighted_bootstrap(self):
        from ...weights import DesignWeights

        part, cnt, pop, df_part, df_cnt = _survey()
        df_p = df_part.copy()
        df_p["w"] = 1.0
        part_w = _quiet(ParticipantData, df_p, id_col="pid", age_col="age", aux_cols="w")
        plain = _quiet(SocialMix, part, cnt, SPECS, pop)
        ones = _quiet(SocialMix, part_w, cnt, SPECS, pop, weight_specs=WeightSpecs(DesignWeights("w")))
        np.testing.assert_allclose(
            ones.cint()["All->All"].central, plain.cint()["All->All"].central
        )
        b_plain = _quiet(plain.run_inference_bootstrap, n_boot=8, random_state=3, progress=False)
        b_ones = _quiet(ones.run_inference_bootstrap, n_boot=8, random_state=3, progress=False)
        assert b_ones.weighted and not b_plain.weighted
        for a, b in zip(b_ones.cint_samples, b_plain.cint_samples):
            np.testing.assert_allclose(a["All->All"], b["All->All"])

    def test_weighted_false_on_weighted_fit(self):
        part, cnt, pop, *_ = _survey()
        plain = _quiet(SocialMix, part, cnt, SPECS, pop)
        sm = _quiet(SocialMix, part, cnt, SPECS, pop, weight_specs=WeightSpecs(LinearAgeFactor()))
        b_plain = _quiet(plain.run_inference_bootstrap, n_boot=5, random_state=1, progress=False)
        b_unw = _quiet(
            sm.run_inference_bootstrap, n_boot=5, random_state=1, progress=False, weighted=False
        )
        assert not b_unw.weighted
        for a, b in zip(b_unw.cint_samples, b_plain.cint_samples):
            np.testing.assert_allclose(a["All->All"], b["All->All"])

    def test_weighted_true_requires_specs(self):
        part, cnt, pop, *_ = _survey()
        sm = _quiet(SocialMix, part, cnt, SPECS, pop)
        with pytest.raises(ValueError, match="weighted=True requires weight_specs"):
            sm.run_inference_bootstrap(n_boot=2, progress=False, weighted=True)

    @pytest.mark.parametrize("sex", [False, True])
    def test_mean_close_to_weighted_point_estimate(self, sex):
        part, cnt, pop, *_ = _survey(sex=sex)
        sm = _quiet(
            SocialMix, part, cnt, SPECS, pop, apply_reciprocity=False,
            weight_specs=WeightSpecs(LinearAgeFactor(slope=1.0)),
        )
        boot = _quiet(sm.run_inference_bootstrap, n_boot=200, random_state=0, progress=False)
        assert boot.weighted
        mean = boot.mean(statistic="cint")
        for label, cs in sm.cint().items():
            np.testing.assert_allclose(mean[label], cs.central, rtol=0.15, atol=0.05)
            # ... and away from the unweighted estimate where they differ.
        unweighted = sm.cint(weighted=False)
        label = next(iter(unweighted))
        assert not np.allclose(sm.cint()[label].central, unweighted[label].central)

    def test_sample_dependent_factors_change_between_resamples(self):
        from ...weights import CategoryWeights, DayOfWeekWeights

        part, cnt, pop, df_part, df_cnt = _survey()
        df_p = df_part.copy()
        df_p["dow"] = np.where(df_p["pid"] % 3 == 0, 0, 2)
        df_p["region"] = np.where(df_p["pid"] % 2 == 0, "A", "B")
        part2 = _quiet(
            ParticipantData, df_p, id_col="pid", age_col="age", aux_cols=["dow", "region"]
        )
        specs = WeightSpecs(
            DayOfWeekWeights(col="dow"), CategoryWeights("region", {"A": 1.0, "B": 2.0})
        )
        sm = _quiet(SocialMix, part2, cnt, SPECS, pop, weight_specs=specs)
        rng = np.random.default_rng(0)
        r1, r2 = (rng.integers(0, 60, 60) for _ in range(2))
        # Participant at row 0 drawn in both resamples: compare its factors.
        r1[0] = r2[0] = 0
        f1, f2 = sm._bound.factor_values(r1), sm._bound.factor_values(r2)
        assert f1[1, 0] == f2[1, 0]  # CategoryWeights: fixed
        assert f1[0, 0] != f2[0, 0]  # DayOfWeekWeights: depends on the counts

    def test_bug_in_factor_propagates(self):
        from ...weights.tests.factors import FailOnResampleFactor

        part, cnt, pop, *_ = _survey()
        sm = _quiet(SocialMix, part, cnt, SPECS, pop, weight_specs=WeightSpecs(FailOnResampleFactor()))
        with pytest.raises(RuntimeError, match="injected"):
            sm.run_inference_bootstrap(n_boot=3, random_state=0, progress=False)

    def test_bootstrap_time_merge_rebinds_and_warns(self):
        from ...weights import WeightWarning

        ages = np.arange(20)  # one participant per year: the stability check merges
        df_part = pd.DataFrame({"pid": ages, "age": ages})
        df_cnt = pd.DataFrame({"pid": np.repeat(ages, 2), "cnt_age": 5})
        part = _quiet(ParticipantData, df_part, id_col="pid", age_col="age")
        cnt = _quiet(ContactData, df_cnt, id_col="pid", age_col="cnt_age")
        sm = _quiet(
            SocialMix, part, cnt, SPECS, apply_reciprocity=False,
            weight_specs=WeightSpecs(LinearAgeFactor()),
        )
        assert sm.C == 2 and len(sm.weights.cells) == 2
        with pytest.warns(WeightWarning, match="merged participant age groups"):
            boot = sm.run_inference_bootstrap(n_boot=3, random_state=0, progress=False)
        assert sm.C == 1 and boot.weighted
        assert len(sm.weights.cells) == 1
        np.testing.assert_allclose(sm.weights.participants["weight"].sum(), 20.0)
        assert sm.W.shape == sm.N.shape == (1,)

    def test_ids_must_match(self):
        from ..classical._socialmix_bootstrap import SocialMixBootstrap

        part, cnt, pop, *_ = _survey()
        sm = _quiet(SocialMix, part, cnt, SPECS, pop, weight_specs=WeightSpecs(LinearAgeFactor()))
        bound = sm._bound
        bound.ids = bound.ids[::-1].copy()
        boot = SocialMixBootstrap(
            sm.part_data, sm.cnt_data, sm.age_group_specs, sm.pop_data, n_boot=2,
            random_state=0, bound_weights=bound,
        )
        with pytest.raises(RuntimeError, match="do not belong"):
            boot.run(progress=False)


class TestPrem2Unaffected:
    """D27: Prem2 shares the SocialMix loader but gets no weighting."""

    def test_prem2_counts_match_unweighted_socialmix(self):
        part, cnt, pop, *_ = _survey(sex=True)
        sm = _quiet(SocialMix, part, cnt, SPECS, apply_reciprocity=False)
        prem2 = _quiet(Prem2, part, cnt, SPECS)
        np.testing.assert_allclose(prem2.Y, sm.Y)
        np.testing.assert_allclose(prem2.N, sm.N)
        assert not hasattr(prem2, "weight_specs")


class TestAuxCols:
    """Auxiliary columns reach the weights and are not strata (dev plan D20)."""

    def _frames(self):
        part, cnt, pop, df_part, df_cnt = _survey()
        df_p = part.data.drop(columns=["part_age_grp"], errors="ignore").copy()
        df_p["part_w"] = 1.0 + (df_p["part_age"] % 4).astype(float)
        return df_p, cnt.data, df_part, df_cnt

    def test_aux_column_drives_weights_end_to_end(self):
        from ...weights.tests.factors import ColumnFactor

        df_p, df_c, df_part, df_cnt = self._frames()
        part = _quiet(ParticipantData, df_p, id_col="id", age_col="part_age", aux_cols="part_w")
        cnt = _quiet(ContactData, df_c, id_col="id", age_col="cnt_age")
        sm = _quiet(
            SocialMix, part, cnt, SPECS, apply_reciprocity=False,
            weight_specs=WeightSpecs(ColumnFactor(col="w", name="w")),
        )
        assert "part_w" in sm.part_data.data.columns
        expected = _expected_weighted_cint(
            df_part, df_cnt, lambda a: 1.0 + (a % 4)
        )[()]
        np.testing.assert_allclose(sm.cint()["All->All"].central, expected)

    def test_from_dataframes_excludes_aux_from_strata(self):
        df_p, df_c, *_ = self._frames()
        df_p["part_sex"] = np.where(df_p["id"] % 2 == 0, "F", "M")
        sm = _quiet(
            SocialMix.from_dataframes, df_p, df_c, SPECS, aux_cols="part_w",
            apply_reciprocity=False,
        )
        assert sm.strat_vars_part == ["sex"]
        assert "part_w" in sm.part_data.data.columns
        # Unprefixed names are accepted too.
        sm2 = _quiet(
            SocialMix.from_dataframes, df_p, df_c, SPECS, aux_cols=["w"],
            apply_reciprocity=False,
        )
        assert sm2.strat_vars_part == ["sex"]

    def test_from_dataframes_warns_on_float_stratum(self):
        df_p, df_c, *_ = self._frames()
        # A float column is not a usable stratum (the fit fails afterwards, as
        # it did before this change); the warning says what to do instead.
        with pytest.warns(UserWarning, match=r"part_w.*aux_cols"):
            with pytest.raises(Exception):
                SocialMix.from_dataframes(df_p, df_c, SPECS, apply_reciprocity=False)


class TestPreview:
    """D21: preview() shows the weights the fit would use, without fitting."""

    @staticmethod
    def _assert_same(preview, sm):
        a, b = preview.participants, sm.weights.participants
        np.testing.assert_array_equal(a["id"].to_numpy(), b["id"].to_numpy())
        assert a["cell"].tolist() == b["cell"].tolist()
        np.testing.assert_allclose(a["weight"], b["weight"])
        pd.testing.assert_frame_equal(preview.cells, sm.weights.cells)

    def test_matches_fit(self):
        part, cnt, pop, *_ = _survey(sex=True)
        specs = WeightSpecs(LinearAgeFactor(), threshold=1.5)
        preview = _quiet(specs.preview, part, SPECS)
        sm = _quiet(SocialMix, part, cnt, SPECS, pop, weight_specs=specs)
        self._assert_same(preview, sm)

    def test_matches_fit_with_merging_of_non_empty_cells(self):
        # F has nobody aged 10+, so [10, 20) is empty in one stratum and is
        # merged in both strata, including M's non-empty cell.
        ages = np.r_[np.tile(np.arange(10), 3), np.tile(np.arange(20), 3)]
        sex = ["F"] * 30 + ["M"] * 60
        df_part = pd.DataFrame({"pid": np.arange(90), "age": ages, "sex": sex})
        df_cnt = pd.DataFrame({"pid": np.arange(90), "cnt_age": 5})
        part = _quiet(ParticipantData, df_part, id_col="pid", age_col="age", strat_var_cols="sex")
        cnt = _quiet(ContactData, df_cnt, id_col="pid", age_col="cnt_age")
        specs = WeightSpecs(LinearAgeFactor())
        preview = _quiet(specs.preview, part, SPECS, adaptive_merge=True)
        sm = _quiet(
            SocialMix, part, cnt, SPECS, apply_reciprocity=False, adaptive_merge=True,
            weight_specs=specs,
        )
        assert sm.C == 1 and len(preview.cells) == 2  # one age group x two strata
        self._assert_same(preview, sm)

    def test_matches_fit_with_out_of_range_participants(self):
        part, cnt, pop, df_part, df_cnt = _survey()
        df_p = df_part.copy()
        df_p.loc[len(df_p)] = {"pid": 999, "age": 50}
        part2 = _quiet(ParticipantData, df_p, id_col="pid", age_col="age")
        specs = WeightSpecs(LinearAgeFactor())
        with pytest.warns(UserWarning, match="Dropped 1 participant"):
            preview = specs.preview(part2, SPECS)
        sm = _quiet(SocialMix, part2, cnt, SPECS, pop, weight_specs=specs)
        assert 999 not in preview.participants["id"].tolist()
        self._assert_same(preview, sm)

    def test_lenient_on_empty_age_group(self):
        ages = np.tile(np.arange(10), 3)  # nobody in [10, 20)
        df_part = pd.DataFrame({"pid": np.arange(30), "age": ages})
        part = _quiet(ParticipantData, df_part, id_col="pid", age_col="age")
        specs = WeightSpecs(LinearAgeFactor())
        from ...weights import WeightWarning

        with pytest.warns(WeightWarning, match="would raise"):
            preview = specs.preview(part, SPECS)
        assert preview.cells["n"].tolist() == [30, 0]
        assert np.isnan(preview.cells["n_eff"].iloc[1])

    def test_does_not_modify_participants(self):
        part, *_ = _survey()
        before = part.data.copy()
        _quiet(WeightSpecs(LinearAgeFactor()).preview, part, SPECS)
        pd.testing.assert_frame_equal(part.data, before)


def test_all_factors_together_end_to_end():
    """DesignWeights x CategoryWeights x PostStratWeights x DayOfWeekWeights x
    AgePostStratWeights with truncation: preview == fit, the weighted bootstrap
    runs, and the weights match a direct computation."""
    from ...weights import (
        AgePostStratWeights, CategoryWeights, DayOfWeekWeights, DesignWeights,
        PostStratWeights,
    )

    part, cnt, pop, df_part, df_cnt = _survey(sex=True)
    rng = np.random.default_rng(7)
    df_p = df_part.copy()
    df_p["dow"] = rng.integers(0, 7, len(df_p)).astype(float)
    df_p.loc[::11, "dow"] = np.nan
    df_p["region"] = rng.choice(["urban", "rural"], len(df_p))
    df_p["country"] = rng.choice(["UK", "DE", "IT"], len(df_p))
    df_p["dw"] = rng.uniform(0.5, 2.0, len(df_p))
    part2 = _quiet(
        ParticipantData, df_p, id_col="pid", age_col="age", strat_var_cols="sex",
        aux_cols=["dow", "region", "country", "dw"],
    )
    pop_sizes = pd.DataFrame({"age": np.arange(30), "P": np.linspace(100, 50, 30)})
    pop_ref = _quiet(PopulationData, pop_sizes, size_col="P", age_col="age")
    specs = WeightSpecs(
        DesignWeights("dw"),
        CategoryWeights("region", {"urban": 1.0, "rural": 2.5}),
        PostStratWeights("country", {"UK": 0.3, "DE": 0.4, "IT": 0.3}),
        DayOfWeekWeights(col="dow", within="cell"),
        AgePostStratWeights(pop_ref),
        threshold=3.0,
    )
    preview = _quiet(specs.preview, part2, SPECS)
    sm = _quiet(SocialMix, part2, cnt, SPECS, pop, weight_specs=specs, apply_reciprocity=False)
    np.testing.assert_allclose(preview.participants["weight"], sm.weights.participants["weight"])
    # Source columns (in factor order), then the factor columns.
    assert list(sm.weights.participants.columns[2:7]) == [
        "part_dw", "part_region", "part_country", "part_dow", "part_age"
    ]
    assert list(sm.weights.participants.columns[7:12]) == list(specs.names)
    pd.testing.assert_series_equal(
        sm.weights.participants.set_index("id")["part_region"],
        part2.data.set_index("id").loc[sm.weights.participants["id"], "part_region"],
        check_names=False,
    )

    # Direct check of the composite: product of the factor columns.
    table = sm.weights.participants
    np.testing.assert_allclose(table["composite"], table[list(specs.names)].prod(axis=1))
    # Weights sum to the cell sizes (up to the single re-normalisation).
    np.testing.assert_allclose(
        table.groupby("cell")["weight"].sum().to_numpy(),
        table.groupby("cell")["weight"].size().to_numpy(),
    )
    boot = _quiet(sm.run_inference_bootstrap, n_boot=20, random_state=0, progress=False)
    assert boot.weighted and boot.n_boot > 0
