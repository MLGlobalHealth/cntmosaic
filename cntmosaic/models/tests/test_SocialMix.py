"""
Unit tests for SocialMix classical contact matrix estimation.

Tests are organised by concern:
  TestInstantiation   – constructor paths: SocialMix() and .from_dataframes()
  TestStratification  – strat_mode and K inferred correctly for each data layout
  TestCint            – .cint() output type (ContactSummary), shape, reciprocity
  TestRate            – .rate() output type (ContactSummary), shape, error handling
  TestAdaptiveMerge   – small-sample behaviour with adaptive_merge=True/False
  TestBootstrap       – run_inference_bootstrap() stores BootstrapResults (N_BOOT=10)
  TestSummariser      – ModelSummariserSocialMix mirrors ModelSummariser interface
"""

import numpy as np
import pytest

from ...analysis.summariser import ContactSummary, ModelSummariserSocialMix
from ...utils import AgeGroupSpecs
from ..classical._SocialMix import SocialMix
from .fixtures import (
    full_large_sample,
    full_multi_strat_large_sample,
    full_small_sample,
    partial_large_sample,
    partial_multi_strat_large_sample,
    partial_small_sample,
    single_large_sample,
    single_small_sample,
)

N_BOOT = 10
AGE_BINS = AgeGroupSpecs(0, 80, 5)  # 16 age groups


# ============================================================================
# Helper
# ============================================================================


def _from_containers(fixture, **kwargs):
    """Build SocialMix with default 5-year age bins."""
    part_data, cnt_data, pop_data = fixture
    return SocialMix(part_data, cnt_data, AGE_BINS, pop_data, **kwargs)


# ============================================================================
# Instantiation
# ============================================================================


class TestInstantiation:
    """Both constructor paths produce valid, equivalent models."""

    def test_from_containers_single(self, single_large_sample):
        sm = _from_containers(single_large_sample)
        assert sm.Y is not None
        assert sm.N is not None
        assert sm.P is not None

    def test_from_dataframes_single(self, single_large_sample):
        part_data, cnt_data, pop_data = single_large_sample
        sm = SocialMix.from_dataframes(
            part_data.data, cnt_data.data, AGE_BINS, pop_data.data
        )
        assert sm.Y is not None

    def test_from_containers_equals_from_dataframes(self, single_large_sample):
        """Both paths produce identical Y and N arrays."""
        part_data, cnt_data, pop_data = single_large_sample
        sm_c = _from_containers(single_large_sample, apply_reciprocity=False)
        sm_df = SocialMix.from_dataframes(
            part_data.data,
            cnt_data.data,
            AGE_BINS,
            pop_data.data,
            apply_reciprocity=False,
        )
        np.testing.assert_array_equal(sm_c.Y, sm_df.Y)
        np.testing.assert_array_equal(sm_c.N, sm_df.N)

    def test_missing_pop_disables_reciprocity(self, single_large_sample):
        part_data, cnt_data, _ = single_large_sample
        with pytest.warns(
            UserWarning,
            match="Reciprocity adjustment requested but no population data",
        ):
            sm = SocialMix(part_data, cnt_data, AGE_BINS, apply_reciprocity=True)
        assert sm.apply_reciprocity is False

    def test_partial_warns_reciprocity_not_applicable(
        self, partial_multi_strat_large_sample
    ):
        with pytest.warns(
            UserWarning,
            match="Reciprocity adjustment is not applicable for partial stratification",
        ):
            _from_containers(partial_multi_strat_large_sample, apply_reciprocity=True)

    def test_reciprocity_off_by_default(self, single_large_sample):
        sm = _from_containers(single_large_sample)
        assert sm.apply_reciprocity is False
        sm_off = _from_containers(single_large_sample, apply_reciprocity=False)
        np.testing.assert_array_equal(
            sm.cint()["All->All"].central, sm_off.cint()["All->All"].central
        )


# ============================================================================
# Stratification mode detection
# ============================================================================


class TestStratification:
    """strat_mode and K are correctly inferred for each data layout."""

    def test_single(self, single_large_sample):
        sm = _from_containers(single_large_sample)
        assert sm.strat_mode == "single"
        assert sm.K == 1
        assert sm.K_part == 1
        assert sm.K_cnt == 1
        assert sm.strat_vars_part == []
        assert sm.strat_vars_cnt == []

    def test_partial(self, partial_large_sample):
        sm = _from_containers(partial_large_sample, apply_reciprocity=False)
        assert sm.strat_mode == "partial"
        assert sm.K == 2
        assert sm.strat_vars_part == ["sex"]
        assert sm.strat_vars_cnt == []
        assert sm.strat_vars_shared == []

    def test_partial_multi_strat(self, partial_multi_strat_large_sample):
        with pytest.warns(UserWarning):
            sm = _from_containers(partial_multi_strat_large_sample)
        assert sm.strat_mode == "partial"
        assert sm.K == 4
        assert sorted(sm.strat_vars_part) == ["ses", "sex"]
        assert sm.strat_vars_cnt == []

    def test_full(self, full_large_sample):
        sm = _from_containers(full_large_sample)
        assert sm.strat_mode == "full"
        assert sm.K == 4
        assert sm.strat_vars_part == ["sex"]
        assert sm.strat_vars_cnt == ["sex"]
        assert sm.strat_vars_shared == ["sex"]

    def test_full_multi_strat(self, full_multi_strat_large_sample):
        sm = _from_containers(full_multi_strat_large_sample)
        assert sm.strat_mode == "full"
        assert sm.K == 16
        assert sorted(sm.strat_vars_part) == ["ses", "sex"]
        assert sorted(sm.strat_vars_cnt) == ["ses", "sex"]


# ============================================================================
# Contact intensity – .cint()
# ============================================================================


class TestCint:
    """cint() returns Dict[str, ContactSummary] with correct shape and math."""

    def test_returns_contact_summary_single(self, single_large_sample):
        sm = _from_containers(single_large_sample)
        result = sm.cint()
        assert isinstance(result, dict)
        assert "All->All" in result
        assert isinstance(result["All->All"], ContactSummary)

    def test_shape_single(self, single_large_sample):
        sm = _from_containers(single_large_sample)
        cs = sm.cint()["All->All"]
        assert cs.central.shape == (16, 16)

    def test_non_negative_single(self, single_large_sample):
        sm = _from_containers(single_large_sample)
        cs = sm.cint()["All->All"]
        assert np.all(cs.central >= 0)

    def test_keys_partial(self, partial_large_sample):
        sm = _from_containers(partial_large_sample, apply_reciprocity=False)
        assert set(sm.cint().keys()) == {"M->All", "F->All"}

    def test_keys_full(self, full_large_sample):
        sm = _from_containers(full_large_sample)
        assert set(sm.cint().keys()) == {"M->M", "M->F", "F->M", "F->F"}

    def test_caching(self, single_large_sample):
        sm = _from_containers(single_large_sample)
        assert sm.cint() is sm.cint()

    def test_reciprocity_single(self, single_large_sample):
        """M[c,d]*P[c] == M[d,c]*P[d] after adjustment."""
        sm = _from_containers(single_large_sample, apply_reciprocity=True)
        M = sm.cint()["All->All"].central
        P = sm.P
        np.testing.assert_allclose(
            M * P[:, np.newaxis],
            M.T * P[np.newaxis, :],
            rtol=1e-8,
        )

    def test_reciprocity_full(self, full_large_sample):
        """For every s,t stratum pair: M^st / P^t == (M^ts / P^s)^T."""
        sm = _from_containers(full_large_sample, apply_reciprocity=True)
        cint = sm.cint()
        labels = sm._create_stratum_labels()
        for i, label_st in enumerate(labels):
            s, t = label_st.split("->")
            reverse = f"{t}->{s}"
            M_st = cint[label_st].central
            M_ts = cint[reverse].central
            k_s = i // sm.K_cnt
            k_t = i % sm.K_cnt
            P_s = sm.P[k_s, :]
            P_t = sm.P[k_t, :]
            np.testing.assert_allclose(
                M_st / P_t[np.newaxis, :],
                (M_ts / P_s[np.newaxis, :]).T,
                rtol=1e-8,
                err_msg=f"Reciprocity violated for {label_st}",
            )


# ============================================================================
# Contact rate – .rate()
# ============================================================================


class TestRate:
    """rate() returns Dict[str, ContactSummary]; raises without population data."""

    def test_raises_without_pop_data(self, single_large_sample):
        part_data, cnt_data, _ = single_large_sample
        with pytest.warns(UserWarning):
            sm = SocialMix(part_data, cnt_data, AGE_BINS, apply_reciprocity=True)
        with pytest.raises(ValueError, match="population data"):
            sm.rate()

    def test_returns_contact_summary_single(self, single_large_sample):
        sm = _from_containers(single_large_sample)
        result = sm.rate()
        assert isinstance(result, dict)
        assert "All->All" in result
        assert isinstance(result["All->All"], ContactSummary)

    def test_shape_single(self, single_large_sample):
        sm = _from_containers(single_large_sample)
        cs = sm.rate()["All->All"]
        assert cs.central.shape == (16, 16)

    def test_non_negative_single(self, single_large_sample):
        sm = _from_containers(single_large_sample)
        assert np.all(sm.rate()["All->All"].central >= 0)

    def test_keys_partial(self, partial_large_sample):
        sm = _from_containers(partial_large_sample, apply_reciprocity=False)
        assert set(sm.rate().keys()) == {"M->All", "F->All"}

    def test_keys_full(self, full_large_sample):
        sm = _from_containers(full_large_sample)
        assert set(sm.rate().keys()) == {"M->M", "M->F", "F->M", "F->F"}


# ============================================================================
# Adaptive merge (small samples)
# ============================================================================


class TestAdaptiveMerge:
    """Small samples raise without adaptive_merge; merge with it."""

    def test_single_small_raises_without_merge(self, single_small_sample):
        part_data, cnt_data, pop_data = single_small_sample
        with pytest.raises(ValueError):
            SocialMix(part_data, cnt_data, AGE_BINS, pop_data, adaptive_merge=False)

    def test_single_small_merges(self, single_small_sample):
        part_data, cnt_data, pop_data = single_small_sample
        with pytest.warns(UserWarning):
            sm = SocialMix(part_data, cnt_data, AGE_BINS, pop_data, adaptive_merge=True)
        cs = sm.cint()["All->All"]
        assert cs.central.shape[0] <= 16
        assert np.all(cs.central >= 0)

    def test_partial_small_merges(self, partial_small_sample):
        part_data, cnt_data, pop_data = partial_small_sample
        with pytest.warns(UserWarning):
            sm = SocialMix(
                part_data,
                cnt_data,
                AGE_BINS,
                pop_data,
                adaptive_merge=True,
                apply_reciprocity=False,
            )
        cint = sm.cint()
        for key in ["M->All", "F->All"]:
            assert key in cint
            assert cint[key].central.shape[0] <= 16

    # def test_full_small_merges(self, full_small_sample):
    #     part_data, cnt_data, pop_data = full_small_sample
    #     with pytest.warns(UserWarning):
    #         sm = SocialMix.from_containers(
    #             part_data, cnt_data, AGE_BINS, pop_data, adaptive_merge=True
    #         )
    #     cint = sm.cint()
    #     for key in ["M->M", "M->F", "F->M", "F->F"]:
    #         assert key in cint
    #         assert cint[key].central.shape[0] <= 16


# ============================================================================
# Bootstrap
# ============================================================================


class TestBootstrap:
    """run_inference_bootstrap() stores BootstrapResults with correct n_boot."""

    def test_single(self, single_large_sample):
        sm = _from_containers(single_large_sample)
        sm.run_inference_bootstrap(n_boot=N_BOOT, random_state=42, progress=False)
        assert sm._boot is not None
        assert sm._boot.n_boot == N_BOOT

    def test_partial(self, partial_large_sample):
        sm = _from_containers(partial_large_sample, apply_reciprocity=False)
        sm.run_inference_bootstrap(n_boot=N_BOOT, random_state=42, progress=False)
        assert sm._boot.n_boot == N_BOOT

    def test_full(self, full_large_sample):
        sm = _from_containers(full_large_sample)
        sm.run_inference_bootstrap(n_boot=N_BOOT, random_state=42, progress=False)
        assert sm._boot.n_boot == N_BOOT

    def test_single_small(self, single_small_sample):
        part_data, cnt_data, pop_data = single_small_sample
        with pytest.warns(UserWarning):
            sm = SocialMix(
                part_data,
                cnt_data,
                AGE_BINS,
                pop_data,
                adaptive_merge=True,
            )
        sm.run_inference_bootstrap(n_boot=N_BOOT, random_state=42, progress=False)
        assert sm._boot.n_boot == N_BOOT


# ============================================================================
# ModelSummariserSocialMix
# ============================================================================


class TestSummariser:
    """ModelSummariserSocialMix mirrors ModelSummariser: returns ContactSummary."""

    @pytest.fixture
    def sm_single(self, single_large_sample):
        sm = _from_containers(single_large_sample)
        sm.run_inference_bootstrap(n_boot=N_BOOT, random_state=42, progress=False)
        return sm

    @pytest.fixture
    def sm_partial(self, partial_large_sample):
        sm = _from_containers(partial_large_sample, apply_reciprocity=False)
        sm.run_inference_bootstrap(n_boot=N_BOOT, random_state=42, progress=False)
        return sm

    @pytest.fixture
    def sm_full(self, full_large_sample):
        sm = _from_containers(full_large_sample)
        sm.run_inference_bootstrap(n_boot=N_BOOT, random_state=42, progress=False)
        return sm

    # --- initialisation ---

    def test_raises_without_bootstrap(self, single_large_sample):
        sm = _from_containers(single_large_sample)
        with pytest.raises(ValueError, match="bootstrap"):
            ModelSummariserSocialMix(sm)

    # --- summarise_cint ---

    def test_summarise_cint_returns_contact_summary(self, sm_single):
        summ = ModelSummariserSocialMix(sm_single)
        result = summ.summarise_cint(alpha=0.05)
        assert isinstance(result, dict)
        assert "All->All" in result
        assert isinstance(result["All->All"], ContactSummary)

    def test_summarise_cint_fields(self, sm_single):
        summ = ModelSummariserSocialMix(sm_single)
        cs = summ.summarise_cint(alpha=0.05)["All->All"]
        assert cs.lower.shape == cs.central.shape == cs.upper.shape
        assert cs.alpha == 0.05
        assert np.all(cs.lower <= cs.central)
        assert np.all(cs.central <= cs.upper)

    def test_summarise_cint_keys_partial(self, sm_partial):
        summ = ModelSummariserSocialMix(sm_partial)
        result = summ.summarise_cint(alpha=0.05)
        assert set(result.keys()) == {"M->All", "F->All"}
        assert isinstance(result["M->All"], ContactSummary)

    def test_summarise_cint_keys_full(self, sm_full):
        summ = ModelSummariserSocialMix(sm_full)
        result = summ.summarise_cint(alpha=0.05)
        assert set(result.keys()) == {"M->M", "M->F", "F->M", "F->F"}

    # --- summarise_rate ---

    def test_summarise_rate_returns_contact_summary(self, sm_single):
        summ = ModelSummariserSocialMix(sm_single)
        result = summ.summarise_rate(alpha=0.05)
        assert isinstance(result["All->All"], ContactSummary)

    def test_summarise_rate_ordering(self, sm_single):
        cs = ModelSummariserSocialMix(sm_single).summarise_rate(alpha=0.05)["All->All"]
        assert np.all(cs.lower <= cs.central)
        assert np.all(cs.central <= cs.upper)

    # --- summarise_mcint ---

    def test_summarise_mcint_returns_contact_summary(self, sm_single):
        summ = ModelSummariserSocialMix(sm_single)
        result = summ.summarise_mcint(alpha=0.05)
        cs = result["All->All"]
        assert isinstance(cs, ContactSummary)
        assert cs.central.ndim == 1

    def test_summarise_mcint_ordering(self, sm_single):
        cs = ModelSummariserSocialMix(sm_single).summarise_mcint(alpha=0.05)["All->All"]
        assert np.all(cs.lower <= cs.central)
        assert np.all(cs.central <= cs.upper)

    # --- caching ---

    def test_caching_same_result(self, sm_single):
        summ = ModelSummariserSocialMix(sm_single)
        r1 = summ.summarise_cint(alpha=0.05)
        n1 = summ.get_cache_info()["n_cached"]
        r2 = summ.summarise_cint(alpha=0.05)
        assert summ.get_cache_info()["n_cached"] == n1  # no new entry
        np.testing.assert_array_equal(r1["All->All"].central, r2["All->All"].central)

    def test_different_alpha_adds_cache_entry(self, sm_single):
        summ = ModelSummariserSocialMix(sm_single)
        summ.summarise_cint(alpha=0.05)
        summ.summarise_cint(alpha=0.10)
        assert summ.get_cache_info()["n_cached"] == 2

    def test_clear_cache(self, sm_single):
        summ = ModelSummariserSocialMix(sm_single)
        summ.summarise_cint(alpha=0.05)
        summ.clear_cache()
        assert summ.get_cache_info()["n_cached"] == 0

    # --- error handling ---

    def test_invalid_alpha_raises(self, sm_single):
        summ = ModelSummariserSocialMix(sm_single)
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                summ.summarise_cint(alpha=bad)
