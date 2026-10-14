"""Tests for the Prem model class."""

import numpy as np
import pandas as pd
import pytest
from jax.random import PRNGKey

from ...analysis.summariser import ContactSummary, ModelSummariserPrem
from ...utils import AgeBins
from ..classical import Prem
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


class TestInit:
    """Test Prem model initialization."""

    def test_single(self, single_large_sample):
        """Test initialization without stratification."""
        part_data, cnt_data, pop_data = single_large_sample
        age_bins = AgeBins(0, 80, 5)
        model = Prem(part_data, cnt_data, age_bins, random_effects=True)

        assert model.strat_vars_part == []
        assert model.strat_dims_part == {}
        assert model.strat_vars_cnt == []
        assert model.strat_dims_cnt == {}
        assert model.strat_vars_shared == []
        assert model.K == 1
        assert model.order == 1

    def test_custom_order(self, single_large_sample):
        """Test that a custom IGMRF2D order is stored on the model."""
        part_data, cnt_data, pop_data = single_large_sample
        age_bins = AgeBins(0, 80, 5)
        model = Prem(part_data, cnt_data, age_bins, order=2)

        assert model.order == 2

    def test_single_small(self, single_small_sample):
        """Test initialization without stratification on small sample."""
        part_data, cnt_data, pop_data = single_small_sample
        age_bins = AgeBins(0, 80, 5)
        model = Prem(part_data, cnt_data, age_bins, random_effects=True)

        assert model.strat_vars_part == []
        assert model.strat_dims_part == {}
        assert model.strat_vars_cnt == []
        assert model.strat_dims_cnt == {}
        assert model.strat_vars_shared == []
        assert model.K == 1

    def test_partial(self, partial_large_sample):
        """Test initialization with participant-only stratification."""
        part_data, cnt_data, pop_data = partial_large_sample
        age_bins = AgeBins(0, 80, 5)
        model = Prem(part_data, cnt_data, age_bins, random_effects=True)

        assert model.strat_vars_part == ["sex"]
        assert model.strat_dims_part == {"sex": 2}
        assert model.strat_vars_cnt == []
        assert model.strat_dims_cnt == {}
        assert model.strat_vars_shared == []
        assert model.K == 2

    def test_partial_small(self, partial_small_sample):
        """Test initialization with participant-only stratification on small sample."""
        part_data, cnt_data, pop_data = partial_small_sample
        age_bins = AgeBins(0, 80, 5)
        model = Prem(part_data, cnt_data, age_bins, random_effects=True)

        assert model.strat_vars_part == ["sex"]
        assert model.strat_dims_part == {"sex": 2}
        assert model.strat_vars_cnt == []
        assert model.strat_dims_cnt == {}
        assert model.strat_vars_shared == []
        assert model.K == 2

    def test_partial_multi_strat(self, partial_multi_strat_large_sample):
        """Test initialization with participant-only multi-variable stratification."""
        part_data, cnt_data, pop_data = partial_multi_strat_large_sample
        age_bins = AgeBins(0, 80, 5)
        model = Prem(part_data, cnt_data, age_bins, random_effects=True)

        assert model.strat_vars_part == ["sex", "ses"]
        assert model.strat_dims_part == {"sex": 2, "ses": 2}
        assert model.strat_vars_cnt == []
        assert model.strat_dims_cnt == {}
        assert model.strat_vars_shared == []
        assert model.K == 4  # 2 (sex) * 2 (ses)

    def test_full(self, full_large_sample):
        """Test initialization with full stratification."""
        part_data, cnt_data, pop_data = full_large_sample
        age_bins = AgeBins(0, 80, 5)
        model = Prem(part_data, cnt_data, age_bins, random_effects=True)

        assert model.strat_vars_part == ["sex"]
        assert model.strat_dims_part == {"sex": 2}
        assert model.strat_vars_cnt == ["sex"]
        assert model.strat_dims_cnt == {"sex": 2}
        assert model.strat_vars_shared == ["sex"]
        assert model.K == 4

    def test_full_multi_strat(self, full_multi_strat_large_sample):
        """Test initialization with full multi-variable stratification."""
        part_data, cnt_data, pop_data = full_multi_strat_large_sample
        age_bins = AgeBins(0, 80, 5)
        model = Prem(part_data, cnt_data, age_bins, random_effects=True)

        assert model.strat_vars_part == ["sex", "ses"]
        assert model.strat_dims_part == {"sex": 2, "ses": 2}
        assert model.strat_vars_cnt == ["sex", "ses"]
        assert model.strat_dims_cnt == {"sex": 2, "ses": 2}
        assert set(model.strat_vars_shared) == {"ses", "sex"}
        assert model.K == 16  # 2 (sex) * 2 (ses) * 2 (sex) * 2 (ses)


class TestDataLoading:
    """Test data loading and index creation."""

    def test_single(self, single_large_sample):
        """Test that _load creates all necessary indices."""
        part_data, cnt_data, _ = single_large_sample
        age_bins = AgeBins(0, 80, 5)
        model = Prem(part_data, cnt_data, age_bins)

        assert model.y is not None
        assert model.iix is not None
        assert model.cix is not None
        assert model.dix is not None
        assert model.six is not None
        assert isinstance(model.y, np.ndarray)
        assert isinstance(model.iix, np.ndarray)
        assert isinstance(model.cix, np.ndarray)
        assert isinstance(model.dix, np.ndarray)
        assert isinstance(model.six, np.ndarray)

    def test_partial(self, partial_large_sample):
        """Test that _load creates all necessary indices with stratification."""
        part_data, cnt_data, _ = partial_large_sample
        age_bins = AgeBins(0, 80, 5)
        model = Prem(part_data, cnt_data, age_bins)

        assert model.y is not None
        assert model.iix is not None
        assert model.cix is not None
        assert model.dix is not None
        assert model.six is not None
        assert isinstance(model.y, np.ndarray)
        assert isinstance(model.iix, np.ndarray)
        assert isinstance(model.cix, np.ndarray)
        assert isinstance(model.dix, np.ndarray)
        assert isinstance(model.six, np.ndarray)

    def test_full(self, full_large_sample):
        """Test that _load creates all necessary indices with full stratification."""
        part_data, cnt_data, _ = full_large_sample
        age_bins = AgeBins(0, 80, 5)
        model = Prem(part_data, cnt_data, age_bins)

        assert model.y is not None
        assert model.iix is not None
        assert model.cix is not None
        assert model.dix is not None
        assert model.six is not None
        assert isinstance(model.y, np.ndarray)
        assert isinstance(model.iix, np.ndarray)
        assert isinstance(model.cix, np.ndarray)
        assert isinstance(model.dix, np.ndarray)
        assert isinstance(model.six, np.ndarray)

    def test_strat_ix_partial(self, partial_large_sample):
        """Test stratum indices for participant-stratified model."""

        part_data, cnt_data, _ = partial_large_sample
        age_bins = AgeBins(0, 80, 5)

        model = Prem(part_data, cnt_data, age_bins)

        # Stratum indices should be valid (0 to K-1)
        unique_strata = np.unique(model.six)
        assert len(unique_strata) >= 1
        assert len(unique_strata) <= model.K
        assert np.all(unique_strata >= 0)
        assert np.all(unique_strata < model.K)

    def test_strat_ix_full(self, full_large_sample):
        """Test stratum indices for stratified model."""

        part_data, cnt_data, _ = full_large_sample
        age_bins = AgeBins(0, 80, 5)

        model = Prem(part_data, cnt_data, age_bins)

        # Stratum indices should be valid (0 to K-1)
        # Note: With participant and contact stratification, we get composite strata
        # For gender: M_part x M_cnt, M_part x F_cnt, F_part x M_cnt, F_part x F_cnt = 4 strata
        unique_strata = np.unique(model.six)
        assert len(unique_strata) >= 1
        assert len(unique_strata) <= model.K
        assert np.all(unique_strata >= 0)
        assert np.all(unique_strata < model.K)

    def test_composite_stratum_creation_partial(self, partial_multi_strat_large_sample):
        """Test composite stratification variable creation with - naming convention."""
        part_data, cnt_data, _ = partial_multi_strat_large_sample
        age_bins = AgeBins(0, 80, 5)

        model = Prem(part_data, cnt_data, age_bins)

        # Should have composite stratum column
        assert "stratum" in model.data.columns

        # Should have up to K unique strata
        unique_strata = model.data["stratum"].nunique()
        assert unique_strata <= model.K

        # Verify naming convention uses "-" separator
        sample_stratum = model.data["stratum"].iloc[0]
        assert (
            "-" in sample_stratum
        ), f"Stratum should use '-' separator, got: {sample_stratum}"

        # For multi-strat, should have format like "M-High", "F-Low"
        parts = sample_stratum.split("-")
        assert (
            len(parts) == 2
        ), f"Stratum should split into 2 parts with '-', got: {sample_stratum}"

    def test_composite_stratum_creation_full(self, full_multi_strat_large_sample):
        """Test composite stratification variable creation with -> naming convention."""
        part_data, cnt_data, _ = full_multi_strat_large_sample
        age_bins = AgeBins(0, 80, 5)

        model = Prem(part_data, cnt_data, age_bins)

        # Should have composite stratum column
        assert "stratum" in model.data.columns

        # Should have up to K unique strata
        unique_strata = model.data["stratum"].nunique()
        assert unique_strata <= model.K

        # Verify naming convention uses "->" separator
        sample_stratum = model.data["stratum"].iloc[0]
        assert (
            "->" in sample_stratum
        ), f"Stratum should use '->' separator, got: {sample_stratum}"

        # For multi-strat, should have format like "M_North->F_South"
        # (participant side on left, contact side on right)
        parts = sample_stratum.split("->")
        assert (
            len(parts) == 2
        ), f"Stratum should split into 2 parts with '->', got: {sample_stratum}"


class TestModel:
    """Test the NumPyro model specification."""

    def test_model_callable_unstratified(self, single_small_sample):
        """Test that model is callable for unstratified case."""
        from numpyro.handlers import seed

        part_data, cnt_data, _ = single_small_sample
        age_bins = AgeBins(0, 80, 5)

        model = Prem(part_data, cnt_data, age_bins)

        # Should not raise an exception
        # NumPyro models need to be called within a seed context
        try:
            with seed(rng_seed=0):
                model.model(y=model.y)
        except Exception as e:
            pytest.fail(f"Model call raised exception: {e}")

    def test_model_callable_stratified(self, full_multi_strat_large_sample):
        """Test that model is callable for stratified case."""
        from numpyro.handlers import seed

        part_data, cnt_data, _ = full_multi_strat_large_sample
        age_bins = AgeBins(0, 80, 5)

        model = Prem(part_data, cnt_data, age_bins)

        # Should not raise an exception
        # NumPyro models need to be called within a seed context
        try:
            with seed(rng_seed=0):
                model.model(y=model.y)
        except Exception as e:
            pytest.fail(f"Model call raised exception: {e}")

    def test_model_callable_second_order(self, single_small_sample):
        """Test that model is callable with a second-order IGMRF2D prior."""
        from numpyro.handlers import seed

        part_data, cnt_data, _ = single_small_sample
        age_bins = AgeBins(0, 80, 5)

        model = Prem(part_data, cnt_data, age_bins, order=2)

        # Should not raise an exception
        # NumPyro models need to be called within a seed context
        try:
            with seed(rng_seed=0):
                model.model(y=model.y)
        except Exception as e:
            pytest.fail(f"Model call raised exception: {e}")

    def test_print_model_shape_unstratified(self, single_small_sample):
        """Test print_model_shape for unstratified model."""
        part_data, cnt_data, _ = single_small_sample
        age_bins = AgeBins(0, 80, 5)

        model = Prem(part_data, cnt_data, age_bins)

        # Should not raise an exception
        try:
            model.print_model_shape()
        except Exception as e:
            pytest.fail(f"print_model_shape raised exception: {e}")

    def test_print_model_shape_stratified(self, full_multi_strat_large_sample):
        """Test print_model_shape for stratified model."""
        part_data, cnt_data, _ = full_multi_strat_large_sample
        age_bins = AgeBins(0, 80, 5)

        model = Prem(part_data, cnt_data, age_bins)

        # Should not raise an exception
        try:
            model.print_model_shape()
        except Exception as e:
            pytest.fail(f"print_model_shape raised exception: {e}")


class TestInference:
    """Test inference methods (basic checks only, not full MCMC runs)."""

    def test_mcmc_initialization_unstratified(self, single_small_sample):
        """Test that MCMC can be initialized (but not run fully)."""
        part_data, cnt_data, _ = single_small_sample
        age_bins = AgeBins(0, 80, 5)

        model = Prem(part_data, cnt_data, age_bins)

        # Just check that the method exists and accepts correct parameters
        assert hasattr(model, "run_inference_mcmc")
        assert callable(model.run_inference_mcmc)

    def test_mcmc_initialization_stratified(self, full_multi_strat_large_sample):
        """Test that MCMC can be initialized for stratified model."""
        part_data, cnt_data, _ = full_multi_strat_large_sample
        age_bins = AgeBins(0, 80, 5)

        model = Prem(part_data, cnt_data, age_bins)

        # Just check that the method exists
        assert hasattr(model, "run_inference_mcmc")
        assert callable(model.run_inference_mcmc)

    def test_svi_method_exists(self, single_small_sample):
        """Test that SVI method exists."""
        part_data, cnt_data, _ = single_small_sample
        age_bins = AgeBins(0, 80, 5)
        model = Prem(part_data, cnt_data, age_bins)

        assert hasattr(model, "run_inference_svi")
        assert callable(model.run_inference_svi)

    def test_posterior_predictive_methods_exist(self, single_small_sample):
        """Test that posterior predictive methods exist."""
        part_data, cnt_data, _ = single_small_sample
        age_bins = AgeBins(0, 80, 5)
        model = Prem(part_data, cnt_data, age_bins)

        assert hasattr(model, "posterior_predictive_mcmc")
        assert hasattr(model, "posterior_predictive_svi")
        assert callable(model.posterior_predictive_mcmc)
        assert callable(model.posterior_predictive_svi)


def _assert_summary_shape(cs, shape):
    """Check a ContactSummary's lower/central/upper bounds all have ``shape``."""
    assert isinstance(cs, ContactSummary)
    assert cs.lower.shape == shape
    assert cs.central.shape == shape
    assert cs.upper.shape == shape


class TestSummarizerIntegration:

    def test_single(self, single_small_sample):
        """Test ModelSummariserPrem integration for unstratified model."""
        part_data, cnt_data, pop_data = single_small_sample
        age_bins = AgeBins(0, 80, 5)
        model = Prem(part_data, cnt_data, age_bins)
        model.run_inference_svi(PRNGKey(0), num_steps=1)

        summariser = ModelSummariserPrem(model, pop_data, num_samples=20)

        # Depixilated summary
        sum_cint = summariser.summarise_cint(return_depixilated=True)
        _assert_summary_shape(sum_cint["All->All"], (81, 81))

        sum_mcint = summariser.summarise_mcint(return_depixilated=True)
        _assert_summary_shape(sum_mcint["All->All"], (81,))

        # Pixilated summary
        sum_cint_pix = summariser.summarise_cint(return_depixilated=False)
        _assert_summary_shape(sum_cint_pix["All->All"], (16, 16))

        sum_mcint_pix = summariser.summarise_mcint(return_depixilated=False)
        _assert_summary_shape(sum_mcint_pix["All->All"], (16,))

        # No population data
        summariser = ModelSummariserPrem(model, num_samples=20)

        sum_cint = summariser.summarise_cint()
        _assert_summary_shape(sum_cint["All->All"], (16, 16))

        with pytest.raises(ValueError):
            # Should raise error if requesting depixilated without pop data
            summariser.summarise_cint(return_depixilated=True)

    def test_partial(self, partial_small_sample):
        """Test ModelSummariserPrem integration for participant-stratified model."""
        part_data, cnt_data, pop_data = partial_small_sample
        age_bins = AgeBins(0, 80, 5)
        model = Prem(part_data, cnt_data, age_bins)
        model.run_inference_svi(PRNGKey(0), num_steps=1)

        summariser = ModelSummariserPrem(model, pop_data, num_samples=20)

        # Depixilated summary
        sum_cint = summariser.summarise_cint(return_depixilated=True)
        for label in sum_cint:
            _assert_summary_shape(sum_cint[label], (81, 81))

        sum_mcint = summariser.summarise_mcint(return_depixilated=True)
        for label in sum_mcint:
            _assert_summary_shape(sum_mcint[label], (81,))

        # Pixilated summary
        sum_cint_pix = summariser.summarise_cint(return_depixilated=False)
        for label in sum_cint_pix:
            _assert_summary_shape(sum_cint_pix[label], (16, 16))

        sum_mcint_pix = summariser.summarise_mcint(return_depixilated=False)
        for label in sum_mcint_pix:
            _assert_summary_shape(sum_mcint_pix[label], (16,))

    def test_partial_multi_strat(self, partial_multi_strat_large_sample):
        """Test ModelSummariserPrem integration for multi-variable participant-stratified model."""
        part_data, cnt_data, pop_data = partial_multi_strat_large_sample
        age_bins = AgeBins(0, 80, 5)
        model = Prem(part_data, cnt_data, age_bins)
        model.run_inference_svi(PRNGKey(0), num_steps=1)

        summariser = ModelSummariserPrem(model, pop_data, num_samples=20)

        # Depixilated summary
        sum_cint = summariser.summarise_cint(return_depixilated=True)
        for label in sum_cint:
            _assert_summary_shape(sum_cint[label], (81, 81))

        sum_mcint = summariser.summarise_mcint(return_depixilated=True)
        for label in sum_mcint:
            _assert_summary_shape(sum_mcint[label], (81,))

        # Pixilated summary
        sum_cint_pix = summariser.summarise_cint(return_depixilated=False)
        for label in sum_cint_pix:
            _assert_summary_shape(sum_cint_pix[label], (16, 16))

        sum_mcint_pix = summariser.summarise_mcint(return_depixilated=False)
        for label in sum_mcint_pix:
            _assert_summary_shape(sum_mcint_pix[label], (16,))
