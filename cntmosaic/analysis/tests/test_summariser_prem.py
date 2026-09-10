"""
Comprehensive tests for ModelSummariserPrem with stratification support.

Tests cover:
- All 4 stratification modes (none, partial, full, mixed)
- Reciprocity adjustments for appropriate stratification modes
- Depixilation with stratified populations
- PopulationData integration
- Output format compatibility with BRC models
- Backward compatibility with legacy API
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from jax.random import PRNGKey

from cntmosaic.analysis import ModelSummariserPrem
from cntmosaic.dataloader.containers import PopulationData
from cntmosaic.datasets import load_age_distribution
from cntmosaic.models import Prem
from cntmosaic.utils import AgeBins

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def age_bins():
    """Create age bins for testing."""
    return AgeBins(min=0, max=24, step=5)


@pytest.fixture
def df_age_dist_fine():
    """Create fine-grained (1-year) age distribution."""
    ages = np.arange(25)
    population = np.random.uniform(1000, 2000, 25)
    return pd.DataFrame({"age": ages, "P": population})


@pytest.fixture
def df_age_dist_coarse(age_bins):
    """Create coarse (5-year) age distribution matching age bins."""
    ages = age_bins.left
    population = np.random.uniform(
        5000, 10000, len(ages)
    )  # Higher totals for aggregated groups
    return pd.DataFrame({"age": ages, "P": population})


@pytest.fixture
def df_age_dist_stratified(age_bins):
    """Create stratified age distribution for full stratification tests."""
    # Create data matching the age bins (5 age groups, 2 genders = 10 rows)
    age_groups = age_bins.left
    ages = np.concatenate([age_groups, age_groups])  # Repeat for M and F
    gender = np.repeat(["M", "F"], len(age_groups))
    population = np.random.uniform(1000, 2000, len(ages))
    return pd.DataFrame({"age": ages, "gender": gender, "P": population})


@pytest.fixture
def df_age_dist_stratified_fine():
    """Create fine-grained stratified age distribution."""
    ages = np.tile(np.arange(25), 2)
    gender = np.repeat(["M", "F"], 25)
    population = np.random.uniform(1000, 2000, 50)
    return pd.DataFrame({"age": ages, "gender": gender, "P": population})


@pytest.fixture
def pop_data_unstratified(df_age_dist_coarse):
    """Create PopulationData for unstratified case (coarse resolution)."""
    return PopulationData(df_age_dist_coarse, age_col="age", size_col="P")


@pytest.fixture
def pop_data_unstratified_fine(df_age_dist_fine):
    """Create PopulationData for unstratified case (fine resolution)."""
    return PopulationData(df_age_dist_fine, age_col="age", size_col="P")


@pytest.fixture
def pop_data_stratified(df_age_dist_stratified):
    """Create PopulationData for stratified case (coarse resolution)."""
    return PopulationData(
        df_age_dist_stratified, age_col="age", size_col="P", strat_var_cols=["gender"]
    )


@pytest.fixture
def pop_data_stratified_fine(df_age_dist_stratified_fine):
    """Create PopulationData for stratified case (fine resolution)."""
    return PopulationData(
        df_age_dist_stratified_fine,
        age_col="age",
        size_col="P",
        strat_var_cols=["gender"],
    )


@pytest.fixture
def mock_prem_unstratified(age_bins):
    """Create mock Prem model for K=1 (unstratified)."""
    # Create minimal mock
    prem = type("Prem", (), {})()
    prem.K = 1
    prem.age_group_specs = age_bins
    prem.data = pd.DataFrame({"stratum": ["All->All"] * 10})

    # Mock posterior samples
    n_samples = 20
    D = C = len(age_bins.left)  # 5 age groups

    prem._mcmc_result = type("MCMCResult", (), {})()
    prem._mcmc_result.get_samples = lambda: {
        "log_cint": np.random.normal(0, 0.5, (n_samples, D, C)),
        "beta0": np.random.normal(0, 0.1, (n_samples,)),
        "tau": np.random.gamma(2, 0.5, (n_samples,)),
    }
    prem._svi_result = None
    prem.inference_type = "mcmc"
    prem.inference_method = "mcmc"
    prem.draw_posterior_samples = lambda rng_key, num_samples=1000: prem._mcmc_result.get_samples()

    return prem


@pytest.fixture
def mock_prem_stratified_full(age_bins):
    """Create mock Prem model for K=4 (full stratification)."""
    prem = type("Prem", (), {})()
    prem.K = 4
    prem.age_group_specs = age_bins
    prem.strat_vars_part = ["gender"]
    prem.strat_vars_cnt = ["gender"]
    prem.inference_type = "mcmc"

    # Create stratum labels
    strata_labels = ["M->M", "M->F", "F->M", "F->F"]
    prem.data = pd.DataFrame({"stratum": np.repeat(strata_labels, 10)})

    # Mock part_data with part_gender column for validation
    part_data_mock = type("ParticipantData", (), {})()
    part_data_mock.data = pd.DataFrame({
        "part_gender": pd.Categorical(["M", "F"] * 20, categories=["M", "F"])
    })
    prem.part_data = part_data_mock

    # Mock posterior samples for stratified model
    n_samples = 20
    K = 4  # 4 strata total (M->M, M->F, F->M, F->F)
    D = C = len(age_bins.left)

    prem._mcmc_result = type("MCMCResult", (), {})()
    prem._mcmc_result.get_samples = lambda: {
        "log_cint": np.random.normal(0, 0.5, (n_samples, K, D, C)),
    }
    prem._svi_result = None
    prem.inference_method = "mcmc"
    prem.draw_posterior_samples = lambda rng_key, num_samples=1000: prem._mcmc_result.get_samples()

    return prem


# ============================================================================
# Test Initialization
# ============================================================================


class TestModelSummariserPremInitialization:
    """Test initialization and stratification detection."""

    def test_init_unstratified(self, mock_prem_unstratified, pop_data_unstratified):
        """Test initialization for K=1."""
        summariser = ModelSummariserPrem(
            mock_prem_unstratified, pop_data=pop_data_unstratified
        )

        assert summariser.K == 1
        assert summariser.strat_mode == "none"
        assert summariser.strata_labels == ["All->All"]
        assert summariser.post_cint_samples is not None

    def test_init_stratified_full(self, mock_prem_stratified_full, pop_data_stratified):
        """Test initialization for K=4 (full stratification)."""
        summariser = ModelSummariserPrem(
            mock_prem_stratified_full, pop_data=pop_data_stratified
        )

        assert summariser.K == 4
        assert summariser.strat_mode == "full"
        assert len(summariser.strata_labels) == 4
        assert "M->M" in summariser.strata_labels
        assert isinstance(summariser.post_cint_samples, dict)

    def test_init_without_inference_fails(self, age_bins):
        """Test that initialization fails if no inference has been run."""
        prem = type("Prem", (), {})()
        prem._mcmc_result = None
        prem._svi_result = None
        prem.inference_method = None
        prem.data = pd.DataFrame()

        with pytest.raises(ValueError, match="Either MCMC or SVI must have been run"):
            ModelSummariserPrem(prem)

    def test_init_with_pop_data(self, mock_prem_unstratified, df_age_dist_fine):
        """Test initialization with PopulationData (new API)."""
        pop_data = PopulationData(df_age_dist_fine, age_col="age", size_col="P")
        summariser = ModelSummariserPrem(mock_prem_unstratified, pop_data=pop_data)

        assert summariser.age_dist is not None
        assert len(summariser.age_dist) == 25


# ============================================================================
# Test Stratum Labels
# ============================================================================


class TestStratumLabels:
    """Test stratum label creation and formatting."""

    def test_labels_unstratified(self, mock_prem_unstratified):
        """Test label creation for K=1."""
        summariser = ModelSummariserPrem(mock_prem_unstratified)
        assert summariser.strata_labels == ["All->All"]

    def test_labels_stratified(self, mock_prem_stratified_full):
        """Test label creation for K>1."""
        summariser = ModelSummariserPrem(mock_prem_stratified_full)

        # Should have participant->contact format
        for label in summariser.strata_labels:
            assert "->" in label
            source, target = label.split("->")
            assert source in ["M", "F"]
            assert target in ["M", "F"]

    def test_labels_ordering(self, mock_prem_stratified_full):
        """Test that labels maintain correct ordering."""
        summariser = ModelSummariserPrem(mock_prem_stratified_full)

        # Labels are sorted alphabetically when not categorical
        expected_labels = sorted(["M->M", "M->F", "F->M", "F->F"])
        assert summariser.strata_labels == expected_labels


# ============================================================================
# Test Reciprocity
# ============================================================================


class TestReciprocityAdjustment:
    """Test reciprocity adjustment for different stratification modes."""

    def test_reciprocity_unstratified(
        self, mock_prem_unstratified, pop_data_unstratified
    ):
        """Test reciprocity for K=1."""
        summariser = ModelSummariserPrem(
            mock_prem_unstratified, pop_data=pop_data_unstratified
        )

        samples = summariser.post_cint_samples
        adjusted = ModelSummariserPrem.apply_reciprocity(
            samples, pop_data_unstratified, strat_mode="none"
        )

        assert adjusted.shape == samples.shape
        # Verify some degree of symmetry (not perfect due to population weighting)
        assert not np.allclose(adjusted, samples)  # Should be different

    def test_reciprocity_partial_warns(self, pop_data_unstratified):
        """Test that reciprocity warns for partial stratification."""
        samples = {"M->All": np.random.rand(10, 5, 5)}

        with pytest.warns(UserWarning, match="Reciprocity not applied for partial"):
            result = ModelSummariserPrem.apply_reciprocity(
                samples,
                pop_data_unstratified,
                strat_mode="partial",
                strata_labels=["M->All"],
            )

        # Should return unchanged
        assert np.array_equal(result["M->All"], samples["M->All"])

    def test_reciprocity_full_stratification(self, pop_data_stratified):
        """Test reciprocity for full stratification."""
        # Create sample data
        n_samples, D, C = 10, 5, 5
        samples = {
            "M->M": np.random.rand(n_samples, D, C),
            "M->F": np.random.rand(n_samples, D, C),
            "F->M": np.random.rand(n_samples, D, C),
            "F->F": np.random.rand(n_samples, D, C),
        }

        adjusted = ModelSummariserPrem.apply_reciprocity(
            samples,
            pop_data_stratified,
            strat_mode="full",
            strata_labels=["M->M", "M->F", "F->M", "F->F"],
        )

        assert isinstance(adjusted, dict)
        assert len(adjusted) == 4
        # Adjusted should be different from original
        assert not np.allclose(adjusted["M->M"], samples["M->M"])

    def test_reciprocity_missing_pop_data_fails(self):
        """Test that reciprocity fails gracefully without pop_data."""
        samples = np.random.rand(10, 5, 5)

        with pytest.raises(ValueError, match="PopulationData required"):
            ModelSummariserPrem.apply_reciprocity(
                samples, pop_data=None, strat_mode="none"
            )


# ============================================================================
# Test Depixilation
# ============================================================================


class TestDepixilation:
    """Test depixilation with stratified populations."""

    def test_depixilate_unstratified(
        self, mock_prem_unstratified, pop_data_unstratified, pop_data_unstratified_fine
    ):
        """Test depixilation for K=1."""
        summariser = ModelSummariserPrem(
            mock_prem_unstratified, pop_data=pop_data_unstratified
        )

        samples = summariser.post_cint_samples
        depix = summariser._depixilate_samples(
            samples,
            pop_data_unstratified_fine,  # Use fine population for depixilation
            strat_mode="none",
        )

        # Should expand from age groups (5) to 1-year ages (25)
        assert depix.shape == (samples.shape[0], 25, 25)

    def test_depixilate_stratified(
        self, mock_prem_stratified_full, pop_data_stratified, pop_data_stratified_fine
    ):
        """Test depixilation for K>1."""
        summariser = ModelSummariserPrem(
            mock_prem_stratified_full, pop_data=pop_data_stratified
        )

        samples = summariser.post_cint_samples
        depix = summariser._depixilate_samples(
            samples,
            pop_data_stratified_fine,  # Use fine population for depixilation
            strat_mode="full",
            strata_labels=summariser.strata_labels,
        )

        assert isinstance(depix, dict)
        for label, depix_sample in depix.items():
            # Should expand to 1-year ages
            assert depix_sample.shape[1:] == (25, 25)

    def test_depixilate_uses_source_population(
        self, mock_prem_stratified_full, pop_data_stratified, pop_data_stratified_fine
    ):
        """Test that depixilation uses source stratum population."""
        summariser = ModelSummariserPrem(
            mock_prem_stratified_full, pop_data=pop_data_stratified
        )

        # The key insight: M->F and M->M should use same source (M) population
        # This is implicit in the implementation but hard to test directly
        # We verify it doesn't crash and produces reasonable output
        samples = summariser.post_cint_samples
        depix = summariser._depixilate_samples(
            samples,
            pop_data_stratified_fine,  # Use fine population for depixilation
            strat_mode="full",
            strata_labels=summariser.strata_labels,
        )

        # Verify all strata processed
        assert "M->M" in depix
        assert "M->F" in depix


# ============================================================================
# Test Output Format
# ============================================================================


class TestSummariseCintOutputFormat:
    """Test output format matches BRC models."""

    def test_output_format_k1(self, mock_prem_unstratified, pop_data_unstratified):
        """Test K=1 returns ContactSummary whose to_array() has shape (3, A, A)."""
        summariser = ModelSummariserPrem(
            mock_prem_unstratified, pop_data=pop_data_unstratified
        )

        summary = summariser.summarise_cint(alpha=0.05)

        # Always returns Dict keyed by strata label
        assert isinstance(summary, dict)
        assert "All->All" in summary
        # to_array() stacks [lower, central, upper] → (3, A, A) where A = 5
        arr = summary["All->All"].to_array()
        assert arr.shape == (3, 5, 5)
        # Order: lower ≤ central ≤ upper
        assert arr[0, 0, 0] <= arr[1, 0, 0] <= arr[2, 0, 0]

    def test_output_format_k_gt_1(self, mock_prem_stratified_full, pop_data_stratified):
        """Test K>1 returns Dict[str, ContactSummary]."""
        summariser = ModelSummariserPrem(
            mock_prem_stratified_full, pop_data=pop_data_stratified
        )

        summary = summariser.summarise_cint(alpha=0.05)

        # Should return Dict
        assert isinstance(summary, dict)
        # Should have all strata
        assert "M->M" in summary
        assert "M->F" in summary

        # Each value should be a ContactSummary; to_array() gives (3, A, A)
        for label, cs in summary.items():
            arr = cs.to_array()
            assert arr.shape == (3, 5, 5)
            # Check ordering: lower ≤ central ≤ upper element-wise
            assert np.all(arr[0] <= arr[1])
            assert np.all(arr[1] <= arr[2])

    def test_summarise_cint_k1_returns_dict(
        self, mock_prem_unstratified, pop_data_unstratified
    ):
        """Test that summarise_cint always returns a dict (even for K=1)."""
        summariser = ModelSummariserPrem(
            mock_prem_unstratified, pop_data=pop_data_unstratified
        )
        summary = summariser.summarise_cint(alpha=0.05)
        assert isinstance(summary, dict)
        assert "All->All" in summary


# ============================================================================
# Test Combined Operations
# ============================================================================


class TestCombinedOperations:
    """Test combinations of reciprocity and depixilation.

    Note: Combined operations require careful handling of population resolution.
    Reciprocity needs coarse-grained population matching model age groups,
    while depixilation needs fine-grained target population.
    """

    @pytest.mark.skip(
        reason="TODO: Requires implementation support for different population resolutions in combined operations"
    )
    def test_reciprocity_then_depixilate_k1(
        self, mock_prem_unstratified, pop_data_unstratified_fine
    ):
        """Test reciprocity + depixilation for K=1.

        TODO: Implement automatic population aggregation/disaggregation when both
        operations are requested with different resolutions.
        """
        summariser = ModelSummariserPrem(
            mock_prem_unstratified, pop_data=pop_data_unstratified_fine
        )

        summary = summariser.summarise_cint(
            alpha=0.05, apply_reciprocity=True, return_depixilated=True
        )

        # Should return fine-grained (25 ages)
        assert summary.shape == (3, 25, 25)

    @pytest.mark.skip(
        reason="TODO: Requires implementation support for different population resolutions in combined operations"
    )
    def test_reciprocity_then_depixilate_full(
        self, mock_prem_stratified_full, pop_data_stratified_fine
    ):
        """Test reciprocity + depixilation for full stratification.

        TODO: Implement automatic population aggregation/disaggregation when both
        operations are requested with different resolutions.
        """
        summariser = ModelSummariserPrem(
            mock_prem_stratified_full, pop_data=pop_data_stratified_fine
        )

        summary = summariser.summarise_cint(
            alpha=0.05, apply_reciprocity=True, return_depixilated=True
        )

        # Should return Dict with fine-grained matrices
        assert isinstance(summary, dict)
        for label, quantiles in summary.items():
            assert quantiles.shape == (3, 25, 25)


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_missing_population_for_reciprocity(self, mock_prem_unstratified):
        """Test error when reciprocity requested without population data."""
        summariser = ModelSummariserPrem(mock_prem_unstratified)

        with pytest.raises(ValueError, match="PopulationData required"):
            summariser.summarise_cint(alpha=0.05, apply_reciprocity=True)

    def test_missing_population_for_depixilation(self, mock_prem_unstratified):
        """Test error when depixilation requested without population data."""
        summariser = ModelSummariserPrem(mock_prem_unstratified)

        with pytest.raises(ValueError, match="pop_data must be provided"):
            summariser.summarise_cint(alpha=0.05, return_depixilated=True)

    def test_invalid_alpha(self, mock_prem_unstratified):
        """Test error for invalid alpha values."""
        summariser = ModelSummariserPrem(mock_prem_unstratified)

        with pytest.raises(ValueError, match="alpha must be in"):
            summariser.summarise_cint(alpha=1.5)

    def test_cache_functionality(self, mock_prem_unstratified, pop_data_unstratified):
        """Test that caching works correctly."""
        summariser = ModelSummariserPrem(
            mock_prem_unstratified, pop_data=pop_data_unstratified
        )

        # First call
        summary1 = summariser.summarise_cint(alpha=0.05)

        # Second call should use cache
        summary2 = summariser.summarise_cint(alpha=0.05)

        # Should be identical (from cache)
        assert summary1.keys() == summary2.keys()
        for key in summary1:
            assert np.array_equal(summary1[key], summary2[key])

        # Cache info should show 1 cached item
        cache_info = summariser.get_cache_info()
        assert cache_info["n_cached"] >= 1

    def test_force_recompute(self, mock_prem_unstratified, pop_data_unstratified):
        """Test force_recompute parameter."""
        summariser = ModelSummariserPrem(
            mock_prem_unstratified, pop_data=pop_data_unstratified
        )

        summary1 = summariser.summarise_cint(alpha=0.05)
        summary2 = summariser.summarise_cint(alpha=0.05, force_recompute=True)

        # Shapes should match but values might differ slightly due to recomputation
        assert summary1["All->All"].to_array().shape == summary2["All->All"].to_array().shape


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
