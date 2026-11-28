from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..summariser._ModelSummariserPrem import ModelSummariserPrem
from ._base import (
    BaseModelEvaluator,
    aggregate_metrics,
    compute_metrics,
    validate_alpha,
)


class ModelEvaluatorPrem(BaseModelEvaluator):
    """
    Evaluator for Prem model performance against ground truth.

    Computes error metrics and uncertainty quantification statistics for estimated
    contact intensity matrices from Bayesian posterior samples. Supports both
    unstratified (K=1) and stratified (K>1) Prem models.

    **Evaluation Strategy**: The Prem model estimates contact matrices at age-group
    resolution (e.g., 5-year bins). To evaluate against fine-grained (1-year) ground
    truth, this evaluator uses **depixilation** to project binned posterior samples
    to fine-grained resolution before computing metrics. This tests how well the
    model reconstructs fine-grained contact patterns, not just binned aggregates.

    This evaluator follows the same design patterns as ModelEvaluatorBRC for consistency,
    with automatic detection of stratification mode and appropriate metric aggregation.

    Parameters
    ----------
    summariser : ModelSummariserPrem
        Summariser containing posterior samples and point estimates from a fitted
        Prem model (MCMC or SVI inference). Must have pop_data or age_dist for
        depixilation.
    cint_matrix_true : NDArray or Dict[str, NDArray]
        Ground truth contact intensity matrix at fine-grained (1-year) resolution.
        - For K=1: NDArray of shape (A_fine, A_fine) where A_fine is max age
        - For K>1: Dict mapping stratum labels to NDArray of shape (A_fine, A_fine)
    alpha : float, default=0.05
        Significance level for interval score and coverage computations.
        Must be in (0, 1).

    Attributes
    ----------
    summariser : ModelSummariserPrem
        Reference to the model summariser
    cint_true : NDArray or Dict[str, NDArray]
        Ground truth contact intensity matrices
    mcint_true : NDArray or Dict[str, NDArray]
        Ground truth marginal contact intensities (computed from cint_true)
    alpha : float
        Significance level for credible intervals
    strat_mode : str
        Stratification mode: "none", "partial", "full", or "mixed"
    K : int
        Number of strata (1 for unstratified)
    age_group_specs : AgeGroupSpecs
        Age bin definition from summariser
    pop_data : PopulationData or None
        Population data from summariser (may be None for legacy usage)
    age_dist : NDArray or None
        Legacy 1-year age distribution (for backward compatibility)

    Examples
    --------
    **Unstratified Prem model evaluation:**

    >>> # Fit model
    >>> prem = Prem(df_part, df_cnt, age_bins)
    >>> prem.run_inference_mcmc(rng_key, num_samples=1000)
    >>>
    >>> # Create summariser and evaluator
    >>> pop_data = PopulationData(df_pop, age_col='age', size_col='P')
    >>> summariser = ModelSummariserPrem(prem, pop_data=pop_data)
    >>> evaluator = ModelEvaluatorPrem(summariser, cint_true, alpha=0.05)
    >>>
    >>> # Evaluate contact intensity
    >>> cint_metrics = evaluator.evaluate_cint()
    >>> print(cint_metrics)
    >>>
    >>> # Evaluate marginal contact intensity
    >>> mcint_metrics = evaluator.evaluate_mcint()
    >>> print(mcint_metrics)
    >>>
    >>> # Get all metrics at once
    >>> all_metrics = evaluator.evaluate()

    **Stratified Prem model evaluation:**

    >>> # For stratified models
    >>> prem = Prem(df_part, df_cnt, age_bins, strat_var='gender')
    >>> prem.run_inference_svi(rng_key, guide, num_steps=5000)
    >>>
    >>> # True matrices organized by stratification
    >>> cint_true = {
    ...     "M->M": matrix_mm,
    ...     "M->F": matrix_mf,
    ...     "F->M": matrix_fm,
    ...     "F->F": matrix_ff
    ... }
    >>>
    >>> pop_data = PopulationData(df_pop, 'age', 'P', strat_var_cols=['gender'])
    >>> summariser = ModelSummariserPrem(prem, pop_data=pop_data)
    >>> evaluator = ModelEvaluatorPrem(summariser, cint_true, alpha=0.05)
    >>>
    >>> # Returns metrics aggregated by stratum label
    >>> metrics = evaluator.evaluate()

    Notes
    -----
    **Depixilation for Evaluation**:
    The Prem model estimates at age-group resolution (e.g., 5-year bins), but
    ground truth is often available at fine-grained (1-year) resolution. This
    evaluator automatically applies depixilation to project posterior samples
    from age-group to fine-grained resolution before computing metrics. This
    approach tests the model's ability to reconstruct fine-grained contact
    patterns, not just fit binned aggregates.

    **Metrics Computed**:
    - RMSE: Root mean squared error
    - MAE: Mean absolute error
    - MAPE: Mean absolute percentage error
    - Interval score: Uncertainty calibration (penalizes width + coverage failures)
    - Coverage: Proportion of true values within credible intervals

    **Implementation Notes**:
    - Caching is used to avoid redundant computation
    - For stratified models, metrics are computed for each stratum pair and aggregated
    - All metrics are returned as pandas DataFrames for easy analysis
    - Requires PopulationData or age_dist in summariser for depixilation

    See Also
    --------
    ModelSummariserPrem : Summarises posterior distributions from Prem models
    ModelEvaluatorBRC : Similar evaluator for BRC models
    """

    def __init__(
        self,
        summariser: ModelSummariserPrem,
        cint_matrix_true: NDArray[np.float64] | Dict[str, NDArray],
        alpha: float = 0.05,
    ) -> None:
        """
        Initialize ModelEvaluatorPrem.

        Parameters
        ----------
        summariser : ModelSummariserPrem
            Summariser with posterior samples from fitted Prem model
        cint_matrix_true : NDArray or Dict[str, NDArray]
            True contact intensity matrix (K=1) or dict of matrices (K>1)
            - K=1: NDArray of shape (A, A) at 1-year resolution
            - K>1: Dict mapping stratum labels to NDArray of shape (A, A)
        alpha : float, default=0.05
            Significance level for interval metrics (must be in (0, 1))

        Raises
        ------
        ValueError
            If alpha not in (0, 1), or if model hasn't been fitted
        TypeError
            If summariser is not ModelSummariserPrem instance, or incompatible types
        """
        super().__init__(summariser, cint_matrix_true, alpha)

        # Prem-specific attributes extracted from the summariser
        self.strat_mode = self.summariser.strat_mode
        self.K = self.summariser.K
        self.age_group_specs = self.summariser.age_group_specs
        self.pop_data = self.summariser.pop_data
        self.age_dist = self.summariser.age_dist
        self._has_fine_age_dist = self.age_dist is not None or (
            self.pop_data is not None and self.pop_data.n_ages >= self.age_group_specs.range
        )

    def _validate_summariser(self, summariser: ModelSummariserPrem) -> None:
        """Validate that summariser has required posterior samples."""
        # Check for required attributes rather than strict instance check
        required_attrs = [
            "post_cint_samples",
            "strat_mode",
            "K",
            "age_group_specs",
            "summarise_cint",
            "summarise_mcint",
        ]
        missing_attrs = [
            attr for attr in required_attrs if not hasattr(summariser, attr)
        ]

        if missing_attrs:
            raise TypeError(
                f"Summariser missing required attributes: {missing_attrs}. "
                f"Expected ModelSummariserPrem instance or compatible object."
            )

        if summariser.post_cint_samples is None:
            raise ValueError(
                "Summariser must have posterior contact intensity samples. "
                "Ensure MCMC or SVI inference was run on the Prem model."
            )

    def _aggregate_true_matrix_to_bins(
        self, cint_true: NDArray | Dict[str, NDArray]
    ) -> NDArray | Dict[str, NDArray]:
        """
        Aggregate fine-grained ground truth matrix to match model's age bins.

        If the ground truth matrix dimensions don't match the model's age bin count,
        this method aggregates it appropriately using the age bin edges.

        Parameters
        ----------
        cint_true : NDArray or Dict[str, NDArray]
            Ground truth contact intensity matrix(ces)

        Returns
        -------
        NDArray or Dict[str, NDArray]
            Aggregated matrix(ces) matching model dimensions
        """
        n_bins = len(self.age_group_specs)

        if isinstance(cint_true, dict):
            # Stratified case
            aggregated = {}
            for label, matrix in cint_true.items():
                if matrix.shape[0] == n_bins:
                    # Already at correct resolution
                    aggregated[label] = matrix
                else:
                    # Need to aggregate
                    aggregated[label] = self._aggregate_single_matrix(matrix)
            return aggregated
        else:
            # Unstratified case
            if cint_true.shape[0] == n_bins:
                # Already at correct resolution
                return cint_true
            else:
                # Need to aggregate
                return self._aggregate_single_matrix(cint_true)

    def _aggregate_single_matrix(self, matrix: NDArray) -> NDArray:
        """
        Aggregate a single fine-grained contact matrix to age bins.

        Uses the age bin edges to aggregate both rows and columns.

        Parameters
        ----------
        matrix : NDArray
            Fine-grained contact intensity matrix of shape (A_fine, A_fine)

        Returns
        -------
        NDArray
            Aggregated matrix of shape (n_bins, n_bins)
        """
        import pandas as _pd

        n_fine = matrix.shape[0]
        n_bins = len(self.age_group_specs)

        # Create age bin edges
        age_edges = self.age_group_specs.left
        age_labels = list(range(n_bins))

        # Create DataFrame for fine-grained ages
        df_ages = _pd.DataFrame({"age": range(n_fine)})

        # Bin the ages
        df_ages["age_bin"] = _pd.cut(
            df_ages["age"],
            bins=age_edges,
            labels=age_labels,
            right=False,
            include_lowest=True,
        )

        # Create aggregated matrix
        aggregated = np.zeros((n_bins, n_bins))

        # Aggregate by summing contacts within each bin pair
        for i_bin in range(n_bins):
            # Get fine-grained ages in this participant age bin
            i_ages = df_ages[df_ages["age_bin"] == i_bin].index.tolist()

            for j_bin in range(n_bins):
                # Get fine-grained ages in this contact age bin
                j_ages = df_ages[df_ages["age_bin"] == j_bin].index.tolist()

                if i_ages and j_ages:
                    # Sum contacts from all (i_age, j_age) pairs in these bins
                    # Then average to get contact intensity per person in age bin
                    aggregated[i_bin, j_bin] = matrix[
                        np.ix_(i_ages, j_ages)
                    ].sum() / len(i_ages)

        return aggregated

    def evaluate_cint(self, alpha: Optional[float] = None) -> pd.DataFrame:
        """
        Evaluate the posterior contact intensity matrix.

        Computes RMSE, MAE, MAPE, interval score, and coverage for the
        estimated contact intensity matrix against ground truth.

        Parameters
        ----------
        alpha : float, optional
            Significance level for interval metrics. If None, uses self.alpha

        Returns
        -------
        pd.DataFrame
            Evaluation metrics:
            - For K=1: Single row DataFrame
            - For K>1: One row per stratum label, plus aggregated "all" row

        Notes
        -----
        Results are cached to avoid redundant computation when called multiple times
        with the same alpha value.

        Examples
        --------
        >>> evaluator = ModelEvaluatorPrem(summariser, cint_true)
        >>> cint_metrics = evaluator.evaluate_cint(alpha=0.05)
        >>> print(f"RMSE: {cint_metrics['rmse'].values[0]:.3f}")
        >>> print(f"Coverage: {cint_metrics['coverage'].values[0]:.1f}%")
        """
        if alpha is None:
            alpha = self.alpha
        else:
            validate_alpha(alpha)

        cache_key = f"cint_alpha{alpha}"

        if cache_key not in self._metrics_cache:
            if self.K == 1:
                # Unstratified Prem model
                # Depixilate binned estimates to fine-grained resolution for comparison
                summary = self.summariser.summarise_cint(
                    alpha=alpha, return_depixilated=True
                )

                y_true = (
                    self.cint_true["All->All"]
                    if isinstance(self.cint_true, dict)
                    else self.cint_true
                )
                y_est = summary["All->All"].central  # median
                y_low = summary["All->All"].lower  # lower bound
                y_high = summary["All->All"].upper  # upper bound

                rmse, mae, mape, int_score, coverage = compute_metrics(
                    y_true, y_est, y_low, y_high, alpha=alpha
                )

                metrics_df = pd.DataFrame(
                    {
                        "cat": ["all"],
                        "rmse": [rmse],
                        "mae": [mae],
                        "mape": [mape],
                        "interval_score": [int_score],
                        "coverage": [coverage],
                    }
                )

            else:  # K > 1 (stratified)
                # Stratified Prem model
                # Depixilate binned estimates to fine-grained resolution for comparison
                summary_dict = self.summariser.summarise_cint(
                    alpha=alpha, return_depixilated=True
                )

                # {label: ContactSummary} -> {label: (lower, median, upper) array}
                summary_arrays = {k: v.to_array() for k, v in summary_dict.items()}
                metrics_df = aggregate_metrics(self.cint_true, summary_arrays, alpha=alpha)

            self._metrics_cache[cache_key] = metrics_df

        return self._metrics_cache[cache_key].copy()

    def evaluate_mcint(self, alpha: Optional[float] = None) -> pd.DataFrame:
        """
        Evaluate the posterior marginal contact intensity.

        Computes RMSE, MAE, MAPE, interval score, and coverage for the
        estimated marginal contact intensity (sum over contact ages) against
        ground truth.

        Parameters
        ----------
        alpha : float, optional
            Significance level for interval metrics. If None, uses self.alpha

        Returns
        -------
        pd.DataFrame
            Evaluation metrics with same structure as evaluate_cint()

        Notes
        -----
        Marginal contact intensity represents the total contacts made by each
        age group, summed across all contact ages. This is often a key quantity
        of interest in epidemiological applications.

        Results are cached to avoid redundant computation.

        Examples
        --------
        >>> evaluator = ModelEvaluatorPrem(summariser, cint_true)
        >>> mcint_metrics = evaluator.evaluate_mcint(alpha=0.05)
        >>> print(f"Marginal RMSE: {mcint_metrics['rmse'].values[0]:.3f}")
        """
        if alpha is None:
            alpha = self.alpha
        else:
            validate_alpha(alpha)

        cache_key = f"mcint_alpha{alpha}"

        if cache_key not in self._metrics_cache:
            if self.K == 1:
                # Unstratified Prem model
                # Use depixilation to project binned estimates to fine-grained resolution
                summary = self.summariser.summarise_mcint(
                    alpha=alpha, return_depixilated=True
                )
                y_true = self.mcint_true
                y_est = summary["All->All"].central  # median
                y_low = summary["All->All"].lower  # lower bound
                y_high = summary["All->All"].upper  # upper bound

                rmse, mae, mape, int_score, coverage = compute_metrics(
                    y_true, y_est, y_low, y_high, alpha=alpha
                )

                metrics_df = pd.DataFrame(
                    {
                        "cat": ["all"],
                        "rmse": [rmse],
                        "mae": [mae],
                        "mape": [mape],
                        "interval_score": [int_score],
                        "coverage": [coverage],
                    }
                )

            else:  # K > 1 (stratified)
                # Stratified Prem model
                # Use depixilation to project binned estimates to fine-grained resolution
                summary_dict = self.summariser.summarise_mcint(
                    alpha=alpha, return_depixilated=True
                )
                summary_arrays = {k: v.to_array() for k, v in summary_dict.items()}
                metrics_df = aggregate_metrics(self.mcint_true, summary_arrays, alpha=alpha)

            self._metrics_cache[cache_key] = metrics_df

        return self._metrics_cache[cache_key].copy()

    def get_point_estimate_error(
        self, quantity: Literal["cint", "mcint"] = "cint"
    ) -> Dict[str, float]:
        """
        Compute error metrics for point estimates (mean from posterior).

        Parameters
        ----------
        quantity : {'cint', 'mcint'}, default='cint'
            Which quantity to compute errors for

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - 'rmse': Root mean squared error
            - 'mae': Mean absolute error
            - 'mape': Mean absolute percentage error
            - 'relative_error': Relative Frobenius norm error

        Notes
        -----
        This method uses the posterior mean instead of median, which may differ
        slightly from the metrics in evaluate_cint() and evaluate_mcint().

        For stratified models, errors are aggregated across all stratum pairs.

        Examples
        --------
        >>> evaluator = ModelEvaluatorPrem(summariser, cint_true)
        >>> point_errors = evaluator.get_point_estimate_error("cint")
        >>> print(f"Point estimate RMSE: {point_errors['rmse']:.3f}")
        """
        if quantity not in ["cint", "mcint"]:
            raise ValueError(f"quantity must be 'cint' or 'mcint', got {quantity}")

        cache_key = f"point_{quantity}"

        if cache_key not in self._metrics_cache:
            # Get point estimates from summariser
            point_est = self.summariser.get_point_estimates(quantity)
            y_hat = point_est["mean"]

            # Get true values
            if quantity == "cint":
                y_true = self.cint_true
            else:
                y_true = self.mcint_true

            # Compute errors
            if self.K == 1:
                diff = y_hat - y_true
                rmse = float(np.sqrt(np.mean(diff**2)))
                mae = float(np.mean(np.abs(diff)))

                # Avoid division by zero in MAPE
                mask = y_true != 0
                if np.any(mask):
                    mape = float(np.mean(np.abs(diff[mask] / y_true[mask])) * 100)
                else:
                    mape = np.inf

                # Relative error - use Frobenius norm for matrices, L2 norm for vectors
                if diff.ndim == 2:
                    norm_diff = np.linalg.norm(diff, "fro")
                    norm_true = np.linalg.norm(y_true, "fro")
                else:
                    norm_diff = np.linalg.norm(diff)
                    norm_true = np.linalg.norm(y_true)
                relative_error = float(
                    norm_diff / norm_true if norm_true > 0 else np.inf
                )

            else:  # K > 1 (stratified)
                # Aggregate across all stratum pairs
                all_diffs = []
                all_true = []

                for label in y_true.keys():
                    diff = y_hat[label] - y_true[label]
                    all_diffs.append(diff.flatten())
                    all_true.append(y_true[label].flatten())

                all_diffs = np.concatenate(all_diffs)
                all_true = np.concatenate(all_true)

                rmse = float(np.sqrt(np.mean(all_diffs**2)))
                mae = float(np.mean(np.abs(all_diffs)))

                mask = all_true != 0
                if np.any(mask):
                    mape = float(
                        np.mean(np.abs(all_diffs[mask] / all_true[mask])) * 100
                    )
                else:
                    mape = np.inf

                norm_diff = np.linalg.norm(all_diffs)
                norm_true = np.linalg.norm(all_true)
                relative_error = float(
                    norm_diff / norm_true if norm_true > 0 else np.inf
                )

            result = {
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "relative_error": relative_error,
            }

            self._metrics_cache[cache_key] = pd.DataFrame([result])

        return self._metrics_cache[cache_key].iloc[0].to_dict()

    def get_cache_info(self) -> Dict[str, any]:
        """
        Get information about cached metrics.

        Returns
        -------
        Dict
            Cache statistics including:
            - 'n_cached': Number of cached results
            - 'cached_metrics': List of cached metric keys
            - 'strat_mode': Stratification mode
            - 'K': Number of strata
            - 'alpha': Current significance level
        """
        return {
            "n_cached": len(self._metrics_cache),
            "cached_metrics": list(self._metrics_cache.keys()),
            "strat_mode": self.strat_mode,
            "K": self.K,
            "alpha": self.alpha,
        }
