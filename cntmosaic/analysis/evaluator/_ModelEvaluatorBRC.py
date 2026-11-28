import warnings
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..summariser._ModelSummariser import ModelSummariser
from ._base import (
    BaseModelEvaluator,
    aggregate_metrics,
    compute_metrics,
    validate_alpha,
)


class ModelEvaluatorBRC(BaseModelEvaluator):
    """
    Evaluator for AgeMix and GenMix model performance against ground truth.

    Computes error metrics and uncertainty quantification statistics for estimated
    contact intensity matrices from Bayesian posterior samples. Supports both
    standard age-only models (AgeMixFF, AgeMixFC, AgeMixCC) and hierarchical
    generalised mixing models (GenMixFF, GenMixFC).

    Parameters
    ----------
    summariser : ModelSummariser
        Summariser containing posterior samples from a fitted AgeMix or GenMix
        model (MCMC or SVI inference).
    cint_matrix_true : NDArray or dict
        Ground truth contact intensity matrix.
        - For agemix models: NDArray of shape (A, A)
        - For genmix models: dict mapping stratum labels to NDArray
    alpha : float, default=0.05
        Significance level for interval score and coverage computations.
        Must be in (0, 1).

    Attributes
    ----------
    summariser : ModelSummariser
        Reference to the model summariser.
    model_type : str
        Either ``"agemix"`` or ``"genmix"``, auto-detected from the summariser.

    Examples
    --------
    >>> model = AgeMixFF(dataloader, priors, likelihood="poisson")
    >>> model.run_inference_mcmc(rng_key, num_samples=1000)
    >>> summariser = ModelSummariser(model)
    >>> evaluator = ModelEvaluatorBRC(summariser, cint_matrix_true, alpha=0.05)
    >>> cint_metrics = evaluator.evaluate_cint()

    See Also
    --------
    ModelSummariser : Summarises posterior distributions from AgeMix/GenMix models.
    ModelEvaluatorPrem : Similar evaluator for Prem models.
    """

    def __init__(
        self,
        summariser: ModelSummariser,
        cint_matrix_true: NDArray[np.float64] | Dict[str, NDArray],
        alpha: float = 0.05,
    ) -> None:
        super().__init__(summariser, cint_matrix_true, alpha)
        self.model_type = self.summariser.model_type

    def _validate_summariser(self, summariser: ModelSummariser) -> None:
        if not isinstance(summariser, ModelSummariser):
            raise TypeError(
                f"Expected ModelSummariser instance, got {type(summariser).__name__}. "
                f"Usage: summariser = ModelSummariser(model); "
                f"evaluator = ModelEvaluatorBRC(summariser, cint_true)"
            )
        if summariser.inference_method is None:
            raise ValueError(
                "Summariser model has no completed inference. "
                "Run model.run_inference_mcmc() or model.run_inference_svi() first."
            )

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
            - For BRC models: Single row DataFrame
            - For HiBRC models: One row per variable-category combination,
              plus aggregated rows per variable and overall

        Notes
        -----
        Results are cached to avoid redundant computation when called multiple times
        with the same alpha value.

        Examples
        --------
        >>> evaluator = ModelEvaluatorBRC(summariser, cint_true)
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
            if self.model_type == "agemix":
                summary = self.summariser.summarise_cint(alpha=alpha)
                cs = summary["All->All"]
                y_true = self.cint_true["All->All"]
                rmse, mae, mape, int_score, coverage = compute_metrics(
                    y_true, cs.central, cs.lower, cs.upper, alpha=alpha
                )
                metrics_df = pd.DataFrame(
                    {
                        "var": ["all"],
                        "cat": ["all"],
                        "rmse": [rmse],
                        "mae": [mae],
                        "mape": [mape],
                        "interval_score": [int_score],
                        "coverage": [coverage],
                    }
                )
            else:  # genmix
                summary_dict = self.summariser.summarise_cint(alpha=alpha)
                # Convert ContactSummary to (3, ...) arrays for aggregate_metrics
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
        >>> evaluator = ModelEvaluatorBRC(summariser, cint_true)
        >>> mcint_metrics = evaluator.evaluate_mcint(alpha=0.05)
        >>> print(f"Marginal RMSE: {mcint_metrics['rmse'].values[0]:.3f}")
        """
        if alpha is None:
            alpha = self.alpha
        else:
            validate_alpha(alpha)

        cache_key = f"mcint_alpha{alpha}"

        if cache_key not in self._metrics_cache:
            if self.model_type == "agemix":
                summary = self.summariser.summarise_mcint(alpha=alpha)
                cs = summary["All->All"]
                y_true = self.mcint_true["All->All"]
                rmse, mae, mape, int_score, coverage = compute_metrics(
                    y_true, cs.central, cs.lower, cs.upper, alpha=alpha
                )
                metrics_df = pd.DataFrame(
                    {
                        "var": ["all"],
                        "cat": ["all"],
                        "rmse": [rmse],
                        "mae": [mae],
                        "mape": [mape],
                        "interval_score": [int_score],
                        "coverage": [coverage],
                    }
                )
            else:  # genmix
                summary_dict = self.summariser.summarise_mcint(alpha=alpha)
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

        For HiBRC models, errors are aggregated across all categories.

        Examples
        --------
        >>> evaluator = ModelEvaluatorBRC(summariser, cint_true)
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

            # Get true values (dicts for BRC with "All->All" key)
            if quantity == "cint":
                y_true = self.cint_true
            else:
                y_true = self.mcint_true

            # Compute errors
            if self.model_type == "agemix":
                y_true_arr = y_true["All->All"] if isinstance(y_true, dict) else y_true
                diff = y_hat - y_true_arr
                rmse = float(np.sqrt(np.mean(diff**2)))
                mae = float(np.mean(np.abs(diff)))

                # Avoid division by zero in MAPE
                mask = y_true_arr != 0
                if np.any(mask):
                    mape = float(np.mean(np.abs(diff[mask] / y_true_arr[mask])) * 100)
                else:
                    mape = np.inf

                # Relative error - use Frobenius norm for matrices, L2 norm for vectors
                if diff.ndim == 2:
                    norm_diff = np.linalg.norm(diff, "fro")
                    norm_true = np.linalg.norm(y_true_arr, "fro")
                else:
                    norm_diff = np.linalg.norm(diff)
                    norm_true = np.linalg.norm(y_true_arr)
                relative_error = float(
                    norm_diff / norm_true if norm_true > 0 else np.inf
                )

            else:  # genmix
                # Aggregate across all categories
                all_diffs = []
                all_true = []

                for var in y_true.keys():
                    for cat in y_true[var].keys():
                        diff = y_hat[var][cat] - y_true[var][cat]
                        all_diffs.append(diff.flatten())
                        all_true.append(y_true[var][cat].flatten())

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
            - 'model_type': Type of model being evaluated
            - 'alpha': Current significance level
        """
        return {
            "n_cached": len(self._metrics_cache),
            "cached_metrics": list(self._metrics_cache.keys()),
            "model_type": self.model_type,
            "alpha": self.alpha,
        }
