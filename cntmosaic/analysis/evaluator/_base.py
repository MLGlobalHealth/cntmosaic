"""
Base classes and shared metric utilities for model evaluators.

This module provides:

- :class:`SummariserProtocol` — a ``typing.Protocol`` describing the minimal interface
  that any summariser must expose so an evaluator can call it. Concrete summariser
  classes (``ModelSummariser``, ``ModelSummariserPrem``, ``ModelSummariserSocialMix``)
  all satisfy this protocol.

- :class:`BaseModelEvaluator` — an abstract base class that consolidates the metric
  computation helpers and structural boilerplate shared by every concrete evaluator.
  Subclasses only need to implement ``_validate_summariser``, ``evaluate_cint``, and
  ``evaluate_mcint``; all shared utilities (``validate_alpha``, ``interval_score``,
  ``compute_metrics``, ``_validate_true_matrix``, ``_compute_marginals``, ``evaluate``,
  ``clear_cache``) are provided here.

Relationship between evaluators and summarisers
-----------------------------------------------
Each model family has a paired summariser and evaluator::

    ModelSummariser          <-->  ModelEvaluatorBRC
    ModelSummariserPrem      <-->  ModelEvaluatorPrem
    ModelSummariserSocialMix <-->  ModelEvaluatorSocialMix

All three evaluators inherit from :class:`BaseModelEvaluator` and therefore share a
common interface: ``evaluate()``, ``evaluate_cint()``, ``evaluate_mcint()``,
``clear_cache()``, and ``get_cache_info()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)


# ---------------------------------------------------------------------------
# SummariserProtocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SummariserProtocol(Protocol):
    """Minimal interface that a summariser object must expose for use with an evaluator.

    Any object that provides ``summarise_cint`` and ``summarise_mcint`` satisfies
    this protocol — no explicit inheritance is required.
    """

    def summarise_cint(self, alpha: float = 0.05, **kwargs) -> Dict:
        """Return posterior/bootstrap summary of the contact intensity matrix."""
        ...

    def summarise_mcint(self, alpha: float = 0.05, **kwargs) -> Dict:
        """Return posterior/bootstrap summary of the marginal contact intensity."""
        ...


# ---------------------------------------------------------------------------
# Shared module-level metric helpers
# ---------------------------------------------------------------------------

def validate_alpha(alpha: float) -> None:
    """Validate alpha parameter.

    Parameters
    ----------
    alpha : float
        Significance level; must satisfy ``0 < alpha < 1``.

    Raises
    ------
    ValueError
        If *alpha* is not in the open interval (0, 1).
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def interval_score(
    y_true: NDArray,
    y_low: NDArray,
    y_high: NDArray,
    alpha: float,
) -> float:
    r"""Mean interval score of central (1 - alpha) prediction intervals.

    .. math::

        IS_\alpha(l, u; y) = (u - l) + \frac{2}{\alpha}(l - y)_+
                             + \frac{2}{\alpha}(y - u)_+

    where :math:`(x)_+ = \max(0, x)` (Gneiting & Raftery, 2007, eq. 43). It is
    the interval width plus a penalty, linear in the distance, for truths that
    fall outside the interval.

    Parameters
    ----------
    y_true : NDArray
        True (ground-truth) values.
    y_low : NDArray
        Lower bound of the prediction interval.
    y_high : NDArray
        Upper bound of the prediction interval.
    alpha : float
        Nominal miscoverage: the interval is the central (1 - alpha) interval.

    Returns
    -------
    float
        Mean interval score over all elements (lower is better).
    """
    validate_alpha(alpha)
    y_true, y_low, y_high = (
        np.asarray(x, dtype=float) for x in (y_true, y_low, y_high)
    )
    width = y_high - y_low
    below = np.maximum(0.0, y_low - y_true)
    above = np.maximum(0.0, y_true - y_high)
    return float(np.mean(width + 2.0 / alpha * (below + above)))


def compute_metrics(
    y_true: NDArray,
    y_est: NDArray,
    y_low: NDArray,
    y_high: NDArray,
    alpha: float = 0.05,
) -> tuple:
    """Compute RMSE, MAE, MAPE, interval score, and coverage.

    Parameters
    ----------
    y_true : NDArray
        True (ground-truth) values.
    y_est : NDArray
        Point estimates (e.g., posterior median).
    y_low : NDArray
        Lower credible-interval bound.
    y_high : NDArray
        Upper credible-interval bound.
    alpha : float, default=0.05
        Nominal miscoverage of the interval ``[y_low, y_high]``, used by the
        interval score.

    Returns
    -------
    tuple
        ``(rmse, mae, mape, int_score, coverage)`` where *mape* is in percent,
        *int_score* uses ``alpha``, and *coverage* is in percent.
    """
    rmse = root_mean_squared_error(y_true, y_est)
    mae = mean_absolute_error(y_true, y_est)
    mape = mean_absolute_percentage_error(y_true, y_est) * 100
    int_score = interval_score(y_true, y_low, y_high, alpha=alpha)
    coverage = np.mean((y_true >= y_low) & (y_true <= y_high)) * 100
    return rmse, mae, mape, int_score, coverage


def process_variable_metrics(
    data_eval: Dict[str, NDArray],
    data_est: Dict[str, NDArray],
    alpha: float = 0.05,
) -> list:
    """Compute metrics for a single variable across its categories and overall.

    Parameters
    ----------
    data_eval : dict
        Mapping from category key to ground-truth array.
    data_est : dict
        Mapping from category key to ``(lower, median, upper)`` triple of arrays.
    alpha : float, default=0.05
        Nominal miscoverage of the intervals, used by the interval score.

    Returns
    -------
    list of dict
        One metrics dict per category, followed by an ``"all"`` aggregate row.
    """
    metrics = []
    for key, values in data_est.items():
        rmse, mae, mape, int_score, coverage = compute_metrics(
            data_eval[key], values[1], values[0], values[2], alpha=alpha
        )
        metrics.append(
            {
                "cat": key,
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "interval_score": int_score,
                "coverage": coverage,
            }
        )

    # Compute overall metrics for the variable
    y_true = np.vstack([data_eval[key] for key in data_est.keys()])
    y_est = np.vstack([values[1] for values in data_est.values()])
    y_low = np.vstack([values[0] for values in data_est.values()])
    y_high = np.vstack([values[2] for values in data_est.values()])

    rmse, mae, mape, int_score, coverage = compute_metrics(
        y_true, y_est, y_low, y_high, alpha=alpha
    )

    metrics.append(
        {
            "cat": "all",
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "interval_score": int_score,
            "coverage": coverage,
        }
    )

    return metrics


def aggregate_metrics(
    data_eval: Dict[str, NDArray],
    data_est: Dict[str, NDArray],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Aggregate metrics for all variables/categories and compute overall metrics.

    Parameters
    ----------
    data_eval : dict
        Mapping from category key to ground-truth array.
    data_est : dict
        Mapping from category key to ``(lower, median, upper)`` triple of arrays.
    alpha : float, default=0.05
        Nominal miscoverage of the intervals, used by the interval score.

    Returns
    -------
    pd.DataFrame
        Per-category metric rows followed by an overall ``"all"`` row.
    """
    all_metrics = []
    all_metrics.extend(process_variable_metrics(data_eval, data_est, alpha=alpha))

    # Compute overall metrics across all variables and categories
    y_true = np.vstack([data_eval[cat] for cat in data_est.keys()])
    y_est = np.vstack([values[1] for values in data_est.values()])
    y_low = np.vstack([values[0] for values in data_est.values()])
    y_high = np.vstack([values[2] for values in data_est.values()])

    rmse, mae, mape, int_score, coverage = compute_metrics(
        y_true, y_est, y_low, y_high, alpha=alpha
    )
    all_metrics.append(
        {
            "cat": "all",
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "interval_score": int_score,
            "coverage": coverage,
        }
    )

    return pd.DataFrame(all_metrics)


# ---------------------------------------------------------------------------
# BaseModelEvaluator abstract base class
# ---------------------------------------------------------------------------

class BaseModelEvaluator(ABC):
    """Abstract base class for all model evaluators.

    Subclasses must implement :meth:`_validate_summariser`, :meth:`evaluate_cint`,
    and :meth:`evaluate_mcint`. All shared validation helpers, metric computation
    utilities, and the top-level :meth:`evaluate` / :meth:`clear_cache` methods are
    provided here.

    Parameters
    ----------
    summariser : SummariserProtocol
        A fitted model summariser exposing ``summarise_cint`` and
        ``summarise_mcint``.
    cint_matrix_true : NDArray or dict
        Ground-truth contact intensity matrix (or dict of matrices for
        stratified / hierarchical models).
    alpha : float, default=0.05
        Significance level for interval score and coverage computations.
        Must be in (0, 1).

    Attributes
    ----------
    summariser : SummariserProtocol
        Reference to the model summariser.
    cint_true : NDArray or dict
        Ground-truth contact intensity matrix/matrices.
    mcint_true : NDArray or dict
        Ground-truth marginal contact intensities derived from *cint_true*.
    alpha : float
        Significance level for credible intervals.
    """

    def __init__(
        self,
        summariser: SummariserProtocol,
        cint_matrix_true: NDArray[np.float64] | Dict[str, NDArray],
        alpha: float = 0.05,
    ) -> None:
        # Validate shared inputs
        validate_alpha(alpha)
        self._validate_summariser(summariser)
        self._validate_true_matrix(cint_matrix_true)

        # Store shared state
        self.summariser = summariser
        self.cint_true = cint_matrix_true
        self.alpha = alpha

        # Derived ground-truth quantity
        self.mcint_true = self._compute_marginals(cint_matrix_true)

        # Metrics cache — keyed by a string such as "cint_alpha0.05"
        self._metrics_cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _validate_summariser(self, summariser) -> None:
        """Validate that *summariser* has the attributes required by this evaluator.

        Subclasses should raise :class:`TypeError` or :class:`ValueError` with a
        descriptive message when validation fails.
        """

    @abstractmethod
    def evaluate_cint(self, alpha: Optional[float] = None) -> pd.DataFrame:
        """Evaluate the posterior/bootstrap contact intensity matrix.

        Parameters
        ----------
        alpha : float, optional
            Override significance level; falls back to ``self.alpha`` when ``None``.

        Returns
        -------
        pd.DataFrame
            Per-category (and overall) evaluation metrics.
        """

    @abstractmethod
    def evaluate_mcint(self, alpha: Optional[float] = None) -> pd.DataFrame:
        """Evaluate the posterior/bootstrap marginal contact intensity.

        Parameters
        ----------
        alpha : float, optional
            Override significance level; falls back to ``self.alpha`` when ``None``.

        Returns
        -------
        pd.DataFrame
            Per-category (and overall) evaluation metrics.
        """

    # ------------------------------------------------------------------
    # Shared concrete helpers
    # ------------------------------------------------------------------

    def _validate_true_matrix(self, cint_true: NDArray | Dict[str, NDArray]) -> None:
        """Validate true matrix dimensions and values.

        Accepts either a single 2-D square NDArray or a dict of such arrays.

        Raises
        ------
        TypeError
            If any array is not an :class:`numpy.ndarray`.
        ValueError
            If any array is not square, 2-D, contains NaN/Inf, or contains
            negative values.
        """
        if isinstance(cint_true, dict):
            for label, matrix in cint_true.items():
                self._validate_single_matrix(matrix, f"cint_true['{label}']")
        else:
            self._validate_single_matrix(cint_true, "cint_true")

    def _validate_single_matrix(self, matrix: NDArray, name: str) -> None:
        """Validate a single contact intensity matrix.

        Parameters
        ----------
        matrix : NDArray
            Array to validate.
        name : str
            Human-readable name used in error messages.

        Raises
        ------
        TypeError
            If *matrix* is not a :class:`numpy.ndarray`.
        ValueError
            If *matrix* is not 2-D, not square, contains NaN/Inf, or contains
            negative values.
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError(
                f"{name} must be a numpy array, got {type(matrix).__name__}"
            )

        if matrix.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {matrix.shape}")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be square, got shape {matrix.shape}")

        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} contains NaN or Inf values")

        if np.any(matrix < 0):
            raise ValueError(f"{name} contains negative values")

    def _compute_marginals(
        self, cint: NDArray | Dict[str, NDArray]
    ) -> NDArray | Dict[str, NDArray]:
        """Compute marginal contact intensities by summing over the contact-age axis.

        Parameters
        ----------
        cint : NDArray or dict
            Contact intensity matrix (or dict of matrices for stratified models).

        Returns
        -------
        NDArray or dict
            Marginal contact intensity vector(s) (sum along axis=1).
        """
        if isinstance(cint, dict):
            return {key: matrix.sum(axis=1) for key, matrix in cint.items()}
        else:
            return cint.sum(axis=1)

    def evaluate(self, alpha: Optional[float] = None) -> pd.DataFrame:
        """Compute all evaluation metrics by combining ``evaluate_cint`` and ``evaluate_mcint``.

        Parameters
        ----------
        alpha : float, optional
            Significance level for interval metrics. Falls back to ``self.alpha``
            when ``None``.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with a ``metric_type`` column (``"cint"`` or
            ``"mcint"``) distinguishing the two sets of metrics.

        Notes
        -----
        Metrics computed:

        - **RMSE** — Root mean squared error
        - **MAE** — Mean absolute error
        - **MAPE** — Mean absolute percentage error (in percent)
        - **interval_score** — Negatively oriented interval score (lower is better)
        - **coverage** — Percentage of true values within the credible interval
        """
        if alpha is None:
            alpha = self.alpha

        cint_metrics = self.evaluate_cint(alpha=alpha)
        mcint_metrics = self.evaluate_mcint(alpha=alpha)

        cint_metrics["metric_type"] = "cint"
        mcint_metrics["metric_type"] = "mcint"

        return pd.concat([cint_metrics, mcint_metrics], ignore_index=True)

    def clear_cache(self) -> None:
        """Clear all cached metric computations."""
        self._metrics_cache.clear()

    @abstractmethod
    def get_cache_info(self) -> Dict:
        """Return statistics about the internal metrics cache.

        Subclasses extend this with model-specific fields (e.g. ``strat_mode``,
        ``model_type``). At minimum the returned dict should contain:

        - ``"n_cached"`` — number of cached entries
        - ``"cached_metrics"`` — list of cached keys
        - ``"alpha"`` — current significance level
        """
