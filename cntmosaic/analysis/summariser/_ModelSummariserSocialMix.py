from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ...models import SocialMix
from ...utils import AgeGroupSpecs, depixilate
from ._summary import ContactSummary

from .._stats import compute_quantiles, validate_alpha


def _get_ci_probs(alpha: float) -> Tuple[float, float, float]:
    """Return (lower, median, upper) quantile probabilities for the given alpha."""
    return (alpha / 2, 0.5, 1 - alpha / 2)


class ModelSummariserSocialMix:
    """
    Statistical summariser for SocialMix bootstrap results.

    Computes quantiles and confidence intervals for contact matrices
    from bootstrap samples, with proper handling of adaptive age binning
    and stratification.

    Parameters
    ----------
    sm : SocialMix
        Fitted SocialMix model with bootstrap results

    Attributes
    ----------
    sm : SocialMix
        Reference to the SocialMix model
    age_bins : AgeGroupSpecs
        Age bins used in the model
    pop_data : PopulationData or None
        Population data container from the model
    age_dist : NDArray or None
        Population age distribution for depixilation

    Examples
    --------
    >>> # Unstratified model
    >>> sm = SocialMix(part_data, cnt_data, age_bins, pop_data)
    >>> sm.run_inference_bootstrap(n_boot=1000, random_state=42)
    >>>
    >>> summariser = ModelSummariserSocialMix(sm)
    >>>
    >>> # Get 95% confidence intervals for contact intensity
    >>> summary = summariser.summarise_cint(alpha=0.05)
    >>> # For unstratified: summary is NDArray of shape (3, C, D)
    >>> lower, median, upper = summary[0], summary[1], summary[2]
    >>>
    >>> # For stratified: summary is Dict[str, NDArray]
    >>> lower_MF = summary['M->F'][0]  # Lower bound for M->F
    >>> median_MF = summary['M->F'][1]  # Median for M->F
    """

    def __init__(self, sm: SocialMix):
        """
        Initialize summariser with a fitted SocialMix model.

        Parameters
        ----------
        sm : SocialMix
            SocialMix model that has been fitted and bootstrapped

        Raises
        ------
        ValueError
            If model has not been fitted
        """
        # Validate model has been fitted and bootstrapped
        if sm.Y is None:
            raise ValueError("SocialMix model has not been fitted")
        if sm._boot is None:
            raise ValueError(
                "Bootstrap has not been run. Call sm.run_inference_bootstrap() first."
            )

        # Store reference to model
        self.sm = sm

        # Reference key attributes
        self.age_group_specs = sm.age_group_specs
        self.pop_data = sm.pop_data

        # Extract population age distribution if available
        if self.pop_data is not None:
            strat_vars = self.pop_data.get_strat_vars()
            pop_df = self.pop_data.data

            if strat_vars:
                # Stratified: create MultiIndex structure [strat_var(s), age] -> P
                if len(strat_vars) > 1:
                    pop_df = pop_df.copy()
                    pop_df["strata"] = pop_df[strat_vars[0]].astype(str)
                    for col in strat_vars[1:]:
                        pop_df["strata"] = pop_df["strata"] + "_" + pop_df[col].astype(str)
                    self.pop_by_strata = pop_df[["strata", "age", "P"]].set_index(["strata", "age"])["P"]
                else:
                    strat_var = strat_vars[0]
                    self.pop_by_strata = pop_df[[strat_var, "age", "P"]].set_index([strat_var, "age"])["P"]

                # Unstratified total: sum P over strata for each age
                self.age_dist = pop_df.groupby("age")["P"].sum().values
            else:
                # Unstratified: just extract the population values
                self.age_dist = pop_df["P"].values
                self.pop_by_strata = None
        else:
            self.age_dist = None
            self.pop_by_strata = None

        # Simple cache: {cache_key: result}
        self._cache: Dict[str, Union[NDArray, Dict[str, NDArray]]] = {}

    def _validate_bootstrap(self) -> None:
        """Validate that bootstrap has been run."""
        if self.sm._boot is None:
            raise ValueError(
                "Bootstrap has not been run. Call sm.run_inference_bootstrap() first."
            )

    def _is_stratified(self) -> bool:
        """Check if model uses stratification."""
        if self.sm.strat_mode != "single":
            return True
        else:
            return False

    def _stack_samples(self, samples_list) -> Dict[str, NDArray]:
        """
        Stack bootstrap samples from List[Dict] to NDArray or Dict[str, NDArray].

        Parameters
        ----------
        samples_list : List[Dict[str, NDArray]]
            List of dictionaries from bootstrap iterations

        Returns
        -------
        Dict[str, NDArray]
            Dict mapping labels to NDArray of shape (n_boot, C, D)
        """
        # Get keys from first sample
        keys = list(samples_list[0].keys())

        stacked = {}
        for key in keys:
            stacked[key] = np.stack([s[key] for s in samples_list])

        return stacked

    def _depixilate_unstratified(self, samples: Dict, use_age_dist: bool) -> NDArray:
        """Depixilate unstratified bootstrap samples."""
        n_boot = samples.shape[0]
        A = self.age_group_specs.range
        depix_samples = np.empty((n_boot, A, A), dtype=np.float64)

        for i in range(n_boot):
            if use_age_dist:
                depix_samples[i] = depixilate(samples[i], self.age_group_specs, self.age_dist)
            else:
                depix_samples[i] = depixilate(samples[i], self.age_group_specs)

        return depix_samples

    def _depixilate_stratified(
        self, samples_dict: Dict[str, NDArray], use_age_dist: bool
    ) -> Dict[str, NDArray]:
        """Depixilate stratified bootstrap samples."""
        A = self.age_group_specs.range
        depix_samples = {}

        for label, samples in samples_dict.items():
            n_boot = samples.shape[0]
            depix_label = np.empty((n_boot, A, A), dtype=np.float64)

            # Extract appropriate population for this stratum
            if use_age_dist and self.pop_by_strata is not None:
                # Parse stratum label (e.g., "M->All", "M->F", "F->All")
                if "->" in label:
                    source_stratum = label.split("->")[0]
                    
                    # Extract source stratum population from MultiIndex
                    if source_stratum in self.pop_by_strata.index.get_level_values(0):
                        stratum_age_dist = self.pop_by_strata.xs(source_stratum, level=0).values
                    else:
                        # Fallback to full population if stratum not found
                        warnings.warn(
                            f"Could not find population for stratum '{source_stratum}'. "
                            "Using total population for depixilation.",
                            UserWarning,
                        )
                        stratum_age_dist = self.age_dist
                else:
                    # No stratification in label, use total population
                    stratum_age_dist = self.age_dist
                
                # Depixilate with stratum-specific population
                for i in range(n_boot):
                    depix_label[i] = depixilate(
                        samples[i], self.age_group_specs, stratum_age_dist
                    )
            else:
                # No age distribution or unstratified
                for i in range(n_boot):
                    if use_age_dist:
                        depix_label[i] = depixilate(
                            samples[i], self.age_group_specs, self.age_dist
                        )
                    else:
                        depix_label[i] = depixilate(samples[i], self.age_group_specs)

            depix_samples[label] = depix_label

        return depix_samples

    def _compute_quantiles(
        self, samples_dict: Dict[str, NDArray], probs: Tuple[float, ...]
    ) -> Dict[str, NDArray]:
        quantiles = {}
        for label, samples in samples_dict.items():
            quantiles[label] = np.quantile(samples, q=probs, axis=0)
        return quantiles

    def _cint_samples(self, return_depixilated: bool) -> Dict[str, NDArray]:
        """
        Stacked bootstrap contact intensity samples, by stratum label.

        With ``return_depixilated``, each sample is first projected from the
        age groups to 1-year ages (weighted by the population age
        distribution); quantiles must be taken afterwards, as depixilation and
        quantiles do not commute.
        """
        stacked = self._stack_samples(self.sm._boot.cint_samples)
        if not return_depixilated:
            return stacked
        if self.age_dist is None:
            raise ValueError(
                "Population data required for depixilation. "
                "Provide pop_data when initializing SocialMix."
            )
        if "All->All" in stacked:
            stacked["All->All"] = self._depixilate_unstratified(
                stacked["All->All"], use_age_dist=True
            )
            return stacked
        return self._depixilate_stratified(stacked, use_age_dist=True)

    def summarise_cint(
        self,
        alpha: float = 0.05,
        return_depixilated: bool = False,
        force_recompute: bool = False,
    ) -> Dict[str, ContactSummary]:
        """
        Compute bootstrap confidence intervals for contact intensity.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level of the intervals.
        return_depixilated : bool, default=False
            Return the summaries at 1-year age resolution (each bootstrap
            sample is depixilated before the quantiles are taken) instead of
            in the model's age groups.
        force_recompute : bool, default=False
            Ignore the cache.

        Returns
        -------
        Dict[str, ContactSummary]
            Dict mapping stratum labels to ContactSummary objects.
        """
        validate_alpha(alpha)
        self._validate_bootstrap()

        cache_key = f"cint_alpha{alpha}_depix{return_depixilated}"
        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        stacked = self._cint_samples(return_depixilated)
        probs = _get_ci_probs(alpha)
        result = {}
        for label, arr in stacked.items():
            q = np.quantile(arr, probs, axis=0)
            result[label] = ContactSummary(
                lower=q[0],
                central=q[1],
                upper=q[2],
                alpha=alpha,
                measure="median",
                age_group_specs=None if return_depixilated else self.sm.age_group_specs,
            )
        self._cache[cache_key] = result
        return result

    def summarise_rate(
        self,
        alpha: float = 0.05,
        force_recompute: bool = False,
    ) -> Dict[str, ContactSummary]:
        """
        Compute bootstrap confidence intervals for contact rate.

        Returns
        -------
        Dict[str, ContactSummary]
            Dict mapping stratum labels to ContactSummary objects.
        """
        validate_alpha(alpha)
        self._validate_bootstrap()

        cache_key = f"rate_alpha{alpha}"
        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        stacked = self._stack_samples(self.sm._boot.rate_samples)
        probs = _get_ci_probs(alpha)
        result = {}
        for label, arr in stacked.items():
            q = np.quantile(arr, probs, axis=0)
            result[label] = ContactSummary(
                lower=q[0],
                central=q[1],
                upper=q[2],
                alpha=alpha,
                measure="median",
                age_group_specs=self.sm.age_group_specs,
            )
        self._cache[cache_key] = result
        return result

    def summarise_mcint(
        self,
        alpha: float = 0.05,
        return_depixilated: bool = False,
        force_recompute: bool = False,
    ) -> Dict[str, ContactSummary]:
        """
        Compute bootstrap confidence intervals for marginal contact intensity.

        Marginal contact intensity m_c = Σ_d M[c,d] is the total average
        number of contacts made by individuals in age group c.

        Returns
        -------
        Dict[str, ContactSummary]
            Dict mapping stratum labels to ContactSummary objects (1-D central arrays).
        """
        validate_alpha(alpha)
        self._validate_bootstrap()

        cache_key = f"mcint_alpha{alpha}_depix{return_depixilated}"
        if not force_recompute and cache_key in self._cache:
            return self._cache[cache_key]

        # Depixilate the full matrices first, then take marginals and quantiles
        stacked = self._cint_samples(return_depixilated)
        probs = _get_ci_probs(alpha)
        result = {}
        for label, arr in stacked.items():
            marginal = arr.sum(axis=-1)  # (n_boot, C)
            q = np.quantile(marginal, probs, axis=0)
            result[label] = ContactSummary(
                lower=q[0],
                central=q[1],
                upper=q[2],
                alpha=alpha,
                measure="median",
                age_group_specs=None if return_depixilated else self.sm.age_group_specs,
            )
        self._cache[cache_key] = result
        return result

    def clear_cache(self) -> None:
        """Clear all cached computations."""
        self._cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached results.

        Returns
        -------
        info : dict
            Dictionary with cache statistics
        """
        return {"n_cached": len(self._cache), "cache_keys": list(self._cache.keys())}

    def _validate(self) -> None:
        """Validate model and bootstrap state."""
        self._validate_bootstrap()
