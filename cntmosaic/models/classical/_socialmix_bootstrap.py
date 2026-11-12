"""
Bootstrap Uncertainty Quantification for SocialMix

This module provides bootstrap resampling for contact matrix estimation.
Uses efficient NumPy operations for maximum performance.
"""

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from ...dataloader import ContactData, ParticipantData, PopulationData
from ...utils import AgeGroupSpecs
from ._socialmix_helpers import (
    apply_reciprocity,
    contact_labels,
    create_stratum_labels,
    infer_strat_mode,
    participant_labels,
    population_by_contact_stratum,
    stratum_codes,
    stratum_dims,
)


@dataclass
class BootstrapResults:
    """
    Container for bootstrap estimation results.

    Attributes
    ----------
    cint_samples : List[Dict[str, NDArray]]
        List of contact intensity matrices from each bootstrap iteration
    rate_samples : List[Dict[str, NDArray]]
        List of contact rate matrices from each bootstrap iteration
    n_boot : int
        Number of bootstrap iterations performed
    success_rate : float
        Fraction of successful bootstrap iterations
    weighted : bool
        Whether participant weights were applied (rebuilt in every replicate)
    """

    cint_samples: List[Dict[str, NDArray]]
    rate_samples: List[Dict[str, NDArray]]
    n_boot: int
    success_rate: float
    weighted: bool = False

    def __repr__(self) -> str:
        """Fast repr for debugger compatibility."""
        n_samples = len(self.cint_samples)
        strata_keys = list(self.cint_samples[0].keys()) if n_samples > 0 else []
        return (
            f"BootstrapResults(n_boot={self.n_boot}, weighted={self.weighted}, "
            f"success_rate={self.success_rate:.2%}, "
            f"samples={n_samples}, strata={strata_keys})"
        )

    def mean(self, statistic: str = "cint") -> Dict[str, NDArray]:
        """
        Compute mean across bootstrap samples.

        Parameters
        ----------
        statistic : str, default "cint"
            Which statistic to compute mean for: "cint" or "rate"

        Returns
        -------
        mean_dict : Dict[str, NDArray]
            Mean of the specified statistic across bootstrap samples
        """
        if statistic == "cint":
            samples = self.cint_samples
        elif statistic == "rate":
            samples = self.rate_samples
        else:
            raise ValueError(f"statistic must be 'cint' or 'rate', got {statistic}")

        mean_dict = {}
        keys = list(samples[0].keys())

        for key in keys:
            stack = np.stack([s[key] for s in samples])
            mean_dict[key] = np.mean(stack, axis=0)

        return mean_dict

    def std(self, statistic: str = "cint") -> Dict[str, NDArray]:
        """
        Compute standard deviation across bootstrap samples.

        Parameters
        ----------
        statistic : str, default "cint"
            Which statistic to compute std for: "cint" or "rate"

        Returns
        -------
        std_dict : Dict[str, NDArray]
            Standard deviation of the specified statistic across bootstrap samples
        """
        if statistic == "cint":
            samples = self.cint_samples
        elif statistic == "rate":
            samples = self.rate_samples
        else:
            raise ValueError(f"statistic must be 'cint' or 'rate', got {statistic}")

        std_dict = {}
        keys = list(samples[0].keys())

        for key in keys:
            stack = np.stack([s[key] for s in samples])
            std_dict[key] = np.std(stack, axis=0, ddof=1)

        return std_dict

    def quantiles(self, q: List[float], statistic: str = "cint") -> Dict[str, NDArray]:
        """
        Compute quantiles across bootstrap samples.

        Parameters
        ----------
        q : List[float]
            Quantiles to compute (e.g., [0.025, 0.975] for 95% CI)
        statistic : str, default "cint"
            Which statistic to compute quantiles for: "cint" or "rate"

        Returns
        -------
        quant_dict : Dict[str, NDArray]
            Quantiles of the specified statistic, shape (len(q), C, D) for each key
        """
        if statistic == "cint":
            samples = self.cint_samples
        elif statistic == "rate":
            samples = self.rate_samples
        else:
            raise ValueError(f"statistic must be 'cint' or 'rate', got {statistic}")

        quant_dict = {}
        keys = list(samples[0].keys())

        for key in keys:
            stack = np.stack([s[key] for s in samples])
            quant_dict[key] = np.quantile(stack, q, axis=0)

        return quant_dict


class SocialMixBootstrap:
    """
    Performs efficient bootstrap uncertainty quantification for SocialMix.

    Uses pre-computed arrays and vectorized operations for maximum performance.
    """

    def __init__(
        self,
        part_data: ParticipantData,
        cnt_data: ContactData,
        age_group_specs: AgeGroupSpecs,
        pop_data: Optional[PopulationData] = None,
        apply_reciprocity: bool = False,
        n_boot: int = 1000,
        random_state: Optional[int] = None,
        bound_weights=None,
    ):
        """
        Initialize bootstrap estimator.

        Parameters
        ----------
        part_data : ParticipantData
            Participant data container
        cnt_data : ContactData
            Contact data container
        age_group_specs : AgeGroupSpecs
            Age stratification bins
        pop_data : PopulationData, optional
            Population data for reciprocity adjustment
        apply_reciprocity : bool, default False
            Whether to apply reciprocity adjustment within each bootstrap iteration
        n_boot : int, default 1000
            Number of bootstrap samples
        random_state : int, optional
            Random seed for reproducibility
        bound_weights : _BoundWeightSpecs, optional
            Participant weights bound to ``part_data`` (same participants, same
            order). When given, the weights are rebuilt on every resample and
            each replicate is the weighted ratio estimator.
        """
        self.part_data = part_data
        self.cnt_data = cnt_data
        self.age_group_specs = age_group_specs
        self.pop_data = pop_data
        self.apply_reciprocity = apply_reciprocity
        self.n_boot = n_boot
        self.rng = np.random.default_rng(random_state)
        self.bound_weights = bound_weights

        # Stratification info (will be set during preparation)
        self.C: int = 0  # Number of participant age groups
        self.D: int = 0  # Number of contact age groups
        self.K_part: int = 1  # Number of participant strata
        self.K_cnt: int = 1  # Number of contact strata
        self.strat_mode: str = "single"
        self.strat_vars_part: List[str] = []
        self.strat_vars_cnt: List[str] = []
        self.P: Optional[NDArray] = None  # Population sizes

        # Assign age groups
        self._assign_age_groups()

    def _assign_age_groups(self) -> None:
        """Assign age groups to participants and contacts if not already provided."""
        # Assign age groups to participants if not present
        if "part_age_grp" not in self.part_data.data.columns:
            if "part_age" in self.part_data.data.columns:
                self.part_data.data["part_age_grp"] = self.age_group_specs.cut(
                    self.part_data.data["part_age"]
                )
            else:
                raise ValueError(
                    "ParticipantData must have either 'part_age' or 'part_age_grp' column."
                )

        # Assign age groups to contacts if not present
        if "cnt_age_grp" not in self.cnt_data.data.columns:
            if "cnt_age" in self.cnt_data.data.columns:
                self.cnt_data.data["cnt_age_grp"] = self.age_group_specs.cut(
                    self.cnt_data.data["cnt_age"]
                )
            else:
                raise ValueError(
                    "ContactData must have either 'cnt_age' or 'cnt_age_grp' column."
                )

        # Assign age groups to population if provided
        if self.pop_data is not None and "age_grp" not in self.pop_data.data.columns:
            if "age" in self.pop_data.data.columns:
                self.pop_data.data["age_grp"] = self.age_group_specs.cut(
                    self.pop_data.data["age"]
                )
            else:
                raise ValueError(
                    "PopulationData must have either 'age' or 'age_grp' column."
                )

    def run(
        self,
        progress: bool = True,
        min_success_rate: float = 0.5,
    ) -> BootstrapResults:
        """
        Run bootstrap resampling with efficient NumPy operations.

        Parameters
        ----------
        progress : bool, default True
            Show progress bar
        min_success_rate : float, default 0.5
            Minimum fraction of successful iterations required

        Returns
        -------
        BootstrapResults
            Container with bootstrap samples and summary statistics

        Raises
        ------
        ValueError
            If too many bootstrap iterations fail
        """
        # Prepare data structures
        Y_raw, age_codes, part_strat_codes, boot_indices = (
            self._prepare_bootstrap_data()
        )

        if self.bound_weights is not None:
            ids = self.part_data.data["id"].to_numpy()
            if not np.array_equal(np.asarray(self.bound_weights.ids), ids):
                raise RuntimeError(
                    "Internal error: the bound weights do not belong to the "
                    "participants being resampled."
                )

        # Storage for results
        cint_samples = []
        rate_samples = []

        iterator = tqdm(range(self.n_boot), desc="Bootstrapping", disable=not progress)

        for b in iterator:
            boot_idx = boot_indices[b]
            # Weights are rebuilt on every resample, outside the try block: an
            # error here is a bug, not a degenerate sample, and must propagate.
            weights = (
                None
                if self.bound_weights is None
                else self.bound_weights.weights(boot_idx)
            )
            try:
                cint_boot, rate_boot = self._bootstrap_iteration(
                    Y_raw, age_codes, part_strat_codes, boot_idx, weights
                )
                cint_samples.append(cint_boot)
                rate_samples.append(rate_boot)
            except (ValueError, ZeroDivisionError, FloatingPointError) as e:
                # Degenerate resamples only; other exceptions propagate.
                if progress:
                    iterator.write(f"Bootstrap iteration {b} failed: {str(e)[:100]}")
                continue

        n_successful = len(cint_samples)
        success_rate = n_successful / self.n_boot

        if success_rate < min_success_rate:
            raise ValueError(
                f"Bootstrap failed: only {n_successful}/{self.n_boot} "
                f"iterations succeeded ({success_rate:.1%}). "
                f"This may indicate insufficient sample sizes in some age groups."
            )

        if progress and n_successful < self.n_boot:
            print(
                f"\nCompleted {n_successful}/{self.n_boot} bootstrap iterations "
                f"({success_rate:.1%} success rate)"
            )

        return BootstrapResults(
            cint_samples=cint_samples,
            rate_samples=rate_samples,
            n_boot=n_successful,
            success_rate=success_rate,
            weighted=self.bound_weights is not None,
        )

    def _prepare_bootstrap_data(
        self,
    ) -> Tuple[NDArray, NDArray, Optional[NDArray], NDArray]:
        """
        Pre-compute arrays for efficient bootstrap resampling.

        Returns
        -------
        Y_raw : NDArray, shape (N, K_cnt, D)
            Contact count tensor per participant
        age_codes : NDArray, shape (N,)
            Participant age group codes
        part_strat_codes : NDArray, shape (N,) or None
            Participant stratum codes (None if no participant stratification)
        boot_indices : NDArray, shape (n_boot, N)
            Pre-generated bootstrap sample indices
        """
        # Extract stratification info
        self.strat_vars_part = self.part_data.get_strat_vars(prefix=False)
        self.strat_vars_cnt = self.cnt_data.get_strat_vars(prefix=False)

        # Determine stratification mode
        self.strat_mode = infer_strat_mode(self.strat_vars_part, self.strat_vars_cnt)

        # Get dimensions
        self.C = len(self.part_data.data["part_age_grp"].cat.categories)
        self.D = len(self.cnt_data.data["cnt_age_grp"].cat.categories)

        # K_part and K_cnt = product of the number of categories (not nunique),
        # matching the stratum labels and the point estimate.
        self.K_part = int(
            np.prod(
                stratum_dims(
                    self.part_data.data, [f"part_{v}" for v in self.strat_vars_part]
                )
            )
        )
        self.K_cnt = int(
            np.prod(
                stratum_dims(
                    self.cnt_data.data, [f"cnt_{v}" for v in self.strat_vars_cnt]
                )
            )
        )

        # Build contact count tensor
        Y_raw = self._build_contact_tensor()  # (N, K_cnt, D)

        # Extract participant age codes
        age_codes = self.part_data.data["part_age_grp"].cat.codes.values.astype(
            np.int32
        )

        # Extract participant stratum codes (if applicable)
        if self.K_part > 1:
            part_strat_codes = self._get_participant_stratum_codes()
        else:
            part_strat_codes = None

        # Pre-generate all bootstrap indices
        N = len(self.part_data.data)
        boot_indices = self.rng.integers(0, N, size=(self.n_boot, N), dtype=np.int32)

        # Load population data if available
        if self.pop_data is not None:
            self._load_population_data()

        return Y_raw, age_codes, part_strat_codes, boot_indices

    def _build_contact_tensor(self) -> NDArray:
        """
        Build contact count tensor of shape (N, K_cnt, D).

        For each participant i, contact stratum k, and contact age group d,
        Y_raw[i, k, d] = total contacts participant i had with age group d in stratum k.
        """
        N = len(self.part_data.data)
        Y_raw = np.zeros((N, self.K_cnt, self.D), dtype=np.float64)

        # Get all participant IDs
        all_ids = self.part_data.data["id"].values
        id_to_idx = {id_val: idx for idx, id_val in enumerate(all_ids)}

        if self.K_cnt == 1:
            # No contact stratification: aggregate by (id, age_grp_cnt)
            df_agg = (
                self.cnt_data.data.groupby(["id", "cnt_age_grp"], observed=False)["y"]
                .sum()
                .reset_index()
            )

            for _, row in df_agg.iterrows():
                i = id_to_idx.get(row["id"])
                if i is not None:
                    d = row["cnt_age_grp"]
                    d_code = self.cnt_data.data["cnt_age_grp"].cat.categories.get_loc(d)
                    Y_raw[i, 0, d_code] = row["y"]
        else:
            # Contact stratification: aggregate by (id, strat_vars_cnt, age_grp_cnt)
            group_cols = (
                ["id"] + [f"cnt_{v}" for v in self.strat_vars_cnt] + ["cnt_age_grp"]
            )
            df_agg = (
                self.cnt_data.data.groupby(group_cols, observed=False)["y"]
                .sum()
                .reset_index()
            )

            # Get contact stratum mapping
            cnt_strat_mapping = self._get_contact_stratum_mapping()

            for _, row in df_agg.iterrows():
                i = id_to_idx.get(row["id"])
                if i is not None:
                    # Get contact stratum code
                    strat_tuple = tuple(row[f"cnt_{v}"] for v in self.strat_vars_cnt)
                    k = cnt_strat_mapping[strat_tuple]

                    # Get age code
                    d = row["cnt_age_grp"]
                    d_code = self.cnt_data.data["cnt_age_grp"].cat.categories.get_loc(d)

                    Y_raw[i, k, d_code] = row["y"]

        return Y_raw

    def _get_participant_stratum_codes(self) -> NDArray:
        """Get stratum codes for each participant.

        Returns a composite integer code in the same order as
        ``itertools.product(*category_lists)`` — i.e. row-major over the
        cartesian product of per-variable categories.
        """
        return stratum_codes(
            self.part_data.data, [f"part_{v}" for v in self.strat_vars_part]
        ).astype(np.int32)

    def _get_participant_stratum_mapping(self) -> Dict:
        """Create mapping from participant stratum tuple to integer code."""
        # Get all category combinations
        categories = []
        for var in self.strat_vars_part:
            col_name = f"part_{var}"
            cats = self.part_data.data[col_name].cat.categories.tolist()
            categories.append(cats)

        # Create mapping
        mapping = {}
        for idx, combo in enumerate(itertools.product(*categories)):
            mapping[combo] = idx

        return mapping

    def _get_contact_stratum_mapping(self) -> Dict:
        """Create mapping from contact stratum tuple to integer code."""
        # Get all category combinations
        categories = []
        for var in self.strat_vars_cnt:
            col_name = f"cnt_{var}"
            cats = self.cnt_data.data[col_name].cat.categories.tolist()
            categories.append(cats)

        # Create mapping
        mapping = {}
        for idx, combo in enumerate(itertools.product(*categories)):
            mapping[combo] = idx

        return mapping

    def _load_population_data(self):
        """Load and process population data for reciprocity adjustment."""
        bin_edges = self.age_group_specs.left + [self.age_group_specs.right[-1] + 1]
        intervals = [
            pd.Interval(left=l, right=r, closed="left")
            for l, r in zip(bin_edges[:-1], bin_edges[1:])
        ]

        if "age_grp" not in self.pop_data.data.columns:
            ages = self.pop_data.data["age"]
            age_grps = pd.cut(ages, bins=bin_edges, right=False, labels=intervals)
            self.pop_data.data["age_grp"] = age_grps

        # Aggregate population by age group and stratum
        if self.K_cnt == 1:
            # No stratification
            pop_sizes = (
                self.pop_data.data.groupby("age_grp", observed=False)["P"]
                .sum()
                .reindex(pd.Index(intervals), fill_value=0)
            )
            self.P = pop_sizes.values.astype(np.float64)
        else:
            # With stratification: match population rows to contact strata by
            # category value, in contact-stratum label order (same helper as the
            # point estimate).
            P = population_by_contact_stratum(
                self.pop_data.data, self.strat_vars_cnt, self.cnt_data.data, n_age=self.D
            )
            if P is None:
                raise ValueError(
                    "Population data must contain every contact stratification "
                    f"variable {self.strat_vars_cnt} to compute stratified rates; "
                    f"got {self.pop_data.get_strat_vars(prefix=False)}."
                )
            self.P = P

    def _bootstrap_iteration(
        self,
        Y_raw: NDArray,
        age_codes: NDArray,
        part_strat_codes: Optional[NDArray],
        boot_idx: NDArray,
        weights: Optional[NDArray] = None,
    ) -> Tuple[Dict[str, NDArray], Dict[str, NDArray]]:
        """
        Execute single bootstrap iteration.

        Parameters
        ----------
        Y_raw : NDArray, shape (N, K_cnt, D)
            Contact count tensor
        age_codes : NDArray, shape (N,)
            Participant age codes
        part_strat_codes : NDArray, shape (N,) or None
            Participant stratum codes
        boot_idx : NDArray, shape (N,)
            Bootstrap sample indices

        Returns
        -------
        cint_boot : Dict[str, NDArray]
            Contact intensity matrices for this bootstrap sample
        rate_boot : Dict[str, NDArray]
            Contact rate matrices for this bootstrap sample
        """
        # Aggregate bootstrap sample into Y_boot and N_boot
        Y_boot, N_boot = self._aggregate_bootstrap_sample(
            Y_raw, age_codes, part_strat_codes, boot_idx, weights
        )

        # Compute contact intensities
        cint_boot = self._compute_cint_from_YN(Y_boot, N_boot)

        # Apply reciprocity if requested
        if self.apply_reciprocity and self.pop_data is not None:
            cint_boot = apply_reciprocity(cint_boot, self.strat_mode, self.K_cnt, self.P)

        # Compute rates
        rate_boot = self._compute_rate_from_cint(cint_boot)

        return cint_boot, rate_boot

    def _aggregate_bootstrap_sample(
        self,
        Y_raw: NDArray,
        age_codes: NDArray,
        part_strat_codes: Optional[NDArray],
        boot_idx: NDArray,
        weights: Optional[NDArray] = None,
    ) -> Tuple[Union[NDArray, Dict[str, NDArray]], Union[NDArray, Dict[str, NDArray]]]:
        """
        Aggregate resampled contacts into Y_boot and N_boot.

        Parameters
        ----------
        Y_raw : NDArray, shape (N, K_cnt, D)
            Contact count tensor
        age_codes : NDArray, shape (N,)
            Participant age codes
        part_strat_codes : NDArray, shape (N,) or None
            Participant stratum codes
        boot_idx : NDArray, shape (N,)
            Bootstrap sample indices
        weights : NDArray, shape (N,), optional
            Participant weight of each resampled row. When given, contacts are
            summed as ``sum_i w_i y_id`` and the denominators are ``sum_i w_i``
            (the weighted ratio estimator).

        Returns
        -------
        Y_boot : NDArray or Dict[str, NDArray]
            Aggregated contact counts
            - Single: (C, D)
            - Partial: Dict with (C, D) arrays
            - Full: Dict with (C, D) arrays
            - Mixed: Dict with (C, D) arrays
        N_boot : NDArray or Dict[str, NDArray]
            Participant counts
            - Single: (C,)
            - Partial: Dict with (C,) arrays
            - Full: Dict with (C,) arrays
            - Mixed: Dict with (C,) arrays
        """
        # Resample
        Y_sampled = Y_raw[boot_idx]  # (N, K_cnt, D)
        age_sampled = age_codes[boot_idx]  # (N,)
        if weights is not None:
            Y_sampled = Y_sampled * np.asarray(weights, dtype=np.float64)[:, None, None]

        def count(mask=None):
            """Participants per age group (or the sum of their weights)."""
            ages = age_sampled if mask is None else age_sampled[mask]
            if weights is None:
                return np.bincount(ages, minlength=self.C).astype(np.int32)
            w = weights if mask is None else weights[mask]
            return np.bincount(ages, weights=w, minlength=self.C)

        Y_boot_dict = {}
        N_boot_dict = {}

        if self.strat_mode == "single":
            label = "All->All"
            Y_boot = np.zeros((self.C, self.D), dtype=np.float64)
            np.add.at(Y_boot, age_sampled, Y_sampled[:, 0, :])
            Y_boot_dict[label] = Y_boot
            N_boot_dict[label] = count()
            return Y_boot_dict, N_boot_dict

        # Stratified cases
        part_strat_sampled = (
            part_strat_codes[boot_idx] if part_strat_codes is not None else None
        )

        # Build stratum labels
        strat_labels = self._create_stratum_labels()

        if self.strat_mode == "partial":
            for p in range(self.K_part):
                Y_boot_dict[strat_labels[p]] = np.zeros((self.C, self.D), dtype=np.float64)
                N_boot_dict[strat_labels[p]] = np.zeros(self.C, dtype=np.int32)

            for p in range(self.K_part):
                mask = part_strat_sampled == p
                age_p = age_sampled[mask]
                np.add.at(Y_boot_dict[strat_labels[p]], age_p, Y_sampled[mask, 0, :])
                N_boot_dict[strat_labels[p]] = count(mask)

        elif self.strat_mode == "full":
            part_labels = self._create_participant_stratum_labels()
            cnt_labels = self._create_contact_stratum_labels()

            for p in range(self.K_part):
                for q in range(self.K_cnt):
                    label = f"{part_labels[p]}->{cnt_labels[q]}"
                    Y_boot_dict[label] = np.zeros((self.C, self.D), dtype=np.float64)
                    N_boot_dict[label] = np.zeros(self.C, dtype=np.int32)

            for p in range(self.K_part):
                mask = part_strat_sampled == p
                age_p = age_sampled[mask]
                Y_p = Y_sampled[mask]  # (N_p, K_cnt, D)
                for q in range(self.K_cnt):
                    label = f"{part_labels[p]}->{cnt_labels[q]}"
                    np.add.at(Y_boot_dict[label], age_p, Y_p[:, q, :])
                    N_boot_dict[label] = count(mask)

        else:  # mixed
            part_labels = self._create_participant_stratum_labels()
            cnt_labels = self._create_contact_stratum_labels()

            for p in range(max(1, self.K_part)):
                for q in range(self.K_cnt):
                    p_label = part_labels[p] if self.K_part > 1 else "All"
                    q_label = cnt_labels[q] if self.K_cnt > 1 else "All"
                    label = f"{p_label}->{q_label}"
                    Y_boot_dict[label] = np.zeros((self.C, self.D), dtype=np.float64)
                    N_boot_dict[label] = np.zeros(self.C, dtype=np.int32)

            for p in range(max(1, self.K_part)):
                mask = (part_strat_sampled == p) if part_strat_sampled is not None else np.ones(len(age_sampled), dtype=bool)
                age_p = age_sampled[mask]
                Y_p = Y_sampled[mask]
                for q in range(self.K_cnt):
                    p_label = part_labels[p] if self.K_part > 1 else "All"
                    q_label = cnt_labels[q] if self.K_cnt > 1 else "All"
                    label = f"{p_label}->{q_label}"
                    np.add.at(Y_boot_dict[label], age_p, Y_p[:, q, :])
                    N_boot_dict[label] = count(mask)

        return Y_boot_dict, N_boot_dict

    def _create_stratum_labels(self) -> List[str]:
        """Create ``"source->target"`` labels for all strata."""
        return create_stratum_labels(
            self.K_part,
            self.K_cnt,
            self.strat_vars_part,
            self.strat_vars_cnt,
            self.part_data,
            self.cnt_data,
        )

    def _create_participant_stratum_labels(self) -> List[str]:
        """Bare source-side labels, e.g. ``["M", "F"]``."""
        return participant_labels(self.strat_vars_part, self.part_data)

    def _create_contact_stratum_labels(self) -> List[str]:
        """Bare target-side labels, e.g. ``["M", "F"]``."""
        return contact_labels(self.strat_vars_cnt, self.cnt_data)

    def _compute_cint_from_YN(
        self,
        Y_boot: Dict[str, NDArray],
        N_boot: Dict[str, NDArray],
    ) -> Dict[str, NDArray]:
        """
        Compute contact intensities from Y_boot and N_boot.

        Parameters
        ----------
        Y_boot : NDArray or Dict[str, NDArray]
            Aggregated contact counts
        N_boot : NDArray or Dict[str, NDArray]
            Participant counts

        Returns
        -------
        cint_boot : Dict[str, NDArray]
            Contact intensity matrices
        """
        cint_boot = {}
        for label in Y_boot.keys():
            cint = Y_boot[label] / N_boot[label][:, np.newaxis]
            cint = np.nan_to_num(cint, nan=0.0, posinf=0.0, neginf=0.0)
            cint_boot[label] = cint

        return cint_boot

    def _compute_rate_from_cint(self, cint: Dict[str, NDArray]) -> Dict[str, NDArray]:
        """
        Compute contact rates from contact intensities.

        Parameters
        ----------
        cint : Dict[str, NDArray]
            Contact intensity matrices

        Returns
        -------
        rate : Dict[str, NDArray]
            Contact rate matrices (per-capita contact rates)
        """
        if self.pop_data is None:
            # Cannot compute rates without population data
            return {label: np.zeros_like(m) for label, m in cint.items()}

        rate = {}

        if self.strat_mode == "single":
            # rate = cint / P[newaxis, :]
            P = self.P
            rate["All->All"] = cint["All->All"] / P[np.newaxis, :]
            rate["All->All"] = np.nan_to_num(
                rate["All->All"], nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            # Stratified
            # Get stratum label mappings
            part_labels = self._create_participant_stratum_labels()
            cnt_labels = self._create_contact_stratum_labels()

            for label, m in cint.items():
                if self.P.ndim == 1:
                    P = self.P
                else:
                    # Extract target stratum
                    parts = label.split("->")
                    target = parts[1]

                    # For "source->All" format, use source (participant) stratum for population
                    if target == "All":
                        source = parts[0]
                        k = part_labels.index(source)
                    else:
                        # For "source->target" format, use target (contact) stratum
                        k = cnt_labels.index(target)

                    P = self.P[k, :]

                rate[label] = m / P[np.newaxis, :]
                rate[label] = np.nan_to_num(
                    rate[label], nan=0.0, posinf=0.0, neginf=0.0
                )

        return rate
