"""
SocialMix Validation Module

Handles validation logic for SocialMix preprocessing, including:
- Reciprocity requirement validation
- Shared stratification variable validation
- Bootstrap requirement validation and adaptive age group merging
"""

import dataclasses
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...dataloader import ContactData, ParticipantData, PopulationData
from ...utils import AgeGroupSpecs
from ._socialmix_age_processing import AgeBinProcessor
from ._socialmix_helpers import infer_strat_mode
from ...utils._participant_cells import build_participant_cells, copy_container


class SocialMixValidator:
    """
    Validates SocialMix inputs and handles adaptive merging.

    This class encapsulates all validation logic for the SocialMix preprocessing
    pipeline, including stratification consistency checks, reciprocity requirements,
    and bootstrap stability validation.

    Parameters
    ----------
    part_data : ParticipantData
        Participant data container
    cnt_data : ContactData
        Contact data container
    age_bins : AgeGroupSpecs
        Age stratification bins
    pop_data : PopulationData, optional
        Population data container (required for reciprocity)
    apply_reciprocity : bool
        Whether to apply reciprocity adjustment
    adaptive_merge : bool
        Whether to automatically merge age groups with insufficient samples
    validate_for_bootstrap : bool, default False
        Whether to validate data for bootstrap stability. If True, performs
        more aggressive age group merging. If False, only merges age groups
        necessary for contact intensity estimation.

    Attributes
    ----------
    age_processor : AgeBinProcessor
        Helper for age group operations and merging
    """

    def __init__(
        self,
        part_data: ParticipantData,
        cnt_data: ContactData,
        age_group_specs: AgeGroupSpecs,
        pop_data: Optional[PopulationData] = None,
        apply_reciprocity: bool = True,
        adaptive_merge: bool = False,
        validate_for_bootstrap: bool = False,
    ):
        self.part_data = part_data
        self.cnt_data = cnt_data
        self.age_group_specs = age_group_specs
        self.pop_data = pop_data
        self.apply_reciprocity = apply_reciprocity
        self.adaptive_merge = adaptive_merge
        self.validate_for_bootstrap = validate_for_bootstrap

        # Initialize age processor
        self.age_processor = AgeBinProcessor(age_group_specs)

        # Participants dropped because their age is outside the age groups
        self.n_dropped: int = 0

        # Stratification variables
        self.strat_vars_part: List[str] = []
        self.strat_vars_cnt: List[str] = []
        self.strat_vars_pop: List[str] = []
        self.strat_vars_shared: List[str] = []
        self.strat_mode: str = None

    def validate_all(self) -> Dict:
        """
        Run all validation checks and return updated components.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'part_data': Updated ParticipantData (may have merged age groups)
            - 'cnt_data': Updated ContactData (may have merged age groups)
            - 'age_bins': Updated AgeGroupSpecs (may have merged bins)
            - 'apply_reciprocity': Final reciprocity flag (may be disabled)

        Notes
        -----
        This method orchestrates all validation steps in the correct order:
        1. Extract stratification variables
        2. Validate reciprocity requirements
        3. Validate shared stratification variables
        4. Validate estimation requirements (no empty age groups)
        5. Validate bootstrap requirements (if validate_for_bootstrap=True)
        """
        # Extract stratification variables
        self._extract_strat_vars()

        # Validate reciprocity requirements
        self.apply_reciprocity = self._validate_reciprocity_requirements()

        # Validate shared stratification variables (if any)
        if self.strat_vars_shared:
            self._validate_shared_strat_vars()

        # Validate estimation requirements (no empty age groups)
        self._validate_estimation_requirements()

        # Validate bootstrap requirements only if requested
        # (more aggressive merging to ensure bootstrap stability)
        if self.validate_for_bootstrap:
            self._validate_bootstrap_requirements()

        # Final participant cells, after any merging. Nothing is dropped or
        # merged here any more, so this emits no warnings.
        cells = dataclasses.replace(
            build_participant_cells(
                self.part_data.data,
                [f"part_{var}" for var in self.strat_vars_part],
                self.age_group_specs,
                adaptive_merge=False,
                strict=True,
            ),
            n_dropped=self.n_dropped,
        )

        return {
            "part_data": self.part_data,
            "cnt_data": self.cnt_data,
            "age_group_specs": self.age_group_specs,
            "apply_reciprocity": self.apply_reciprocity,
            "cells": cells,
        }

    def _extract_strat_vars(self) -> None:
        """Extract stratification variables from data containers."""
        self.strat_vars_part = self.part_data.get_strat_vars(prefix=False)
        self.strat_vars_cnt = self.cnt_data.get_strat_vars(prefix=False)
        self.strat_vars_pop = (
            self.pop_data.get_strat_vars(prefix=False)
            if self.pop_data is not None
            else []
        )

        # Identify shared variables
        self.strat_vars_shared = sorted(
            list(set(self.strat_vars_part) & set(self.strat_vars_cnt))
        )

        # Infer stratification mode
        self._infer_strat_mode()

    def _infer_strat_mode(self) -> None:
        """Infer and cache the stratification mode."""
        self.strat_mode = infer_strat_mode(self.strat_vars_part, self.strat_vars_cnt)

    def _validate_reciprocity_requirements(self) -> bool:
        """
        Validate that reciprocity adjustment can be applied.

        Ensures that the inputs contain all necessary data for reciprocity adjustment.
        Returns updated reciprocity flag (may disable if requirements not met).

        Returns
        -------
        bool
            Whether reciprocity can be applied
        """
        if not self.apply_reciprocity:
            return False

        # Single stratum without population data
        if self.strat_mode == "single" and self.pop_data is None:
            warnings.warn(
                "Reciprocity adjustment requested but no population data provided. "
                "Proceeding without reciprocity adjustment.",
                UserWarning,
            )
            return False

        # Partial stratification (not supported)
        if self.strat_mode == "partial":
            warnings.warn(
                "Reciprocity adjustment is not applicable for partial stratification. "
                "Proceeding without reciprocity adjustment.",
                UserWarning,
            )
            return False

        # Mixed stratification (not supported)
        if self.strat_mode == "mixed":
            warnings.warn(
                "Reciprocity adjustment is not supported for mixed stratification. "
                "Proceeding without reciprocity adjustment.",
                UserWarning,
            )
            return False

        # Full stratification requires population data
        if self.strat_mode == "full":
            if self.pop_data is None:
                warnings.warn(
                    "Reciprocity adjustment requires population data when "
                    "both participant and contact data are stratified. "
                    "Proceeding without reciprocity adjustment.",
                    UserWarning,
                )
                return False

            # Population must have same stratification as shared variables
            if set(self.strat_vars_shared) != set(self.strat_vars_pop):
                warnings.warn(
                    f"Reciprocity adjustment requires population data to have "
                    f"the same stratification variables as participants and contacts. "
                    f"Population vars: {self.strat_vars_pop}, "
                    f"Expected: {self.strat_vars_shared}. "
                    f"Missing vars: {set(self.strat_vars_shared) - set(self.strat_vars_pop)}. "
                    f"Proceeding without reciprocity adjustment.",
                    UserWarning,
                )
                return False

        return True

    def _validate_shared_strat_vars(self) -> None:
        """
        Validate and align shared stratification variables.

        For variables that appear in participant, contact, and population data, this ensures:
        1. Both sides have the same unique categories
        2. Categories are encoded in the same order (using participant order as reference)

        If categories match but order differs, contact data is automatically reordered
        to match participant data encoding.

        Raises
        ------
        ValueError
            If categories don't match (different sets of values).
        """
        for var in self.strat_vars_shared:
            # Get column names
            col_part = f"part_{var}"
            col_cnt = f"cnt_{var}"

            # Get categories from both sides
            part_col = self.part_data.data[col_part]
            cnt_col = self.cnt_data.data[col_cnt]

            # Convert to categorical if not already
            if not hasattr(part_col, "cat"):
                part_col = part_col.astype("category")
                self.part_data.data[col_part] = part_col
            if not hasattr(cnt_col, "cat"):
                cnt_col = cnt_col.astype("category")
                self.cnt_data.data[col_cnt] = cnt_col

            # Get categories in encoding order
            part_cats = list(part_col.cat.categories)
            cnt_cats = list(cnt_col.cat.categories)

            # Check if categories match as sets
            part_set = set(part_cats)
            cnt_set = set(cnt_cats)

            if part_set != cnt_set:
                # Different categories - this is an error
                only_part = part_set - cnt_set
                only_cnt = cnt_set - part_set
                raise ValueError(
                    f"Shared stratification variable '{var}' has different categories:\n"
                    f"  Participant side: {part_cats}\n"
                    f"  Contact side: {cnt_cats}\n"
                    f"  Only in participants: {sorted(only_part) if only_part else 'None'}\n"
                    f"  Only in contacts: {sorted(only_cnt) if only_cnt else 'None'}\n"
                    f"For shared variables, both sides must have identical categories."
                )

            # Same categories but possibly different order
            # Use participant ordering as reference and reorder contact data
            if part_cats != cnt_cats:
                self.cnt_data.data[col_cnt] = self.cnt_data.data[
                    col_cnt
                ].cat.reorder_categories(part_cats, ordered=False)

            # Population data (if present) must also match participant categories
            if self.pop_data is not None and var in self.strat_vars_pop:
                col_pop = var
                pop_col = self.pop_data.data[col_pop]

                if not hasattr(pop_col, "cat"):
                    pop_col = pop_col.astype("category")
                    self.pop_data.data[col_pop] = pop_col

                pop_cats = list(pop_col.cat.categories)
                pop_set = set(pop_cats)

                if part_set != pop_set:
                    self.apply_reciprocity = False

                    only_part = part_set - pop_set
                    only_pop = pop_set - part_set
                    warnings.warn(
                        f"Shared stratification variable '{var}' has different categories in population data:\n"
                        f"  Participant side: {part_cats}\n"
                        f"  Population side: {pop_cats}\n"
                        f"  Only in participants: {sorted(only_part) if only_part else 'None'}\n"
                        f"  Only in population: {sorted(only_pop) if only_pop else 'None'}\n"
                        f"For shared variables, both sides must have identical categories. "
                        f"Proceeding without reciprocity adjustment.",
                        UserWarning,
                    )

                if part_cats != pop_cats:
                    self.pop_data.data[col_pop] = self.pop_data.data[
                        col_pop
                    ].cat.reorder_categories(part_cats, ordered=False)

    def _validate_estimation_requirements(self) -> None:
        """
        Validate that contact intensity matrix can be estimated.

        Ensures that there are no empty participant age groups, which would
        cause division by zero when computing contact intensities.

        Strategy:
        1. Check for empty age groups across all strata
        2. If adaptive_merge=True, merge empty age groups until none remain
        3. If adaptive_merge=False, raise an error

        Raises
        ------
        ValueError
            If there are empty age groups and adaptive_merge=False

        Notes
        -----
        This validation must run before bootstrap validation since it may
        modify the age bins.
        """
        # Verify that the number of bins in age_group_specs matches the actual
        # number of age-group categories in the data.  A mismatch typically means
        # the data was pre-binned with pd.cut(..., bins=range(0, N, step)) whose
        # N is an *exclusive* upper edge, while AgeGroupSpecs(0, N, step) treats
        # N as an *inclusive* maximum age, producing one extra bin.
        n_bins = len(self.age_group_specs.left)
        n_part_cats = len(self.part_data.data["part_age_grp"].cat.categories)
        n_cnt_cats = len(self.cnt_data.data["cnt_age_grp"].cat.categories)
        if n_part_cats != n_bins or n_cnt_cats != n_bins:
            raise ValueError(
                f"age_group_specs defines {n_bins} age bins, but the data has "
                f"{n_part_cats} participant age group(s) and {n_cnt_cats} contact "
                f"age group(s). "
                f"If you pre-binned the data with "
                f"pd.cut(..., bins=range(0, {self.age_group_specs.max + 1}, {self.age_group_specs.step or 'step'}), right=False), "
                f"note that AgeGroupSpecs treats 'max' as an inclusive upper bound. "
                f"Use AgeGroupSpecs(0, {self.age_group_specs.max - (self.age_group_specs.step or 1)}, "
                f"{self.age_group_specs.step or 'step'}) so that the last bin ends at "
                f"{self.age_group_specs.max - 1}."
            )

        # Bin, drop participants outside the age groups (D23) and merge empty
        # age groups, through the shared helper that previews also use.
        cells = build_participant_cells(
            self.part_data.data,
            [f"part_{var}" for var in self.strat_vars_part],
            self.age_group_specs,
            adaptive_merge=self.adaptive_merge,
            strict=True,
        )
        self.n_dropped = cells.n_dropped
        # Replace the data instead of rebuilding the container, so that every
        # column survives (D24).
        self.part_data = copy_container(self.part_data, cells.part)
        if cells.merged:
            self.cnt_data = copy_container(
                self.cnt_data,
                self.age_processor.reassign_age_groups(
                    self.cnt_data.data,
                    age_grp_col="cnt_age_grp",
                    merged_intervals=cells.age_groups,
                    new_group_col="cnt_age_grp",
                ),
            )
            self.age_group_specs = cells.age_group_specs
            self.age_processor = AgeBinProcessor(self.age_group_specs)

    def _validate_bootstrap_requirements(self) -> None:
        """
        Validate that bootstrap can be performed with current stratification.

        Ensures that bootstrap resampling does not fail due to insufficient data,
        accounting for both age groups and participant stratification variables.

        Strategy:
        1. Groups by strat_vars_part + age_grp_part to get counts for each (stratum, age) combination
        2. Identifies age groups that are empty in ANY stratum (must be globally merged)
        3. Calculates failure probability using Bonferroni union bound (sum across all strata)
        4. If adaptive_merge=True, recursively merges age groups until failure probability is acceptable

        Notes
        -----
        Age bins must be consistent across all strata - if an age group is empty in any
        stratum, it must be merged globally across all strata.

        This method may modify:
        - self.part_data (reassigned age groups)
        - self.cnt_data (reassigned age groups)
        - self.age_group_specs (merged bins)
        """
        # Determine grouping columns based on stratification
        group_cols = [f"part_{var}" for var in self.strat_vars_part] + ["part_age_grp"]

        # Compute counts for all (stratum, age_group) combinations
        if self.strat_vars_part:
            # Stratified case: group by strat vars + age
            strata_age_counts = self.part_data.data.groupby(
                group_cols, observed=False
            ).size()

            # For each age group, find minimum count across all strata
            # An age group is problematic if it's empty (count=0) in ANY stratum
            age_min_counts = strata_age_counts.groupby(
                "part_age_grp", observed=False
            ).min()

            # Reindex to ensure all age groups present
            age_min_counts = age_min_counts.reindex(
                pd.Index(self.part_data.data["part_age_grp"].cat.categories),
                fill_value=0,
            )

            empty_groups = age_min_counts[age_min_counts == 0].index.tolist()
        else:
            # No stratification: simple age group counts
            age_grp_counts = (
                self.part_data.data.groupby("part_age_grp", observed=False)
                .size()
                .reindex(
                    pd.Index(self.part_data.data["part_age_grp"].cat.categories),
                    fill_value=0,
                )
            )

            empty_groups = age_grp_counts[age_grp_counts == 0].index.tolist()
            age_min_counts = age_grp_counts  # For consistency in later code

        # Handle empty age groups
        if empty_groups:
            if self.adaptive_merge:
                # Merge empty age groups globally (must be consistent across all strata)
                self._merge_empty_age_groups(age_min_counts)

                # Recalculate counts after merging
                if self.strat_vars_part:
                    strata_age_counts = self.part_data.data.groupby(
                        group_cols, observed=False
                    ).size()
                    age_min_counts = strata_age_counts.groupby("part_age_grp", observed=False).min()
                    age_min_counts = age_min_counts.reindex(
                        pd.Index(self.part_data.data["part_age_grp"].cat.categories),
                        fill_value=0,
                    )
                else:
                    age_min_counts = (
                        self.part_data.data.groupby("part_age_grp", observed=False)
                        .size()
                        .reindex(
                            pd.Index(
                                self.part_data.data["part_age_grp"].cat.categories
                            ),
                            fill_value=0,
                        )
                    )
            else:
                strata_info = ""
                if self.strat_vars_part:
                    strata_info = f" across {len(self.strat_vars_part)} stratification variable(s)"

                raise ValueError(
                    f"\nCannot run bootstrap: found empty participant age group(s){strata_info}: {empty_groups}. "
                    f"\nEmpty age groups cause division by zero during bootstrap resampling. "
                    f"\nPlease either:\n"
                    f"  1. Set adaptive_merge=True to automatically merge empty age groups, or\n"
                    f"  2. Use coarser age bins that avoid empty groups, or\n"
                    f"  3. Collect more participant data to fill all age groups."
                )

        # Calculate failure probability using Bonferroni union bound
        if self.strat_vars_part:
            # Recompute full strata_age_counts for failure probability calculation
            strata_age_counts = self.part_data.data.groupby(
                group_cols, observed=False
            ).size()
            # failure_prob = sum over all (stratum, age) of exp(-n_stratum_age)
            failure_probs = np.exp(-strata_age_counts.values)
            failure_prob = np.sum(failure_probs)
        else:
            # No stratification: simple sum over age groups
            failure_probs = np.exp(-age_min_counts.values)
            failure_prob = np.sum(failure_probs)

        threshold = 0.00005  # 5% failure rate for 1000 iterations

        if failure_prob >= threshold:
            # Recursively merge smallest age groups until failure probability drops
            self._adaptive_merge_for_stability(
                group_cols, age_min_counts, failure_prob, threshold
            )

    def _merge_empty_age_groups(self, age_min_counts: pd.Series) -> None:
        """
        Merge empty age groups globally across all strata.

        Parameters
        ----------
        age_min_counts : pd.Series
            Minimum counts per age group across all strata
        """
        merged_intervals = self.age_processor.merge_zero_groups(
            list(age_min_counts.index), age_min_counts.values
        )

        # Reassign age groups in participant and contact data
        df_part_updated = self.age_processor.reassign_age_groups(
            self.part_data.data,
            age_grp_col="part_age_grp",
            merged_intervals=merged_intervals,
            new_group_col="part_age_grp",
        )

        df_cnt_updated = self.age_processor.reassign_age_groups(
            self.cnt_data.data,
            age_grp_col="cnt_age_grp",
            merged_intervals=merged_intervals,
            new_group_col="cnt_age_grp",
        )

        # Replace the data instead of rebuilding the containers, so that every
        # column survives (D24).
        self.part_data = copy_container(self.part_data, df_part_updated)
        self.cnt_data = copy_container(self.cnt_data, df_cnt_updated)

        # Replace age bins with a new AgeGroupSpecs built from merged bounds
        new_left = [interval.left for interval in merged_intervals]
        new_right = [interval.right - 1 for interval in merged_intervals]
        self.age_group_specs = self.age_group_specs.replace(
            age_min=new_left, age_max=new_right
        )

    def _adaptive_merge_for_stability(
        self,
        group_cols: List[str],
        age_min_counts: pd.Series,
        failure_prob: float,
        threshold: float,
    ) -> None:
        """
        Recursively merge age groups to achieve bootstrap stability.

        Parameters
        ----------
        group_cols : list of str
            Grouping columns for stratified counts
        age_min_counts : pd.Series
            Minimum counts per age group
        failure_prob : float
            Current failure probability
        threshold : float
            Target failure probability threshold
        """
        current_intervals = list(age_min_counts.index)
        current_counts = age_min_counts.values.copy()

        merge_iterations = 0
        max_iterations = len(current_intervals) - 1  # Safety limit

        while failure_prob >= threshold and merge_iterations < max_iterations:
            # Merge age group with smallest minimum count across all strata
            try:
                merged_intervals = self.age_processor._merge_smallest_age_group(
                    current_intervals, current_counts
                )
            except ValueError:
                # Only one group left, can't merge further
                warnings.warn(
                    f"Cannot merge further (only one age group remains), "
                    f"but failure probability ({failure_prob:.6f}) is still above "
                    f"threshold ({threshold:.6f}).",
                    UserWarning,
                )
                break

            # Reassign age groups and recalculate counts
            df_part_updated = self.age_processor.reassign_age_groups(
                self.part_data.data,
                age_grp_col="part_age_grp",
                merged_intervals=merged_intervals,
                new_group_col="part_age_grp",
            )

            df_cnt_updated = self.age_processor.reassign_age_groups(
                self.cnt_data.data,
                age_grp_col="cnt_age_grp",
                merged_intervals=merged_intervals,
                new_group_col="cnt_age_grp",
            )

            # Replace the data instead of rebuilding the containers, so that
            # every column survives (D24).
            self.part_data = copy_container(self.part_data, df_part_updated)
            self.cnt_data = copy_container(self.cnt_data, df_cnt_updated)

            # Update validator's stratification tracking to match new containers
            self.strat_vars_part = self.part_data.get_strat_vars(prefix=False)
            self.strat_vars_cnt = self.cnt_data.get_strat_vars(prefix=False)

            # Rebuild group_cols with updated stratification info
            if self.strat_vars_part:
                group_cols = [f"part_{var}" for var in self.strat_vars_part] + [
                    "part_age_grp"
                ]
            else:
                group_cols = ["part_age_grp"]

            # Update current state
            current_intervals = merged_intervals

            # Recalculate counts and failure probability
            if self.strat_vars_part:
                strata_age_counts = self.part_data.data.groupby(
                    group_cols, observed=False
                ).size()
                age_min_counts = strata_age_counts.groupby(
                    "part_age_grp", observed=False
                ).min()
                age_min_counts = age_min_counts.reindex(
                    pd.Index(self.part_data.data["part_age_grp"].cat.categories),
                    fill_value=0,
                )
                current_counts = age_min_counts.values.copy()

                # Recalculate failure probability (sum across all strata)
                failure_probs = np.exp(-strata_age_counts.values)
                failure_prob = np.sum(failure_probs)
            else:
                age_min_counts = (
                    self.part_data.data.groupby("part_age_grp", observed=False)
                    .size()
                    .reindex(
                        pd.Index(self.part_data.data["part_age_grp"].cat.categories),
                        fill_value=0,
                    )
                )
                current_counts = age_min_counts.values.copy()

                # Recalculate failure probability
                failure_probs = np.exp(-current_counts)
                failure_prob = np.sum(failure_probs)

            merge_iterations += 1

        # Replace age bins with a new AgeGroupSpecs built from merged bounds
        if merge_iterations > 0:
            new_left = [interval.left for interval in merged_intervals]
            new_right = [interval.right - 1 for interval in merged_intervals]
            self.age_group_specs = self.age_group_specs.replace(
                age_min=new_left, age_max=new_right
            )

            # Format age bins for display
            left_str = ", ".join(str(x) for x in self.age_group_specs.left)
            right_str = ", ".join(str(x) for x in self.age_group_specs.right)

            strata_info = ""
            if self.strat_vars_part:
                strata_age_counts = self.part_data.data.groupby(
                    group_cols, observed=False
                ).size()
                n_strata = len(strata_age_counts) // len(self.age_group_specs.left)
                strata_info = f" across {n_strata} strata"

            warnings.warn(
                f"\nAdaptively merged {merge_iterations} age group(s) to ensure bootstrap stability{strata_info}.\n"
                f"  Final failure probability for 1000 iterations: {1000*failure_prob:.2f} (threshold: {1000*threshold:.2f})\n"
                f"  Final age bins:\n"
                f"    left:  [{left_str}]\n"
                f"    right: [{right_str}]",
                UserWarning,
            )
