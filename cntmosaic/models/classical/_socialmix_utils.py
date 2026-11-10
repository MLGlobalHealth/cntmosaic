"""
Data Loading Utilities for SocialMix

This module provides helper classes for loading and processing stratified contact data.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...dataloader import ContactData, ParticipantData, PopulationData
from ...utils import AgeGroupSpecs
from ._socialmix_age_processing import AgeBinProcessor
from ._socialmix_helpers import population_by_contact_stratum, ravel_stratum_codes

if TYPE_CHECKING:
    from ._SocialMix import SocialMix


class SocialMixDataLoader:
    """
    Helper class for loading and processing stratified contact data.

    This class handles the conversion of contact survey data into the
    numerical arrays needed for matrix estimation, with full support for
    stratification.
    """

    def __init__(self, socialmix_instance: "SocialMix"):
        """
        Initialize data processor.

        Parameters
        ----------
        socialmix_instance : SocialMix
            Reference to parent SocialMix instance for accessing data and state
        """
        self.sm = socialmix_instance

    def load_data(self) -> None:
        """
        Orchestrate the data loading process.

        Sets the following attributes on the parent SocialMix instance:
        - Y: Contact count matrix/tensor
        - N: Participant count matrix/vector
        - P: Population size matrix/vector (if applicable)
        - C, D, K_part, K_cnt: Dimensions
        """
        # 1. Calculate dimensions
        self._calculate_dimensions()

        # 2. Compute N: participant counts
        self._compute_N()

        # 3. Compute Y: contact counts
        self._compute_Y()

        # 4. Compute P: population sizes (if pop_data provided)
        if self.sm.pop_data is not None:
            self._assign_age_groups_to_population()
            self._compute_P()
        else:
            self.sm.P = None

    def _calculate_dimensions(self) -> None:
        """
        Calculate and store stratification dimensions.
        """
        # Get age group dimensions
        self.sm.C = len(self.sm.part_data.data["part_age_grp"].cat.categories)
        self.sm.D = len(self.sm.cnt_data.data["cnt_age_grp"].cat.categories)

        # Calculate K_part: number of participant strata
        if self.sm.strat_vars_part:
            self.sm.K_part = 1
            for var in self.sm.strat_vars_part:
                self.sm.K_part *= self.sm.strat_dims_part[var]
        else:
            self.sm.K_part = 1

        # Calculate K_cnt: number of contact strata
        if self.sm.strat_vars_cnt:
            self.sm.K_cnt = 1
            for var in self.sm.strat_vars_cnt:
                self.sm.K_cnt *= self.sm.strat_dims_cnt[var]
        else:
            self.sm.K_cnt = 1

    def _compute_N(self) -> None:
        """
        Compute participant counts per age group and stratum.

        Sets
        ----
        self.sm.N : NDArray
            - Shape (C,) if K_part == 1 (no stratification)
            - Shape (K_part, C) if K_part > 1 (stratified)
        """
        df_part = self.sm.part_data.data
        age_grps_part = df_part["part_age_grp"].cat.categories

        if self.sm.K_part == 1:
            # No stratification: simple count by age group
            counts = (
                df_part.groupby("part_age_grp", observed=False)
                .size()
                .reindex(pd.Index(age_grps_part), fill_value=0)
            )
            self.sm.N = counts.values.astype(np.int64)
        else:
            # Stratified: group by strat vars + age
            group_cols = [f"part_{var}" for var in self.sm.strat_vars_part] + [
                "part_age_grp"
            ]
            counts = df_part.groupby(group_cols, observed=False).size()

            # Reshape to (K_part, C) matrix
            self.sm.N = self._fill_simple_array(
                counts,
                shape=(self.sm.K_part, self.sm.C),
                strat_vars=self.sm.strat_vars_part,
                suffix="part",
            )

    def _compute_Y(self) -> None:
        """
        Compute contact count matrix with stratification.

        When ``part_data.weight_col`` is set, each contact record is multiplied
        by the participant's survey weight before summation.  If
        ``self.sm.normalise_weights`` is True, weights are first normalized
        within each (age_group, stratum) cell so that
        ``sum(w_i in cell) == N_cell``, preserving the contact intensity
        estimator ``M = Y / N``. When False (the default), the raw weights
        supplied by the user are used as-is.

        Sets
        ----
        self.sm.Y : NDArray
            - Shape (C, D) if K_part == 1 and K_cnt == 1 (no stratification)
            - Shape (K_part, C, D) if K_part > 1 and K_cnt == 1 (partial)
            - Shape (K_part, K_cnt, C, D) if both K_part > 1 and K_cnt > 1 (full/mixed)
        """
        df_part = self.sm.part_data.data
        df_cnt = self.sm.cnt_data.data
        age_grps_part = df_part["part_age_grp"].cat.categories
        age_grps_cnt = df_cnt["cnt_age_grp"].cat.categories

        # Normalize weights within (age_group, stratum) cells when present,
        # unless the caller opted out via SocialMix(normalise_weights=False).
        has_weights = self.sm.part_data.weight_col is not None
        if has_weights and self.sm.normalise_weights:
            grp_cols = ["part_age_grp"] + [
                f"part_{v}" for v in self.sm.strat_vars_part
            ]
            if not self.sm.part_data.check_weights_normalized(grp_cols):
                self.sm.part_data.normalize_weights(grp_cols)

        # Merge participants with contacts
        part_cols = ["id", "part_age_grp"] + [
            f"part_{v}" for v in self.sm.strat_vars_part
        ]
        if has_weights:
            part_cols.append("part_weight")
        merged = df_cnt.merge(df_part[part_cols], on="id", how="inner")

        if len(merged) == 0:
            raise ValueError(
                "No valid contacts after merging with participants. "
                "Check that contact IDs match participant IDs."
            )

        # Create the column to aggregate: weighted or raw contact counts
        if has_weights:
            merged["_y"] = merged["y"] * merged["part_weight"]
        else:
            merged["_y"] = merged["y"]

        if self.sm.K_part == 1 and self.sm.K_cnt == 1:
            # No stratification: simple (C, D) matrix
            Y_agg = (
                merged.groupby(["part_age_grp", "cnt_age_grp"], observed=False)["_y"]
                .sum()
                .unstack(fill_value=0)
            )
            Y_df = Y_agg.reindex(
                index=pd.Index(age_grps_part),
                columns=pd.Index(age_grps_cnt),
                fill_value=0,
            )
            self.sm.Y = Y_df.values.astype(np.float64)

        elif self.sm.K_part > 1 and self.sm.K_cnt == 1:
            # Partial stratification: (K_part, C, D) tensor
            group_cols = [f"part_{v}" for v in self.sm.strat_vars_part] + [
                "part_age_grp",
                "cnt_age_grp",
            ]
            Y_agg = merged.groupby(group_cols, observed=False)["_y"].sum()

            self.sm.Y = self._fill_partial_array(
                Y_agg,
                shape=(self.sm.K_part, self.sm.C, self.sm.D),
                strat_vars_part=self.sm.strat_vars_part,
            )

        else:
            # Full/mixed stratification: (K_part, K_cnt, C, D) tensor
            group_cols = (
                [f"part_{v}" for v in self.sm.strat_vars_part]
                + ["part_age_grp"]
                + [f"cnt_{v}" for v in self.sm.strat_vars_cnt]
                + ["cnt_age_grp"]
            )
            Y_agg = merged.groupby(group_cols, observed=False)["_y"].sum()

            self.sm.Y = self._fill_full_array(
                Y_agg,
                shape=(self.sm.K_part, self.sm.K_cnt, self.sm.C, self.sm.D),
                strat_vars_part=self.sm.strat_vars_part,
                strat_vars_cnt=self.sm.strat_vars_cnt,
            )

    def _compute_P(self) -> None:
        """
        Compute population sizes per age group and stratum.

        Sets
        ----
        self.sm.P : NDArray
            - Shape (D,) if K_cnt == 1 (no contact stratification)
            - Shape (K_cnt, D) if K_cnt > 1 (contact stratified)
        """
        df_pop = self.sm.pop_data.data
        age_grps = df_pop["age_grp"].cat.categories

        if self.sm.K_cnt == 1:
            # No stratification: simple sum by age group
            pop_sizes = (
                df_pop.groupby("age_grp", observed=False)["P"]
                .sum()
            )

            if (pop_sizes == 0).any():
                zero_groups = pop_sizes[pop_sizes == 0].index.tolist()
                raise ValueError(
                    f"Age groups have zero population: {zero_groups}. "
                    f"Check that population data covers all age bins."
                )

            self.sm.P = pop_sizes.values.astype(np.float64)
        else:
            # Stratified: match population rows to contact strata by category
            # value, in contact-stratum label order.
            P = population_by_contact_stratum(
                df_pop,
                self.sm.strat_vars_cnt,
                self.sm.cnt_data.data,
                n_age=self.sm.D,
            )
            if P is None:
                # Population lacks some contact stratification variable:
                # keep the previous behaviour (rows filled by population codes).
                group_cols = self.sm.strat_vars_pop + ["age_grp"]
                pop_sizes = df_pop.groupby(group_cols, observed=False)["P"].sum()
                P = self._fill_simple_array(
                    pop_sizes,
                    shape=(self.sm.K_cnt, self.sm.D),
                    strat_vars=self.sm.strat_vars_pop,
                    suffix=None,  # Population data has no part_/cnt_ prefix
                )
            self.sm.P = P

    def _assign_age_groups_to_population(self) -> None:
        """
        Assign age groups to population data using same bins as contacts.
        """
        import warnings

        if "age_grp" in self.sm.pop_data.data.columns:
            return  # Already assigned

        if "pop_age_grp" in self.sm.pop_data.data.columns:
            expected_intervals = list(
                pd.IntervalIndex.from_breaks(self.sm.age_group_specs.bins, closed="left")
            )
            pop_intervals = set(self.sm.pop_data.data["pop_age_grp"].dropna().unique())

            if pop_intervals.issubset(set(expected_intervals)):
                # Groups already align — copy directly
                self.sm.pop_data.data["age_grp"] = self.sm.pop_data.data["pop_age_grp"]
            else:
                # Fine intervals need merging up to the new (post-adaptive-merge) bins
                self._rebin_population_age_groups(expected_intervals)
            return

        # Population is in 1-year resolution — it will be binned and summed.
        warnings.warn(
            "Population data is provided in 1-year age resolution and will be "
            "aggregated into coarse bins matching age_group_specs. "
            "If you intended to use pre-grouped population counts, provide "
            "PopulationData with age_grp_col instead of age_col.",
            UserWarning,
            stacklevel=4,
        )

        self.sm.pop_data.data["age_grp"] = self.sm.age_group_specs.cut(
            self.sm.pop_data.data["age"]
        )

    def _rebin_population_age_groups(self, merged_intervals: list) -> None:
        """Remap pop_age_grp to merged_intervals, summing P values across merged groups.

        Used when adaptive age-group merging has coarsened age_group_specs after
        the initial load, leaving pop_age_grp at a finer resolution than expected.
        AgeBinProcessor.reassign_age_groups raises ValueError if any pop interval
        cannot be contained within a merged interval (incompatible binning).
        """
        df_pop = AgeBinProcessor.reassign_age_groups(
            self.sm.pop_data.data,
            age_grp_col="pop_age_grp",
            merged_intervals=merged_intervals,
            new_group_col="age_grp",
        )
        self.sm.pop_data.data = df_pop

    def _fill_simple_array(
        self,
        series: pd.Series,
        shape: Tuple[int, ...],
        strat_vars: List[str],
        suffix: Optional[str],
    ) -> NDArray:
        """Fill a (K, age) array for N (participants) or P (population).

        Parameters
        ----------
        series : pd.Series
            Multi-indexed series from a groupby on [strat_vars..., age_grp].
        shape : tuple
            Target array shape ``(K, age_bins)``.
        strat_vars : list of str
            Stratification variable names without prefix.
        suffix : str or None
            Column prefix — ``"part"`` → ``"part_{var}"``, ``None`` → ``"{var}"``
            (used for population data which carries no prefix).
        """
        arr = np.zeros(shape, dtype=np.float64)
        df = series.reset_index()
        df.columns = [*df.columns[:-1], "value"]

        for var in strat_vars:
            col_name = f"{suffix}_{var}" if suffix else var
            df[f"{var}_code"] = df[col_name].cat.codes

        if "part_age_grp" in df.columns:
            age_col = "part_age_grp"
        elif "age_grp" in df.columns:
            age_col = "age_grp"
        else:
            raise ValueError("No age group column found in series index")

        df["age_code"] = df[age_col].cat.codes

        if strat_vars:
            strat_dims = [
                self.sm.strat_dims_part.get(v) or self.sm.strat_dims_cnt.get(v)
                for v in strat_vars
            ]
            df["k"] = self._create_composite_index(
                df, [f"{v}_code" for v in strat_vars], strat_dims
            )
        else:
            df["k"] = 0

        for _, row in df.iterrows():
            arr[int(row["k"]), int(row["age_code"])] = row["value"]

        return arr

    def _fill_partial_array(
        self,
        series: pd.Series,
        shape: Tuple[int, ...],
        strat_vars_part: List[str],
    ) -> NDArray:
        """Fill a (K_part, C, D) array for Y under partial stratification.

        Parameters
        ----------
        series : pd.Series
            Multi-indexed series from a groupby on
            [part_{v}..., part_age_grp, cnt_age_grp].
        shape : tuple
            Target array shape ``(K_part, C, D)``.
        strat_vars_part : list of str
            Participant stratification variable names without prefix.
        """
        arr = np.zeros(shape, dtype=np.float64)
        df = series.reset_index()
        df.columns = [*df.columns[:-1], "value"]

        for var in strat_vars_part:
            df[f"{var}_code"] = df[f"part_{var}"].cat.codes
        df["part_age_grp_code"] = df["part_age_grp"].cat.codes
        df["cnt_age_grp_code"] = df["cnt_age_grp"].cat.codes

        df["k"] = self._create_composite_index(
            df,
            [f"{v}_code" for v in strat_vars_part],
            [self.sm.strat_dims_part[v] for v in strat_vars_part],
        )

        for _, row in df.iterrows():
            arr[
                int(row["k"]),
                int(row["part_age_grp_code"]),
                int(row["cnt_age_grp_code"]),
            ] = row["value"]

        return arr

    def _fill_full_array(
        self,
        series: pd.Series,
        shape: Tuple[int, ...],
        strat_vars_part: List[str],
        strat_vars_cnt: List[str],
    ) -> NDArray:
        """Fill a (K_part, K_cnt, C, D) array for Y under full stratification.

        Parameters
        ----------
        series : pd.Series
            Multi-indexed series from a groupby on
            [part_{v}..., part_age_grp, cnt_{v}..., cnt_age_grp].
        shape : tuple
            Target array shape ``(K_part, K_cnt, C, D)``.
        strat_vars_part : list of str
            Participant stratification variable names without prefix.
        strat_vars_cnt : list of str
            Contact stratification variable names without prefix.
        """
        arr = np.zeros(shape, dtype=np.float64)
        df = series.reset_index()
        df.columns = [*df.columns[:-1], "value"]

        for var in strat_vars_part:
            df[f"{var}_part_code"] = df[f"part_{var}"].cat.codes
        for var in strat_vars_cnt:
            df[f"{var}_cnt_code"] = df[f"cnt_{var}"].cat.codes
        df["part_age_grp_code"] = df["part_age_grp"].cat.codes
        df["cnt_age_grp_code"] = df["cnt_age_grp"].cat.codes

        df["k_part"] = self._create_composite_index(
            df,
            [f"{v}_part_code" for v in strat_vars_part],
            [self.sm.strat_dims_part[v] for v in strat_vars_part],
        )
        df["k_cnt"] = self._create_composite_index(
            df,
            [f"{v}_cnt_code" for v in strat_vars_cnt],
            [self.sm.strat_dims_cnt[v] for v in strat_vars_cnt],
        )

        for _, row in df.iterrows():
            arr[
                int(row["k_part"]),
                int(row["k_cnt"]),
                int(row["part_age_grp_code"]),
                int(row["cnt_age_grp_code"]),
            ] = row["value"]

        return arr

    @staticmethod
    def _create_composite_index(
        df: pd.DataFrame, code_cols: List[str], dims: List[int]
    ) -> pd.Series:
        """
        Create composite index from multiple categorical codes.

        Similar to np.ravel_multi_index but for pandas.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with code columns
        code_cols : list
            Names of columns containing categorical codes
        dims : list
            Dimensions for each categorical variable

        Returns
        -------
        pd.Series
            Composite integer index
        """
        if not code_cols:
            return pd.Series(0, index=df.index)

        # First variable most significant: the order of the stratum labels
        # (itertools.product) and of the bootstrap.
        return pd.Series(
            ravel_stratum_codes([df[c].to_numpy() for c in code_cols], dims),
            index=df.index,
        )
