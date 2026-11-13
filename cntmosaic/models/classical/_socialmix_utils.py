"""
Data Loading Utilities for SocialMix

This module provides helper classes for loading and processing stratified contact data,
and the proportionate-mixing factorisation of a contact matrix
(:class:`PropMixFactorisation`, :func:`prop_mix_factorise`).
"""

import warnings
from dataclasses import dataclass
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
        """Compute the (unweighted) contact count array and store it as ``Y``."""
        self.sm.Y = self.compute_Y_array()

    def compute_Y_array(self, part_weights: Optional[pd.Series] = None) -> NDArray:
        """
        Compute the contact count array with stratification.

        Parameters
        ----------
        part_weights : pd.Series, optional
            Weight per participant, indexed by participant id. When given,
            each contact counts ``w_i`` instead of 1, so the result holds
            weighted sums ``sum_i w_i y_id``. The loader holds no weight state
            itself; callers pass weights explicitly (``Prem2`` shares this
            loader and never does).

        Returns
        -------
        NDArray
            - Shape (C, D) if K_part == 1 and K_cnt == 1 (no stratification)
            - Shape (K_part, C, D) if K_part > 1 and K_cnt == 1 (partial)
            - Shape (K_part, K_cnt, C, D) if both K_part > 1 and K_cnt > 1 (full/mixed)
        """
        df_part = self.sm.part_data.data
        df_cnt = self.sm.cnt_data.data
        age_grps_part = df_part["part_age_grp"].cat.categories
        age_grps_cnt = df_cnt["cnt_age_grp"].cat.categories

        # Merge participants with contacts
        part_cols = ["id", "part_age_grp"] + [
            f"part_{v}" for v in self.sm.strat_vars_part
        ]
        merged = df_cnt.merge(df_part[part_cols], on="id", how="inner")

        if len(merged) == 0:
            raise ValueError(
                "No valid contacts after merging with participants. "
                "Check that contact IDs match participant IDs."
            )

        if part_weights is None:
            merged["_y"] = merged["y"]
        else:
            w = merged["id"].map(part_weights)
            if w.isna().any():
                raise ValueError("Missing participant weights for some contacts.")
            merged["_y"] = merged["y"] * w.to_numpy(dtype=np.float64)

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
            return Y_df.values.astype(np.float64)

        elif self.sm.K_part > 1 and self.sm.K_cnt == 1:
            # Partial stratification: (K_part, C, D) tensor
            group_cols = [f"part_{v}" for v in self.sm.strat_vars_part] + [
                "part_age_grp",
                "cnt_age_grp",
            ]
            Y_agg = merged.groupby(group_cols, observed=False)["_y"].sum()

            return self._fill_partial_array(
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

            return self._fill_full_array(
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


# ============================================================================
# Proportionate-mixing factorisation
# ============================================================================


@dataclass(frozen=True, eq=False)
class PropMixFactorisation:
    """Proportionate-mixing factorisation of a contact intensity matrix,
    ``m[i, j] = c * q * d[i] * a[i, j] * p[j]``.

    Returned by :meth:`SocialMix.prop_mix_factorisation`; see there for the
    definitions and the mapping to socialmixr's ``split_matrix()``.

    Attributes
    ----------
    matrix : np.ndarray, shape (C, C)
        The factorised contact intensity matrix ``m``.
    assortativity : np.ndarray, shape (C, C)
        ``a[i, j] = m[i, j] / (contacts[i] * population_share[j])``. Each row
        satisfies ``sum_j a[i, j] * population_share[j] = 1``.
    contacts : np.ndarray, shape (C,)
        Mean number of contacts of a member of each age group (row sums of ``m``).
    population_share : np.ndarray, shape (C,)
        Share of the population in each age group, ``p[j]``.
    mean_contacts : float
        Population-weighted mean number of contacts, ``c``.
    spectral_radius : float
        Largest eigenvalue of ``m``, ``c * q``.
    normalisation : float
        ``q = spectral_radius / mean_contacts``.
    age_group_specs : AgeGroupSpecs
        Age groups of the rows and columns.
    weighted : bool
        Whether the factorised matrix is the weighted estimate.
    """

    matrix: NDArray
    assortativity: NDArray
    contacts: NDArray
    population_share: NDArray
    mean_contacts: float
    spectral_radius: float
    normalisation: float
    age_group_specs: AgeGroupSpecs
    weighted: bool

    @property
    def relative_contacts(self) -> NDArray:
        """``d[i] = contacts[i] / spectral_radius`` (socialmixr's ``$contacts``)."""
        return self.contacts / self.spectral_radius

    def reconstruct(self) -> NDArray:
        """``c * q * d[i] * a[i, j] * p[j]``; equals ``matrix`` up to rounding."""
        return (
            self.mean_contacts
            * self.normalisation
            * self.relative_contacts[:, np.newaxis]
            * self.assortativity
            * self.population_share[np.newaxis, :]
        )

    def __repr__(self) -> str:
        return (
            f"PropMixFactorisation({len(self.contacts)} age groups, "
            f"mean_contacts={self.mean_contacts:.3g}, "
            f"spectral_radius={self.spectral_radius:.3g}, "
            f"normalisation={self.normalisation:.3g}, weighted={self.weighted})"
        )


def prop_mix_factorise(
    matrix: NDArray,
    population: NDArray,
    age_group_specs: AgeGroupSpecs,
    weighted: bool = False,
) -> PropMixFactorisation:
    """Proportionate-mixing factorisation of a square contact intensity matrix.

    Computes ``m[i, j] = c * q * d[i] * a[i, j] * p[j]`` as socialmixr's
    ``split_mean_norm_contacts()``: ``r = m.sum(1)``, ``p = P / P.sum()``,
    ``c = sum(P * r) / sum(P)``, ``rho`` the largest eigenvalue of ``m``,
    ``q = rho / c`` and ``a[i, j] = m[i, j] / (r[i] * p[j])``.

    Parameters
    ----------
    matrix : np.ndarray, shape (C, C)
        Contact intensity matrix (rows: participant age groups; columns:
        contact age groups; the same age groups on both sides).
    population : np.ndarray, shape (C,)
        Population size of each age group.
    age_group_specs : AgeGroupSpecs
        Age groups of the rows and columns.
    weighted : bool, default False
        Recorded on the result.

    Returns
    -------
    PropMixFactorisation

    Warns
    -----
    UserWarning
        If an age group has no contacts; its row of ``assortativity`` is NaN.
    """
    m = np.asarray(matrix, dtype=np.float64)
    P = np.asarray(population, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or P.shape != (m.shape[0],):
        raise ValueError(
            f"prop_mix_factorise needs a square matrix and one population size per "
            f"age group; got matrix {m.shape} and population {P.shape}."
        )

    r = m.sum(axis=1)
    p = P / P.sum()
    c = float(np.sum(P * r) / P.sum())
    eigenvalues = np.linalg.eigvals(np.nan_to_num(m))
    rho = float(np.real(eigenvalues[np.argmax(np.abs(eigenvalues))]))

    no_contacts = r == 0
    if no_contacts.any():
        groups = [
            f"[{lo}, {hi + 1})"
            for lo, hi, z in zip(age_group_specs.left, age_group_specs.right, no_contacts)
            if z
        ]
        warnings.warn(
            f"Age group(s) {groups} report no contacts; their rows of the "
            f"assortativity matrix are undefined (NaN).",
            UserWarning,
            stacklevel=2,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        a = m / (r[:, np.newaxis] * p[np.newaxis, :])
    a[no_contacts, :] = np.nan

    return PropMixFactorisation(
        matrix=m,
        assortativity=a,
        contacts=r,
        population_share=p,
        mean_contacts=c,
        spectral_radius=rho,
        normalisation=rho / c,
        age_group_specs=age_group_specs,
        weighted=bool(weighted),
    )
