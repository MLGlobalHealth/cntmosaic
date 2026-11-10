"""
Social Contact Matrix Estimation

This module implements the socialmixr algorithm for estimating age-structured
social contact matrices from survey data. Based on Funk et al. (2024).

Key Features:
- Contact intensity and rate matrix estimation
- Optional reciprocity (symmetry) adjustment
- Adaptive merging of zero-sample age groups
- Bootstrap uncertainty quantification
- Comprehensive input validation
"""

import dataclasses
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ...analysis.summariser._summary import ContactSummary
from ...dataloader import ContactData, ParticipantData, PopulationData
from ...utils import AgeGroupSpecs
from ._base import DeterministicContactModel
from ...utils._age_bin_processor import AgeBinProcessor
from ...utils._participant_cells import bin_ages, copy_container
from ._socialmix_bootstrap import BootstrapResults, SocialMixBootstrap
from ._socialmix_helpers import apply_reciprocity, create_stratum_labels, infer_strat_mode
from ._socialmix_utils import SocialMixDataLoader
from ._socialmix_validation import SocialMixValidator

# ============================================================================
# Main Class
# ============================================================================


class SocialMix(DeterministicContactModel):
    """
    Estimate age-structured social contact matrices from survey data.

    Implements the socialmixr algorithm (Funk et al. 2024) for computing
    contact intensity and contact rate matrices from participant and contact
    survey data, with optional reciprocity adjustment and bootstrap uncertainty
    quantification.

    Parameters
    ----------
    part_data : ParticipantData
        Validated participant data container. Must have either a ``part_age``
        (1-year resolution) or ``part_age_grp`` (coarse intervals) column.
        When ``part_data.weight_col`` is set, contact counts in ``Y`` are
        weighted by individual survey weights, normalized within each
        age-group / stratum cell.
    cnt_data : ContactData
        Validated contact data container. Must have either a ``cnt_age`` or
        ``cnt_age_grp`` column, and a ``y`` column with contact counts.
    age_group_specs : AgeGroupSpecs
        Age stratification bins used to group raw ages. If participant or
        contact ages are already binned, the bins must be consistent with
        these specs.
    pop_data : PopulationData, optional
        Validated population data container. Required for contact rate
        computation and reciprocity adjustment. When provided with 1-year
        resolution (``age_col``), counts are silently aggregated into coarse
        bins matching ``age_group_specs``.
    apply_reciprocity : bool, default True
        Apply population-weighted reciprocity adjustment so that
        ``M[c,d] * P[c] == M[d,c] * P[d]``. Automatically disabled if
        ``pop_data`` is not provided or stratification mode is incompatible
        (partial/mixed).
    adaptive_merge : bool, default False
        Automatically merge age groups with zero participants to prevent
        division-by-zero errors. When ``False`` and empty groups are detected,
        a ``ValueError`` is raised instead.
    normalise_weights : bool, default False
        Only relevant when ``part_data.weight_col`` is set. When ``True``,
        survey weights are rescaled within each (age_group, stratum) cell so
        that ``sum(w_i in cell) == N_cell`` before being applied to ``Y``,
        preserving the contact intensity estimator ``M = Y / N``. When
        ``False`` (the default), the raw weights supplied by the user are
        used as-is, without rescaling to the cell sample size.

    Attributes
    ----------
    Y : NDArray
        Aggregated contact count tensor.
        Shape ``(C, D)`` for single mode; ``(K_part, C, D)`` for partial;
        ``(K_part, K_cnt, C, D)`` for full/mixed.
    N : NDArray
        Participant counts per age group and stratum.
        Shape ``(C,)`` for single mode; ``(K_part, C)`` for stratified.
    P : NDArray or None
        Population sizes per age group (``None`` if ``pop_data`` not provided).
        Shape ``(D,)`` for single/partial; ``(K_cnt, D)`` for full/mixed.
    C : int
        Number of participant age groups after any adaptive merging.
    D : int
        Number of contact age groups after any adaptive merging.
    age_group_specs : AgeGroupSpecs
        Age bins, updated in-place if adaptive merging occurred.
    strat_mode : str
        Stratification mode: ``"single"``, ``"partial"``, ``"full"``, or
        ``"mixed"``.

    Methods
    -------
    cint()
        Compute the contact intensity matrix M[c,d].
    rate()
        Compute the contact rate matrix ω[c,d] = M[c,d] / P[d].
    run_inference_bootstrap(n_boot, random_state, progress, min_success_rate)
        Quantify uncertainty via bootstrap resampling.
    predict()
        Alias for :meth:`cint`.

    Examples
    --------
    >>> from cntmosaic.dataloader import ParticipantData, ContactData, PopulationData
    >>> from cntmosaic.utils import AgeGroupSpecs
    >>>
    >>> age_bins = AgeGroupSpecs(left=[0, 20, 40, 60], right=[19, 39, 59, 79])
    >>> sm = SocialMix(part_data, cnt_data, age_bins, pop_data)
    >>>
    >>> # Point estimates
    >>> cint_dict = sm.cint()          # Dict[str, ContactSummary]
    >>> M = cint_dict["All->All"].central
    >>>
    >>> rate_dict = sm.rate()
    >>> omega = rate_dict["All->All"].central
    >>>
    >>> # Bootstrap uncertainty
    >>> boot = sm.run_inference_bootstrap(n_boot=1000, random_state=42)
    >>> M_mean = boot.mean(statistic="cint")["All->All"]
    >>> M_ci = boot.quantiles(q=[0.025, 0.975], statistic="cint")["All->All"]

    Notes
    -----
    Contact intensity ``M[c,d]`` is the average number of contacts that an
    individual in age group ``c`` has with individuals in age group ``d``
    during the survey period.

    Contact rate ``ω[c,d] = M[c,d] / P[d]`` is the per-capita contact rate
    from age group ``c`` to age group ``d``.

    The reciprocity adjustment ensures ``M[c,d] * P[c] == M[d,c] * P[d]``,
    i.e. total contacts are symmetric across the population.

    If ``part_data`` carries a ``weight_col``, each contact record is
    multiplied by the participant's survey weight before aggregation, so that
    ``Y[c,d] = sum_i( w_i * y_i_d )``. When ``normalise_weights=True``, weights
    are first rescaled by
    :meth:`~cntmosaic.dataloader.ParticipantData.normalize_weights` so that
    ``sum_i(w_i in cell c) == N[c]``; when ``False`` (the default), the raw
    weights are used unmodified.
    """

    def __init__(
        self,
        part_data: ParticipantData,
        cnt_data: ContactData,
        age_group_specs: AgeGroupSpecs,
        pop_data: Optional[PopulationData] = None,
        apply_reciprocity: bool = True,
        adaptive_merge: bool = False,
        normalise_weights: bool = False,
    ):
        # Store parameters
        self.part_data = part_data
        self.cnt_data = cnt_data
        self.age_group_specs = age_group_specs
        self.pop_data = pop_data
        self.apply_reciprocity = apply_reciprocity
        self.adaptive_merge = adaptive_merge
        self.normalise_weights = normalise_weights

        # Stratification attributes (initialized in _preprocess)
        self.strat_vars_part: List[str] = []
        self.strat_vars_cnt: List[str] = []
        self.strat_vars_pop: List[str] = []
        self.strat_vars_shared: List[str] = []
        self.strat_vars_part_only: List[str] = []
        self.strat_vars_cnt_only: List[str] = []
        self.strat_mode: str = None
        self.strat_dims_part: Dict[str, int] = {}
        self.strat_dims_cnt: Dict[str, int] = {}
        self.K: int = 1  # Total number of strata

        # Initialize helper classes
        self.age_processor = AgeBinProcessor(age_group_specs)

        # Computed attributes (initialized in pipeline)
        self._cint: Optional[NDArray] = None
        self._rate: Optional[NDArray] = None
        self._boot: Optional[BootstrapResults] = None
        self._cells = None  # ParticipantCells, set in _preprocess

        # Data arrays (initialized in _load)
        self.Y: Optional[NDArray] = None  # Contact counts
        self.N: Optional[NDArray] = None  # Participant counts
        self.P: Optional[NDArray] = None  # Population sizes
        self.C: int = 0  # Number of participant age groups
        self.D: int = 0  # Number of contact age groups
        self.K_part: int = 1  # Number of participant strata
        self.K_cnt: int = 1  # Number of contact strata

        # Run processing pipeline
        self.fit()

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframes(
        cls,
        df_part: pd.DataFrame,
        df_cnt: pd.DataFrame,
        age_group_specs: AgeGroupSpecs,
        df_pop: Optional[pd.DataFrame] = None,
        apply_reciprocity: bool = True,
        adaptive_merge: bool = False,
        normalise_weights: bool = False,
    ) -> "SocialMix":
        """Construct SocialMix directly from DataFrames, bypassing manual container creation.

        Use this constructor when you have raw DataFrames rather than pre-built
        :class:`ParticipantData` / :class:`ContactData` / :class:`PopulationData`
        containers. The constructor auto-detects age and stratification columns from
        the ``part_*`` / ``cnt_*`` / ``pop_*`` column naming convention and wraps
        the DataFrames in the appropriate container objects internally.

        Parameters
        ----------
        df_part : pd.DataFrame
            Participant DataFrame. Must contain ``id`` and either ``part_age``
            (1-year resolution) or ``part_age_grp`` (coarse intervals).
            Extra ``part_*`` columns are treated as stratification variables.
        df_cnt : pd.DataFrame
            Contact DataFrame. Must contain ``id``, ``y``, and either ``cnt_age``
            or ``cnt_age_grp``. Extra ``cnt_*`` columns are treated as
            stratification variables.
        age_group_specs : AgeGroupSpecs
            Age stratification bins.
        df_pop : pd.DataFrame, optional
            Population DataFrame. Must contain ``P`` and either ``age``
            (1-year resolution) or ``pop_age_grp`` (coarse intervals).
        apply_reciprocity : bool, default True
            See :class:`SocialMix`.
        adaptive_merge : bool, default False
            See :class:`SocialMix`.
        normalise_weights : bool, default False
            See :class:`SocialMix`.
        """
        _CORE_PART = {"id", "part_age", "part_age_grp", "part_age_min", "part_age_max"}
        _CORE_CNT = {"id", "cnt_age", "cnt_age_grp", "cnt_age_min", "cnt_age_max", "y"}
        _CORE_POP = {"age", "age_min", "age_max", "P", "pop_age_grp"}

        part_strat = [
            c for c in df_part.columns if c.startswith("part_") and c not in _CORE_PART
        ]
        cnt_strat = [
            c for c in df_cnt.columns if c.startswith("cnt_") and c not in _CORE_CNT
        ]

        if "part_age" in df_part.columns:
            part_age_kwargs: dict = {"age_col": "part_age"}
        elif "part_age_grp" in df_part.columns:
            part_age_kwargs = {"age_grp_col": "part_age_grp"}
        else:
            raise ValueError("df_part must contain 'part_age' or 'part_age_grp' column.")

        if "cnt_age" in df_cnt.columns:
            cnt_age_kwargs: dict = {"age_col": "cnt_age"}
        elif "cnt_age_grp" in df_cnt.columns:
            cnt_age_kwargs = {"age_grp_col": "cnt_age_grp"}
        else:
            raise ValueError("df_cnt must contain 'cnt_age' or 'cnt_age_grp' column.")

        part_data = ParticipantData(
            df_part,
            id_col="id",
            strat_var_cols=part_strat or None,
            **part_age_kwargs,
        )
        cnt_data = ContactData(
            df_cnt,
            id_col="id",
            strat_var_cols=cnt_strat or None,
            **cnt_age_kwargs,
        )

        pop_data = None
        if df_pop is not None:
            if "pop_age_grp" in df_pop.columns:
                pop_age_kwargs: dict = {"age_grp_col": "pop_age_grp"}
            else:
                pop_age_kwargs = {"age_col": "age"}
            pop_strat = [c for c in df_pop.columns if c not in _CORE_POP]
            pop_data = PopulationData(
                df_pop,
                size_col="P",
                strat_var_cols=pop_strat or None,
                **pop_age_kwargs,
            )

        return cls(
            part_data=part_data,
            cnt_data=cnt_data,
            age_group_specs=age_group_specs,
            pop_data=pop_data,
            apply_reciprocity=apply_reciprocity,
            adaptive_merge=adaptive_merge,
            normalise_weights=normalise_weights,
        )

    # ------------------------------------------------------------------
    # DeterministicContactModel interface
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """
        Prepare and load data for matrix estimation.

        Runs the full preprocessing and loading pipeline:
        1. Assigns age groups to participants / contacts
        2. Validates stratification, reciprocity requirements, and age bins
        3. Aggregates contact counts into Y, N, P arrays
        """
        self._preprocess()
        self._load()

    def predict(self) -> Dict[str, NDArray]:
        """
        Compute and return the contact intensity matrix.

        This is a convenience wrapper around :meth:`cint`.

        Returns
        -------
        Dict[str, NDArray]
            Dictionary mapping stratum labels to contact intensity matrices.
        """
        return self.cint()

    # ------------------------------------------------------------------
    # Internal preprocessing / loading
    # ------------------------------------------------------------------

    def _preprocess(self) -> None:
        """
        Extract and validate stratification information from data containers.

        This method:
        1. Extracts stratification variables from both containers
        2. Identifies shared, participant-only, and contact-only variables
        3. Computes stratification dimensions and expected number of strata
        4. Assigns age groups to raw ages if needed

        Notes
        -----
        Stratification variables do NOT need to match between participants and contacts.
        Four modes are supported:
        - No stratification: Both empty → K=1
        - Partial: Only participant vars → K=product of participant categories
        - Full: Same vars on both sides → K=product of squares
        - Mixed: Some overlap → K=complex product
        """
        # Work on copies so that the user's containers are never modified
        # (dev plan Problem 11). Copies keep every column (no re-preprocessing).
        self.part_data = copy_container(self.part_data)
        self.cnt_data = copy_container(self.cnt_data)
        if self.pop_data is not None:
            self.pop_data = copy_container(self.pop_data)

        # Assign age groups first (needed for validation)
        self._assign_age_groups()

        # Use validator to handle validation logic
        # Bootstrap stability validation runs lazily inside run_inference_bootstrap()
        validator = SocialMixValidator(
            self.part_data,
            self.cnt_data,
            self.age_group_specs,
            self.pop_data,
            self.apply_reciprocity,
            self.adaptive_merge,
            validate_for_bootstrap=False,
        )

        # Run all validations and get updated components
        validated = validator.validate_all()

        # Update instance with validated components (age_group_specs may be a new object
        # if adaptive merging occurred — keep age_processor in sync)
        self.part_data = validated["part_data"]
        self.cnt_data = validated["cnt_data"]
        self.age_group_specs = validated["age_group_specs"]
        self.age_processor = AgeBinProcessor(self.age_group_specs)
        self.apply_reciprocity = validated["apply_reciprocity"]
        # Participant cells (age group x participant strata), shared with
        # WeightSpecs.preview() via build_participant_cells.
        self._cells = validated["cells"]

        # Extract stratification variables from validated data
        self.strat_vars_part = self.part_data.get_strat_vars(prefix=False)
        self.strat_vars_cnt = self.cnt_data.get_strat_vars(prefix=False)
        self.strat_vars_pop = (
            self.pop_data.get_strat_vars(prefix=False)
            if self.pop_data is not None
            else []
        )

        # Identify shared and unique variables
        self.strat_vars_shared = sorted(
            list(set(self.strat_vars_part) & set(self.strat_vars_cnt))
        )
        self.strat_vars_part_only = sorted(
            list(set(self.strat_vars_part) - set(self.strat_vars_cnt))
        )
        self.strat_vars_cnt_only = sorted(
            list(set(self.strat_vars_cnt) - set(self.strat_vars_part))
        )

        # Stratification dimensions = number of categories (not nunique), so
        # they agree with the stratum labels, which enumerate every category.
        self.strat_dims_part = {
            var: len(self.part_data.data[f"part_{var}"].cat.categories)
            for var in self.strat_vars_part
        }
        self.strat_dims_cnt = {
            var: len(self.cnt_data.data[f"cnt_{var}"].cat.categories)
            for var in self.strat_vars_cnt
        }

        # Calculate expected number of strata
        self.K = self._calculate_K()

    def _infer_strat_mode(self) -> str:
        """Infer and cache the stratification mode."""
        self.strat_mode = infer_strat_mode(self.strat_vars_part, self.strat_vars_cnt)
        return self.strat_mode

    def _calculate_K(self) -> int:
        """
        Calculate the number of strata based on stratification mode.

        Returns
        -------
        int
            Expected number of unique strata:
            - Case 1 (no stratification): 1
            - Case 2 (partial): product of participant categories
            - Case 3 (mixed): product of (part-only x shared^2 x cnt-only)
            - Case 4 (full): product of squares of categories
        """
        if self._infer_strat_mode() == "single":
            # Case 1: No stratification
            return 1

        K = 1

        # Participant-only variables (partial mode)
        for var in self.strat_vars_part_only:
            K *= self.strat_dims_part[var]

        # Contact-only variables
        for var in self.strat_vars_cnt_only:
            K *= self.strat_dims_cnt[var]

        # Shared variables (full mode for these variables)
        for var in self.strat_vars_shared:
            # For shared vars, we get all combinations: n_categories × n_categories
            K *= self.strat_dims_part[var] ** 2

        return int(K)

    def _assign_age_groups(self) -> None:
        """
        Assign age groups to participants and contacts if not already provided.

        Uses the age_bins provided during initialization to categorize
        raw ages into age groups. Operates on the working copies made in
        :meth:`_preprocess`. Single-year ages are always re-binned, so a stale
        age-group column can never be reused (dev plan Problem 11); pre-binned
        age groups are kept as given.
        """
        # Participants
        if "part_age" in self.part_data.data.columns:
            self.part_data.data["part_age_grp"] = bin_ages(
                self.part_data.data["part_age"], self.age_group_specs
            )
        elif "part_age_grp" not in self.part_data.data.columns:
            raise ValueError(
                "ParticipantData must have either 'part_age' or 'part_age_grp' column."
            )

        # Contacts
        if "cnt_age" in self.cnt_data.data.columns:
            self.cnt_data.data["cnt_age_grp"] = bin_ages(
                self.cnt_data.data["cnt_age"], self.age_group_specs
            )
        elif "cnt_age_grp" not in self.cnt_data.data.columns:
            raise ValueError(
                "ContactData must have either 'cnt_age' or 'cnt_age_grp' column."
            )

    def _load(self) -> None:
        """
        Prepare contact count matrices for estimation.

        Delegates to SocialMixDataProcessor for loading stratified data arrays.

        Computes stratified contact count arrays based on stratification mode:

        - Single mode: Y (C, D), N (C,), P (D,)
        - Partial mode: Y (K_part, C, D), N (K_part, C), P (D,)
        - Full/Mixed mode: Y (K_part, K_cnt, C, D), N (K_part, C), P (K_cnt, D)

        Where:
            - C: number of participant age groups
            - D: number of contact age groups
            - K_part: number of participant strata
            - K_cnt: number of contact strata

        Sets
        ----
        self.Y : NDArray
            Contact count matrix/tensor
        self.N : NDArray
            Participant count matrix/vector
        self.P : NDArray or None
            Population size matrix/vector (if pop_data available)
        self.C : int
            Number of participant age groups
        self.D : int
            Number of contact age groups
        self.K_part : int
            Number of participant strata
        self.K_cnt : int
            Number of contact strata
        """
        # Delegate to data processor
        loader = SocialMixDataLoader(self)
        loader.load_data()

    def _create_stratum_labels(self) -> List[str]:
        """Create ``"source->target"`` stratum labels matching array index order."""
        return create_stratum_labels(
            self.K_part,
            self.K_cnt,
            self.strat_vars_part,
            self.strat_vars_cnt,
            self.part_data,
            self.cnt_data,
        )


    def cint(self) -> Dict[str, "ContactSummary"]:
        """
        Compute contact intensity matrix.

        M[c,d] represents the average number of contacts that individuals
        in age group c have with individuals in age group d.

        If apply_reciprocity=True, applies post-hoc reciprocity adjustment
        to ensure contact symmetry (for single/full stratification modes).

        Returns
        -------
        Dict[str, ContactSummary]
            Dictionary mapping stratum labels to ContactSummary objects.
            Keys follow "source->target" format:
            - "All->All" for unstratified data
            - "M->All", "F->All" for participant-only stratification
            - "M->M", "M->F", "F->M", "F->F" for full stratification
        """
        if self._cint is None:
            # Get stratum labels
            labels = self._create_stratum_labels()

            # Compute raw intensity matrices based on data structure
            raw = {}

            if self.K_part == 1 and self.K_cnt == 1:
                # Single stratum: Y is (C, D), N is (C,)
                raw["All->All"] = self.Y / self.N[:, np.newaxis]

            elif self.K_part > 1 and self.K_cnt == 1:
                # Partial stratification: Y is (K_part, C, D), N is (K_part, C)
                for k in range(self.K_part):
                    raw[labels[k]] = self.Y[k] / self.N[k, :, np.newaxis]

            else:
                # Full/mixed stratification: Y is (K_part, K_cnt, C, D), N is (K_part, C)
                idx = 0
                for k_part in range(self.K_part):
                    for k_cnt in range(self.K_cnt):
                        raw[labels[idx]] = (
                            self.Y[k_part, k_cnt] / self.N[k_part, :, np.newaxis]
                        )
                        idx += 1

            # Apply reciprocity adjustment if requested
            if self.apply_reciprocity and self.P is not None:
                raw = apply_reciprocity(raw, self.strat_mode, self.K_cnt, self.P)

            _nan = np.full(next(iter(raw.values())).shape, np.nan)
            self._cint = {
                key: ContactSummary(
                    lower=_nan.copy(),
                    central=arr,
                    upper=_nan.copy(),
                    alpha=np.nan,
                    measure="mean",
                    age_group_specs=self.age_group_specs,
                )
                for key, arr in raw.items()
            }

        return self._cint

    def rate(self) -> Dict[str, "ContactSummary"]:
        """
        Compute contact rate matrix.

        ω[c,d] represents the per-capita rate at which individuals in
        age group c contact individuals in age group d.

        The rate is computed from the contact intensity matrix by dividing
        by the population size of the target age group:
            ω[c,d] = M[c,d] / P[d]

        Returns
        -------
        Dict[str, ContactSummary]
            Dictionary of contact rate ContactSummary objects, one per stratum.
            Keys follow the format "source->target" (e.g., "All->All", "M->F").

        Raises
        ------
        ValueError
            If population data was not provided during initialization.
        """
        if self._rate is None:
            # Check that population data is available
            if self.P is None:
                raise ValueError(
                    "Cannot compute contact rates without population data. "
                    "Please provide 'pop_data' when initializing SocialMix."
                )

            # Get contact intensity matrices (central values)
            cint_dict = self.cint()

            # Compute rate for each stratum: rate = cint / P
            raw = {}
            labels = self._create_stratum_labels()
            for key, cs in cint_dict.items():
                M = cs.central
                if self.K_cnt == 1:
                    raw[key] = M / self.P[np.newaxis, :]
                else:
                    stratum_idx = labels.index(key)
                    k_cnt = stratum_idx % self.K_cnt
                    raw[key] = M / self.P[k_cnt, :][np.newaxis, :]

            _nan = np.full(next(iter(raw.values())).shape, np.nan)
            self._rate = {
                key: ContactSummary(
                    lower=_nan.copy(),
                    central=arr,
                    upper=_nan.copy(),
                    alpha=np.nan,
                    measure="mean",
                    age_group_specs=self.age_group_specs,
                )
                for key, arr in raw.items()
            }

        return self._rate

    def run_inference_bootstrap(
        self,
        n_boot: int = 1000,
        random_state: Optional[int] = None,
        progress: bool = True,
        min_success_rate: float = 0.5,
    ) -> BootstrapResults:
        """
        Estimate uncertainty via bootstrap resampling.

        Runs bootstrap stability validation before resampling — age groups that are
        too small for reliable resampling are merged automatically when
        ``adaptive_merge=True``, or a ``ValueError`` is raised otherwise. If merging
        occurs at this stage, cached point estimates (``cint``, ``rate``) are
        invalidated and recomputed at the merged resolution.

        Parameters
        ----------
        n_boot : int, default 1000
            Number of bootstrap resamples.
        random_state : int, optional
            Random seed for reproducibility.
        progress : bool, default True
            Show a tqdm progress bar during resampling.
        min_success_rate : float, default 0.5
            Minimum fraction of iterations that must succeed. Raises ``ValueError``
            if the actual success rate falls below this threshold.

        Returns
        -------
        BootstrapResults
            Container with per-iteration samples and summary methods:

            - ``mean(statistic='cint'|'rate')`` — mean across samples
            - ``std(statistic='cint'|'rate')`` — standard deviation
            - ``quantiles(q, statistic='cint'|'rate')`` — arbitrary quantiles

            Results are also stored in ``self._boot`` for later access.

        Raises
        ------
        ValueError
            If empty age groups remain after adaptive merging, or if
            ``success_rate < min_success_rate``.

        Examples
        --------
        >>> sm = SocialMix(part_data, cnt_data, age_bins, pop_data)
        >>> boot = sm.run_inference_bootstrap(n_boot=1000, random_state=42)
        >>> cint_mean = boot.mean(statistic='cint')
        >>> cint_ci = boot.quantiles(q=[0.025, 0.975], statistic='cint')
        """
        # Run bootstrap stability validation lazily.
        # This may adaptively merge more age groups than estimation required.
        boot_validator = SocialMixValidator(
            self.part_data,
            self.cnt_data,
            self.age_group_specs,
            self.pop_data,
            self.apply_reciprocity,
            self.adaptive_merge,
            validate_for_bootstrap=True,
        )
        validated = boot_validator.validate_all()

        # If bootstrap validation merged additional age groups, update instance state
        # and invalidate cached point estimates so they stay consistent.
        if validated["age_group_specs"] is not self.age_group_specs:
            self.part_data = validated["part_data"]
            self.cnt_data = validated["cnt_data"]
            self.age_group_specs = validated["age_group_specs"]
            self.age_processor = AgeBinProcessor(self.age_group_specs)
            self.apply_reciprocity = validated["apply_reciprocity"]
            # Participants were already dropped at fit time; keep that count.
            self._cells = dataclasses.replace(
                validated["cells"], n_dropped=self._cells.n_dropped
            )
            self._cint = None
            self._rate = None
            # Drop stale age_grp column so _assign_age_groups_to_population()
            # recomputes it with the new (post-merge) intervals.
            if self.pop_data is not None and "age_grp" in self.pop_data.data.columns:
                self.pop_data.data.drop(columns=["age_grp"], inplace=True)
            self._load()

        # Create bootstrap estimator and run
        bootstrap = SocialMixBootstrap(
            part_data=self.part_data,
            cnt_data=self.cnt_data,
            age_group_specs=self.age_group_specs,
            pop_data=self.pop_data,
            apply_reciprocity=self.apply_reciprocity,
            n_boot=n_boot,
            random_state=random_state,
        )

        self._boot = bootstrap.run(
            progress=progress,
            min_success_rate=min_success_rate,
        )
        return self._boot
