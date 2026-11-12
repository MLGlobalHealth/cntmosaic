"""
Social Contact Matrix Estimation

This module estimates age-structured social contact matrices from survey data.

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
from ...weights import BindContext, WeightResult, WeightSpecs
from ...weights._warnings import warn as warn_weights

# ============================================================================
# Main Class
# ============================================================================


class SocialMix(DeterministicContactModel):
    """
    Estimate age-structured social contact matrices from survey data.

    Computes contact intensity and contact rate matrices from participant and
    contact survey data, with optional reciprocity adjustment and bootstrap
    uncertainty quantification.

    Parameters
    ----------
    part_data : ParticipantData
        Validated participant data container. Must have either a ``part_age``
        (1-year resolution) or ``part_age_grp`` (coarse intervals) column.
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
    apply_reciprocity : bool, default False
        Apply population-weighted reciprocity adjustment so that
        ``M[c,d] * P[c] == M[d,c] * P[d]``. Off by default: whether the
        adjustment is appropriate depends on the analysis. Automatically
        disabled if ``pop_data`` is not provided or stratification mode is
        incompatible (partial/mixed).
    adaptive_merge : bool, default False
        Automatically merge age groups with zero participants to prevent
        division-by-zero errors. When ``False`` and empty groups are detected,
        a ``ValueError`` is raised instead. This flag applies only to empty
        age groups. :meth:`run_inference_bootstrap` also merges age groups
        that are too small for stable resampling, whatever the value of this
        flag.
    weight_specs : WeightSpecs, optional
        Participant weights (see :mod:`cntmosaic.weights`). Weighting is
        opt-in: without ``weight_specs`` the estimate is unweighted. With it,
        the weights are computed on the fitted participants and cells, and
        :meth:`cint` / :meth:`rate` return the weighted estimate
        ``M[c,d] = sum_i w_i y_id / sum_i w_i`` by default; the unweighted
        estimate stays available through ``weighted=False``.

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
    Y_w, W : NDArray or None
        Weighted counterparts of ``Y`` and ``N`` (``sum_i w_i y_id`` and
        ``sum_i w_i``), or ``None`` without ``weight_specs``.
    weights : WeightResult or None
        Weight diagnostics, or ``None`` without ``weight_specs``.

    Methods
    -------
    cint(weighted=None)
        Compute the contact intensity matrix M[c,d].
    rate(weighted=None)
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
    >>>
    >>> # Participant weights (opt-in; see cntmosaic.weights)
    >>> from cntmosaic.weights import WeightSpecs, DayOfWeekWeights
    >>> part_data = ParticipantData(df_part, id_col="part_id", age_col="part_age",
    ...                             aux_cols="dayofweek")
    >>> sm_w = SocialMix(part_data, cnt_data, age_bins, pop_data,
    ...                  weight_specs=WeightSpecs(DayOfWeekWeights()))
    >>> sm_w.cint()                  # weighted
    >>> sm_w.cint(weighted=False)    # unweighted, same fit
    >>> sm_w.weights.cells           # weight diagnostics per cell

    Notes
    -----
    Contact intensity ``M[c,d]`` is the average number of contacts that an
    individual in age group ``c`` has with individuals in age group ``d``
    during the survey period.

    Contact rate ``ω[c,d] = M[c,d] / P[d]`` is the per-capita contact rate
    from age group ``c`` to age group ``d``.

    The reciprocity adjustment ensures ``M[c,d] * P[c] == M[d,c] * P[d]``,
    i.e. total contacts are symmetric across the population.
    """

    def __init__(
        self,
        part_data: ParticipantData,
        cnt_data: ContactData,
        age_group_specs: AgeGroupSpecs,
        pop_data: Optional[PopulationData] = None,
        apply_reciprocity: bool = False,
        adaptive_merge: bool = False,
        weight_specs: Optional[WeightSpecs] = None,
    ):
        if weight_specs is not None and not isinstance(weight_specs, WeightSpecs):
            raise TypeError(
                f"weight_specs must be a WeightSpecs or None; got {type(weight_specs).__name__}."
            )

        # Store parameters
        self.part_data = part_data
        self.cnt_data = cnt_data
        self.age_group_specs = age_group_specs
        self.pop_data = pop_data
        self.apply_reciprocity = apply_reciprocity
        self.adaptive_merge = adaptive_merge
        self.weight_specs = weight_specs

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
        # cint / rate caches, one entry per value of ``weighted``
        self._cint: Dict[bool, Dict[str, ContactSummary]] = {}
        self._rate: Dict[bool, Dict[str, ContactSummary]] = {}
        self._boot: Optional[BootstrapResults] = None
        self._cells = None  # ParticipantCells, set in _preprocess

        # Weights (set in _fit_weights when weight_specs is given)
        self._bound = None  # private _BoundWeightSpecs
        self._weight_result: Optional[WeightResult] = None
        self.Y_w: Optional[NDArray] = None  # sum_i w_i y_id
        self.W: Optional[NDArray] = None  # sum_i w_i per cell

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
        apply_reciprocity: bool = False,
        adaptive_merge: bool = False,
        weight_specs: Optional[WeightSpecs] = None,
        aux_cols: Optional[Union[List[str], str]] = None,
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
            Extra ``part_*`` columns are treated as stratification variables,
            except those listed in ``aux_cols``.
        df_cnt : pd.DataFrame
            Contact DataFrame. Must contain ``id``, ``y``, and either ``cnt_age``
            or ``cnt_age_grp``. Extra ``cnt_*`` columns are treated as
            stratification variables.
        age_group_specs : AgeGroupSpecs
            Age stratification bins.
        df_pop : pd.DataFrame, optional
            Population DataFrame. Must contain ``P`` and either ``age``
            (1-year resolution) or ``pop_age_grp`` (coarse intervals).
        apply_reciprocity : bool, default False
            See :class:`SocialMix`.
        adaptive_merge : bool, default False
            See :class:`SocialMix`.
        weight_specs : WeightSpecs, optional
            See :class:`SocialMix`.
        aux_cols : str or list of str, optional
            Participant columns kept for weighting (e.g. ``"part_dayofweek"``
            or a design-weight column). They are passed to
            ``ParticipantData(aux_cols=...)`` and are **not** treated as
            stratification variables.
        """
        # Resolve each auxiliary name to the column present in df_part (it may
        # be given with or without the "part_" prefix).
        aux_list = []
        for c in [aux_cols] if isinstance(aux_cols, str) else list(aux_cols or []):
            prefixed = c if c.startswith("part_") else f"part_{c}"
            if c not in df_part.columns and prefixed in df_part.columns:
                c = prefixed
            aux_list.append(c)
        aux_names = set(aux_list)
        _CORE_PART = {"id", "part_age", "part_age_grp", "part_age_min", "part_age_max"}
        _CORE_CNT = {"id", "cnt_age", "cnt_age_grp", "cnt_age_min", "cnt_age_max", "y"}
        _CORE_POP = {"age", "age_min", "age_max", "P", "pop_age_grp"}

        part_strat = [
            c
            for c in df_part.columns
            if c.startswith("part_") and c not in _CORE_PART and c not in aux_names
        ]
        float_strat = [c for c in part_strat if pd.api.types.is_float_dtype(df_part[c])]
        if float_strat:
            warnings.warn(
                f"Participant column(s) {float_strat} have a floating-point dtype and "
                f"will be used as stratification variables. If they are weighting "
                f"variables (e.g. design weights), pass them in aux_cols instead.",
                UserWarning,
                stacklevel=2,
            )
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
            aux_cols=aux_list or None,
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
            weight_specs=weight_specs,
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
        4. If ``weight_specs`` is given, computes the weights and the weighted
           arrays Y_w, W
        """
        self._preprocess()
        self._load()
        self._fit_weights()

    def _fit_weights(self, warn_user: bool = True) -> None:
        """Bind ``weight_specs`` to the fitted cells and compute Y_w and W.

        Lives in SocialMix, not in the SocialMixDataLoader shared with Prem2,
        which gets no weighting (dev plan D27). ``warn_user=False`` is used
        when rebinding after a bootstrap-time merge, so bind-time warnings
        are not repeated.
        """
        self._bound = None
        self._weight_result = None
        self.Y_w = None
        self.W = None
        self._cint = {}
        self._rate = {}
        if self.weight_specs is None:
            return

        cells = self._cells
        ctx = BindContext(
            part=cells.part.copy(),
            cell=cells.cell,
            n_cells=cells.n_cells,
            cell_labels=cells.cell_labels,
            ids=cells.ids,
        )
        self._bound = self.weight_specs._bind(ctx, warn_user=warn_user)
        w = self._bound.weights()

        loader = SocialMixDataLoader(self)
        self.Y_w = loader.compute_Y_array(part_weights=pd.Series(w, index=cells.ids))
        W = np.bincount(cells.cell, weights=w, minlength=cells.n_cells).reshape(
            self.K_part, self.C
        )
        self.W = W[0] if self.N.ndim == 1 else W
        if self.W.shape != self.N.shape or self.Y_w.shape != self.Y.shape:
            raise RuntimeError(
                "Internal error: weighted arrays do not match the unweighted ones "
                f"(W {self.W.shape} vs N {self.N.shape}, "
                f"Y_w {self.Y_w.shape} vs Y {self.Y.shape})."
            )

    @property
    def weights(self) -> Optional[WeightResult]:
        """Weight diagnostics (:class:`~cntmosaic.weights.WeightResult`), or
        ``None`` when the fit is unweighted."""
        if self._bound is None:
            return None
        if self._weight_result is None:
            self._weight_result = self._bound.result()
        return self._weight_result

    def _resolve_weighted(self, weighted: Optional[bool]) -> bool:
        """``None`` means weighted iff ``weight_specs`` was given."""
        if weighted is None:
            return self.weight_specs is not None
        if weighted and self.weight_specs is None:
            raise ValueError(
                "weighted=True requires weight_specs; this SocialMix was fitted "
                "without weights."
            )
        return bool(weighted)

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


    def cint(self, weighted: Optional[bool] = None) -> Dict[str, "ContactSummary"]:
        """
        Compute contact intensity matrix.

        M[c,d] represents the average number of contacts that individuals
        in age group c have with individuals in age group d.

        If apply_reciprocity=True, applies post-hoc reciprocity adjustment
        to ensure contact symmetry (for single/full stratification modes).

        Parameters
        ----------
        weighted : bool, optional
            ``None`` (default): weighted if ``weight_specs`` was given,
            otherwise unweighted. ``False``: always the unweighted estimate
            ``Y / N``. ``True``: the weighted estimate ``Y_w / W``, i.e.
            ``sum_i w_i y_id / sum_i w_i`` per cell; requires ``weight_specs``.

        Returns
        -------
        Dict[str, ContactSummary]
            Dictionary mapping stratum labels to ContactSummary objects.
            Keys follow "source->target" format:
            - "All->All" for unstratified data
            - "M->All", "F->All" for participant-only stratification
            - "M->M", "M->F", "F->M", "F->F" for full stratification

        Raises
        ------
        ValueError
            If ``weighted=True`` but the fit has no ``weight_specs``.
        """
        weighted = self._resolve_weighted(weighted)
        if weighted not in self._cint:
            Y, N = (self.Y_w, self.W) if weighted else (self.Y, self.N)
            labels = self._create_stratum_labels()

            # Compute raw intensity matrices based on data structure
            raw = {}

            if self.K_part == 1 and self.K_cnt == 1:
                # Single stratum: Y is (C, D), N is (C,)
                raw["All->All"] = Y / N[:, np.newaxis]

            elif self.K_part > 1 and self.K_cnt == 1:
                # Partial stratification: Y is (K_part, C, D), N is (K_part, C)
                for k in range(self.K_part):
                    raw[labels[k]] = Y[k] / N[k, :, np.newaxis]

            else:
                # Full/mixed stratification: Y is (K_part, K_cnt, C, D), N is (K_part, C)
                idx = 0
                for k_part in range(self.K_part):
                    for k_cnt in range(self.K_cnt):
                        raw[labels[idx]] = Y[k_part, k_cnt] / N[k_part, :, np.newaxis]
                        idx += 1

            # Apply reciprocity adjustment if requested
            if self.apply_reciprocity and self.P is not None:
                raw = apply_reciprocity(raw, self.strat_mode, self.K_cnt, self.P)

            self._cint[weighted] = self._to_summaries(raw)

        return self._cint[weighted]

    def rate(self, weighted: Optional[bool] = None) -> Dict[str, "ContactSummary"]:
        """
        Compute contact rate matrix.

        ω[c,d] represents the per-capita rate at which individuals in
        age group c contact individuals in age group d.

        The rate is computed from the contact intensity matrix by dividing
        by the population size of the target age group:
            ω[c,d] = M[c,d] / P[d]

        Parameters
        ----------
        weighted : bool, optional
            As in :meth:`cint`; passed through to it.

        Returns
        -------
        Dict[str, ContactSummary]
            Dictionary of contact rate ContactSummary objects, one per stratum.
            Keys follow the format "source->target" (e.g., "All->All", "M->F").

        Raises
        ------
        ValueError
            If population data was not provided during initialization, or if
            ``weighted=True`` but the fit has no ``weight_specs``.
        """
        weighted = self._resolve_weighted(weighted)
        if weighted not in self._rate:
            # Check that population data is available
            if self.P is None:
                raise ValueError(
                    "Cannot compute contact rates without population data. "
                    "Please provide 'pop_data' when initializing SocialMix."
                )

            # Get contact intensity matrices (central values)
            cint_dict = self.cint(weighted=weighted)

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

            self._rate[weighted] = self._to_summaries(raw)

        return self._rate[weighted]

    def _to_summaries(self, raw: Dict[str, NDArray]) -> Dict[str, "ContactSummary"]:
        """Wrap point estimates in ContactSummary objects (no intervals)."""
        _nan = np.full(next(iter(raw.values())).shape, np.nan)
        return {
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

    def run_inference_bootstrap(
        self,
        n_boot: int = 1000,
        random_state: Optional[int] = None,
        progress: bool = True,
        min_success_rate: float = 0.5,
        weighted: Optional[bool] = None,
    ) -> BootstrapResults:
        """
        Estimate uncertainty via bootstrap resampling.

        Before resampling, the participant age groups are checked in two
        stages. Either stage may merge age groups:

        1. **Empty age groups.** If an age group has no participants in some
           participant stratum, it is merged with a neighbour when
           ``adaptive_merge=True``. Otherwise a ``ValueError`` is raised.
        2. **Small age groups.** Let ``n_{s,c}`` be the number of participants
           in stratum ``s`` and age group ``c`` (without stratification, the
           number in age group ``c``). A resample leaves cell ``(s, c)`` empty
           with probability about ``exp(-n_{s,c})``. While
           ``sum over (s, c) of exp(-n_{s,c})`` is at least ``5e-5``, the age
           group with the smallest count (its minimum over strata) is merged
           with the smaller of its neighbours. This stage runs **whatever
           the value of** ``adaptive_merge``.

        A merge persists after the bootstrap. It changes ``age_group_specs``,
        the cached point estimates (``cint``, ``rate``) are recomputed at the
        merged resolution, and for a weighted fit the weights are recomputed
        on the merged cells (see ``weighted``).

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
        weighted : bool, optional
            As in :meth:`cint`: ``None`` (default) means weighted if
            ``weight_specs`` was given. A weighted bootstrap rebuilds the
            weights on every resample (counts such as the number of weekday
            participants change between resamples). If the
            bootstrap merges age groups, the weights are rebound to the new
            cells and a :class:`~cntmosaic.weights.WeightWarning` says that
            ``sm.weights`` has changed.

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
            If an age group is empty in some participant stratum and
            ``adaptive_merge=False``, if ``success_rate < min_success_rate``,
            or if ``weighted=True`` but the fit has no ``weight_specs``.

        Examples
        --------
        >>> sm = SocialMix(part_data, cnt_data, age_bins, pop_data)
        >>> boot = sm.run_inference_bootstrap(n_boot=1000, random_state=42)
        >>> cint_mean = boot.mean(statistic='cint')
        >>> cint_ci = boot.quantiles(q=[0.025, 0.975], statistic='cint')
        """
        weighted = self._resolve_weighted(weighted)

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
            self._cint = {}
            self._rate = {}
            # Drop stale age_grp column so _assign_age_groups_to_population()
            # recomputes it with the new (post-merge) intervals.
            if self.pop_data is not None and "age_grp" in self.pop_data.data.columns:
                self.pop_data.data.drop(columns=["age_grp"], inplace=True)
            self._load()
            # Rebind the weights to the merged cells (D25); bind-time warnings
            # were already emitted at fit time.
            self._fit_weights(warn_user=False)
            if self.weight_specs is not None:
                warn_weights(
                    "The bootstrap merged participant age groups, so the weights were "
                    "recomputed on the merged cells: sm.weights and the weighted point "
                    "estimates have changed."
                )

        # Create bootstrap estimator and run
        bootstrap = SocialMixBootstrap(
            part_data=self.part_data,
            cnt_data=self.cnt_data,
            age_group_specs=self.age_group_specs,
            pop_data=self.pop_data,
            apply_reciprocity=self.apply_reciprocity,
            n_boot=n_boot,
            random_state=random_state,
            bound_weights=self._bound if weighted else None,
        )

        self._boot = bootstrap.run(
            progress=progress,
            min_success_rate=min_success_rate,
        )
        return self._boot
