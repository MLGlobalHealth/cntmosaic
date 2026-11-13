from typing import Any, Dict, List, Optional

import pandas as pd
from numpy.typing import NDArray

from ...dataloader import ContactData, ParticipantData
from ...utils import AgeGroupSpecs
from .._base import ContactModel
from ..numpyro import Prem2NumPyroMixin
from ._socialmix_age_processing import AgeBinProcessor
from ._socialmix_helpers import create_stratum_labels, infer_strat_mode
from ._socialmix_utils import SocialMixDataLoader
from ._prem2_validation import Prem2Validator


class Prem2(Prem2NumPyroMixin, ContactModel):
    """
    Estimate age-structured social contact matrices using aggregated counts.

    Reformulates :class:`Prem` to operate on aggregated contact count cells
    (following the same ``(C, D)``-style tensors as :class:`SocialMix`)
    instead of individual-level long-format data. This class handles data
    preparation only; the Bayesian inference model itself is defined by
    :class:`~cntmosaic.models.numpyro.Prem2NumPyroMixin`, whose ``model()``
    is a placeholder for now and will be filled in with an
    ``AgeMixCC``/``GenMixCC``-style Poisson/negative-binomial model
    (without the ``log_P`` population-size offset, since Prem2 has no
    population data) in a follow-up task.

    Parameters
    ----------
    part_data : ParticipantData
        Validated participant data container. Should include age groups
        (part_age_grp) and optional stratification variables.
    cnt_data : ContactData
        Validated contact data container. Should include contact age groups
        (cnt_age_grp) and matching stratification variables.
    age_group_specs : AgeGroupSpecs
        Age binning scheme to categorize ages into age groups.
        Used to assign age groups if raw ages are provided in the containers.
    likelihood : {"poisson", "negbin"}, default="negbin"
        Observation likelihood used by :meth:`model`.
    backend : InferenceBackend, optional
        Pluggable inference engine. See :class:`ContactModel`.

    Attributes
    ----------
    Y : NDArray
        Aggregated contact count tensor.
        Shape ``(C, D)`` for single mode; ``(K_part, C, D)`` for partial;
        ``(K_part, K_cnt, C, D)`` for full/mixed.
    N : NDArray
        Participant counts per age group and stratum.
        Shape ``(C,)`` for single mode; ``(K_part, C)`` for stratified.
    P : None
        Always ``None`` — Prem2 has no population data.
    C : int
        Number of participant age groups.
    D : int
        Number of contact age groups.
    K_part : int
        Number of participant strata.
    K_cnt : int
        Number of contact strata.
    strat_vars_part : List[str]
        Names of participant stratification variables.
    strat_vars_cnt : List[str]
        Names of contact stratification variables.
    strat_vars_shared : List[str]
        Stratification variables present in both participant and contact data.
    strat_vars_part_only : List[str]
        Stratification variables only in participant data.
    strat_vars_cnt_only : List[str]
        Stratification variables only in contact data.
    K : int
        Total number of strata.

    Examples
    --------
    >>> from cntmosaic.dataloader import ParticipantData, ContactData
    >>> from cntmosaic.utils import AgeGroupSpecs
    >>>
    >>> age_bins = AgeGroupSpecs(0, 80, 5)
    >>> model = Prem2(part_data, cnt_data, age_bins)
    >>> model.Y.shape
    (16, 16)

    See Also
    --------
    Prem : Individual-level long-format Bayesian model this class reformulates.
    SocialMix : Deterministic aggregated-count model this class's data pipeline
        mirrors.
    """

    ALLOWED_LIKELIHOODS = ["poisson", "negbin"]

    def __init__(
        self,
        part_data: ParticipantData,
        cnt_data: ContactData,
        age_group_specs: AgeGroupSpecs,
        likelihood: str = "negbin",
        backend: Optional[Any] = None,
    ):
        super().__init__(backend=backend)

        if likelihood not in self.ALLOWED_LIKELIHOODS:
            raise ValueError(
                f"likelihood must be one of: {self.ALLOWED_LIKELIHOODS}, "
                f"got '{likelihood}'"
            )

        # Store parameters
        self.part_data = part_data
        self.cnt_data = cnt_data
        self.age_group_specs = age_group_specs
        self.likelihood = likelihood

        # Fixed internal-only attribute so SocialMixDataLoader (reused for
        # _load) works unmodified — Prem2 has no pop_data concept.
        self.pop_data = None

        # Stratification attributes (initialized in _preprocess)
        self.strat_vars_part: List[str] = []
        self.strat_vars_cnt: List[str] = []
        self.strat_vars_shared: List[str] = []
        self.strat_vars_part_only: List[str] = []
        self.strat_vars_cnt_only: List[str] = []
        self.strat_mode: str = None
        self.strat_dims_part: Dict[str, int] = {}
        self.strat_dims_cnt: Dict[str, int] = {}
        self.K: int = 1  # Total number of strata

        # Initialize helper classes
        self.age_processor = AgeBinProcessor(age_group_specs)

        # Data arrays (initialized in _load)
        self.Y: Optional[NDArray] = None  # Contact counts
        self.N: Optional[NDArray] = None  # Participant counts
        self.P: Optional[NDArray] = None  # Always None (no pop_data)
        self.C: int = 0  # Number of participant age groups
        self.D: int = 0  # Number of contact age groups
        self.K_part: int = 1  # Number of participant strata
        self.K_cnt: int = 1  # Number of contact strata

        # Run processing pipeline
        self._preprocess()
        self._load()

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframes(
        cls,
        df_part: pd.DataFrame,
        df_cnt: pd.DataFrame,
        age_group_specs: AgeGroupSpecs,
    ) -> "Prem2":
        """Construct Prem2 directly from DataFrames, bypassing manual container creation.

        Use this constructor when you have raw DataFrames rather than pre-built
        :class:`ParticipantData` / :class:`ContactData` containers. The
        constructor auto-detects age and stratification columns from the
        ``part_*`` / ``cnt_*`` column naming convention and wraps the
        DataFrames in the appropriate container objects internally.

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
        """
        _CORE_PART = {"id", "part_age", "part_age_grp", "part_age_min", "part_age_max"}
        _CORE_CNT = {"id", "cnt_age", "cnt_age_grp", "cnt_age_min", "cnt_age_max", "y"}

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

        return cls(
            part_data=part_data,
            cnt_data=cnt_data,
            age_group_specs=age_group_specs,
        )

    # ------------------------------------------------------------------
    # Internal preprocessing / loading
    # ------------------------------------------------------------------

    def _preprocess(self) -> None:
        """
        Extract and validate stratification information from data containers.

        This method:
        1. Assigns age groups to raw ages if needed
        2. Extracts stratification variables from both containers
        3. Identifies shared, participant-only, and contact-only variables
        4. Computes stratification dimensions and expected number of strata
        """
        # Assign age groups first (needed for validation)
        self._assign_age_groups()

        # Use validator to handle validation logic
        validator = Prem2Validator(
            self.part_data,
            self.cnt_data,
            self.age_group_specs,
        )

        # Run all validations and get updated components
        validated = validator.validate_all()

        # Update instance with validated components
        self.part_data = validated["part_data"]
        self.cnt_data = validated["cnt_data"]
        self.age_group_specs = validated["age_group_specs"]
        self.age_processor = AgeBinProcessor(self.age_group_specs)

        # Extract stratification variables from validated data
        self.strat_vars_part = self.part_data.get_strat_vars(prefix=False)
        self.strat_vars_cnt = self.cnt_data.get_strat_vars(prefix=False)

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
        # they agree with the stratum labels used by the shared SocialMix loader.
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
        raw ages into age groups.
        """
        bin_edges = self.age_group_specs.left + [self.age_group_specs.right[-1] + 1]

        # Interval labels: right bound is exclusive (right[i] + 1)
        intervals = [
            pd.Interval(left=l, right=r + 1, closed="left")
            for l, r in zip(self.age_group_specs.left, self.age_group_specs.right)
        ]

        # Assign age groups to participants if not present
        if "part_age_grp" not in self.part_data.data.columns:
            if "part_age" in self.part_data.data.columns:
                # Create age groups from raw ages
                ages = self.part_data.data["part_age"]
                age_grps = pd.cut(
                    ages,
                    bins=bin_edges,
                    right=False,
                    labels=intervals,
                )
                self.part_data.data["part_age_grp"] = age_grps
            else:
                raise ValueError(
                    "ParticipantData must have either 'part_age' or 'part_age_grp' column."
                )

        # Assign age groups to contacts if not present
        if "cnt_age_grp" not in self.cnt_data.data.columns:
            if "cnt_age" in self.cnt_data.data.columns:
                # Create age groups from raw ages
                ages = self.cnt_data.data["cnt_age"]
                age_grps = pd.cut(
                    ages,
                    bins=bin_edges,
                    right=False,
                    labels=intervals,
                )
                self.cnt_data.data["cnt_age_grp"] = age_grps
            else:
                raise ValueError(
                    "ContactData must have either 'cnt_age' or 'cnt_age_grp' column."
                )

    def _load(self) -> None:
        """
        Prepare contact count matrices for the aggregated-cell model.

        Delegates to :class:`SocialMixDataLoader` for loading stratified data
        arrays. Since ``pop_data`` is always ``None``, ``P`` stays ``None``.

        Computes stratified contact count arrays based on stratification mode:

        - Single mode: Y (C, D), N (C,)
        - Partial mode: Y (K_part, C, D), N (K_part, C)
        - Full/Mixed mode: Y (K_part, K_cnt, C, D), N (K_part, C)

        Sets
        ----
        self.Y : NDArray
            Contact count matrix/tensor
        self.N : NDArray
            Participant count matrix/vector
        self.P : None
            Always None (no population data)
        self.C : int
            Number of participant age groups
        self.D : int
            Number of contact age groups
        self.K_part : int
            Number of participant strata
        self.K_cnt : int
            Number of contact strata
        self.y : NDArray
            Alias for ``self.Y`` — satisfies :class:`ContactModel`'s
            ``run_inference_mcmc``/``run_inference_svi``, which read
            observation data from ``self.y``.
        """
        SocialMixDataLoader(self).load_data()

        # ContactModel.run_inference_mcmc/run_inference_svi read observation
        # data from self.y (see cntmosaic/models/_base.py) and pass it as the
        # `y` kwarg to model(). Keep it as an alias for self.Y.
        self.y = self.Y

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
