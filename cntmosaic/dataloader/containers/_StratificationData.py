import warnings
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..._types import StratMode
from ._stratification_preprocessing import preprocess_stratification_data
from ._stratification_validation import validate_stratification_data


@dataclass
class StratificationData:
    """
    Container for stratified population data required for stratified contact matrix inference.

    This class handles the population data required to satisfy structural constratains
    in the social contact model. More specifically, it takes either the stratified population
    sizes or proportions and computes the necessary demographic opportunity multipliers (product of population proportions)
    which are used to calculate the contact rate modifiers for each stratification variable.
    This class supports both the partially stratified (PARTIAL) and the fully stratified (FULL) scenarios.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing stratified population proportions.
        Must have columns for: age (or age range), stratification variable(s), and proportions.
        Proportions within each age group must sum to 1.0.
    age_col : str, optional
        Name of the column containing exact age values (e.g. 0, 1, 2, …).
        Use this OR ``age_min_col``/``age_max_col``, not both.
    age_min_col : str, optional
        Name of the column containing the lower bound of each age range.
        Must be used together with ``age_max_col``. Mutually exclusive with ``age_col`` and ``age_grp_col``.
    age_max_col : str, optional
        Name of the column containing the upper bound of each age range.
        Must be used together with ``age_min_col``. Mutually exclusive with ``age_col`` and ``age_grp_col``.
    age_grp_col : str, optional
        Name of the column containing pre-built age groups (e.g. pd.IntervalIndex or categorical).
        Mutually exclusive with ``age_col`` and ``age_min_col``/``age_max_col``.
        In the processed data this column is renamed to ``"age_grp_strat"``.
    strat_var_cols : Union[str, List[str]], optional
        Name(s) of the stratification variable column(s) in the DataFrame.
    prop_col : str, default='prop'
        Name of the column containing population proportions.
        Values must be in [0, 1] and sum to 1.0 within each age group.
        See also: from_counts() class method for automatic proportion computation from counts.

    Attributes
    ----------
    var_name : str
        The standardized name of the stratification variable (read-only).
        - For single column: same as the column name
        - For multiple columns: underscore-joined (e.g., 'gender_region')
        This is automatically determined and used by DataLoader for matching.
    strat_col : str
        The final stratification column name in the processed data (read-only).
        After merging, this contains the composite variable name.

    Methods
    -------
    validate()
        Validates the population proportion data structure and values.
    from_counts(data, age_col, strat_var_cols, count_col)
        Class method to create StratificationData from population counts.
    compute_marginal_demopty(strat_modes)
        Computes marginal multipliers for each stratification variable.
    compute_demopty(strat_modes)
        Computes stratified multipliers for contact modifier weighting.

    Raises
    ------
    ValueError
        If required columns are missing from the dataframe.
        If proportions don't sum to 1.0 within each age group (tolerance 1e-6).
        If proportion values are outside [0, 1] range.

    Examples
    --------
    >>> # Single stratification variable (auto-detected)
    >>> df_strat = pd.DataFrame({
    ...     'age': [0, 0, 1, 1, 2, 2],
    ...     'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    ...     'proportion': [0.51, 0.49, 0.51, 0.49, 0.50, 0.50]
    ... })
    >>> pop_data = StratificationData(data=df_strat, age_col='age', strat_var_cols='gender', prop_col='proportion')
    >>> pop_data.strat_var_cols
    ['gender']
    >>>
    >>> # Multiple columns (automatic composite)
    >>> df_strat = pd.DataFrame({
    ...     'age': [0, 0, 0, 0, 1, 1, 1, 1],
    ...     'gender': ['M', 'M', 'F', 'F', 'M', 'M', 'F', 'F'],
    ...     'region': ['North', 'South', 'North', 'South', 'North', 'South', 'North', 'South'],
    ...     'prop': [0.26, 0.24, 0.25, 0.25, 0.26, 0.24, 0.25, 0.25]
    ... })
    >>> strat_data = StratificationData(
    ...     data=df_strat,
    ...     age_col='age',
    ...     strat_var_cols=['gender', 'region'],
    ...     prop_col='prop'
    ... )
    >>>
    >>> # From population counts (auto-computes proportions)
    >>> df_counts = pd.DataFrame({
    ...     'age': [0, 0, 1, 1],
    ...     'gender': ['M', 'F', 'M', 'F'],
    ...     'count': [510, 490, 505, 495]
    ... })
    >>> strat_prop = StratificationData.from_counts(
    ...     data=df_counts,
    ...     age_col='age',
    ...     strat_var_cols='gender',
    ...     count_col='count'
    ... )
    """

    data: pd.DataFrame
    age_col: Optional[str] = None
    strat_var_cols: Optional[Union[str, List[str]]] = None
    prop_col: str = "prop"
    age_min_col: Optional[str] = None
    age_max_col: Optional[str] = None
    age_grp_col: Optional[str] = None

    strat_col: str = None  # legacy field; not used internally

    def __post_init__(self) -> None:
        """Process stratification columns and validate after initialization."""
        # Convert single string to list for uniform processing
        if isinstance(self.strat_var_cols, str):
            self.strat_var_cols = [self.strat_var_cols]

        # Validate mutual exclusivity of age specifications.
        _has_exact = self.age_col is not None
        _has_range = self.age_min_col is not None or self.age_max_col is not None
        _has_grp = self.age_grp_col is not None

        if not _has_exact and not _has_range and not _has_grp:
            raise ValueError(
                "Must specify an age representation:\n"
                "  'age_col' for exact ages,\n"
                "  both 'age_min_col' and 'age_max_col' for age ranges,\n"
                "  or 'age_grp_col' for age groups."
            )
        if sum([_has_exact, _has_range, _has_grp]) > 1:
            raise ValueError(
                "Age specification forms are mutually exclusive — provide exactly one:\n"
                "  'age_col', 'age_min_col'/'age_max_col', or 'age_grp_col'.\n"
                f"  Got: age_col={self.age_col!r}, age_min_col={self.age_min_col!r}, "
                f"age_max_col={self.age_max_col!r}, age_grp_col={self.age_grp_col!r}"
            )
        if _has_range and (self.age_min_col is None or self.age_max_col is None):
            raise ValueError(
                "Both 'age_min_col' and 'age_max_col' must be specified together.\n"
                f"  Got: age_min_col={self.age_min_col!r}, age_max_col={self.age_max_col!r}"
            )

        # Delegate column validation and categorical conversion
        self.data = preprocess_stratification_data(
            self.data,
            self.age_col,
            self.strat_var_cols,
            self.prop_col,
            self.age_min_col,
            self.age_max_col,
            self.age_grp_col,
        )

        self.validate()

    def validate(self) -> None:
        """
        Validate population proportion data structure and values.

        Checks:
        1. All required columns are present in the dataframe
        2. Proportions are in valid range [0, 1]
        3. Proportions sum to 1.0 within each age group (tolerance 1e-6)

        Raises
        ------
        ValueError
            If any validation check fails.

        Examples
        --------
        >>> pop_prop = StratificationData(df, 'age', 'gender', 'proportion')
        >>> # Validation happens automatically, but can be called explicitly:
        >>> pop_prop.validate()
        """
        validate_stratification_data(
            self.data,
            self.age_col,
            self.strat_var_cols,
            self.prop_col,
            self.age_min_col,
            self.age_max_col,
            self.age_grp_col,
        )

    @property
    def _age_groupby_cols(self) -> List[str]:
        """Return the list of column(s) used as the age groupby key."""
        if self.age_col:
            return [self.age_col]
        if self.age_grp_col:
            return ["age_grp_strat"]
        return [self.age_min_col, self.age_max_col]

    @classmethod
    def from_counts(
        cls,
        data: pd.DataFrame,
        age_col: Optional[str] = None,
        strat_var_cols: Optional[Union[str, List[str]]] = None,
        count_col: str = "count",
        age_min_col: Optional[str] = None,
        age_max_col: Optional[str] = None,
        age_grp_col: Optional[str] = None,
    ) -> "StratificationData":
        """
        Create StratificationData from population counts (automatically computes proportions).

        This is a convenience constructor that automatically normalizes population counts
        to proportions within each age group. More intuitive than manually computing
        proportions.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with population counts stratified by age and category.
        age_col : str
            Name of age column.
        strat_var_cols : Union[str, List[str]], optional
            Name(s) of stratification variable column(s) in the DataFrame.
            If None, will be auto-detected from columns (excluding age_col and count_col).
        count_col : str, default='count'
            Name of column containing population counts.
        prop_col : str, default='proportion'
            Name to assign to the computed proportion column.

        Returns
        -------
        StratificationData
            New instance with proportions computed from counts.

        Examples
        --------
        >>> # Single variable (auto-detected)
        >>> df = pd.DataFrame({
        ...     'age': [0, 0, 1, 1, 2, 2],
        ...     'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
        ...     'population': [5100, 4900, 5050, 4950, 5000, 5000]
        ... })
        >>> pop_prop = StratificationData.from_counts(
        ...     data=df,
        ...     age_col='age',
        ...     count_col='population'
        ... )  # 'gender' auto-detected
        >>>
        >>> # Multiple variables (automatic composite)
        >>> df_multi = pd.DataFrame({
        ...     'age': [0, 0, 0, 0],
        ...     'gender': ['M', 'M', 'F', 'F'],
        ...     'region': ['North', 'South', 'North', 'South'],
        ...     'count': [2600, 2400, 2500, 2500]
        ... })
        >>> pop_prop = StratificationData.from_counts(
        ...     data=df_multi,
        ...     age_col='age',
        ...     strat_var_cols=['gender', 'region'],
        ...     count_col='count'
        ... )
        >>> pop_prop.var_name  # 'gender_region'
        """
        if strat_var_cols is None:
            raise ValueError("strat_var_cols must be specified.")

        # Convert to list for uniform processing
        if isinstance(strat_var_cols, str):
            strat_var_cols = [strat_var_cols]

        # Resolve age columns
        _has_exact = age_col is not None
        _has_range = age_min_col is not None or age_max_col is not None
        _has_grp = age_grp_col is not None
        if not _has_exact and not _has_range and not _has_grp:
            raise ValueError(
                "Must specify 'age_col', both 'age_min_col' and 'age_max_col', or 'age_grp_col'."
            )
        if sum([_has_exact, _has_range, _has_grp]) > 1:
            raise ValueError(
                "Provide exactly one of 'age_col', 'age_min_col'/'age_max_col', or 'age_grp_col'."
            )
        if _has_range and (age_min_col is None or age_max_col is None):
            raise ValueError("Both 'age_min_col' and 'age_max_col' must be specified together.")

        if age_col:
            age_groupby_cols = [age_col]
        elif age_grp_col:
            age_groupby_cols = [age_grp_col]
        else:
            age_groupby_cols = [age_min_col, age_max_col]

        # Validate required columns
        required_cols = age_groupby_cols + [count_col] + strat_var_cols
        missing = [col for col in required_cols if col not in data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Available: {list(data.columns)}"
            )

        # Compute proportions within each age group
        df_with_props = data.copy()
        df_with_props["prop"] = df_with_props.groupby(age_groupby_cols, observed=False)[
            count_col
        ].transform(lambda x: x / x.sum())

        return cls(
            data=df_with_props,
            age_col=age_col,
            age_min_col=age_min_col,
            age_max_col=age_max_col,
            age_grp_col=age_grp_col,
            strat_var_cols=strat_var_cols,
        )

    def get_strat_vars(self) -> List[str]:
        """
        Return a list of stratification variables

        Returns
        -------
        List[str]
            List of stratification variable column names.

        Examples
        --------
        >>> strat_data = StratificationData(df, "age", "sex", "prop")
        >>> strat_data.get_strat_vars()
        ["sex"]
        """
        return self.strat_var_cols

    def get_strat_var_schema(self) -> Dict[str, Dict[str, List[Union[str, int]]]]:
        """
        Return a dictionary of stratification variables and their unique category and code values.
        The output is to be used in the DataLoader to ensure that the category and codes are consistent
        between the participant, contact, population, and demographic data.

        Returns
        -------
        Dict[str, Dict[str, List[str | int]]]
            A dictionary where keys are the original stratification variable names (without '_part' suffix)
            and values are dictionaries with keys 'categories' and 'codes' containing lists of unique category
            values and their corresponding codes.
        """
        schema = {}
        for var in self.strat_var_cols:
            if var in self.data.columns:
                categories = self.data[var].cat.categories.tolist()
                codes = sorted(self.data[var].cat.codes.unique().tolist())
                schema[var] = {"categories": categories, "codes": codes}

        return schema

    def _build_Q_and_labels(
        self,
        vars_list: List[str],
        strat_labels_for_vars: Optional[List] = None,
    ) -> Tuple[NDArray, List]:
        """
        Build ordered Q matrix and corresponding labels for given stratification variable(s).

        Parameters
        ----------
        vars_list : List[str]
            List of stratification variable column names to combine (length 1 for single-var case).
        strat_labels_for_vars : List, optional
            Optional ordering of labels to reindex the pivot (preserves order).

        Returns
        -------
        Q : NDArray
            Array shape (n_strata, A) giving proportions by stratum and age.
        labels : List
            Ordered list of stratum labels corresponding to rows of Q.
        """
        age_groupby = self._age_groupby_cols
        # When using age ranges, create a composite age key for pivoting
        df_Q_source = self.data[age_groupby + vars_list + [self.prop_col]].copy()

        if len(age_groupby) > 1:
            df_Q_source["_age_key"] = list(
                zip(df_Q_source[age_groupby[0]], df_Q_source[age_groupby[1]])
            )
            age_key = "_age_key"
        else:
            age_key = age_groupby[0]

        sort_vars = [age_key] + vars_list
        df_Q_source = df_Q_source.sort_values(by=sort_vars)

        if len(vars_list) == 1:
            var = vars_list[0]
            df_pivot = (
                df_Q_source.groupby([age_key, var], observed=False)[self.prop_col]
                .sum()
                .reset_index()
                .pivot(index=var, columns=age_key, values=self.prop_col)
            )
            if strat_labels_for_vars is not None:
                # If caller passed full cross-product labels like 'A->B',
                # deduplicate by splitting on '->' to obtain source labels
                if any("->" in str(l) for l in strat_labels_for_vars):
                    strat_labels_for_vars = list(
                        dict.fromkeys(l.split("->")[0] for l in strat_labels_for_vars)
                    )
                df_pivot = df_pivot.reindex(strat_labels_for_vars)
                labels = strat_labels_for_vars
            else:
                labels = df_pivot.index.tolist()
        else:
            # Combine multiple vars into single label
            df_Q_source["strat_combined"] = df_Q_source[vars_list].apply(
                lambda row: "_".join(row.astype(str)), axis=1
            )
            df_pivot = (
                df_Q_source.groupby([age_key, "strat_combined"], observed=False)[
                    self.prop_col
                ]
                .sum()
                .reset_index()
                .pivot(
                    index="strat_combined", columns=age_key, values=self.prop_col
                )
            )
            if strat_labels_for_vars is not None:
                df_pivot = df_pivot.reindex(strat_labels_for_vars)
                labels = strat_labels_for_vars
            else:
                labels = df_Q_source["strat_combined"].unique().tolist()

        Q = df_pivot.values
        return Q, labels

    def compute_marginal_demopty(
        self, strat_modes: Dict[str, StratMode], strat_labels: Dict[str, List] = None
    ) -> Dict[str, NDArray]:
        """
        Compute the marginal multipliers for each stratification variable.
        These marginals are used for centering the contact rate modifier priors in the latent space.

        Parameters
        ----------
        strat_modes : Dict[str, StratMode]
            Dictionary mapping stratification variable names to their StratMode (PARTIAL or FULL).
        strat_labels : Dict[str, List], optional
            Optional mapping of variable -> label ordering to preserve.

        Returns
        -------
        Dict[str, NDArray]
            Dictionary mapping each stratification variable to its marginal multipliers array.
        """
        strat_vars = list(strat_modes.keys())
        margin_dempoty: Dict[str, NDArray] = {}

        for var in strat_vars:
            labels_for_var = None
            if (
                strat_labels is not None
                and isinstance(strat_labels, dict)
                and var in strat_labels
            ):
                labels_for_var = strat_labels[var]

            Q, strat_labels_source = self._build_Q_and_labels(
                [var], strat_labels_for_vars=labels_for_var
            )

            if strat_modes[var] == StratMode.PARTIAL:
                result = Q[:, :, None]
                margin_dempoty[var] = self._handle_nans(
                    result, strat_labels_source, None, mode="PARTIAL"
                )

            elif strat_modes[var] == StratMode.FULL:
                mult_array = Q[:, None, :, None] * Q[None, :, None, :]
                n_strata = Q.shape[0]
                A = Q.shape[1]
                reshaped = mult_array.reshape(n_strata * n_strata, A, A)

                # Row n = s * n_strata + t pairs source s with target t.
                pair_sources = [s for s in strat_labels_source for _ in strat_labels_source]
                pair_targets = [t for _ in strat_labels_source for t in strat_labels_source]

                margin_dempoty[var] = self._handle_nans(
                    reshaped, pair_sources, pair_targets, mode="FULL"
                )

            else:
                raise ValueError(
                    f"Unknown StratMode: {strat_modes[var]} for variable {var}"
                )

        return margin_dempoty

    def compute_demopty(
        self, strat_modes: Dict[str, StratMode], strat_labels: List = None
    ) -> NDArray:
        """
        Compute demographic opportunity P^{s,t}_{a,b} / (P_a * P_b) (for FULL) or P^{s}_{a} / P_a (for PARTIAL).

        This method takes the dictionary of stratification modes and computes the demographic opportunity
        by age and stratification variables. It supports both PARTIAL, FULL, and mixed scenarios.

        Parameters
        ----------
        strat_modes : Dict[str, StratMode]
            Dictionary mapping stratification variable names to their StratMode (PARTIAL or FULL).
        strat_labels : List, optional
            Full stratification labels (``"<source>-><target>"``) as produced by
            ``infer_full_strat_labels``. FULL rows are returned in this order,
            which must match the model's ``log_delta``. If None, the same order
            is inferred from the data.

        Notes
        -----
        The .from_counts() constructor handles normalization from counts to proportions.
        So this method will only work with proportions, not raw counts.

        Returns
        -------
        NDArray
            - PARTIAL: shape (n_strata, A, 1)
            - FULL: shape (len(strat_labels), A, A), row n for strat_labels[n]
        """
        strat_vars_source = list(strat_modes.keys())
        strat_vars_target = [
            var for var in strat_vars_source if strat_modes[var] == StratMode.FULL
        ]
        if len(strat_vars_source) == 0:
            raise ValueError("No stratification variables provided")

        # =======================================================================
        # PARTIAL case: no FULL variables, one row per source stratum
        # =======================================================================
        if len(strat_vars_target) == 0:
            strat_labels_source = None
            if strat_labels is not None and not isinstance(strat_labels, dict):
                strat_labels_source = list(
                    dict.fromkeys(l.split("->")[0] for l in strat_labels)
                )
            Q_sa, strat_labels_source = self._build_Q_and_labels(
                strat_vars_source, strat_labels_for_vars=strat_labels_source
            )
            result = Q_sa[:, :, None]  # shape (n_strata, A, 1)
            return self._handle_nans(result, strat_labels_source, None, mode="PARTIAL")

        # =======================================================================
        # FULL case: one source x target outer product per full label
        # =======================================================================
        # Row n of the result must match ``strat_labels[n]``: the model
        # subtracts log(multipliers) from log_delta by position, and log_delta
        # is ordered like the full labels. Those interleave source and target
        # per variable (e.g. sex, sex', ethn, ethn'), so the multipliers are
        # looked up by label rather than by reshaping a (source, target)
        # outer product, whose flat order (sex, ethn, sex', ethn') differs
        # as soon as a FULL variable is followed by another variable.
        if strat_labels is None or isinstance(strat_labels, dict):
            strat_labels = self._default_full_labels(strat_modes)

        pair_sources = [label.split("->")[0] for label in strat_labels]
        pair_targets = [label.split("->")[1] for label in strat_labels]

        Q_sa, strat_labels_source = self._build_Q_and_labels(
            strat_vars_source,
            strat_labels_for_vars=list(dict.fromkeys(pair_sources)),
        )
        Q_tb, strat_labels_target = self._build_Q_and_labels(
            strat_vars_target,
            strat_labels_for_vars=list(dict.fromkeys(pair_targets)),
        )
        source_ix = {label: i for i, label in enumerate(strat_labels_source)}
        target_ix = {label: i for i, label in enumerate(strat_labels_target)}
        s_ix = [source_ix[label] for label in pair_sources]
        t_ix = [target_ix[label] for label in pair_targets]

        # result[n, a, b] = Q_sa[s_n, a] * Q_tb[t_n, b]
        result = Q_sa[s_ix][:, :, None] * Q_tb[t_ix][:, None, :]

        return self._handle_nans(result, pair_sources, pair_targets, mode="FULL")

    def _default_full_labels(self, strat_modes: Dict[str, StratMode]) -> List[str]:
        """
        Full stratification labels in the order the model uses.

        Mirrors ``infer_full_strat_labels``: the per-variable labels
        (``"a->b"`` for FULL, ``"a->All"`` for PARTIAL) are combined with the
        rightmost variable varying fastest.

        Parameters
        ----------
        strat_modes : Dict[str, StratMode]
            Stratification variables and their modes, in model order.

        Returns
        -------
        List[str]
            Labels of the form ``"<source>-><target>"``.
        """
        per_var_labels = []
        for var, mode in strat_modes.items():
            _, categories = self._build_Q_and_labels([var])
            if mode == StratMode.FULL:
                per_var_labels.append([(a, b) for a in categories for b in categories])
            else:
                per_var_labels.append([(a, None) for a in categories])

        full_labels = []
        for combo in product(*per_var_labels):
            source = "_".join(str(a) for a, _ in combo)
            targets = [str(b) for _, b in combo if b is not None]
            target = "_".join(targets) if targets else "All"
            full_labels.append(f"{source}->{target}")
        return full_labels

    def _handle_nans(
        self,
        multipliers: NDArray,
        strat_labels_source: NDArray,
        strat_labels_target: Optional[NDArray],
        mode: str,
    ) -> NDArray:
        """
        Check for NaN values in multipliers and replace with neutral value (1.0).

        Parameters
        ----------
        multipliers : NDArray
            The computed multipliers array that may contain NaN values.
        strat_labels_source : NDArray
            PARTIAL: label of each stratum (row). FULL: source label of each
            row of ``multipliers``.
        strat_labels_target : NDArray, optional
            FULL only: target label of each row of ``multipliers``.
        mode : str
            Either 'PARTIAL' or 'FULL' to determine warning message format.

        Returns
        -------
        NDArray
            Multipliers with NaN values replaced by 1.0.
        """
        if not np.isnan(multipliers).any():
            return multipliers

        # Identify NaN locations
        nan_mask = np.isnan(multipliers)
        nan_count = nan_mask.sum()

        # Generate warning message with affected strata combinations
        if mode == "PARTIAL":
            # Shape: (n_strata, A, 1)
            affected_strata = set()
            for strat_idx in range(multipliers.shape[0]):
                if nan_mask[strat_idx].any():
                    affected_strata.add(strat_labels_source[strat_idx])

            warning_msg = (
                f"Found {nan_count} NaN values in stratified multipliers (PARTIAL mode). "
                f"These have been replaced with 1.0 (neutral multiplier). "
                f"Affected strata: {sorted(affected_strata)}. "
                f"This typically occurs when certain age-strata combinations have no population data. "
                f"Verify your population proportion data completeness."
            )
        else:  # FULL mode
            # Shape: (n_pairs, A, A); row n pairs strat_labels_source[n]
            # with strat_labels_target[n].
            affected_pairs = {
                (strat_labels_source[n], strat_labels_target[n])
                for n in range(multipliers.shape[0])
                if nan_mask[n].any()
            }

            warning_msg = (
                f"Found {nan_count} NaN values in stratified multipliers (FULL mode). "
                f"These have been replaced with 1.0 (neutral multiplier). "
                f"Affected strata pairs (source, target): {sorted(affected_pairs)}. "
                f"This typically occurs when certain age-strata combinations have no population data. "
                f"Verify your population proportion data completeness."
            )

        warnings.warn(warning_msg, UserWarning, stacklevel=3)

        # Replace NaNs with neutral multiplier
        multipliers_clean = multipliers.copy()
        multipliers_clean[nan_mask] = 1.0

        return multipliers_clean

    def validate_for_mode(self, mode: StratMode) -> None:
        """
        Validate data structure matches stratification mode.

        For PARTIAL: Requires age column(s), strat_var_cols, and prop_col.
        For FULL: Same requirements (outer product computed internally).
        """
        if self.age_col:
            age_cols = [self.age_col]
        elif self.age_grp_col:
            age_cols = ["age_grp_strat"]
        else:
            age_cols = [self.age_min_col, self.age_max_col]
        required_cols = age_cols + (self.strat_var_cols or []) + [self.prop_col]
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for {mode.name} mode: {missing}\n"
                f"Required: {required_cols}\n"
                f"Available: {list(self.data.columns)}"
            )
