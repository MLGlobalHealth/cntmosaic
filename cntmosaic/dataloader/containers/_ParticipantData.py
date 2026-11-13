from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ._participant_preprocessing import preprocess_participant_data
from ._participant_validation import validate_participant_data


@dataclass
class ParticipantData:
    """
    A container class for participant data from social contact surveys.

    This class wraps participant DataFrames and handles validation, type conversion, and
    column standardization. It automatically preprocesses data and provides convenient
    properties and methods to access participant information.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing participant information. Each row represents one participant.
        Must contain columns specified by id_col, age_col (or age_grp_col), and optionally
        strat_var_cols, repeat_col, and amb_cnt_col.
    id_col : str
        Name of the column containing unique participant identifiers.
        Values must be unique for each participant. Renamed to 'id' internally.
    age_col : Optional[str], default=None
        Name of the column containing participant ages as numeric values.
        Use this OR age_grp_col, not both. Ages should be non-negative.
        Renamed to 'part_age' internally.
    age_min_col : Optional[str], default=None
        Name of the column containing minimum age of participants (for age ranges).
        Use this with age_max_col for age range representation. Both are required together.
        Renamed to 'part_age_min' internally.
    age_max_col : Optional[str], default=None
        Name of the column containing maximum age of participants (for age ranges).
        Use this with age_min_col for age range representation. Both are required together.
        Renamed to 'part_age_max' internally.
    age_grp_col : Optional[str], default=None
        Name of the column containing participant age groups as pd.IntervalIndex.
        Use this OR age_col, not both. Must be categorical with IntervalIndex categories.
        Renamed to 'part_age_grp' internally.
    strat_var_cols : Optional[Union[List[str], str]], default=None
        Stratification variable column name(s) for participants.
        Can be a single string or list of strings. Examples: 'gender', ['gender', 'region'].
        Each variable is renamed with the ``part_`` prefix (e.g., ``gender`` → ``part_gender``).
    repeat_col : Optional[str], default=None
        Name of the column indicating repeat interviews/waves.
        Used to track longitudinal data where participants are surveyed multiple times.
        Renamed to 'part_repeat' internally.
    amb_cnt_col : Optional[str], default=None
        Name of the column containing ambiguous/group contact counts.
        If None, no ambiguous contact column is added to the DataFrame.
    aux_cols : Optional[Union[List[str], str]], default=None
        Auxiliary column name(s) kept for participant weighting (see
        :mod:`cntmosaic.weights`), e.g. ``"dayofweek"`` or a design-weight
        column. Each is renamed with the ``part_`` prefix (``"dayofweek"`` →
        ``"part_dayofweek"``). Auxiliary columns are **not** stratification
        variables, and they **keep their missing values**: rows are never
        dropped because of them. They must not overlap any other declared
        column or a reserved name (``id``, ``part_age``, ``part_age_grp``,
        ``part_age_min``, ``part_age_max``, ``part_repeat``).

    Attributes
    ----------
    data : pd.DataFrame
        The validated and preprocessed participant DataFrame.
    n : int
        Returns the number of participants (after preprocessing and validation).
    age_range : Tuple[float, float]
        Returns (min_age, max_age). Only available when using age_col.
    strat_vars : List[str]
        Returns list of original stratification variable names (empty list if none).

    Methods
    -------
    validate()
        Performs comprehensive validation of the participant data.
        Called automatically during initialization.
    get_strat_vars(prefix=False)
        Returns stratification variable names, optionally with '_part' suffix.
    get_aux_cols(prefix=True)
        Returns auxiliary column names, by default with the ``part_`` prefix.
    get_sample_sizes(stratify=False)
        Returns DataFrame with participant counts, optionally stratified by all variables.
    summary()
        Returns dictionary with summary statistics about the participant data.

    Examples
    --------
    >>> # Basic usage with individual ages
    >>> df = pd.DataFrame({
    ...     'participant_id': [1, 2, 3, 4],
    ...     'age': [25, 34, 45, 52],
    ...     'gender': ['M', 'F', 'M', 'F']
    ... })
    >>> part_data = ParticipantData(
    ...     data=df,
    ...     id_col='participant_id',
    ...     age_col='age',
    ...     strat_var_cols='gender'
    ... )
    >>> part_data.n
    4
    >>> part_data.age_range
    (25.0, 52.0)
    >>> # Columns are renamed with standardized names
    >>> list(part_data.data.columns)
    ['id', 'part_age', 'part_gender']
    >>> part_data.data['part_gender'].dtype.name
    'category'
    >>>
    >>> # With age groups and multiple stratification variables
    >>> df = pd.DataFrame({
    ...     'pid': [1, 2, 3],
    ...     'age_group': pd.Categorical.from_codes(
    ...         [0, 1, 2],
    ...         categories=pd.IntervalIndex.from_tuples([(0, 5), (5, 10), (10, 15)])
    ...     ),
    ...     'gender': ['M', 'F', 'M'],
    ...     'region': ['North', 'South', 'East']
    ... })
    >>> part_data = ParticipantData(
    ...     data=df,
    ...     id_col='pid',
    ...     age_grp_col='age_group',
    ...     strat_var_cols=['gender', 'region']
    ... )
    >>> part_data.strat_vars
    ['gender', 'region']
    >>> part_data.get_strat_vars(prefix=True)
    ['part_gender', 'part_region']
    >>> list(part_data.data.columns)
    ['id', 'part_age_grp', 'part_gender', 'part_region']
    >>>
    >>> # Get sample sizes
    >>> sample_sizes = part_data.get_sample_sizes(stratify=True)
    >>> sample_sizes.columns.tolist()
    ['part_age_grp', 'part_gender', 'part_region', 'N']

    Notes
    -----
    **Automatic Preprocessing:**

    1. Creates a copy of the input DataFrame (original is not modified)
    2. Drops rows with missing values in required columns (with warning)
    3. Converts object-type columns (except ID) to categorical dtype
    4. Renames columns to standardized names:
       - id_col → 'id' (only if id_col != 'id')
       - age_col → ``part_age`` (only if not already starting with ``part_``)
       - age_grp_col → ``part_age_grp`` (only if not already starting with ``part_``)
       - Each strat_var_cols → '{var}_part' (only if not already ending with '_part')
       - repeat_col → ``part_repeat`` (only if not already starting with ``part_``)
    5. Validates all data constraints

    **Processed DataFrame Columns:**

    - id: Participant identifier (standardized from id_col)
    - part_age: Participant age (if using age_col)
    - part_age_grp: Participant age group (if using age_grp_col)
    - {var}_part: Each stratification variable with _part suffix
    - part_repeat: Repeat interview indicator (if specified)
    - {amb_cnt_col}: Ambiguous contact count column (only if specified)
    - ``part_{aux}``: Each auxiliary column (only if specified; may contain NaN)

    **Validation Checks:**

    - Exactly one of age_col or age_grp_col must be specified
    - All required columns must exist in data
    - Participant IDs must be unique (no duplicate rows)
    - Age values (if using age_col) must be non-negative numeric
    - Age groups (if using age_grp_col) must be categorical with IntervalIndex categories
    - Repeat values (if specified) must be non-negative numeric
    - Ambiguous contact counts (if specified) must be non-negative numeric
    - No missing values in required columns (removed during preprocessing);
      auxiliary columns may contain missing values
    - Auxiliary columns must not overlap other declared columns or reserved names

    **Warnings:**

    - UserWarning: If rows are dropped due to missing values
    - UserWarning: If object columns are converted to categorical

    See Also
    --------
    ContactData : Container for contact data from social contact surveys
    DataLoader : Main data loader that combines participant and contact data
    """

    data: pd.DataFrame
    id_col: str
    age_col: Optional[str] = None
    age_min_col: Optional[str] = None
    age_max_col: Optional[str] = None
    age_grp_col: Optional[str] = None
    strat_var_cols: Optional[Union[List[str], str]] = None
    repeat_col: Optional[str] = None
    amb_cnt_col: Optional[str] = None
    aux_cols: Optional[Union[List[str], str]] = None

    def __post_init__(self) -> None:
        """
        Post-initialization processing and validation.

        Automatically called after dataclass initialization to:
        1. Validate input types
        2. Normalize strat_var_cols to list format
        3. Validate mutual exclusivity of age_col and age_grp_col
        4. Preprocess the DataFrame (copy, clean, type conversion)
        5. Perform comprehensive data validation

        Raises
        ------
        TypeError
            If df_part is not a pandas DataFrame.
        ValueError
            If both or neither of age_col and age_grp_col are specified.
            If DataFrame becomes empty after preprocessing.
        """
        # Type validation
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError(
                f"data must be a pandas DataFrame, got {type(self.data).__name__}"
            )

        # Normalize strat_var_cols to list format for consistent handling
        if isinstance(self.strat_var_cols, str):
            object.__setattr__(self, "strat_var_cols", [self.strat_var_cols])
        elif self.strat_var_cols is None:
            object.__setattr__(self, "strat_var_cols", [])

        # Normalize aux_cols to list format
        if isinstance(self.aux_cols, str):
            object.__setattr__(self, "aux_cols", [self.aux_cols])
        elif self.aux_cols is None:
            object.__setattr__(self, "aux_cols", [])
        else:
            object.__setattr__(self, "aux_cols", list(self.aux_cols))

        # Validate mutual exclusivity of age specifications.
        # Exactly one of the three forms must be used:
        #   (a) age_col only
        #   (b) age_grp_col only
        #   (c) age_min_col + age_max_col (both required together)
        _has_exact = self.age_col is not None
        _has_grp = self.age_grp_col is not None
        _has_range = self.age_min_col is not None or self.age_max_col is not None

        _n_forms = sum([_has_exact, _has_grp, _has_range])

        if _n_forms == 0:
            raise ValueError(
                "Must specify exactly one age representation:\n"
                "  'age_col' for exact integer ages (e.g., 25, 34, 45),\n"
                "  'age_grp_col' for age groups (e.g., pd.IntervalIndex or categorical),\n"
                "  or both 'age_min_col' and 'age_max_col' for age ranges."
            )
        if _n_forms > 1:
            raise ValueError(
                "Age specification forms are mutually exclusive — provide exactly one:\n"
                "  'age_col', 'age_grp_col', or 'age_min_col'/'age_max_col'.\n"
                f"  Got: age_col={self.age_col!r}, age_grp_col={self.age_grp_col!r}, "
                f"age_min_col={self.age_min_col!r}, age_max_col={self.age_max_col!r}"
            )
        if _has_range and (self.age_min_col is None or self.age_max_col is None):
            raise ValueError(
                "Both 'age_min_col' and 'age_max_col' must be specified together.\n"
                f"  Got: age_min_col={self.age_min_col!r}, age_max_col={self.age_max_col!r}"
            )

        # Delegate column validation, NaN removal, dtype coercion, and renaming
        object.__setattr__(
            self,
            "data",
            preprocess_participant_data(
                self.data,
                self.id_col,
                self.age_col,
                self.age_min_col,
                self.age_max_col,
                self.age_grp_col,
                self.strat_var_cols,
                self.repeat_col,
                self.amb_cnt_col,
                self.aux_cols,
            ),
        )

        # Perform domain validation on the cleaned data
        self.validate()

    def validate(self) -> None:
        """
        Perform comprehensive validation of participant data.

        Validation checks:
        1. Participant IDs are unique (no duplicates)
        2. (If age_col specified) Age column contains valid non-negative numeric values
        3. (If age_grp_col specified) Age group column contains valid pd.IntervalIndex or categorical values
        4. (If repeat_col specified) Repeat interview values are non-negative integers
        5. (If amb_cnt_col specified) Ambiguous contact count values are non-negative integers

        Note: Missing value checks are handled during preprocessing in _preprocess(),
        so this method assumes clean data without NaNs.

        Raises
        ------
        KeyError
            If required columns are missing from the DataFrame.
        ValueError
            If duplicate participant IDs are found.
            If age values are invalid (negative or non-numeric).

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age')
        >>> # Validation happens automatically, but can be called explicitly:
        >>> part_data.validate()
        """
        validate_participant_data(
            self.data,
            self.age_col,
            self.age_min_col,
            self.age_max_col,
            self.age_grp_col,
            self.repeat_col,
            self.amb_cnt_col,
        )

    @property
    def n(self) -> int:
        """
        Return the number of participants in the dataset.

        Returns
        -------
        int
            Total number of participants (rows in the DataFrame).

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age')
        >>> part_data.n
        500
        """
        return len(self.data)

    @property
    def age_range(self) -> Tuple[float, float]:
        """
        Return the age range (min, max) of participants.

        Only available when using age_col (exact ages), not age_grp_col.

        Returns
        -------
        Tuple[float, float]
            Tuple of (minimum_age, maximum_age).

        Raises
        ------
        ValueError
            If age_grp_col is used instead of age_col.

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age')
        >>> part_data.age_range
        (0, 85)
        """
        if self.age_col is None:
            raise ValueError(
                "age_range is only available when using 'age_col' (exact ages).\n"
                f"Currently using 'age_grp_col': {self.age_grp_col}"
            )

        ages = self.data["part_age"]
        return (float(ages.min()), float(ages.max()))

    @property
    def strat_vars(self) -> List[str] | str:
        """
        Return list of stratification variable names.

        Returns
        -------
        List[str]
            List of stratification variable column names (empty if none).

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age', strat_var_cols=['gender', 'region'])
        >>> part_data.strat_vars
        ['gender', 'region']
        """
        return self.strat_var_cols if self.strat_var_cols else []

    def get_strat_vars(self, prefix: bool = False) -> List[str]:
        """
        Return list of stratification variable names, optionally with the ``part_`` prefix.

        Parameters
        ----------
        prefix : bool, default=False
            If True, returns stratification variable names with the ``part_`` prefix.
            If False, returns original stratification variable names.

        Returns
        -------
        List[str]
            List of stratification variable column names.

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age', strat_var_cols=['gender', 'region'])
        >>> part_data.get_strat_vars(prefix=True)
        ['part_gender', 'part_region']
        """
        if not self.strat_var_cols:
            return []

        if prefix:
            return [
                f"part_{var}" if not var.startswith("part_") else var
                for var in self.strat_var_cols
            ]
        else:
            return [var.removeprefix("part_") for var in self.strat_var_cols]

    def get_aux_cols(self, prefix: bool = True) -> List[str]:
        """
        Return auxiliary column names.

        Parameters
        ----------
        prefix : bool, default=True
            If True, return the names as they appear in ``data`` (with the
            ``part_`` prefix). If False, return the names without the prefix.

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age', aux_cols='dayofweek')
        >>> part_data.get_aux_cols()
        ['part_dayofweek']
        """
        names = [c if c.startswith("part_") else f"part_{c}" for c in self.aux_cols]
        return names if prefix else [c.removeprefix("part_") for c in names]

    def get_strat_var_schema(self) -> Dict[str, Dict[str, List[str | int]]]:
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
        if self.strat_var_cols is not None:
            schema = {}
            for var in self.strat_var_cols:
                col = f"part_{var}" if not var.startswith("part_") else var
                if col in self.data.columns:
                    categories = self.data[col].cat.categories.tolist()
                    codes = sorted(self.data[col].cat.codes.unique().tolist())
                    base = var.removeprefix("part_")
                    schema[base] = {"categories": categories, "codes": codes}

            return schema
        else:
            return {}

    def get_sample_sizes(self, stratify=False) -> pd.DataFrame:
        """
        Return a DataFrame with sample sizes per stratification group.

        Parameters
        ----------
        stratify : bool, default=False
            If True, returns counts of participants stratified by all stratification variables.
            If False, returns counts of participants stratified by age only.

        Returns
        -------
        pd.DataFrame
            DataFrame with counts of participants per stratification group.
            If stratify=False, returns a DataFrame where counts of participants are stratified by age only.

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age')
        >>> sample_sizes = part_data.get_sample_sizes()
        >>> print(sample_sizes.head())
        age
        0     15
        1     18
        2     20
        ...
        """
        age_column = "part_age" if self.age_col else "part_age_grp"
        if stratify and self.strat_var_cols:
            group_cols = [age_column] + [f"part_{var}" for var in self.strat_var_cols]
            return (
                self.data.groupby(group_cols, observed=False)
                .agg(N=("id", "count"))
                .reset_index()
            )

        return (
            self.data.groupby(age_column, observed=False)
            .agg(N=("id", "count"))
            .reset_index()
        )

    def summary(self) -> Dict[str, Any]:
        """
        Return summary statistics about the participant data.

        Provides a comprehensive overview including sample size, age statistics
        (if applicable), and stratification information.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - n_participants: Total number of participants
            - id_col: Name of ID column
            - age_col: Name of age column (or None)
            - age_grp_col: Name of age group column (or None)
            - age_range: Tuple of (min, max) age if using age_col
            - strat_vars: List of stratification variables

        Examples
        --------
        >>> part_data = ParticipantData(df, 'id', age_col='age', strat_var_cols='gender')
        >>> summary = part_data.summary()
        >>> print(summary)
        {
            'n_participants': 500,
            'id_col': 'id',
            'age_col': 'age',
            'age_grp_col': None,
            'age_range': (0, 85),
            'strat_vars': ['gender']
        }
        """
        summary_dict = {
            "n": self.n,
            "id_col": self.id_col,
            "age_col": self.age_col,
            "age_grp_col": self.age_grp_col,
            "strat_vars": self.strat_vars,
        }

        # Add age range if using exact ages
        if self.age_col:
            summary_dict["age_range"] = self.age_range

        return summary_dict
