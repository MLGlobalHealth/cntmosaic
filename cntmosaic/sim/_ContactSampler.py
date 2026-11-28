from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .._types import StratumLabel  # noqa: F401


class ContactSampler:
    """
    Generate synthetic contact count data from contact intensity matrices and participant data.

    This class simulates contact counts for participants based on contact intensity matrices,
    supporting both Poisson and Negative Binomial sampling distributions with optional
    individual random effects.

    The generator works with outputs from:
    - ParticipantSampler: provides participant demographics (df_part)
    - MatrixSampler: provides contact intensity matrices (cint_matrices)

    Three modes of operation:
    1. Single population: One matrix for all participants (key: "All->All")
    2. Partial case: One matrix per stratum to general population (keys: "{label}->All")
    3. Full case: All pairwise stratum matrices (keys: "{source}->{target}")

    Examples
    --------
    >>> import numpy as np
    >>> from cntmosaic.sim import (
    ...     Stratification, Population, MatrixSampler,
    ...     ParticipantSampler, ContactSampler
    ... )
    >>> from cntmosaic.datasets import load_template_patterns

    **Example 1: Single population**

    >>> templates = load_template_patterns('United_States', max_age=50)
    >>> n_ages = templates['household'].shape[0]
    >>> ref_age_dist = np.random.rand(n_ages) * 1000
    >>>
    >>> # Create stratification and population
    >>> strat = Stratification('group', 1, ref_age_dist, labels=['All'], seed=42)
    >>> pop = Population(strat)
    >>>
    >>> # Generate participants and matrix
    >>> part_gen = ParticipantSampler(pop, n_participants=100)
    >>> participants = part_gen.sample(seed=42)
    >>>
    >>> mat_gen = MatrixSampler(templates)
    >>> matrices = mat_gen.generate_single(pop, mean_intensity=15.0, seed=42)
    >>>
    >>> # Generate contacts
    >>> contact_gen = ContactSampler(participants, matrices)
    >>> contacts = contact_gen.sample(seed=42)
    >>> print(contacts.head())
       id  cnt_age   y
    0   1        2  12
    1   1        3   8
    2   1        4   3
    3   2        1   5
    4   2        3  15

    **Example 2: Multiple strata (Partial case)**

    >>> # Create stratification
    >>> region_strat = Stratification(
    ...     'region', 2, ref_age_dist, labels=['Urban', 'Rural'], seed=42
    ... )
    >>> pop = Population(region_strat)
    >>>
    >>> # Generate participants and matrices
    >>> part_gen = ParticipantSampler(pop, n_participants=150)
    >>> participants = part_gen.sample(seed=42)
    >>>
    >>> mat_gen = MatrixSampler(templates)
    >>> matrices = mat_gen.generate_partial(pop, mean_intensity=15.0, seed=42)
    >>> # matrices = {"Urban->All": M_urban, "Rural->All": M_rural}
    >>>
    >>> contact_gen = ContactSampler(participants, matrices)
    >>> contacts = contact_gen.sample(seed=42)
    >>> print(contacts.head())
       id  cnt_age   y
    0   1        2  15
    1   1        3   9
    2   2        1   6
    3   3        0  12
    4   3        2   8
    >>> # Note: No 'region' column because contacts are with general population

    **Example 3: Full case (All stratum interactions)**

    >>> # Create stratification
    >>> region_strat = Stratification(
    ...     'region', 2, ref_age_dist, labels=['Urban', 'Rural'], seed=42
    ... )
    >>> pop = Population(region_strat)
    >>>
    >>> # Generate participants and matrices
    >>> part_gen = ParticipantSampler(pop, n_participants=150)
    >>> participants = part_gen.sample(seed=42)
    >>>
    >>> mat_gen = MatrixSampler(templates)
    >>> matrices = mat_gen.generate_full(pop, mean_intensity=15.0, seed=42)
    >>> # matrices = {"Urban->Urban": M_uu, "Urban->Rural": M_ur,
    >>> #            "Rural->Urban": M_ru, "Rural->Rural": M_rr}
    >>>
    >>> contact_gen = ContactSampler(participants, matrices)
    >>> contacts = contact_gen.sample(seed=42)
    >>> print(contacts.head())
       id  cnt_age  cnt_region   y
    0   1        2       Urban  12
    1   1        3       Urban   8
    2   1        1       Rural   3
    3   2        0       Urban   7
    4   2        2       Rural   5
    >>>
    >>> # Urban participants have contacts with both Urban and Rural strata
    >>> print(contacts[contacts['id'] == 1]['cnt_region'].value_counts())
    cnt_region
    Urban    3
    Rural    2
    """

    ALLOWED_MODELS = ["poisson", "negbin"]

    def __init__(
        self,
        df_part: pd.DataFrame,
        cint_matrices: Dict[str, NDArray],
        model: str = "poisson",
        odisp: float | None = None,
        random_effects: bool = False,
        random_effects_shape: float = 5.0,
        random_effects_rate: float = 5.0,
    ):
        """
        Initialize ContactSampler with participant data and contact matrices.

        Parameters
        ----------
        df_part : pd.DataFrame
                Participant data from ParticipantSampler.generate().
                Must contain columns:
                - 'id': Unique participant identifier
                - 'age': Participant age (continuous)
                - Stratification columns (e.g., 'gender', 'region') if using stratified matrices

        cint_matrices : dict of str to NDArray
                Contact intensity matrices from MatrixSampler.
                Dictionary mapping string keys to contact matrices:
                - Single case: {"All->All": matrix}
                - Partial case: {"{label}->All": matrix} for each stratum
                  (e.g., {"Urban->All": M_urban, "Rural->All": M_rural})
                - Full case: {"{source}->{target}": matrix} for all stratum pairs
                  (e.g., {"Urban->Urban": M_uu, "Urban->Rural": M_ur, ...})

        model : str, default='poisson'
                Contact count sampling distribution. Options:
                - 'poisson': Poisson distribution (equidispersed)
                - 'negbin': Negative binomial distribution (overdispersed)

        odisp : float, optional
                Overdispersion parameter for negative binomial model.
                Required if model='negbin'. Higher values = more overdispersion.
                Typical range: 0.1 to 2.0

        random_effects : bool, default=False
                Whether to include individual-level random effects.
                If True, multiplies contact intensities by Gamma(shape, rate) random variates
                to model heterogeneity in individual contact behavior.

        random_effects_shape : float, default=5.0
                Shape parameter for Gamma distribution of random effects.
                Mean of random effects = shape / rate.

        random_effects_rate : float, default=5.0
                Rate parameter for Gamma distribution of random effects.
                With shape=rate=5, mean=1 and variance=0.2.

        Raises
        ------
        ValueError
                If model is not in ALLOWED_MODELS
                If odisp is not provided when model='negbin'
                If cint_matrices structure doesn't match df_part subgroups

        Notes
        -----
        For the full case (generate_full output), only within-subgroup contacts are
        generated, using the diagonal matrices M_{k,k}. To generate between-subgroup
        contacts, use the matrices with different source and target indices.
        """
        if model not in self.ALLOWED_MODELS:
            raise ValueError(
                f"Model '{model}' is not supported. "
                f"Allowed models: {self.ALLOWED_MODELS}"
            )

        if model == "negbin" and odisp is None:
            raise ValueError(
                "Overdispersion parameter 'odisp' must be provided for negative binomial model."
            )

        self.df_part = df_part.copy()
        self.cint_matrices = cint_matrices
        self.model = model
        self.odisp = odisp
        self.random_effects = random_effects
        self.random_effects_shape = random_effects_shape
        self.random_effects_rate = random_effects_rate

        # Validate and parse matrix structure
        self._validate_matrices()

    def _validate_matrices(self) -> None:
        """Validate that matrices match participant data structure."""
        if not isinstance(self.cint_matrices, dict):
            raise TypeError(
                f"cint_matrices must be a dictionary with string keys, got {type(self.cint_matrices)}"
            )

        if len(self.cint_matrices) == 0:
            raise ValueError("cint_matrices cannot be empty")

        # Check if all keys are strings
        if not all(isinstance(k, str) for k in self.cint_matrices.keys()):
            raise TypeError("All keys in cint_matrices must be strings")

        # Determine case based on key format
        first_key = next(iter(self.cint_matrices.keys()))

        # Single population case: "All->All"
        if first_key == "All->All":
            if len(self.cint_matrices) != 1:
                raise ValueError(
                    "Single population case should have exactly one key: 'All->All'"
                )
            self.is_full_case = False
            self.is_single_case = True
            self.strat_column = None
            self.strat_columns = []  # No stratification columns
            return

        # Check for partial case: "{label}->All"
        if "->All" in first_key:
            self.is_full_case = False
            self.is_single_case = False

            # Extract stratum labels from keys
            matrix_labels = set(
                key.replace("->All", "") for key in self.cint_matrices.keys()
            )

            # Find stratification column in df_part
            # Look for columns that match the extracted labels
            strat_cols = [
                col
                for col in self.df_part.columns
                if col not in ["id", "age", "age_group"]
            ]

            if len(strat_cols) == 0:
                raise ValueError(
                    "Partial case matrices provided but df_part has no stratification columns"
                )

            # For multi-stratification, labels are combined with underscores
            # We need to find which combination of columns produces these labels
            found_match = False
            for col in strat_cols:
                part_labels = set(self.df_part[col].astype(str).unique())
                if part_labels == matrix_labels:
                    self.strat_column = col
                    self.strat_columns = [col]  # Single stratification
                    self.strat_labels = sorted(matrix_labels)
                    found_match = True
                    break

            # Try multi-stratification (combined labels)
            if not found_match and len(strat_cols) > 1:
                # Create combined labels from df_part
                df_temp = self.df_part.copy()
                combined_labels = df_temp[strat_cols].astype(str).agg("_".join, axis=1)
                part_labels_combined = set(combined_labels.unique())

                if part_labels_combined == matrix_labels:
                    self.strat_column = "_".join(strat_cols)
                    self.strat_columns = strat_cols  # Multiple stratifications
                    self.strat_labels = sorted(matrix_labels)
                    # Add combined column to df_part for easier processing
                    self.df_part[self.strat_column] = combined_labels
                    found_match = True

            if not found_match:
                raise ValueError(
                    f"Cannot match matrix labels {matrix_labels} with participant stratifications. "
                    f"Available columns: {strat_cols}"
                )
            return

        # Full case: "{source}->{target}"
        if "->" in first_key and "->All" not in first_key:
            self.is_full_case = True
            self.is_single_case = False

            # Extract unique stratum labels from keys
            all_labels = set()
            for key in self.cint_matrices.keys():
                source, target = key.split("->", 1)
                all_labels.add(source)
                all_labels.add(target)

            # Find stratification column(s) in df_part
            strat_cols = [
                col
                for col in self.df_part.columns
                if col not in ["id", "age", "age_group"]
            ]

            if len(strat_cols) == 0:
                raise ValueError(
                    "Full case matrices provided but df_part has no stratification columns"
                )

            # Try to match labels
            found_match = False
            for col in strat_cols:
                part_labels = set(self.df_part[col].astype(str).unique())
                if part_labels == all_labels:
                    self.strat_column = col
                    self.strat_columns = [col]  # Single stratification
                    self.strat_labels = sorted(all_labels)
                    found_match = True
                    break

            # Try multi-stratification
            if not found_match and len(strat_cols) > 1:
                df_temp = self.df_part.copy()
                combined_labels = df_temp[strat_cols].astype(str).agg("_".join, axis=1)
                part_labels_combined = set(combined_labels.unique())

                if part_labels_combined == all_labels:
                    self.strat_column = "_".join(strat_cols)
                    self.strat_columns = strat_cols  # Multiple stratifications
                    self.strat_labels = sorted(all_labels)
                    self.df_part[self.strat_column] = combined_labels
                    found_match = True

            if not found_match:
                raise ValueError(
                    f"Cannot match matrix labels {all_labels} with participant stratifications. "
                    f"Available columns: {strat_cols}"
                )

            # Validate all stratum pairs exist
            for source in self.strat_labels:
                for target in self.strat_labels:
                    key = f"{source}->{target}"
                    if key not in self.cint_matrices:
                        raise ValueError(
                            f"Missing matrix for {key} in full case matrices"
                        )
            return

        raise ValueError(
            f"Cannot determine matrix case from key format: {first_key}. "
            "Expected 'All->All', '{label}->All', or '{source}->{target}'"
        )

    def _generate_single(
        self,
        df_part: pd.DataFrame,
        cint_matrix: NDArray,
        rng: np.random.Generator,
        target_label: str = None,
        strat_col_name: str = None,
    ) -> pd.DataFrame:
        """
        Generate contact counts for a single group of participants.

        Parameters
        ----------
        df_part : pd.DataFrame
                Participant data for this group
        cint_matrix : NDArray
                Contact intensity matrix (A×A)
        rng : np.random.Generator
                Random number generator
        target_label : str, optional
                Stratum label of contacts (for full case only)
                If None, contacts are with general population (single or partial case)
        strat_col_name : str, optional
                Name of stratification column for output (for full case only)

        Returns
        -------
        pd.DataFrame
                Contact data with columns:
                - ['id', 'cnt_age', 'y'] for single or partial case
                - ['id', 'cnt_age', 'cnt_{strat_col_name}', 'y'] for full case
        """
        # Get age groups for all participants
        age_groups = df_part["age"].astype(int).values

        # Extract contact intensities for each participant's age group
        # lambda_[i, j] = expected contacts from participant i to age group j
        lambda_ = cint_matrix[age_groups, :]

        # Apply individual random effects if requested
        if self.random_effects:
            gamma_effects = rng.gamma(
                shape=self.random_effects_shape,
                scale=1.0 / self.random_effects_rate,
                size=lambda_.shape,
            )
            lambda_ = lambda_ * gamma_effects

        # Sample contact counts
        if self.model == "poisson":
            samples = rng.poisson(lambda_)

        elif self.model == "negbin":
            # Negative binomial parameterization: n_success, p_success
            # Mean = n * (1-p) / p, where we want Mean = lambda_
            # With overdispersion odisp, n = 1/odisp
            n_success = 1.0 / self.odisp
            p_success = n_success / (n_success + lambda_)
            samples = rng.negative_binomial(n_success, p_success)

        # Convert to long format: one row per (participant, age_group) pair with contact
        data = []
        participant_ids = df_part["id"].values

        for i, participant_id in enumerate(participant_ids):
            # Find age groups where this participant has contacts
            age_groups_with_contacts = np.where(samples[i, :] > 0)[0]

            for age_cnt in age_groups_with_contacts:
                contact_count = samples[i, age_cnt]

                # Determine columns based on case
                if target_label is not None:
                    # Full case: include stratification column(s)
                    # For multi-stratification, split combined label
                    if "_" in target_label and len(self.strat_columns) > 1:
                        # Multi-stratification: split label and create separate columns
                        label_parts = target_label.split("_")
                        row_data = (
                            [participant_id, age_cnt] + label_parts + [contact_count]
                        )
                    else:
                        # Single stratification
                        row_data = [
                            participant_id,
                            int(age_cnt),
                            target_label,
                            contact_count,
                        ]
                    data.append(row_data)
                else:
                    # Single or partial case: no stratum info
                    data.append([participant_id, age_cnt, contact_count])

        # Create DataFrame
        if target_label is not None:
            # For multi-stratification, create separate columns
            if len(self.strat_columns) > 1:
                strat_cnt_cols = [f"cnt_{col}" for col in self.strat_columns]
                columns = ["id", "cnt_age"] + strat_cnt_cols + ["y"]
            else:
                # Single stratification
                strat_cnt_col = (
                    f"cnt_{strat_col_name}" if strat_col_name else "cnt_stratum"
                )
                columns = ["id", "cnt_age", strat_cnt_col, "y"]
        else:
            columns = ["id", "cnt_age", "y"]

        df = pd.DataFrame(data, columns=columns)

        # Ensure integer types for id, cnt_age, and y columns
        # This is important when data is empty or when pd.concat mixes types
        if len(df) > 0:
            df["cnt_age"] = df["cnt_age"].astype(int)
            df["y"] = df["y"].astype(int)

        return df

    def sample(self, seed: int | None = None) -> pd.DataFrame:
        """
        Sample contact count data for all participants.

        Parameters
        ----------
        seed : int, optional
                Random seed for reproducibility

        Returns
        -------
        pd.DataFrame
                Contact data with columns:
                - 'id': Participant identifier (matches df_part)
                - 'cnt_age': Age group of contact
                - 'y': Number of contacts with this age group
                - 'cnt_{strat}': Contact stratum for each stratification variable
                  (only for full case; column name matches the stratification, e.g.
                  'cnt_region' for a 'region' stratification)

                Each row represents contacts between a participant and an age group.
                Only non-zero contact counts are included.

                Note: For partial case, there is no 'cnt_{strat}' column because
                contacts are with the general population, not specific subgroups.

        Examples
        --------
        >>> # Single population
        >>> contacts = contact_gen.sample(seed=42)
        >>> print(contacts.head())
           id  cnt_age   y
        0   1        2  12
        1   1        3   8
        2   2        1   5

        >>> # Partial case (multiple subgroups, contacts with general population)
        >>> contacts = contact_gen.sample(seed=42)
        >>> print(contacts.head())
           id  cnt_age   y
        0   1        2  15
        1   1        3   9
        2   2        1   6
        3   3        0  12
        4   3        2   8

        >>> # Full case (all stratum interactions, e.g. 'region' stratification)
        >>> contacts = contact_gen.sample(seed=42)
        >>> print(contacts.head())
           id  cnt_age  cnt_region   y
        0   1        2       Urban  12
        1   1        3       Urban   8
        2   1        1       Rural   3
        3   2        0       Urban   7
        4   2        2       Rural   5

        Notes
        -----
        For participants with many contacts, the output can be large. Consider
        aggregating by age group or filtering rare contact patterns for analysis.

        In the full case, each participant can have contacts with all strata.
        For example, an Urban participant will have contacts generated using both
        the Urban->Urban and Urban->Rural matrices.
        """
        rng = np.random.default_rng(seed)

        # Single population case
        if self.is_single_case:
            return self._generate_single(
                self.df_part,
                self.cint_matrices["All->All"],
                rng,
                target_label=None,
                strat_col_name=None,
            )

        # Partial case - stratified participants, contacts with general population
        if not self.is_full_case:
            dfs = []
            for label in self.strat_labels:
                # Filter participants for this stratum
                df_part_sub = self.df_part[
                    self.df_part[self.strat_column] == label
                ].copy()

                # Generate contacts for this stratum (with general population)
                df_contacts = self._generate_single(
                    df_part_sub,
                    self.cint_matrices[f"{label}->All"],
                    rng,
                    target_label=None,
                    strat_col_name=None,
                )
                dfs.append(df_contacts)

            # Combine all strata
            return pd.concat(dfs, ignore_index=True)

        # Full case - all stratum pair interactions
        dfs = []
        for source_label in self.strat_labels:
            # Filter participants for this source stratum
            df_part_source = self.df_part[
                self.df_part[self.strat_column] == source_label
            ].copy()

            # Generate contacts with each target stratum
            for target_label in self.strat_labels:
                # Get the appropriate contact matrix
                matrix = self.cint_matrices[f"{source_label}->{target_label}"]

                # Generate contacts from source to target
                df_contacts = self._generate_single(
                    df_part_source,
                    matrix,
                    rng,
                    target_label=target_label,
                    strat_col_name=self.strat_column,
                )

                dfs.append(df_contacts)

        # Combine all source-target pairs
        return pd.concat(dfs, ignore_index=True)

    def get_contact_matrix_empirical(
        self, contacts: pd.DataFrame = None, normalize: bool = False
    ) -> dict[str, NDArray]:  # keys are "source->target" matrix labels
        """
        Calculate empirical contact matrix from generated contact data.

        This computes the actual contact patterns observed in the generated data,
        which can be compared to the input contact intensity matrices.

        Parameters
        ----------
        contacts : pd.DataFrame, optional
                Contact data from sample(). If None, will call sample() with default seed.
        normalize : bool, default=False
                If True, normalize by the number of participants in each age group

        Returns
        -------
        dict of str to NDArray
                Empirical contact matrix/matrices where element [i,j] represents
                total (or average if normalized) contacts from age group i to age group j

                - For single case: {"All->All": matrix}
                - For partial case: {"{label}->All": matrix} for each stratum
                - For full case: {"{source}->{target}": matrix} for all stratum pairs

        Examples
        --------
        >>> # Single case
        >>> contacts = contact_gen.sample(seed=42)
        >>> empirical = contact_gen.get_contact_matrix_empirical(contacts, normalize=True)
        >>> matrix = empirical["All->All"]
        >>> print(f"Empirical contacts from age 0 to age 1: {matrix[0, 1]:.2f}")

        >>> # Partial case
        >>> contacts = contact_gen.sample(seed=42)
        >>> empirical = contact_gen.get_contact_matrix_empirical(contacts, normalize=True)
        >>> urban_matrix = empirical['Urban->All']
        >>> print(f"Urban contacts (age 0→1): {urban_matrix[0, 1]:.2f}")

        >>> # Full case
        >>> contacts = contact_gen.sample(seed=42)
        >>> empirical = contact_gen.get_contact_matrix_empirical(contacts, normalize=True)
        >>> urban_to_rural = empirical['Urban->Rural']
        >>> print(f"Urban to rural contacts (age 0→1): {urban_to_rural[0, 1]:.2f}")
        """
        if contacts is None:
            contacts = self.sample()

        # Determine number of age groups from contact matrix
        first_matrix = next(iter(self.cint_matrices.values()))
        n_age_groups = first_matrix.shape[0]

        # Single population case
        if self.is_single_case:
            # Create empirical matrix
            empirical = np.zeros((n_age_groups, n_age_groups))

            # Merge with participant ages
            contacts_with_age = contacts.merge(self.df_part[["id", "age"]], on="id")

            # Aggregate contacts
            for _, row in contacts_with_age.iterrows():
                age_from = int(row["age"])
                age_to = int(row["cnt_age"])
                empirical[age_from, age_to] += row["y"]

            # Normalize if requested
            if normalize:
                age_counts = self.df_part["age"].value_counts().sort_index()
                for i in range(n_age_groups):
                    if i in age_counts.index and age_counts[i] > 0:
                        empirical[i, :] /= age_counts[i]

            return {"All->All": empirical}

        # Full case - create matrices for each (source, target) pair
        if self.is_full_case:
            empirical_matrices = {}

            for source_label in self.strat_labels:
                for target_label in self.strat_labels:
                    # Create empirical matrix for this pair
                    empirical = np.zeros((n_age_groups, n_age_groups))

                    # Filter participants from source subgroup
                    df_part_source = self.df_part[
                        self.df_part[self.strat_column] == source_label
                    ]

                    # Filter contacts to target subgroup based on stratification structure
                    if len(self.strat_columns) > 1:
                        # Multi-stratification: filter by each column
                        target_parts = target_label.split("_")
                        contacts_pair = contacts.copy()
                        for i, col in enumerate(self.strat_columns):
                            col_cnt = f"cnt_{col}"
                            contacts_pair = contacts_pair[
                                contacts_pair[col_cnt] == target_parts[i]
                            ]
                    else:
                        # Single stratification
                        strat_col_name = f"cnt_{self.strat_column}"
                        contacts_pair = contacts[
                            contacts[strat_col_name] == target_label
                        ]

                    # Merge with participant ages
                    contacts_with_age = contacts_pair.merge(
                        df_part_source[["id", "age"]], on="id"
                    )

                    # Aggregate contacts
                    for _, row in contacts_with_age.iterrows():
                        age_from = int(row["age"])
                        age_to = int(row["cnt_age"])
                        empirical[age_from, age_to] += row["y"]

                    # Normalize if requested
                    if normalize:
                        age_counts = df_part_source["age"].value_counts().sort_index()
                        for i in range(n_age_groups):
                            if i in age_counts.index and age_counts[i] > 0:
                                empirical[i, :] /= age_counts[i]

                    empirical_matrices[f"{source_label}->{target_label}"] = empirical

            return empirical_matrices

        # Partial case - create matrices for each subgroup
        empirical_matrices = {}

        for label in self.strat_labels:
            # Filter participants for this subgroup
            df_part_sub = self.df_part[self.df_part[self.strat_column] == label]

            # Filter contacts for this subgroup
            # Note: In partial case, we need to get contacts for participants in this subgroup
            contacts_sub = contacts[contacts["id"].isin(df_part_sub["id"])]

            # Create empirical matrix
            empirical = np.zeros((n_age_groups, n_age_groups))

            # Merge with participant ages
            contacts_with_age = contacts_sub.merge(df_part_sub[["id", "age"]], on="id")

            # Aggregate contacts
            for _, row in contacts_with_age.iterrows():
                age_from = int(row["age"])
                age_to = int(row["cnt_age"])
                empirical[age_from, age_to] += row["y"]

            # Normalize if requested
            if normalize:
                age_counts = df_part_sub["age"].value_counts().sort_index()
                for i in range(n_age_groups):
                    if i in age_counts.index and age_counts[i] > 0:
                        empirical[i, :] /= age_counts[i]

            empirical_matrices[f"{label}->All"] = empirical

        return empirical_matrices
