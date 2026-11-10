"""
Age Group Processing Utilities

This module provides utilities for age binning and adaptive merging of age
groups. It lives in ``cntmosaic.utils`` (not ``models.classical``) so that
participant-cell construction can use it without importing the models package.
"""

from typing import List

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._AgeGroupSpecs import AgeGroupSpecs


class AgeBinProcessor:
    """Handles age binning and zero-group merging."""

    def __init__(self, age_group_specs: AgeGroupSpecs):
        """
        Initialize age bin processor.

        Parameters
        ----------
        age_group_specs : AgeGroupSpecs
            Age stratification bins
        """
        self.age_group_specs = age_group_specs

    def assign_age_groups(
        self, df: pd.DataFrame, age_col: str, group_col: str
    ) -> pd.DataFrame:
        """
        Assign age groups to dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with age column
        age_col : str
             Name of age column
        group_col : str
            Name for new age group column

        Returns
        -------
        pd.DataFrame
            Dataframe with age group column added
        """
        df = df.copy()
        bins = self.age_group_specs.left + [self.age_group_specs.right[-1] + 1]

        df[group_col] = pd.cut(df[age_col], bins=bins, right=False, include_lowest=True)

        return df

    @staticmethod
    def merge_zero_groups(
        intervals: List[pd.Interval], counts: NDArray[np.int64]
    ) -> List[pd.Interval]:
        """
        Merge zero-count intervals with adjacent non-zero intervals.

        Strategy:
        1. Zero-count groups are merged with the previous non-zero group if available
        2. Leading zero-count groups are merged with the first non-zero group
        3. If all groups are zero, raise an error

        Parameters
        ----------
        intervals : list of pd.Interval
            Sorted list of age intervals
        counts : NDArray
            Sample counts for each interval

        Returns
        -------
        list of pd.Interval
            Merged intervals

        Raises
        ------
        ValueError
            If all counts are zero

        Examples
        --------
        >>> intervals = [pd.Interval(0, 5), pd.Interval(5, 10),
        ...              pd.Interval(10, 15), pd.Interval(15, 20)]
        >>> counts = np.array([10, 0, 15, 0])
        >>> merged = AgeBinProcessor.merge_zero_groups(intervals, counts)
        >>> len(merged)
        2
        """
        if not np.any(counts > 0):
            raise ValueError(
                "All age groups have zero participants. Cannot create contact matrices."
            )

        merged = []
        n = len(intervals)
        i = 0

        while i < n:
            if counts[i] > 0:
                # Non-zero count: start new merged interval
                merged.append(intervals[i])
                i += 1
            else:
                # Zero count: merge with previous or look ahead
                if merged:
                    # Extend the last merged interval
                    prev = merged[-1]
                    merged[-1] = pd.Interval(
                        left=prev.left, right=intervals[i].right, closed="left"
                    )
                    i += 1
                else:
                    # No previous interval: find first non-zero and merge all leading zeros
                    j = i
                    while j < n and counts[j] == 0:
                        j += 1

                    if j >= n:
                        # Should never happen due to earlier check, but be defensive
                        raise ValueError("No non-zero age groups found")

                    # Create merged interval from first zero to first non-zero
                    merged.append(
                        pd.Interval(
                            left=intervals[i].left,
                            right=intervals[j].right,
                            closed="left",
                        )
                    )
                    i = j + 1

        return merged

    @staticmethod
    def reassign_age_groups(
        df: pd.DataFrame,
        age_grp_col: str,
        merged_intervals: List[pd.Interval],
        new_group_col: str = None,
    ) -> pd.DataFrame:
        """
        Reassign age groups based on merged intervals from merge_zero_groups.

        This method takes a dataframe with existing age group assignments and
        remaps them to the new merged age groups. Original intervals that were
        merged together will be assigned to the same new group.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe with existing age group column
        age_grp_col : str
            Name of existing age group column (contains pd.Interval objects)
        merged_intervals : list of pd.Interval
            Output from merge_zero_groups - the new merged intervals
        new_group_col : str, optional
            Name for new age group column. If None, overwrites age_grp_col

        Returns
        -------
        pd.DataFrame
            Dataframe with reassigned age groups

        Examples
        --------
        >>> # Original dataframe with fine-grained age groups
        >>> df = pd.DataFrame({
        ...     'id': [1, 2, 3, 4],
        ...     'age_grp': [pd.Interval(0, 5), pd.Interval(5, 10),
        ...                 pd.Interval(10, 15), pd.Interval(15, 20)]
        ... })
        >>>
        >>> # Merge zero groups
        >>> original_intervals = df['age_grp'].unique()
        >>> counts = np.array([2, 0, 1, 0])
        >>> merged = AgeBinProcessor.merge_zero_groups(original_intervals, counts)
        >>> # merged = [Interval(0, 10), Interval(10, 15)]
        >>>
        >>> # Reassign based on merged intervals
        >>> df_new = AgeBinProcessor.reassign_age_groups(
        ...     df, 'age_grp', merged, 'age_grp_merged'
        ... )
        >>> df_new['age_grp_merged'].unique()
        array([Interval(0, 10, closed='left'), Interval(10, 20, closed='left')])

        Notes
        -----
        The reassignment works by checking which merged interval contains each
        original interval. An original interval is contained in a merged interval
        if its left boundary is >= merged.left and its right boundary <= merged.right.
        """
        df = df.copy()

        # Determine output column name
        output_col = new_group_col if new_group_col is not None else age_grp_col

        # Get original intervals from the dataframe
        original_intervals = df[age_grp_col]

        # Create mapping from original intervals to merged intervals
        def find_merged_interval(original_interval):
            """
            Find which merged interval contains the original interval.

            An original interval is contained in a merged interval if:
            - original.left >= merged.left
            - original.right <= merged.right
            """
            if pd.isna(original_interval):
                return pd.NA

            for merged_interval in merged_intervals:
                # Check if original interval is contained within merged interval
                if (
                    original_interval.left >= merged_interval.left
                    and original_interval.right <= merged_interval.right
                ):
                    return merged_interval

            # If no match found, this shouldn't happen with valid data
            # but handle gracefully
            raise ValueError(
                f"Original interval {original_interval} does not fit within "
                f"any merged interval. This may indicate inconsistent age bins."
            )

        # Apply the mapping to create new column
        df[output_col] = original_intervals.apply(find_merged_interval)

        # Convert to categorical with ordered categories
        df[output_col] = pd.Categorical(
            df[output_col], categories=merged_intervals, ordered=True
        )

        return df

    @staticmethod
    def create_interval_mapping(
        original_intervals: List[pd.Interval], merged_intervals: List[pd.Interval]
    ) -> dict:
        """
        Create explicit mapping from original intervals to merged intervals.

        This is a helper method that can be useful for understanding how
        intervals were merged or for applying the same mapping to multiple
        dataframes.

        Parameters
        ----------
        original_intervals : list of pd.Interval
            Original fine-grained intervals
        merged_intervals : list of pd.Interval
            Merged intervals from merge_zero_groups

        Returns
        -------
        dict
            Mapping from original intervals to merged intervals

        Examples
        --------
        >>> original = [pd.Interval(0, 5), pd.Interval(5, 10),
        ...             pd.Interval(10, 15)]
        >>> merged = [pd.Interval(0, 10), pd.Interval(10, 15)]
        >>> mapping = AgeBinProcessor.create_interval_mapping(original, merged)
        >>> mapping[pd.Interval(0, 5)]
        Interval(0, 10, closed='left')
        >>> mapping[pd.Interval(5, 10)]
        Interval(0, 10, closed='left')
        """
        mapping = {}

        for orig_interval in original_intervals:
            for merged_interval in merged_intervals:
                if (
                    orig_interval.left >= merged_interval.left
                    and orig_interval.right <= merged_interval.right
                ):
                    mapping[orig_interval] = merged_interval
                    break
            else:
                raise ValueError(
                    f"Original interval {orig_interval} does not fit within "
                    f"any merged interval."
                )

        return mapping

    @staticmethod
    def _merge_smallest_age_group(
        intervals: List[pd.Interval], counts: NDArray[np.int64]
    ) -> List[pd.Interval]:
        """
        Merge the smallest age group with its neighbor.

        Strategy:
        1. Find the age group with the smallest count
        2. Merge it with the neighbor that has the smaller count
           (prefer left neighbor if tied or at boundary)
        3. Return new merged intervals

        Parameters
        ----------
        intervals : list of pd.Interval
            Sorted list of age intervals
        counts : NDArray
            Sample counts for each interval

        Returns
        -------
        list of pd.Interval
            Intervals with smallest group merged

        Examples
        --------
        >>> intervals = [pd.Interval(0, 5), pd.Interval(5, 10),
        ...              pd.Interval(10, 15), pd.Interval(15, 20)]
        >>> counts = np.array([10, 2, 15, 8])
        >>> merged = AgeBinProcessor._merge_smallest_age_group(intervals, counts)
        >>> # Interval(5, 10) merged with neighbor
        """
        if len(intervals) <= 1:
            raise ValueError("Cannot merge - only one age group remains")

        # Find index of smallest group
        smallest_idx = np.argmin(counts)

        # Determine merge direction
        if smallest_idx == 0:
            # First group: merge with right neighbor
            merged = [
                pd.Interval(
                    left=intervals[0].left, right=intervals[1].right, closed="left"
                )
            ] + intervals[2:]
        elif smallest_idx == len(intervals) - 1:
            # Last group: merge with left neighbor
            merged = intervals[:-2] + [
                pd.Interval(
                    left=intervals[-2].left, right=intervals[-1].right, closed="left"
                )
            ]
        else:
            # Middle group: merge with smaller neighbor
            left_count = counts[smallest_idx - 1]
            right_count = counts[smallest_idx + 1]

            if left_count <= right_count:
                # Merge with left neighbor
                merged = (
                    intervals[: smallest_idx - 1]
                    + [
                        pd.Interval(
                            left=intervals[smallest_idx - 1].left,
                            right=intervals[smallest_idx].right,
                            closed="left",
                        )
                    ]
                    + intervals[smallest_idx + 1 :]
                )
            else:
                # Merge with right neighbor
                merged = (
                    intervals[:smallest_idx]
                    + [
                        pd.Interval(
                            left=intervals[smallest_idx].left,
                            right=intervals[smallest_idx + 1].right,
                            closed="left",
                        )
                    ]
                    + intervals[smallest_idx + 2 :]
                )

        return merged
