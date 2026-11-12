"""
Thin orchestrator for contact survey data loading.

Internal API — not exported from ``cntmosaic.dataloader``. Provides
``ContactSurveyLoader``, the preferred entry point replacing the
``DataLoader`` / ``BaseLoader`` hierarchy. It delegates all heavy lifting
to the free functions in ``_observation`` and ``_stratification``.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from .._types import StratMode
from ._ColumnSpec import ColumnSpec
from ._DataFrameSurveySource import DataFrameSurveySource
from ._observation import (
    align_age_range,
    build_contact_counts,
    build_contact_offsets,
    build_observation_grid,
    build_participant_counts,
    construct_log_P,
)
from ._stratification import assemble_strat_kwargs
from ._utils import expand_ix_array, make_idarrs_for_intervals
from .containers._ModelData import ModelData
from .containers._ContactData import ContactData
from .containers._ParticipantData import ParticipantData
from .containers._PopulationData import PopulationData
from .containers._StratificationData import StratificationData


class ContactSurveyLoader:
    """
    Orchestrates the contact survey data loading pipeline.

    This is the preferred entry point for building ``ModelData`` from contact
    survey DataFrames. It replaces the ``DataLoader`` / ``BaseLoader`` class
    hierarchy with a flat design: a ``DataFrameSurveySource`` holds validated
    data; ``ContactSurveyLoader`` runs the pipeline.

    Parameters
    ----------
    source : DataFrameSurveySource
        Validated survey data source containing merged data, column spec,
        population data, and optional stratification data.
    smooth_amb_cnt_offsets : bool, default=True
        Whether to apply Gaussian smoothing to the ambiguous contact offsets
        (V) before use as log-offsets. Bandwidth selected by leave-one-out
        cross-validation.

    Attributes
    ----------
    col_map : ColumnSpec
        Column mapping for this loader instance.
    min_age : int
        Minimum age in the aligned age range (set after ``__init__``).
    max_age : int
        Maximum age in the aligned age range (set after ``__init__``).
    model_data : ModelData or None
        Populated after calling ``load()``.

    Examples
    --------
    >>> source = DataFrameSurveySource(part_data, cnt_data, pop_data)
    >>> loader = ContactSurveyLoader(source)
    >>> md = loader.load()
    >>> md.y.shape
    (12345,)

    >>> # Convenience factory
    >>> loader = ContactSurveyLoader.from_containers(
    ...     part_data, cnt_data, pop_data, strat_data=strat_data
    ... )
    >>> md = loader.load()
    """

    def __init__(
        self,
        source: DataFrameSurveySource,
        smooth_amb_cnt_offsets: bool = True,
    ) -> None:
        self.col_map: ColumnSpec = source.col_map
        self.pop_data: pd.DataFrame = source.pop_data
        self.strat_data: Optional[StratificationData] = source.strat_data
        self.smooth_amb_cnt_offsets: bool = smooth_amb_cnt_offsets

        # Auto-create the ambiguous contact count column if missing
        data = source.data
        if self.col_map.z is not None and self.col_map.z not in data.columns:
            data = data.copy()
            data[self.col_map.z] = 0
        self.data: pd.DataFrame = data

        self._align_age_range()
        self._df_full_cache: Optional[pd.DataFrame] = None
        self.model_data: Optional[ModelData] = None

    def _align_age_range(self) -> None:
        """Align age ranges between sample and population data."""
        self.data, min_age, max_age = align_age_range(
            self.data, self.pop_data, self.col_map
        )
        self.min_age: int = min_age
        self.max_age: int = max_age

    @property
    def df_full(self) -> pd.DataFrame:
        """Full observation grid (cached after ``load()``)."""
        if self._df_full_cache is not None:
            return self._df_full_cache
        return self._build_df_full()

    def _build_df_full(self) -> pd.DataFrame:
        """Build the full Cartesian observation grid from scratch."""
        df_n = build_participant_counts(self.data, self.col_map)
        df_V = build_contact_offsets(
            self.data, self.col_map, self.smooth_amb_cnt_offsets
        )
        df_y = build_contact_counts(self.data, self.col_map)
        return build_observation_grid(
            self.data, self.col_map, self.min_age, self.max_age, df_n, df_y, df_V
        )

    def load(self) -> ModelData:
        """
        Run the full loading pipeline and return a ``ModelData``.

        Returns
        -------
        ModelData
            Flat container with all arrays and optional stratification metadata
            required by model classes.
        """
        # Cache df_full so all downstream calls use the same row ordering
        self._df_full_cache = self._build_df_full()
        df_full = self._df_full_cache

        # Build required observation arrays
        if self.col_map.part_age_grp:
            aid = None
        else:
            aid = df_full[self.col_map.part_age].to_numpy()

        # cid: coarse participant age group index
        cid = None
        if self.col_map.part_age_grp:
            cid = df_full[self.col_map.part_age].cat.codes.to_numpy()

        # did: coarse contact age group index
        did = None
        if self.col_map.cnt_age_grp:
            did = df_full[self.col_map.cnt_age_grp].cat.codes.to_numpy()

        bid = None
        aid_exp = None
        bid_pad = None
        if self.col_map.cnt_age:
            bid = df_full[self.col_map.cnt_age].to_numpy()
        elif self.col_map.cnt_age_grp:
            # aid_exp uses aid for fine-participant case; cid for coarse-participant case
            _aid_for_exp: np.ndarray = cid if self.col_map.part_age_grp else aid  # type: ignore[assignment]
            aid_exp, bid_pad = make_idarrs_for_intervals(
                df_full, self.col_map.cnt_age_grp, _aid_for_exp
            )

        if self.col_map.part_age_grp and self.col_map.cnt_age:
            warnings.warn(
                "Coarse participant age groups combined with fine-resolution contact ages "
                "is not currently supported by any model. Consider binning contact ages "
                "to match the participant age groups.",
                UserWarning,
                stacklevel=2,
            )

        rid = None
        if self.col_map.part_repeat is not None:
            rid = df_full[self.col_map.part_repeat].astype(int).to_numpy()

        # Build stratification kwargs
        strat_kwargs: Dict = {}
        if len(self.col_map.part_strat_vars) > 0:
            strat_kwargs = assemble_strat_kwargs(df_full, self.col_map, self.strat_data)
            if self.col_map.cnt_age_grp:
                strat_kwargs["flat_ix_exp"] = expand_ix_array(
                    strat_kwargs["flat_ix"], bid_pad.shape[1]
                )

        self.model_data = ModelData(
            y=df_full["y"].to_numpy(),
            log_N=np.log(df_full["N"].to_numpy()),
            log_P=construct_log_P(self.pop_data, self.col_map),
            age_min=self.min_age,
            age_max=self.max_age,
            aid=aid,
            cid=cid,
            did=did,
            bid=bid,
            log_V=df_full["log_V"].to_numpy(),
            aid_exp=aid_exp,
            bid_pad=bid_pad,
            rid=rid,
            **strat_kwargs,
        )

        return self.model_data

    @classmethod
    def from_containers(
        cls,
        part_data: ParticipantData,
        cnt_data: ContactData,
        pop_data: PopulationData,
        strat_data: Optional[StratificationData] = None,
        smooth_amb_cnt_offsets: bool = True,
    ) -> ContactSurveyLoader:
        """
        Build a ``ContactSurveyLoader`` directly from container objects.

        This is the primary entry point for users. It validates the containers,
        builds a ``DataFrameSurveySource``, and returns a ready-to-use loader.

        Parameters
        ----------
        part_data : ParticipantData
        cnt_data : ContactData
        pop_data : PopulationData
        strat_data : StratificationData or None, optional
        smooth_amb_cnt_offsets : bool, default=True

        Returns
        -------
        ContactSurveyLoader
        """
        source = DataFrameSurveySource(part_data, cnt_data, pop_data, strat_data)
        return cls(source, smooth_amb_cnt_offsets)
