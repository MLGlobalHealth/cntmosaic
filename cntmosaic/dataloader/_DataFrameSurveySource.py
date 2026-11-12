"""
Validated survey data source backed by DataFrames.

Internal API — not exported from ``cntmosaic.dataloader``. Provides
``DataFrameSurveySource``, a thin container that validates the four container
objects, merges participant/contact data, and builds a ``ColumnSpec``.
It is the canonical input for ``ContactSurveyLoader``.
"""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd

from ._ColumnSpec import ColumnSpec
from ._DataValidator import DataValidator
from .containers._ContactData import ContactData
from .containers._ParticipantData import ParticipantData
from .containers._PopulationData import PopulationData
from .containers._StratificationData import StratificationData


class DataFrameSurveySource:
    """
    Validated contact survey data source backed by pandas DataFrames.

    Accepts the four container objects, runs ``DataValidator``, merges the
    participant and contact DataFrames, and exposes the resulting ``data``,
    ``pop_data``, ``col_map``, and ``strat_data`` attributes ready for
    consumption by ``ContactSurveyLoader``.

    Parameters
    ----------
    part_data : ParticipantData
        Validated participant data container.
    cnt_data : ContactData
        Validated contact data container.
    pop_data : PopulationData
        Validated population data container.
    strat_data : StratificationData or None, optional
        Stratified population proportion data.

    Attributes
    ----------
    data : pd.DataFrame
        Merged participant-contact DataFrame with categorical dtypes restored.
    pop_data : pd.DataFrame
        Population DataFrame (from the validated ``PopulationData`` container).
    col_map : ColumnSpec
        Column mapping built from the validated containers.
    strat_data : StratificationData or None
        Validated stratification data, or ``None`` if not provided.
    part_data : ParticipantData
        Validated participant data container (post-validation copy).
    cnt_data : ContactData
        Validated contact data container (post-validation copy).
    """

    def __init__(
        self,
        part_data: ParticipantData,
        cnt_data: ContactData,
        pop_data: PopulationData,
        strat_data: Optional[StratificationData] = None,
    ) -> None:
        self.part_data, self.cnt_data, self.pop_data_container, self.strat_data = (
            DataValidator(
                part_data=part_data,
                cnt_data=cnt_data,
                pop_data=pop_data,
                strat_data=strat_data,
            ).validate()
        )

        self.col_map: ColumnSpec = ColumnSpec.from_containers(
            self.part_data, self.cnt_data, self.pop_data_container
        )

        self.data: pd.DataFrame = self._merge_data()
        self.pop_data: pd.DataFrame = self.pop_data_container.data

    def _merge_data(self) -> pd.DataFrame:
        """Merge contact and participant DataFrames on 'id', restoring categoricals."""
        data = pd.merge(self.cnt_data.data, self.part_data.data, on="id")

        for col in data.columns:
            if col in self.part_data.data.columns:
                if isinstance(data[col].dtype, pd.CategoricalDtype):
                    data[col] = pd.Categorical(
                        data[col],
                        categories=self.part_data.data[col].cat.categories,
                        ordered=self.part_data.data[col].cat.ordered,
                    )
            elif col in self.cnt_data.data.columns:
                if isinstance(data[col].dtype, pd.CategoricalDtype):
                    data[col] = pd.Categorical(
                        data[col],
                        categories=self.cnt_data.data[col].cat.categories,
                        ordered=self.cnt_data.data[col].cat.ordered,
                    )

        return data
