"""Data loading utilities for the Prem model."""

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._socialmix_helpers import row_stratum_labels

if TYPE_CHECKING:
    from ._Prem import Prem


class PremDataLoader:
    """Build the flat index arrays required by the Prem NumPyro model.

    The Prem model uses plate-based indexing rather than dense tensors, so
    the loaded representation is a long-format frame where every row is a
    (participant, contact-age-group[, stratum]) observation, together with
    integer index arrays that map each row into the right model dimensions.

    The frame includes *structural zeros*: every participant contributes one
    row per contact age group (and contact stratum), with ``y = 0`` where no
    such contact was recorded.  A zero here means the participant did not meet
    anyone in that contact age group, not that the contact went unreported.

    Parameters
    ----------
    prem : Prem
        The parent :class:`Prem` instance.  The loader reads preprocessing
        state from *prem* and writes the resulting arrays back onto it.
    """

    def __init__(self, prem: "Prem") -> None:
        self.prem = prem

    def load_data(self) -> None:
        """Orchestrate the full loading pipeline.

        Sets the following attributes on the parent :class:`Prem` instance:

        - ``data``  — aggregated long-format DataFrame
        - ``y``     — observed contact counts ``(n_obs,)``
        - ``iix``   — participant index per row ``(n_obs,)``
        - ``cix``   — participant age-group index per row ``(n_obs,)``
        - ``dix``   — contact age-group index per row ``(n_obs,)``
        - ``six``   — stratum index per row ``(n_obs,)``
        - ``N``     — number of unique participants
        - ``C``     — number of participant age groups
        - ``D``     — number of contact age groups
        """
        df_cnt_full = self._build_full_contact_frame()
        self._merge_participant_data(df_cnt_full)
        self._aggregate()
        self._encode_strata()
        self._extract_arrays()

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _build_full_contact_frame(self) -> pd.DataFrame:
        """Build a complete (id × cnt_age_grp [× cnt_strat]) Cartesian frame.

        Every (participant, contact-age-group, contact-stratum) combination is
        represented even if no contact was recorded; missing counts are filled
        with zero (structural zeros) so the model likelihood sees the full
        observation grid.  A participant contributes a zero for a contact age
        group because they did not meet anyone in it, not because they failed
        to report the contact.
        """
        prem = self.prem
        df_cnt = prem.cnt_data.data.copy()

        if not isinstance(df_cnt["cnt_age_grp"].dtype, pd.CategoricalDtype):
            df_cnt["cnt_age_grp"] = pd.Categorical(df_cnt["cnt_age_grp"], ordered=True)

        # Build Cartesian dimension dict: id × age group [× contact strat vars]
        coords: dict = {
            "id": df_cnt["id"].unique(),
            "cnt_age_grp": df_cnt["cnt_age_grp"].cat.categories,
        }
        for var in prem.strat_vars_cnt:
            col = f"cnt_{var}"
            if col in df_cnt.columns:
                if not isinstance(df_cnt[col].dtype, pd.CategoricalDtype):
                    df_cnt[col] = pd.Categorical(df_cnt[col])
                coords[col] = df_cnt[col].cat.categories

        index = pd.MultiIndex.from_product(
            [list(v) for v in coords.values()], names=list(coords.keys())
        )
        df_full = pd.DataFrame(index.to_frame(index=False), columns=list(coords.keys()))

        merge_keys = ["id", "cnt_age_grp"] + [f"cnt_{v}" for v in prem.strat_vars_cnt]
        df_full = pd.merge(df_full, df_cnt, on=merge_keys, how="left")
        df_full["y"] = df_full["y"].fillna(0).astype(int)

        # Restore categorical dtypes lost during merge
        df_full["cnt_age_grp"] = pd.Categorical(
            df_full["cnt_age_grp"],
            categories=df_cnt["cnt_age_grp"].cat.categories,
            ordered=True,
        )
        for var in prem.strat_vars_cnt:
            col = f"cnt_{var}"
            if col in df_cnt.columns:
                df_full[col] = pd.Categorical(
                    df_full[col],
                    categories=df_cnt[col].cat.categories,
                    ordered=getattr(df_cnt[col].cat, "ordered", False),
                )

        return df_full

    def _merge_participant_data(self, df_cnt_full: pd.DataFrame) -> None:
        """Left-join the contact frame with participant data on participant ID."""
        prem = self.prem
        prem.data = pd.merge(df_cnt_full, prem.part_data.data, on="id", how="left")

        if prem.data["part_age_grp"].isnull().any():
            missing = prem.data[prem.data["part_age_grp"].isna()]["id"].unique()
            warnings.warn(
                f"Missing age group information for participant IDs: {missing}. "
                "These participants will be dropped from the analysis.",
                UserWarning,
                stacklevel=2,
            )
            prem.data = prem.data[prem.data["part_age_grp"].notna()]

        if not isinstance(prem.data["part_age_grp"].dtype, pd.CategoricalDtype):
            prem.data["part_age_grp"] = pd.Categorical(
                prem.data["part_age_grp"], ordered=True
            )

    def _aggregate(self) -> None:
        """Sum contact counts within each (participant × part_age_grp × cnt_age_grp [× strat]) cell.

        ``observed=True`` so only cells materialised by the Cartesian contact
        frame are kept.  ``part_age_grp`` and the ``part_*`` strata are 1:1 with
        the participant ``id``; expanding them (``observed=False``) would emit
        spurious zero rows at the wrong participant age group.
        """
        prem = self.prem
        groupby_cols = ["id", "part_age_grp", "cnt_age_grp"]
        for col in [f"part_{v}" for v in prem.strat_vars_part] + [
            f"cnt_{v}" for v in prem.strat_vars_cnt
        ]:
            if col not in groupby_cols:
                groupby_cols.append(col)

        prem.data = (
            prem.data.groupby(groupby_cols, observed=True)["y"].sum().reset_index()
        )

    def _encode_strata(self) -> None:
        """Create integer stratum index ``six`` from composite stratum labels."""
        prem = self.prem
        if prem.strat_vars_part or prem.strat_vars_cnt:
            part_cols = [f"part_{v}" for v in prem.strat_vars_part]
            cnt_cols = [f"cnt_{v}" for v in prem.strat_vars_cnt]
            prem.data["stratum"] = row_stratum_labels(prem.data, part_cols, cnt_cols)
            prem.data["stratum"] = pd.Categorical(prem.data["stratum"], ordered=True)
            prem.six = np.array(prem.data["stratum"].cat.codes, dtype=np.int32)
            # Actual K may be less than the Cartesian product if some combos are absent
            prem.K = int(prem.data["stratum"].nunique())
        else:
            prem.six = np.zeros(len(prem.data), dtype=np.int32)

    def _extract_arrays(self) -> None:
        """Extract NumPy index arrays and dimension counts from the aggregated frame."""
        prem = self.prem
        prem.data["id_cat"] = pd.Categorical(prem.data["id"])
        prem.data["iix"] = prem.data["id_cat"].cat.codes

        prem.y = np.array(prem.data["y"].values)
        prem.iix = np.array(prem.data["iix"].values, dtype=np.int32)
        prem.cix = np.array(prem.data["part_age_grp"].cat.codes, dtype=np.int32)
        prem.dix = np.array(prem.data["cnt_age_grp"].cat.codes, dtype=np.int32)

        prem.N = prem.data["id"].nunique()
        prem.C = prem.data["part_age_grp"].cat.categories.size
        prem.D = prem.data["cnt_age_grp"].cat.categories.size
