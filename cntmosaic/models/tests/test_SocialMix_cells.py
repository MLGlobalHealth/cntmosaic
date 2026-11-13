"""
Tests for participant cells and the SocialMix fixes that come with them
(dev plan prerequisites; D23, D24, Problems 8, 9 and 11).

- ``build_participant_cells`` assigns cells (age group x participant strata),
  drops participants outside the age groups and merges empty age groups.
- ``SocialMix`` works on copies: the user's containers are never modified.
- Merging replaces the data instead of rebuilding containers, so columns
  such as ``part_age`` and ``part_repeat`` survive.
- Participants outside the age groups are dropped with one warning, and the
  bootstrap no longer fails because of them.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from ...dataloader import ContactData, ParticipantData, PopulationData
from ...utils import AgeGroupSpecs
from ...utils._participant_cells import build_participant_cells, copy_container
from ..classical._SocialMix import SocialMix

SPECS = AgeGroupSpecs(0, 20, 10)  # [0, 10), [10, 21): max age 20 is inclusive


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _survey(ages, sex=None, n_contacts=2, repeat=False):
    """Participants with the given ages; each reports n_contacts contacts aged 5."""
    df_part = pd.DataFrame({"pid": np.arange(len(ages)), "age": ages})
    kwargs = {}
    if sex is not None:
        df_part["sex"] = sex
        kwargs["strat_var_cols"] = "sex"
    if repeat:
        df_part["wave"] = 1
        kwargs["repeat_col"] = "wave"
    df_cnt = pd.DataFrame(
        {"pid": np.repeat(df_part["pid"].to_numpy(), n_contacts), "cnt_age": 5}
    )
    part = _quiet(ParticipantData, df_part, id_col="pid", age_col="age", **kwargs)
    cnt = _quiet(ContactData, df_cnt, id_col="pid", age_col="cnt_age")
    return part, cnt


# ============================================================================
# build_participant_cells
# ============================================================================


class TestBuildParticipantCells:
    def test_cell_codes_and_labels(self):
        part, _ = _survey(np.tile(np.arange(20), 2), sex=["F"] * 20 + ["M"] * 20)
        cells = build_participant_cells(part.data, ["part_sex"], SPECS)

        assert cells.n_cells == 4
        assert cells.cell_labels == (
            "F, [0, 10)", "F, [10, 21)", "M, [0, 10)", "M, [10, 21)"
        )
        sex = cells.part["part_sex"].astype(str).to_numpy()
        young = cells.part["part_age"].to_numpy() < 10
        expected = np.where(sex == "F", 0, 2) + np.where(young, 0, 1)
        np.testing.assert_array_equal(cells.cell, expected)
        np.testing.assert_array_equal(cells.ids, cells.part["id"].to_numpy())

    def test_input_not_modified(self):
        part, _ = _survey(np.arange(25))
        before = part.data.copy()
        _quiet(build_participant_cells, part.data, [], SPECS)
        pd.testing.assert_frame_equal(part.data, before)

    def test_drops_out_of_range_with_one_warning(self):
        part, _ = _survey(np.arange(30))  # ages 21-29 are outside [0, 20]
        with pytest.warns(UserWarning, match="Dropped 9 participant") as record:
            cells = build_participant_cells(part.data, [], SPECS)
        assert len([w for w in record if "Dropped" in str(w.message)]) == 1
        assert cells.n_dropped == 9
        assert len(cells.part) == 21
        assert cells.part["part_age_grp"].notna().all()

    def test_empty_group_strict_raises(self):
        part, _ = _survey(np.arange(10))  # nobody in [10, 21)
        with pytest.raises(ValueError, match="empty participant age group"):
            build_participant_cells(part.data, [], SPECS, strict=True)

    def test_empty_group_lenient_reports(self):
        part, _ = _survey(np.arange(10))
        cells = build_participant_cells(part.data, [], SPECS, strict=False)
        assert cells.empty_age_groups == (pd.Interval(10, 21, closed="left"),)
        assert not cells.merged
        assert cells.n_cells == 2

    def test_adaptive_merge_merges_and_keeps_columns(self):
        part, _ = _survey(np.arange(10), repeat=True)
        with pytest.warns(UserWarning, match="Merging age groups"):
            cells = build_participant_cells(part.data, [], SPECS, adaptive_merge=True)
        assert cells.merged
        assert cells.n_cells == 1
        assert cells.age_group_specs.left == [0]
        for col in ("part_age", "part_repeat"):
            assert col in cells.part.columns


# ============================================================================
# SocialMix integration
# ============================================================================


class TestUserContainersUnchanged:
    """Problem 11: SocialMix must not write into the user's containers."""

    def test_fit_and_bootstrap_leave_containers_unchanged(self):
        part, cnt = _survey(np.tile(np.arange(20), 3))
        df_pop = pd.DataFrame({"age": np.arange(20), "P": 100.0})
        pop = _quiet(PopulationData, df_pop, size_col="P", age_col="age")
        before = {
            "part": part.data.copy(),
            "cnt": cnt.data.copy(),
            "pop": pop.data.copy(),
        }
        sm = _quiet(SocialMix, part, cnt, SPECS, pop)
        _quiet(sm.rate)
        _quiet(sm.run_inference_bootstrap, n_boot=3, random_state=0, progress=False)

        pd.testing.assert_frame_equal(part.data, before["part"])
        pd.testing.assert_frame_equal(cnt.data, before["cnt"])
        pd.testing.assert_frame_equal(pop.data, before["pop"])

    def test_refit_with_other_age_groups_rebins(self):
        part, cnt = _survey(np.tile(np.arange(20), 3))
        _quiet(SocialMix, part, cnt, SPECS)
        sm = _quiet(SocialMix, part, cnt, AgeGroupSpecs(0, 20, 5))
        assert sm.C == 4  # not the 2 groups of the first fit


class TestOutOfRangeParticipants:
    """D23 / Problem 8: participants outside the age groups are dropped."""

    def test_point_estimate_warns_once_and_excludes(self):
        part, cnt = _survey(np.concatenate([np.tile(np.arange(20), 3), [25, 40, 70]]))
        with pytest.warns(UserWarning) as record:
            sm = SocialMix(part, cnt, SPECS, apply_reciprocity=False)
        dropped = [w for w in record if "Dropped 3 participant" in str(w.message)]
        assert len(dropped) == 1
        assert sm.N.sum() == 60
        assert len(sm.part_data.data) == 60
        assert sm._cells.n_dropped == 3

    def test_bootstrap_succeeds(self):
        part, cnt = _survey(np.concatenate([np.tile(np.arange(20), 3), [25, 40, 70]]))
        sm = _quiet(SocialMix, part, cnt, SPECS, apply_reciprocity=False)
        boot = _quiet(
            sm.run_inference_bootstrap, n_boot=5, random_state=0, progress=False
        )
        assert boot.n_boot == 5
        # Every participant reports 2 contacts, all in age group [0, 10).
        for label, mean in boot.mean(statistic="cint").items():
            np.testing.assert_allclose(mean.sum(axis=1), 2.0)


class TestMergeKeepsColumns:
    """D24 / Problem 9: merging must not delete columns."""

    def test_adaptive_merge_at_fit(self):
        # Nobody in [10, 21), so the two age groups are merged at fit time.
        part, cnt = _survey(np.tile(np.arange(10), 3), repeat=True)
        with pytest.warns(UserWarning, match="Merging age groups"):
            sm = SocialMix(part, cnt, SPECS, apply_reciprocity=False, adaptive_merge=True)
        assert sm.C == 1
        for col in ("part_age", "part_repeat", "part_age_grp"):
            assert col in sm.part_data.data.columns
        assert "cnt_age" in sm.cnt_data.data.columns
        assert sm._cells.n_cells == 1

    def test_bootstrap_time_merge(self):
        # Few participants per group: the bootstrap stability check merges.
        part, cnt = _survey(np.arange(20), repeat=True)
        sm = _quiet(SocialMix, part, cnt, SPECS, apply_reciprocity=False)
        assert sm.C == 2
        _quiet(sm.run_inference_bootstrap, n_boot=3, random_state=0, progress=False)
        assert sm.C == 1
        for col in ("part_age", "part_repeat"):
            assert col in sm.part_data.data.columns
        assert sm._cells.n_cells == 1


def test_copy_container_keeps_columns_and_does_not_share_data():
    part, _ = _survey(np.arange(20), repeat=True)
    copied = copy_container(part)
    assert copied is not part
    assert list(copied.data.columns) == list(part.data.columns)
    copied.data["extra"] = 1
    assert "extra" not in part.data.columns


def test_normalise_weights_argument_is_gone():
    """`normalise_weights` was removed with `weight_col` (dev plan D15, D28)."""
    part, cnt = _survey(np.tile(np.arange(20), 3))
    with pytest.raises(TypeError, match="normalise_weights"):
        SocialMix(part, cnt, SPECS, normalise_weights=True)
    with pytest.raises(TypeError, match="normalise_weights"):
        SocialMix.from_dataframes(part.data, cnt.data, SPECS, normalise_weights=True)
