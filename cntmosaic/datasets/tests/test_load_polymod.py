import pytest

from .._base import load_polymod_germany, load_polymod_uk


def test_load_polymod_germany():
    """Tests that the function works."""

    data = load_polymod_germany()

    assert "contacts" in data
    assert "participants" in data
    assert "population" in data
    assert data["contacts"].shape[0] > 0
    assert data["participants"].shape[0] > 0
    assert data["population"].shape[0] > 0


def test_load_polymod_uk():
    """Tests that the function works."""

    data = load_polymod_uk()

    assert "contacts" in data
    assert "participants" in data
    assert "population" in data
    assert data["contacts"].shape[0] > 0
    assert data["participants"].shape[0] > 0
    assert data["population"].shape[0] > 0


def test_load_polymod_uk_layout():
    """Columns used by the tutorials are present; no precomputed age groups."""

    data = load_polymod_uk()
    part, cnt, pop = data["participants"], data["contacts"], data["population"]

    assert {"part_id", "part_age", "part_gender", "dayofweek"} <= set(part.columns)
    assert {"part_id", "cnt_age_exact", "cnt_gender", "cnt_school"} <= set(cnt.columns)
    assert list(pop.columns) == ["age", "gender", "P"]
    assert not {"part_age_grp", "cnt_age_grp", "age_grp"} & (
        set(part.columns) | set(cnt.columns) | set(pop.columns)
    )

    assert part["part_id"].is_unique
    assert set(cnt["part_id"]) <= set(part["part_id"])
    assert pop["age"].min() == 0 and pop["age"].max() == 90
    assert set(pop["gender"]) == {"M", "F"}
    assert len(pop) == 2 * 91
