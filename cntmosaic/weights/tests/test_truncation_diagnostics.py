"""
Tests for truncation (threshold), diagnostics (WeightResult.cells, the
"factor has no effect" warning, repr) and warning behaviour
(dev plan: Truncation, Diagnostics, Warnings; D18).
"""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from .. import CategoryWeights, DesignWeights, WeightSpecs, WeightWarning
from .factors import ConstantFactor, LinearAgeFactor
from .helpers import make_ctx


def _part(w, sex=None):
    df = pd.DataFrame({"id": np.arange(len(w)) + 1, "part_age": np.arange(len(w)), "part_w": w})
    if sex is not None:
        df["part_sex"] = sex
    return df


class TestThreshold:
    def test_validation(self):
        for bad in (1.0, 0.5, np.inf):
            with pytest.raises(ValueError, match="greater than 1"):
                WeightSpecs(DesignWeights("w"), threshold=bad)
        with pytest.raises(TypeError):
            WeightSpecs(DesignWeights("w"), threshold="3")

    def test_value_semantics(self):
        a = WeightSpecs(DesignWeights("w"), threshold=3)
        assert a.threshold == 3.0
        assert a != WeightSpecs(DesignWeights("w"))
        assert a == WeightSpecs(DesignWeights("w"), threshold=3.0)
        assert pickle.loads(pickle.dumps(a)) == a
        assert "threshold=3" in repr(a)

    def test_cap_then_renormalise_once(self):
        part = _part([1.0, 1.0, 1.0, 5.0])  # normalised: 0.5, 0.5, 0.5, 2.5
        bound = WeightSpecs(DesignWeights("w"), threshold=2.0)._bind(make_ctx(part))
        w = bound.weights()
        expected = np.array([0.5, 0.5, 0.5, 2.0]) * 4 / 3.5
        np.testing.assert_allclose(w, expected)
        assert w[3] > 2.0  # can end slightly above the threshold (as socialmixr)
        assert w.sum() == pytest.approx(4.0)
        result = bound.result()
        assert result.participants["capped"].tolist() == [False, False, False, True]
        assert result.cells["n_capped"].tolist() == [1]

    def test_no_capping_needed(self):
        part = _part([1.0, 1.2, 0.9])
        a = WeightSpecs(DesignWeights("w"), threshold=10)._bind(make_ctx(part)).weights()
        b = WeightSpecs(DesignWeights("w"))._bind(make_ctx(part)).weights()
        np.testing.assert_allclose(a, b)

    def test_applied_per_cell_and_in_resamples(self):
        part = _part([1.0, 1.0, 1.0, 5.0, 1.0, 1.0])
        cell = [0, 0, 0, 0, 1, 1]
        bound = WeightSpecs(DesignWeights("w"), threshold=2.0)._bind(make_ctx(part, cell=cell))
        w = bound.weights(np.array([3, 3, 0, 4]))  # resample with repeats
        # Cell 0 rows: 5, 5, 1 -> normalised 3*[5,5,1]/11; capped at 2 and renormalised.
        raw = np.array([5, 5, 1]) * 3 / 11
        capped = np.minimum(raw, 2.0) * 3 / np.minimum(raw, 2.0).sum()
        np.testing.assert_allclose(w[:3], capped)
        assert w[3] == pytest.approx(1.0)


class TestDiagnostics:
    def test_cells_table(self):
        w = np.array([1.0, 3.0, 2.0, 2.0])
        cell = [0, 0, 1, 1]
        result = WeightSpecs(DesignWeights("w"))._bind(make_ctx(_part(w), cell=cell)).result()
        cells = result.cells
        assert cells["cell"].tolist() == ["c0", "c1"]
        assert cells["n"].tolist() == [2, 2]
        np.testing.assert_allclose(cells["min_w"], [0.5, 1.0])
        np.testing.assert_allclose(cells["max_w"], [1.5, 1.0])
        # Kish: (sum w)^2 / sum w^2
        np.testing.assert_allclose(cells["n_eff"], [4 / (0.25 + 2.25), 2.0])
        np.testing.assert_allclose(cells["deff"], [2 / 1.6, 1.0])

    def test_empty_cell_row(self):
        result = WeightSpecs(DesignWeights("w"))._bind(
            make_ctx(_part([1.0, 2.0]), cell=[0, 0], n_cells=2)
        ).result()
        row = result.cells.iloc[1]
        assert row["n"] == 0 and np.isnan(row["n_eff"]) and np.isnan(row["max_w"])
        assert "1 empty" in repr(result)

    def test_repr(self):
        result = WeightSpecs(LinearAgeFactor(), threshold=5)._bind(
            make_ctx(_part([1.0, 1.0, 1.0]))
        ).result()
        text = repr(result)
        for piece in ("3 participants", "1 cells", "smallest n_eff", "largest weight", "capped"):
            assert piece in text

    def test_factor_without_effect_warns(self):
        part = _part([1.0] * 4, sex=["F", "F", "M", "M"])
        cell = [0, 0, 1, 1]  # cells are the sex strata
        specs = WeightSpecs(CategoryWeights("sex", {"F": 1.0, "M": 3.0}))
        with pytest.warns(WeightWarning, match="no effect"):
            specs._bind(make_ctx(part, cell=cell))
        with pytest.warns(WeightWarning, match="no effect"):
            WeightSpecs(ConstantFactor())._bind(make_ctx(part))

    def test_varying_factor_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", WeightWarning)
            WeightSpecs(LinearAgeFactor())._bind(make_ctx(_part([1.0, 1.0, 1.0])))

    def test_rebind_does_not_repeat_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", WeightWarning)
            WeightSpecs(ConstantFactor())._bind(make_ctx(_part([1.0, 2.0])), warn_user=False)

    def test_warning_points_at_caller(self):
        with pytest.warns(WeightWarning) as record:
            WeightSpecs(ConstantFactor())._bind(make_ctx(_part([1.0, 2.0])))
        assert record[0].filename == __file__

    def test_weight_warning_is_user_warning(self):
        assert issubclass(WeightWarning, UserWarning)
