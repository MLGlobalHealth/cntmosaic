"""
Tests for WeightSpecs and its private bound form (dev plan, Core item).
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from .. import BindContext, BoundFactor, WeightFactor, WeightResult, WeightSpecs
from .factors import BoundValues, ColumnFactor, ConstantFactor, LinearAgeFactor


def _ctx(ages, cell, n_cells=None, extra=None):
    part = pd.DataFrame({"id": np.arange(len(ages)) + 100, "part_age": ages})
    for k, v in (extra or {}).items():
        part[k] = v
    cell = np.asarray(cell)
    n_cells = int(cell.max()) + 1 if n_cells is None else n_cells
    return BindContext(
        part=part,
        cell=cell,
        n_cells=n_cells,
        cell_labels=tuple(f"c{k}" for k in range(n_cells)),
        ids=part["id"].to_numpy(),
    )


class TestConstruction:
    def test_needs_a_factor(self):
        with pytest.raises(ValueError, match="at least one"):
            WeightSpecs()

    def test_rejects_non_factors(self):
        with pytest.raises(TypeError, match="WeightFactor"):
            WeightSpecs(ConstantFactor(), "age")

    def test_duplicate_names(self):
        with pytest.raises(ValueError, match="name="):
            WeightSpecs(ConstantFactor(), ConstantFactor(value=3.0))
        WeightSpecs(ConstantFactor(), ConstantFactor(value=3.0, name="other"))

    def test_factors_and_names(self):
        a, b = ConstantFactor(), LinearAgeFactor()
        specs = WeightSpecs(a, b)
        assert specs.factors == (a, b)
        assert specs.names == ("constant", "linear_age")


class TestValueSemantics:
    def test_immutable(self):
        specs = WeightSpecs(ConstantFactor())
        with pytest.raises(AttributeError):
            specs._factors = ()
        with pytest.raises(AttributeError):
            specs.threshold = 3

    def test_equality_and_hash(self):
        a = WeightSpecs(ConstantFactor(), LinearAgeFactor())
        b = WeightSpecs(ConstantFactor(), LinearAgeFactor())
        c = WeightSpecs(LinearAgeFactor(), ConstantFactor())
        assert a == b and hash(a) == hash(b)
        assert a != c  # order is part of the specification
        assert len({a, b}) == 1

    def test_pickle_round_trip(self):
        specs = WeightSpecs(ConstantFactor(), LinearAgeFactor(slope=2.0))
        assert pickle.loads(pickle.dumps(specs)) == specs

    def test_repr_lists_factors(self):
        assert "LinearAgeFactor" in repr(WeightSpecs(LinearAgeFactor()))

    def test_factor_contract_types(self):
        assert isinstance(ConstantFactor(), WeightFactor)
        assert ConstantFactor.sample_dependent is False
        assert isinstance(BoundValues(np.ones(2)), BoundFactor)


class TestBind:
    def test_missing_column_lists_column_and_factor(self):
        specs = WeightSpecs(ColumnFactor(col="dayofweek", name="dow"))
        with pytest.raises(KeyError, match="dayofweek.*dow"):
            specs._bind(_ctx([10, 20], [0, 0]))

    def test_prefix_resolution(self):
        specs = WeightSpecs(ColumnFactor(col="w"))
        bound = specs._bind(_ctx([10, 20], [0, 0], extra={"part_w": [1.0, 3.0]}))
        np.testing.assert_allclose(bound.composite(), [1.0, 3.0])

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_non_positive_or_non_finite_factor_raises(self, bad):
        specs = WeightSpecs(ColumnFactor(col="w"))
        ctx = _ctx([10, 20, 30], [0, 0, 0], extra={"part_w": [1.0, bad, 2.0]})
        with pytest.raises(ValueError, match=r"ids \[101\]"):
            specs._bind(ctx)

    def test_bind_does_not_modify_specs(self):
        specs = WeightSpecs(LinearAgeFactor())
        before = repr(specs)
        specs._bind(_ctx([10, 20], [0, 0]))
        assert repr(specs) == before


class TestWeights:
    def test_product_of_factors(self):
        specs = WeightSpecs(ConstantFactor(value=2.0), LinearAgeFactor(slope=1.0))
        bound = specs._bind(_ctx([1, 2, 3], [0, 0, 0]))
        np.testing.assert_allclose(bound.composite(), 2.0 * np.array([2.0, 3.0, 4.0]))

    def test_normalised_within_cells(self):
        ages = np.array([1, 2, 3, 10, 20])
        cell = np.array([0, 0, 0, 1, 1])
        bound = WeightSpecs(LinearAgeFactor(slope=1.0))._bind(_ctx(ages, cell))
        w = bound.weights()
        np.testing.assert_allclose(np.bincount(cell, weights=w), [3.0, 2.0])
        # Within a cell, weights keep the ratios of the composite weights.
        np.testing.assert_allclose(w[:3] / w[0], (ages[:3] + 1) / (ages[0] + 1))

    def test_constant_factor_gives_unit_weights(self):
        bound = WeightSpecs(ConstantFactor(value=7.0))._bind(_ctx([1, 2, 3], [0, 1, 1]))
        np.testing.assert_allclose(bound.weights(), 1.0)

    def test_rows_with_repeats_count_multiplicity(self):
        ages = np.array([1, 2, 3])
        bound = WeightSpecs(LinearAgeFactor(slope=1.0))._bind(_ctx(ages, [0, 0, 0]))
        rows = np.array([0, 0, 2])  # participant 0 drawn twice, participant 1 not at all
        w = bound.weights(rows)
        assert w.sum() == pytest.approx(3.0)
        composite = ages[rows] + 1.0
        np.testing.assert_allclose(w, composite * 3 / composite.sum())

    def test_result_table(self):
        specs = WeightSpecs(ConstantFactor(value=2.0), LinearAgeFactor(slope=1.0))
        result = specs._bind(_ctx([1, 2], [0, 1], n_cells=2)).result()
        assert isinstance(result, WeightResult)
        assert result.specs is specs
        assert list(result.participants.columns) == [
            "id", "cell", "part_age", "constant", "linear_age", "composite", "weight",
            "capped",
        ]
        np.testing.assert_array_equal(result.participants["part_age"], [1, 2])
        np.testing.assert_allclose(result.participants["composite"], [4.0, 6.0])
        np.testing.assert_allclose(result.participants["weight"], [1.0, 1.0])
        assert list(result.participants["cell"]) == ["c0", "c1"]
        assert "2 respondents" in repr(result)

    def test_result_table_source_columns_once_in_factor_order(self):
        ctx = _ctx([1, 2, 3], [0, 0, 1], extra={"part_x": [1.0, 2.0, 3.0]})
        specs = WeightSpecs(
            ColumnFactor(col="x", name="x1"),  # given without the prefix
            LinearAgeFactor(),
            ColumnFactor(col="part_x", name="x2"),  # same column again
        )
        table = specs._bind(ctx).result().participants
        assert list(table.columns[:4]) == ["id", "cell", "part_x", "part_age"]
        assert list(table.columns[4:7]) == ["x1", "linear_age", "x2"]
        np.testing.assert_allclose(table["part_x"], [1.0, 2.0, 3.0])

    def test_factor_name_clashing_with_source_column_raises(self):
        ctx = _ctx([1, 2, 3], [0, 0, 1], extra={"part_x": [1.0, 2.0, 3.0]})
        specs = WeightSpecs(ColumnFactor(col="x", name="part_x"))
        with pytest.raises(ValueError, match="clash"):
            specs._bind(ctx)

    def test_bound_specs_pickle(self):
        bound = WeightSpecs(LinearAgeFactor())._bind(_ctx([1, 2, 3], [0, 0, 1]))
        restored = pickle.loads(pickle.dumps(bound))
        np.testing.assert_allclose(restored.weights(), bound.weights())
