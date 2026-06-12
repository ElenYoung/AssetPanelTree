"""
Tests for WeightedR2DiffCriterion (M3, priority item).

The default ``R2DiffCriterion`` behaviour must remain unchanged; the new
weighted criterion is an opt-in, stricter alternative.
"""

from __future__ import annotations

import pytest

from ptree.criteria import R2DiffCriterion, WeightedR2DiffCriterion


class TestWeightedR2Diff:
    def test_reduces_to_balanced_r2diff(self):
        """With default options it equals R2Diff times the balance term."""
        left = {"r2": 0.8, "n_samples": 100}
        right = {"r2": 0.2, "n_samples": 300}
        w = WeightedR2DiffCriterion(balance=True, shrinkage_k=0.0)
        expected = abs(0.8 - 0.2) * (100 / 400)
        assert w.calculate_score(left, right) == pytest.approx(expected)

    def test_balance_off_equals_raw_diff(self):
        left = {"r2": 0.8, "n_samples": 100}
        right = {"r2": 0.2, "n_samples": 300}
        w = WeightedR2DiffCriterion(balance=False, shrinkage_k=0.0)
        assert w.calculate_score(left, right) == pytest.approx(0.6)

    def test_shrinkage_penalises_small_nodes(self):
        big = {"r2": 0.9, "n_samples": 1000}
        small = {"r2": 0.1, "n_samples": 1000}
        tiny_l = {"r2": 0.9, "n_samples": 6}
        tiny_r = {"r2": 0.1, "n_samples": 1994}
        w = WeightedR2DiffCriterion(balance=False, shrinkage_k=10.0)
        score_big = w.calculate_score(big, small)
        score_tiny = w.calculate_score(tiny_l, tiny_r)
        # Tiny minority node should be shrunk far more aggressively.
        assert score_tiny < score_big

    def test_adjusted_r2_degrades_high_dim(self):
        # High R² from many features on few samples is penalised.
        left = {"r2": 0.95, "n_samples": 12, "n_features": 8}
        right = {"r2": 0.10, "n_samples": 500, "n_features": 8}
        raw = WeightedR2DiffCriterion(
            balance=False, use_adjusted_r2=False
        ).calculate_score(left, right)
        adj = WeightedR2DiffCriterion(
            balance=False, use_adjusted_r2=True
        ).calculate_score(left, right)
        assert adj < raw

    def test_default_criterion_unchanged(self):
        """Sanity: the default criterion is untouched."""
        left = {"r2": 0.8, "n_samples": 50}
        right = {"r2": 0.2, "n_samples": 50}
        assert R2DiffCriterion().calculate_score(left, right) == pytest.approx(0.6)

    def test_metric_key(self):
        assert WeightedR2DiffCriterion().metric_key() == "r2"

    def test_repr(self):
        r = repr(WeightedR2DiffCriterion(shrinkage_k=5.0))
        assert "WeightedR2DiffCriterion" in r and "shrinkage_k=5.0" in r
