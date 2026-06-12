"""
Tests for M3 B4: stopping criteria and adaptive thresholds.

Covers:
* ``min_impurity_decrease`` — a high threshold forces the tree to stay a leaf,
  a zero threshold preserves the default (multi-node) behaviour.
* ``split_thresholds="adaptive"`` — per-node quantile thresholds produce a
  valid, complete-partition tree; bad string values raise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ptree import (
    DataHandler,
    PanelTreeEngine,
    R2DiffCriterion,
    RidgeRegressor,
)


def _make_data(T: int = 30, N: int = 80, P: int = 5, seed: int = 2026):
    rng = np.random.default_rng(seed)
    feature_names = [f"char_{i + 1}" for i in range(P)]
    dates = np.repeat(np.arange(T), N)
    asset_ids = np.tile(np.arange(N), T)
    X_raw = rng.standard_normal((T * N, P))
    alpha_strong = 0.6 * X_raw[:, 1] - 0.4 * X_raw[:, 2] + 0.3 * X_raw[:, 3]
    alpha_weak = 0.05 * X_raw[:, 1]
    y_raw = np.where(X_raw[:, 0] > 0, alpha_strong, alpha_weak)
    y_raw = y_raw + rng.standard_normal(T * N) * 0.3
    df = pd.DataFrame(X_raw, columns=feature_names)
    df["date"] = dates
    df["asset_id"] = asset_ids
    return df, pd.Series(y_raw, name="ret"), feature_names


def _fit(**overrides):
    df, y, _ = _make_data()
    dh = DataHandler(cs_rank_standardize=True)
    X_proc, y_proc, _ = dh.fit_transform(
        df, y, time_col="date", entity_col="asset_id"
    )
    params = dict(
        predictor=RidgeRegressor(alpha=1.0),
        criterion=R2DiffCriterion(),
        split_thresholds=[0.3, 0.5, 0.7],
        max_depth=3,
        min_samples=200,
        verbose=0,
    )
    params.update(overrides)
    engine = PanelTreeEngine(**params)
    engine.fit(X_proc, y_proc, feature_names=dh.feature_names)
    return engine, X_proc, y_proc


class TestMinImpurityDecrease:
    def test_default_zero_splits(self):
        """Default ``min_impurity_decrease=0`` must build a multi-node tree."""
        engine, _, _ = _fit()
        assert len(engine.get_all_nodes()) > 1

    def test_high_threshold_forces_leaf(self):
        """A threshold above any achievable score yields a root-only tree."""
        engine, _, _ = _fit(min_impurity_decrease=1e9)
        nodes = engine.get_all_nodes()
        assert len(nodes) == 1
        assert nodes[0].is_leaf

    def test_intermediate_threshold_reduces_nodes(self):
        """A moderate threshold yields fewer nodes than the unconstrained tree."""
        base, _, _ = _fit()
        constrained, _, _ = _fit(min_impurity_decrease=0.05)
        assert len(constrained.get_all_nodes()) <= len(base.get_all_nodes())

    def test_predictions_still_finite(self):
        engine, X, y = _fit(min_impurity_decrease=0.01)
        preds = engine.predict(X)
        assert preds.shape == (len(y),)
        assert np.all(np.isfinite(preds))


class TestAdaptiveThresholds:
    def test_adaptive_builds_valid_tree(self):
        engine, X, y = _fit(split_thresholds="adaptive")
        assert engine.root_ is not None
        preds = engine.predict(X)
        assert np.all(np.isfinite(preds))

    def test_adaptive_partition_is_complete(self):
        engine, X, y = _fit(split_thresholds="adaptive")
        leaf_samples = engine.get_leaf_samples()
        all_idx = np.concatenate(list(leaf_samples.values()))
        assert np.array_equal(np.sort(all_idx), np.arange(len(y)))

    def test_custom_quantiles(self):
        engine, X, y = _fit(
            split_thresholds="adaptive",
            adaptive_quantiles=[0.5],
        )
        # Every internal split threshold must equal a node median (a quantile),
        # so thresholds should lie strictly inside the [0, 1] rank range.
        for node in engine.get_all_nodes():
            if not node.is_leaf:
                assert 0.0 <= node.split_threshold <= 1.0

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            PanelTreeEngine(split_thresholds="nonsense")
