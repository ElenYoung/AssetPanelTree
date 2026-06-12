"""
Tests for M3 B3: cost-complexity pruning.

Covers:
* ``prune(ccp_alpha)`` — ccp_alpha=0 is a no-op; a large alpha collapses the
  tree to its root; pruning is monotone in alpha; predictions stay valid.
* ``cost_complexity_pruning_path()`` — returns increasing alphas, decreasing
  leaf counts, and does not mutate the fitted tree.
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


class TestPrune:
    def test_zero_alpha_is_noop(self):
        engine, _, _ = _fit()
        n_before = len(engine.get_all_nodes())
        engine.prune(0.0)
        assert len(engine.get_all_nodes()) == n_before

    def test_large_alpha_collapses_to_root(self):
        engine, _, _ = _fit()
        engine.prune(1e9)
        nodes = engine.get_all_nodes()
        assert len(nodes) == 1
        assert nodes[0].is_leaf

    def test_pruning_is_monotone(self):
        e_small, _, _ = _fit()
        e_small.prune(0.001)
        e_big, _, _ = _fit()
        e_big.prune(0.05)
        assert len(e_big.get_leaves()) <= len(e_small.get_leaves())

    def test_predictions_valid_after_prune(self):
        engine, X, y = _fit()
        engine.prune(0.02)
        preds = engine.predict(X)
        assert preds.shape == (len(y),)
        assert np.all(np.isfinite(preds))

    def test_negative_alpha_raises(self):
        engine, _, _ = _fit()
        with pytest.raises(ValueError):
            engine.prune(-1.0)


class TestPruningPath:
    def test_path_keys_and_monotonicity(self):
        engine, _, _ = _fit()
        path = engine.cost_complexity_pruning_path()
        assert set(path.keys()) == {"ccp_alphas", "n_leaves", "total_scores"}
        alphas = path["ccp_alphas"]
        leaves = path["n_leaves"]
        # Alphas non-decreasing; leaf counts non-increasing toward the root.
        assert np.all(np.diff(alphas) >= -1e-12)
        assert np.all(np.diff(leaves) <= 0)
        assert leaves[0] == len(engine.get_leaves())
        assert leaves[-1] == 1

    def test_path_does_not_mutate_tree(self):
        engine, _, _ = _fit()
        n_before = len(engine.get_all_nodes())
        engine.cost_complexity_pruning_path()
        assert len(engine.get_all_nodes()) == n_before

    def test_path_first_alpha_is_zero(self):
        engine, _, _ = _fit()
        path = engine.cost_complexity_pruning_path()
        assert path["ccp_alphas"][0] == pytest.approx(0.0)
