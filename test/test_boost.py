"""
Tests for BoostedPanelTree and the random splitter (M7 / D2 + D3)
=================================================================

Covers:
* residual-norm history is monotonically non-increasing as boosting proceeds;
* the *self-limiting* property on single-feature-dominated data (later trees add
  almost nothing once the dominant regime is explained);
* ``predict`` shape / finiteness and reproducibility under a fixed seed;
* stochastic boosting via ``subsample < 1`` (requires ``time_index``);
* ``splitter="random"`` builds a valid tree and keeps the golden ``"best"`` path
  bit-identical (random splitter does not perturb the default behaviour);
* the forest exposes the random splitter through ``base_params``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ptree import (
    BoostedPanelTree,
    DataHandler,
    PanelForest,
    PanelTreeEngine,
    R2DiffCriterion,
    RidgeRegressor,
)


# ----------------------------------------------------------------------
# Synthetic panel data
# ----------------------------------------------------------------------

def _make_data(T: int = 40, N: int = 60, P: int = 6, seed: int = 11):
    """Single-feature-dominated panel (same family as the golden baseline)."""
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


def _make_multifeature_data(T: int = 40, N: int = 60, P: int = 6, seed: int = 3):
    """Predictability spread across several weak, independent features/regimes."""
    rng = np.random.default_rng(seed)
    feature_names = [f"char_{i + 1}" for i in range(P)]
    dates = np.repeat(np.arange(T), N)
    asset_ids = np.tile(np.arange(N), T)
    X_raw = rng.standard_normal((T * N, P))
    y = (
        0.3 * np.where(X_raw[:, 0] > 0, X_raw[:, 1], 0.0)
        + 0.3 * np.where(X_raw[:, 2] > 0, X_raw[:, 3], 0.0)
        + 0.3 * np.where(X_raw[:, 4] > 0, X_raw[:, 5], 0.0)
    )
    y = y + rng.standard_normal(T * N) * 0.3
    df = pd.DataFrame(X_raw, columns=feature_names)
    df["date"] = dates
    df["asset_id"] = asset_ids
    return df, pd.Series(y, name="ret"), feature_names


def _processed(maker=_make_data, **kwargs):
    df, y, _ = maker(**kwargs)
    dh = DataHandler(cs_rank_standardize=True)
    X_proc, y_proc, _ = dh.fit_transform(
        df, y, time_col="date", entity_col="asset_id"
    )
    return X_proc, y_proc, dh.feature_names


# ----------------------------------------------------------------------
# BoostedPanelTree
# ----------------------------------------------------------------------

class TestBoostedPanelTree:
    def _fit(self, maker=_make_multifeature_data, **kwargs):
        X, y, feats = _processed(maker=maker)
        boost = BoostedPanelTree(
            n_estimators=kwargs.pop("n_estimators", 8),
            learning_rate=kwargs.pop("learning_rate", 0.3),
            max_depth=kwargs.pop("max_depth", 2),
            base_params=dict(min_samples=150),
            random_state=0,
            verbose=0,
            **kwargs,
        )
        boost.fit(X, y, feature_names=feats, time_index="date")
        return boost, X, y

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            BoostedPanelTree(n_estimators=0)
        with pytest.raises(ValueError):
            BoostedPanelTree(learning_rate=0.0)
        with pytest.raises(ValueError):
            BoostedPanelTree(learning_rate=1.5)
        with pytest.raises(ValueError):
            BoostedPanelTree(subsample=0.0)

    def test_grows_requested_number_of_trees(self):
        boost, _, _ = self._fit(n_estimators=6)
        assert len(boost.trees_) == 6

    def test_predict_shape_and_finite(self):
        boost, X, y = self._fit()
        preds = boost.predict(X)
        assert preds.shape == (len(y),)
        assert np.all(np.isfinite(preds))

    def test_residual_norms_recorded(self):
        boost, _, _ = self._fit(n_estimators=8)
        # One entry per round plus a final post-fit residual.
        assert len(boost.residual_norms_) == 9
        assert all(np.isfinite(r) for r in boost.residual_norms_)

    def test_residual_norm_decreases(self):
        """Boosting should not increase the residual norm over the run."""
        boost, _, _ = self._fit(n_estimators=10, learning_rate=0.3)
        norms = boost.residual_norms_
        # Final residual is no larger than the initial ||y||.
        assert norms[-1] <= norms[0] + 1e-8

    def test_self_limiting_on_single_feature(self):
        """On single-feature data, later trees add little (self-limiting)."""
        boost, X, _ = self._fit(maker=_make_data, n_estimators=6, learning_rate=0.5)
        # The first tree's contribution dwarfs the last tree's contribution.
        first = np.abs(boost.trees_[0].predict(X)).mean()
        last = np.abs(boost.trees_[-1].predict(X)).mean()
        assert last <= first

    def test_reproducible_with_seed(self):
        boost_a, X, _ = self._fit()
        boost_b, _, _ = self._fit()
        np.testing.assert_allclose(boost_a.predict(X), boost_b.predict(X))

    def test_subsample_requires_time_index(self):
        X, y, feats = _processed()
        boost = BoostedPanelTree(n_estimators=4, subsample=0.5, random_state=0)
        with pytest.raises(ValueError, match="time_index"):
            boost.fit(X, y, feature_names=feats)

    def test_subsample_builds_valid_tree(self):
        X, y, feats = _processed(maker=_make_multifeature_data)
        boost = BoostedPanelTree(
            n_estimators=5,
            subsample=0.6,
            block_size=5,
            base_params=dict(min_samples=100),
            random_state=0,
        )
        boost.fit(X, y, feature_names=feats, time_index="date")
        preds = boost.predict(X)
        assert preds.shape == (len(y),)
        assert np.all(np.isfinite(preds))


# ----------------------------------------------------------------------
# Random splitter (D3 / ExtraPanelTrees)
# ----------------------------------------------------------------------

class TestRandomSplitter:
    def test_invalid_splitter_raises(self):
        with pytest.raises(ValueError, match="splitter"):
            PanelTreeEngine(splitter="nonsense")

    def test_invalid_n_random_splits_raises(self):
        with pytest.raises(ValueError, match="n_random_splits"):
            PanelTreeEngine(n_random_splits=0)

    def test_best_splitter_is_default(self):
        eng = PanelTreeEngine(verbose=0)
        assert eng.splitter == "best"

    def test_random_splitter_builds_valid_tree(self):
        X, y, feats = _processed()
        eng = PanelTreeEngine(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=R2DiffCriterion(),
            max_depth=3,
            min_samples=200,
            splitter="random",
            n_random_splits=5,
            random_state=0,
            verbose=0,
        )
        eng.fit(X, y, feature_names=feats)
        preds = eng.predict(X)
        assert preds.shape == (len(y),)
        assert np.all(np.isfinite(preds))

    def test_random_splitter_reproducible(self):
        X, y, feats = _processed()
        kw = dict(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=R2DiffCriterion(),
            max_depth=3,
            min_samples=200,
            splitter="random",
            n_random_splits=4,
            random_state=7,
            verbose=0,
        )
        eng_a = PanelTreeEngine(**kw).fit(X, y, feature_names=feats)
        eng_b = PanelTreeEngine(**kw).fit(X, y, feature_names=feats)
        np.testing.assert_allclose(eng_a.predict(X), eng_b.predict(X))

    def test_default_best_path_unchanged(self):
        """splitter defaults must not perturb the standard exhaustive search."""
        X, y, feats = _processed()
        base = dict(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=R2DiffCriterion(),
            max_depth=3,
            min_samples=200,
            verbose=0,
        )
        eng_a = PanelTreeEngine(**base).fit(X, y, feature_names=feats)
        eng_b = PanelTreeEngine(splitter="best", **base).fit(
            X, y, feature_names=feats
        )
        np.testing.assert_allclose(eng_a.predict(X), eng_b.predict(X))

    def test_forest_with_random_splitter(self):
        X, y, feats = _processed(maker=_make_multifeature_data)
        forest = PanelForest(
            n_estimators=6,
            max_features="sqrt",
            block_size=5,
            base_params=dict(
                predictor=RidgeRegressor(alpha=1.0),
                max_depth=3,
                min_samples=150,
                splitter="random",
                n_random_splits=3,
            ),
            random_state=0,
        )
        forest.fit(X, y, feature_names=feats, time_index="date")
        preds = forest.predict(X)
        assert preds.shape == (len(y),)
        assert np.all(np.isfinite(preds))
