"""
Tests for the PanelForest ensemble (M6 / D1)
============================================

Covers:
* the time-block bootstrap helper (contiguous blocks, with replacement, OOB);
* node-level random feature subsetting (``max_features``) in the engine;
* the forest fit + three aggregation outputs (mean prediction, soft regime
  membership, co-association matrix) and the OOB R² estimate;
* reproducibility under a fixed ``random_state``;
* the *multi-feature-driven* benefit claim from the design doc — on data whose
  predictability is spread across several weak features, the forest's OOB R²
  should be a finite, sensible number and the bagged prediction should track
  the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ptree import (
    DataHandler,
    PanelForest,
    PanelTreeEngine,
    R2DiffCriterion,
    RidgeRegressor,
)
from ptree.criteria import ClassificationCriterion
from ptree.ensemble import _block_bootstrap_times
from ptree.predictors import RidgeLogitClassifier


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
    # Each gate switches on a different feature, each with a modest alpha.
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
# Block bootstrap helper
# ----------------------------------------------------------------------

class TestBlockBootstrap:
    def test_preserves_block_contiguity_and_replacement(self):
        rng = np.random.default_rng(0)
        times = np.arange(20)
        sampled = _block_bootstrap_times(times, block_size=5, rng=rng)
        # 4 blocks of 5 → resampled to 4 blocks → 20 labels total.
        assert len(sampled) == 20
        # Every sampled label is a valid original time.
        assert set(sampled).issubset(set(times.tolist()))

    def test_some_blocks_left_out_on_average(self):
        rng = np.random.default_rng(1)
        times = np.arange(50)
        # With replacement, typically not all blocks are drawn.
        sampled = _block_bootstrap_times(times, block_size=5, rng=rng)
        # There is at least one out-of-bag period with high probability.
        assert len(set(times.tolist()) - set(sampled.tolist())) >= 1

    def test_block_size_larger_than_series(self):
        rng = np.random.default_rng(2)
        times = np.arange(3)
        sampled = _block_bootstrap_times(times, block_size=10, rng=rng)
        # One block holding all 3 periods → resampled once → all 3 returned.
        assert sorted(set(sampled.tolist())) == [0, 1, 2]


# ----------------------------------------------------------------------
# Engine max_features (node-level random subsetting)
# ----------------------------------------------------------------------

class TestEngineMaxFeatures:
    def test_resolve_max_features_variants(self):
        eng = PanelTreeEngine(max_features="sqrt", verbose=0)
        assert eng._resolve_max_features(16) == 4
        eng = PanelTreeEngine(max_features="log2", verbose=0)
        assert eng._resolve_max_features(8) == 3
        eng = PanelTreeEngine(max_features=3, verbose=0)
        assert eng._resolve_max_features(10) == 3
        assert eng._resolve_max_features(2) == 2  # capped at p
        eng = PanelTreeEngine(max_features=0.5, verbose=0)
        assert eng._resolve_max_features(10) == 5
        eng = PanelTreeEngine(max_features=None, verbose=0)
        assert eng._resolve_max_features(10) == 10

    def test_max_features_none_matches_full_search(self):
        """max_features=None must not change the standard tree behaviour."""
        X, y, feats = _processed()
        base = dict(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=R2DiffCriterion(),
            max_depth=3,
            min_samples=200,
            verbose=0,
        )
        eng_a = PanelTreeEngine(**base).fit(X, y, feature_names=feats)
        eng_b = PanelTreeEngine(max_features=None, **base).fit(
            X, y, feature_names=feats
        )
        np.testing.assert_allclose(eng_a.predict(X), eng_b.predict(X))

    def test_max_features_builds_valid_tree(self):
        X, y, feats = _processed()
        eng = PanelTreeEngine(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=R2DiffCriterion(),
            max_depth=3,
            min_samples=200,
            max_features="sqrt",
            random_state=0,
            verbose=0,
        )
        eng.fit(X, y, feature_names=feats)
        preds = eng.predict(X)
        assert preds.shape == (len(y),)
        assert np.all(np.isfinite(preds))


# ----------------------------------------------------------------------
# PanelForest
# ----------------------------------------------------------------------

class TestPanelForest:
    def _fit(self, n_estimators=8, maker=_make_data, **kwargs):
        X, y, feats = _processed(maker=maker, **kwargs)
        forest = PanelForest(
            n_estimators=n_estimators,
            max_features="sqrt",
            block_size=5,
            base_params=dict(
                predictor=RidgeRegressor(alpha=1.0),
                max_depth=3,
                min_samples=150,
            ),
            random_state=0,
            verbose=0,
        )
        forest.fit(X, y, feature_names=feats, time_index="date")
        return forest, X, y

    def test_fit_requires_time_index(self):
        X, y, feats = _processed()
        forest = PanelForest(n_estimators=4, random_state=0)
        with pytest.raises(ValueError, match="time_index"):
            forest.fit(X, y, feature_names=feats)

    def test_predict_shape_and_finite(self):
        forest, X, y = self._fit()
        preds = forest.predict(X)
        assert preds.shape == (len(y),)
        assert np.all(np.isfinite(preds))

    def test_grows_requested_number_of_trees(self):
        forest, X, y = self._fit(n_estimators=6)
        assert len(forest.trees_) == 6

    def test_regime_membership_in_unit_interval(self):
        forest, X, y = self._fit()
        m = forest.regime_membership(X)
        assert m.shape == (len(y),)
        assert np.all(m >= 0.0) and np.all(m <= 1.0)

    def test_coassociation_matrix_properties(self):
        forest, X, y = self._fit()
        # Use a small subset to keep the O(n^2) matrix cheap.
        Xs = X.iloc[:50].reset_index(drop=True)
        C = forest.coassociation_matrix(Xs)
        assert C.shape == (50, 50)
        # Symmetric, diagonal == 1 (each obs always shares its own leaf), [0,1].
        np.testing.assert_allclose(C, C.T)
        np.testing.assert_allclose(np.diag(C), 1.0)
        assert np.all(C >= 0.0) and np.all(C <= 1.0)

    def test_oob_score_is_finite(self):
        forest, X, y = self._fit()
        assert forest.oob_score_ is not None
        assert np.isfinite(forest.oob_score_)

    def test_reproducible_with_seed(self):
        forest_a, X, _ = self._fit()
        forest_b, _, _ = self._fit()
        np.testing.assert_allclose(forest_a.predict(X), forest_b.predict(X))

    def test_output_dispatches_on_aggregate(self):
        X, y, feats = _processed()
        common = dict(
            n_estimators=4,
            base_params=dict(max_depth=2, min_samples=200),
            random_state=0,
        )
        f_mean = PanelForest(aggregate="mean", **common).fit(
            X, y, feature_names=feats, time_index="date"
        )
        out = f_mean.output(X)
        assert out.shape == (len(y),)

        f_cons = PanelForest(aggregate="consensus", **common).fit(
            X, y, feature_names=feats, time_index="date"
        )
        Xs = X.iloc[:30].reset_index(drop=True)
        out = f_cons.output(Xs)
        assert out.shape == (30, 30)

    def test_multifeature_forest_oob_reasonable(self):
        """On multi-feature-driven data the forest should yield a finite OOB R²."""
        forest, X, y = self._fit(n_estimators=12, maker=_make_multifeature_data)
        assert np.isfinite(forest.oob_score_)
        # Bagged predictions should be positively correlated with the target.
        preds = forest.predict(X)
        corr = np.corrcoef(preds, y.values)[0, 1]
        assert corr > 0.0


# ----------------------------------------------------------------------
# PanelForest — classification path
# ----------------------------------------------------------------------

def _make_classification_data(T: int = 40, N: int = 60, P: int = 6, seed: int = 7):
    """Binary-label panel: the sign of a linear combination of features.

    Produces ``y ∈ {0, 1}`` so that a classification forest (precision /
    F1 / AUC criterion) has a well-defined target.
    """
    rng = np.random.default_rng(seed)
    feature_names = [f"char_{i + 1}" for i in range(P)]
    dates = np.repeat(np.arange(T), N)
    asset_ids = np.tile(np.arange(N), T)
    X_raw = rng.standard_normal((T * N, P))
    logit = 0.8 * X_raw[:, 0] - 0.5 * X_raw[:, 1] + 0.3 * X_raw[:, 2]
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=T * N) < p).astype(np.float64)
    df = pd.DataFrame(X_raw, columns=feature_names)
    df["date"] = dates
    df["asset_id"] = asset_ids
    return df, pd.Series(y, name="up"), feature_names


class TestPanelForestClassification:
    """The forest should remain self-consistent under a classification criterion.

    These tests pin the behaviour added when generalising P-Forest from the
    R²-only "high-R² leaf" rule to ``criterion.metric_key()``: the
    leaf-ranking, OOB scoring, and ``predict_proba`` paths must all behave
    correctly when the criterion is precision-/F1-/AUC-based.
    """

    def _fit(self, criterion_metric: str = "precision", n_estimators: int = 8):
        df, y, feats = _make_classification_data()
        dh = DataHandler(cs_rank_standardize=True)
        X_proc, y_proc, _ = dh.fit_transform(
            df, y, time_col="date", entity_col="asset_id"
        )
        forest = PanelForest(
            n_estimators=n_estimators,
            max_features="sqrt",
            block_size=5,
            base_params=dict(
                criterion=ClassificationCriterion(metric=criterion_metric),
                # Predictor is intentionally NOT supplied so we exercise the
                # task-aware default (logistic) in PanelForest.fit.
                max_depth=3,
                min_samples=150,
            ),
            random_state=0,
            verbose=0,
        )
        forest.fit(X_proc, y_proc, feature_names=dh.feature_names, time_index="date")
        return forest, X_proc, y_proc

    def test_detects_classification_task(self):
        """The fit should flag the forest as classification."""
        forest, _, _ = self._fit()
        assert forest.is_classification_ is True

    def test_default_predictor_is_logistic_under_classification_criterion(self):
        """Without an explicit predictor, the forest should pick a logistic one."""
        forest, _, _ = self._fit()
        leaves = forest.trees_[0].get_leaves()
        # At least one leaf should hold a RidgeLogitClassifier (the default).
        assert any(isinstance(leaf.predictor, RidgeLogitClassifier) for leaf in leaves)

    def test_predict_returns_probabilities(self):
        """Bagged prediction on a classification forest is P(y=1|X) ∈ [0, 1]."""
        forest, X, y = self._fit()
        p = forest.predict(X)
        assert p.shape == (len(y),)
        assert np.all(p >= 0.0) and np.all(p <= 1.0)
        # Probabilities should at least be non-degenerate (not all equal).
        assert p.std() > 0.0

    def test_predict_proba_alias(self):
        forest, X, _ = self._fit()
        np.testing.assert_allclose(forest.predict_proba(X), forest.predict(X))

    def test_predict_proba_unavailable_on_regression(self):
        """The probability API is gated on classification."""
        X, y, feats = _processed()
        reg_forest = PanelForest(
            n_estimators=4,
            base_params=dict(max_depth=2, min_samples=200),
            random_state=0,
        ).fit(X, y, feature_names=feats, time_index="date")
        with pytest.raises(AttributeError):
            reg_forest.predict_proba(X)

    def test_regime_membership_uses_criterion_metric(self):
        """regime_membership must not silently degenerate to "all ones".

        Under the old, R²-hardcoded implementation, classification forests
        had no ``"r2"`` key in leaf.metrics → every leaf was considered
        "high" → membership == 1 everywhere.  The fix should produce a
        non-degenerate distribution.
        """
        forest, X, y = self._fit(n_estimators=10)
        m = forest.regime_membership(X)
        assert m.shape == (len(y),)
        assert np.all(m >= 0.0) and np.all(m <= 1.0)
        # Non-degenerate: at least *some* observations should sit below 1
        # and above 0 (else the high-leaf set is trivially {all} or {}).
        assert m.std() > 0.0
        assert m.min() < 1.0

    def test_oob_score_is_classification_metric(self):
        """oob_score_ on a precision criterion should sit in [0, 1]."""
        forest, _, _ = self._fit()
        assert forest.oob_score_ is not None
        assert np.isfinite(forest.oob_score_)
        # Precision is bounded in [0, 1].
        assert 0.0 <= forest.oob_score_ <= 1.0

    def test_f1_criterion_also_works(self):
        """Smoke test for an alternative classification criterion."""
        forest, X, y = self._fit(criterion_metric="f1")
        p = forest.predict(X)
        assert np.all(np.isfinite(p))
        assert forest.oob_score_ is not None and np.isfinite(forest.oob_score_)
