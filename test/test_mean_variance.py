"""
Tests for the mean-variance criterion and SDF factor output (M4)
================================================================

Covers:
* ``MeanVarianceCriterion.calculate_score`` — Sharpe of the tangency
  combination of two child portfolio series, including the degenerate
  guards (missing series, too-few overlapping periods).
* The engine pipeline that attaches ``_port_ret`` to leaf metrics when the
  mean-variance criterion is used, and the ``time_index`` plumbing in ``fit``.
* ``PanelTreeEngine.build_sdf_factor`` — leaf long-short portfolios combined
  into a single SDF return series with tangency weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ptree import (
    DataHandler,
    PanelTreeEngine,
    MeanVarianceCriterion,
    R2DiffCriterion,
    RidgeRegressor,
)


# ----------------------------------------------------------------------
# Synthetic panel data (same generator family as the golden baseline)
# ----------------------------------------------------------------------

def _make_data(T: int = 40, N: int = 60, P: int = 5, seed: int = 7):
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


def _processed(seed: int = 7):
    df, y, _ = _make_data(seed=seed)
    dh = DataHandler(cs_rank_standardize=True)
    X_proc, y_proc, _ = dh.fit_transform(
        df, y, time_col="date", entity_col="asset_id"
    )
    # ``date`` survives processing as a column we can feed as the time index.
    return X_proc, y_proc, dh.feature_names


# ----------------------------------------------------------------------
# Criterion-level unit tests
# ----------------------------------------------------------------------

class TestMeanVarianceCriterion:
    def test_missing_series_scores_zero(self):
        crit = MeanVarianceCriterion()
        assert crit.calculate_score({"r2": 0.1}, {"r2": 0.2}) == 0.0

    def test_too_few_periods_scores_zero(self):
        crit = MeanVarianceCriterion(min_periods=3)
        left = {"_port_ret": {0: 0.01, 1: 0.02}}
        right = {"_port_ret": {0: -0.01, 1: 0.00}}
        assert crit.calculate_score(left, right) == 0.0

    def test_positive_sharpe_for_profitable_portfolios(self):
        crit = MeanVarianceCriterion(annualization=12.0)
        rng = np.random.default_rng(0)
        times = list(range(60))
        # Two positive-mean, weakly-correlated return streams.
        rl = {t: 0.01 + 0.02 * rng.standard_normal() for t in times}
        rr = {t: 0.008 + 0.02 * rng.standard_normal() for t in times}
        score = crit.calculate_score({"_port_ret": rl}, {"_port_ret": rr})
        assert score > 0.0
        assert np.isfinite(score)

    def test_complementary_beats_redundant(self):
        """Two de-correlated streams should out-score two identical ones."""
        crit = MeanVarianceCriterion()
        rng = np.random.default_rng(1)
        times = list(range(80))
        a = {t: 0.01 + 0.02 * rng.standard_normal() for t in times}
        b = {t: 0.01 + 0.02 * rng.standard_normal() for t in times}
        # Redundant: right == left.
        redundant = crit.calculate_score({"_port_ret": a}, {"_port_ret": dict(a)})
        complementary = crit.calculate_score({"_port_ret": a}, {"_port_ret": b})
        assert complementary >= redundant

    def test_metric_key_is_r2(self):
        assert MeanVarianceCriterion().metric_key() == "r2"


# ----------------------------------------------------------------------
# Engine integration
# ----------------------------------------------------------------------

class TestEngineMeanVariance:
    def test_fit_requires_time_index(self):
        X, y, feats = _processed()
        engine = PanelTreeEngine(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=MeanVarianceCriterion(),
            max_depth=2,
            min_samples=200,
            verbose=0,
        )
        with pytest.raises(ValueError, match="time_index"):
            engine.fit(X, y, feature_names=feats)

    def test_fit_with_time_index_builds_tree(self):
        X, y, feats = _processed()
        engine = PanelTreeEngine(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=MeanVarianceCriterion(),
            max_depth=2,
            min_samples=200,
            verbose=0,
        )
        engine.fit(X, y, feature_names=feats, time_index="date")
        # The tree should have split at least once (root + 2 children).
        assert engine.root_ is not None
        assert len(engine.get_all_nodes()) >= 1
        preds = engine.predict(X)
        assert preds.shape == (len(y),)
        assert np.all(np.isfinite(preds))

    def test_time_index_as_array(self):
        X, y, feats = _processed()
        t = X["date"].values
        engine = PanelTreeEngine(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=MeanVarianceCriterion(),
            max_depth=2,
            min_samples=200,
            verbose=0,
        )
        engine.fit(X, y, feature_names=feats, time_index=t)
        assert engine.root_ is not None

    def test_leaf_metrics_carry_port_ret(self):
        X, y, feats = _processed()
        engine = PanelTreeEngine(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=MeanVarianceCriterion(),
            max_depth=2,
            min_samples=200,
            verbose=0,
        )
        engine.fit(X, y, feature_names=feats, time_index="date")
        # Root metrics are built with the (full) time series attached.
        assert "_port_ret" in engine.root_.metrics
        assert isinstance(engine.root_.metrics["_port_ret"], dict)


# ----------------------------------------------------------------------
# SDF factor output (C3)
# ----------------------------------------------------------------------

class TestBuildSdfFactor:
    def _fitted(self):
        X, y, feats = _processed()
        engine = PanelTreeEngine(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=MeanVarianceCriterion(),
            max_depth=3,
            min_samples=150,
            verbose=0,
        )
        engine.fit(X, y, feature_names=feats, time_index="date")
        return engine, X, y

    def test_sdf_factor_shapes_and_keys(self):
        engine, X, y = self._fitted()
        out = engine.build_sdf_factor()
        for key in ("weights", "leaf_ids", "times", "sdf_returns", "sharpe"):
            assert key in out
        n_leaves = len(out["leaf_ids"])
        assert out["weights"].shape == (n_leaves,)
        assert out["sdf_returns"].shape == out["times"].shape
        # Tangency weights are L1-normalised.
        assert np.isclose(np.abs(out["weights"]).sum(), 1.0)

    def test_sdf_sharpe_is_finite(self):
        engine, X, y = self._fitted()
        out = engine.build_sdf_factor()
        assert np.isfinite(out["sharpe"])

    def test_sdf_works_with_r2_tree_given_explicit_time(self):
        """build_sdf_factor must work even on a tree fit with R2Diff, as long
        as time labels are supplied explicitly."""
        df, y, _ = _make_data()
        dh = DataHandler(cs_rank_standardize=True)
        X_proc, y_proc, _ = dh.fit_transform(
            df, y, time_col="date", entity_col="asset_id"
        )
        engine = PanelTreeEngine(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=R2DiffCriterion(),
            max_depth=3,
            min_samples=200,
            verbose=0,
        )
        engine.fit(X_proc, y_proc, feature_names=dh.feature_names)
        out = engine.build_sdf_factor(
            X=X_proc, y=y_proc, time_index=X_proc["date"].values
        )
        assert np.isfinite(out["sharpe"])
        assert out["sdf_returns"].shape == out["times"].shape

    def test_build_sdf_requires_time(self):
        """An R2Diff tree fit without time labels cannot build an SDF unless
        time is provided at call time."""
        df, y, _ = _make_data()
        dh = DataHandler(cs_rank_standardize=True)
        X_proc, y_proc, _ = dh.fit_transform(
            df, y, time_col="date", entity_col="asset_id"
        )
        engine = PanelTreeEngine(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=R2DiffCriterion(),
            max_depth=2,
            min_samples=200,
            verbose=0,
        )
        engine.fit(X_proc, y_proc, feature_names=dh.feature_names)
        with pytest.raises(ValueError, match="time"):
            engine.build_sdf_factor()


# ----------------------------------------------------------------------
# Portfolio-returns helper
# ----------------------------------------------------------------------

class TestPortfolioReturns:
    def test_dollar_neutral_weighting(self):
        y_true = np.array([1.0, -1.0, 0.5, -0.5])
        y_pred = np.array([2.0, -2.0, 1.0, -1.0])
        time = np.array([0, 0, 1, 1])
        out = PanelTreeEngine._portfolio_returns(y_true, y_pred, time)
        assert set(out.keys()) == {0, 1}
        assert np.all(np.isfinite(list(out.values())))

    def test_single_obs_period_skipped(self):
        y_true = np.array([1.0, -1.0, 0.5])
        y_pred = np.array([2.0, -2.0, 1.0])
        time = np.array([0, 0, 1])  # period 1 has a single observation
        out = PanelTreeEngine._portfolio_returns(y_true, y_pred, time)
        assert 1 not in out
        assert 0 in out

    def test_zero_net_weight_skipped(self):
        # Identical predictions ⇒ de-meaned weights are all zero ⇒ skipped.
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([5.0, 5.0])
        time = np.array([0, 0])
        out = PanelTreeEngine._portfolio_returns(y_true, y_pred, time)
        assert out == {}
