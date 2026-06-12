"""
Tests for M5: ElasticNetRegressor, PLSRegressor, and the logloss metric.

These predictors do not participate in the A1 incremental Ridge path; the
engine fits them directly, so the tests also confirm end-to-end tree building.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ptree import (
    DataHandler,
    ElasticNetRegressor,
    PLSRegressor,
    PanelTreeEngine,
    R2DiffCriterion,
    ClassificationCriterion,
    RidgeRegressor,
)
from ptree.criteria import evaluate_classification


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 6))
    beta = np.array([1.5, 0.0, -0.8, 0.0, 0.3, 0.0])
    y = X @ beta + rng.standard_normal(200) * 0.1
    return X, y, beta


class TestElasticNet:
    def test_fits_and_predicts(self, regression_data):
        X, y, _ = regression_data
        model = ElasticNetRegressor(alpha=0.01, l1_ratio=0.5)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (200,)
        assert model.get_coefficients().shape == (6,)

    def test_l1_induces_sparsity(self, regression_data):
        X, y, _ = regression_data
        # Strong L1 should zero out at least one (truly irrelevant) coefficient.
        model = ElasticNetRegressor(alpha=0.5, l1_ratio=1.0)
        model.fit(X, y)
        coefs = model.get_coefficients()
        assert np.sum(np.abs(coefs) < 1e-8) >= 1

    def test_recovers_signal(self, regression_data):
        X, y, beta = regression_data
        model = ElasticNetRegressor(alpha=0.001, l1_ratio=0.1)
        model.fit(X, y)
        # The strongest true coefficient (index 0) should be clearly positive.
        assert model.get_coefficients()[0] > 0.5

    def test_weights_supported(self, regression_data):
        X, y, _ = regression_data
        w = np.abs(np.random.default_rng(1).standard_normal(200)) + 0.1
        model = ElasticNetRegressor(alpha=0.01)
        model.fit(X, y, weights=w)
        assert np.all(np.isfinite(model.predict(X)))


class TestPLS:
    def test_fits_and_predicts(self, regression_data):
        X, y, _ = regression_data
        model = PLSRegressor(n_components=2)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (200,)
        assert model.get_coefficients().shape == (6,)

    def test_components_capped_at_features(self, regression_data):
        X, y, _ = regression_data
        model = PLSRegressor(n_components=99)
        model.fit(X, y)  # must not raise even though n_components > p
        assert np.all(np.isfinite(model.predict(X)))

    def test_reasonable_fit(self, regression_data):
        X, y, _ = regression_data
        model = PLSRegressor(n_components=4)
        model.fit(X, y)
        preds = model.predict(X)
        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        assert 1.0 - ss_res / ss_tot > 0.8


class TestEngineWithNewPredictors:
    @pytest.fixture
    def panel(self):
        rng = np.random.default_rng(2026)
        T, N, P = 25, 60, 4
        feats = [f"char_{i+1}" for i in range(P)]
        dates = np.repeat(np.arange(T), N)
        ids = np.tile(np.arange(N), T)
        Xr = rng.standard_normal((T * N, P))
        alpha = 0.5 * Xr[:, 0] + 0.3 * Xr[:, 1]
        y = np.where(Xr[:, 0] > 0, alpha, 0.1 * Xr[:, 2]) + rng.standard_normal(T * N) * 0.3
        df = pd.DataFrame(Xr, columns=feats)
        df["date"] = dates
        df["asset_id"] = ids
        return df, pd.Series(y, name="ret")

    def test_engine_elasticnet(self, panel):
        df, y = panel
        dh = DataHandler(cs_rank_standardize=True)
        Xp, yp, _ = dh.fit_transform(df, y, time_col="date", entity_col="asset_id")
        engine = PanelTreeEngine(
            predictor=ElasticNetRegressor(alpha=0.01),
            criterion=R2DiffCriterion(),
            max_depth=2, min_samples=100, verbose=0,
        )
        engine.fit(Xp, yp, feature_names=dh.feature_names)
        preds = engine.predict(Xp)
        assert np.all(np.isfinite(preds))

    def test_engine_pls(self, panel):
        df, y = panel
        dh = DataHandler(cs_rank_standardize=True)
        Xp, yp, _ = dh.fit_transform(df, y, time_col="date", entity_col="asset_id")
        engine = PanelTreeEngine(
            predictor=PLSRegressor(n_components=2),
            criterion=R2DiffCriterion(),
            max_depth=2, min_samples=100, verbose=0,
        )
        engine.fit(Xp, yp, feature_names=dh.feature_names)
        preds = engine.predict(Xp)
        assert np.all(np.isfinite(preds))


class TestLogLoss:
    def test_logloss_in_metrics(self):
        rng = np.random.default_rng(0)
        y = (rng.random(100) > 0.5).astype(int)
        proba = rng.random(100)
        m = evaluate_classification(y, proba)
        assert "logloss" in m
        assert m["logloss"] > 0

    def test_perfect_prediction_low_logloss(self):
        y = np.array([0, 1, 0, 1, 1])
        proba = np.array([0.01, 0.99, 0.02, 0.98, 0.97])
        m = evaluate_classification(y, proba)
        assert m["logloss"] < 0.1

    def test_criterion_accepts_logloss(self):
        crit = ClassificationCriterion(metric="logloss")
        left = {"logloss": 0.2, "n_samples": 50}
        right = {"logloss": 0.6, "n_samples": 50}
        assert crit.calculate_score(left, right) == pytest.approx(0.4)
        assert crit.metric_key() == "logloss"

    def test_invalid_metric_still_raises(self):
        with pytest.raises(ValueError):
            ClassificationCriterion(metric="nonsense")
