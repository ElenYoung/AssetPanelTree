"""Basic tests for ptree package."""

import numpy as np
import pandas as pd
import pytest

from ptree import (
    __version__,
    DataHandler,
    RidgeRegressor,
    VolWeightedRidgeRegressor,
    RidgeLogitClassifier,
    R2DiffCriterion,
    ClassificationCriterion,
    PanelTreeEngine,
    PanelTreeNode,
    NodeReporter,
    MosaicVisualizer,
)


class TestImports:
    """Test that all public API imports work correctly."""

    def test_version(self):
        assert __version__ == "0.2.1"

    def test_all_classes_importable(self):
        assert DataHandler is not None
        assert RidgeRegressor is not None
        assert VolWeightedRidgeRegressor is not None
        assert RidgeLogitClassifier is not None
        assert R2DiffCriterion is not None
        assert ClassificationCriterion is not None
        assert PanelTreeEngine is not None
        assert PanelTreeNode is not None
        assert NodeReporter is not None
        assert MosaicVisualizer is not None


class TestDataHandler:
    """Test DataHandler functionality."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n_samples = 100
        n_assets = 10
        n_periods = 10

        dates = np.repeat(np.arange(n_periods), n_assets)
        asset_ids = np.tile(np.arange(n_assets), n_periods)

        df = pd.DataFrame({
            "date": dates,
            "asset_id": asset_ids,
            "char_1": np.random.randn(n_samples),
            "char_2": np.random.randn(n_samples),
            "char_3": np.random.randn(n_samples),
        })
        y = pd.Series(np.random.randn(n_samples), name="ret")
        return df, y

    def test_fit_transform(self, sample_data):
        df, y = sample_data
        dh = DataHandler(cs_rank_standardize=True)
        X_proc, y_proc, vol_weights = dh.fit_transform(
            df, y, time_col="date", entity_col="asset_id"
        )
        assert X_proc is not None
        assert y_proc is not None
        assert dh.feature_names == ["char_1", "char_2", "char_3"]

    def test_rank_standardization(self, sample_data):
        df, y = sample_data
        dh = DataHandler(cs_rank_standardize=True)
        X_proc, _, _ = dh.fit_transform(df, y, time_col="date", entity_col="asset_id")
        # After rank standardization, values should be in [0, 1]
        for col in dh.feature_names:
            assert X_proc[col].min() >= 0.0
            assert X_proc[col].max() <= 1.0


class TestPredictors:
    """Test predictor models."""

    @pytest.fixture
    def regression_data(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X @ np.array([1.0, -0.5, 0.3]) + np.random.randn(100) * 0.1
        return X, y

    def test_ridge_regressor(self, regression_data):
        X, y = regression_data
        model = RidgeRegressor(alpha=1.0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)
        assert model.get_coefficients() is not None
        assert len(model.get_coefficients()) == 3

    def test_vol_weighted_ridge(self, regression_data):
        X, y = regression_data
        weights = np.abs(np.random.randn(100)) + 0.1
        model = VolWeightedRidgeRegressor(alpha=1.0)
        model.fit(X, y, weights=weights)
        preds = model.predict(X)
        assert preds.shape == (100,)


class TestCriteria:
    """Test split criteria."""

    def test_r2_diff_criterion(self):
        criterion = R2DiffCriterion()
        left_metrics = {"r2": 0.8, "n_samples": 50}
        right_metrics = {"r2": 0.2, "n_samples": 50}
        score = criterion.calculate_score(left_metrics, right_metrics)
        assert score == pytest.approx(0.6)

    def test_classification_criterion(self):
        criterion = ClassificationCriterion(metric="precision")
        left_metrics = {"precision": 0.9, "n_samples": 50}
        right_metrics = {"precision": 0.5, "n_samples": 50}
        score = criterion.calculate_score(left_metrics, right_metrics)
        assert score == pytest.approx(0.4)


class TestPanelTreeEngine:
    """Test the main engine."""

    @pytest.fixture
    def panel_data(self):
        np.random.seed(2026)
        T, N, P = 20, 50, 3
        feature_names = [f"char_{i+1}" for i in range(P)]

        dates = np.repeat(np.arange(T), N)
        asset_ids = np.tile(np.arange(N), T)
        X_raw = np.random.randn(T * N, P)

        # Create predictability pattern
        alpha = 0.5 * X_raw[:, 0] + 0.3 * X_raw[:, 1]
        y_raw = np.where(X_raw[:, 0] > 0, alpha, 0.1 * X_raw[:, 2])
        y_raw += np.random.randn(T * N) * 0.3

        df = pd.DataFrame(X_raw, columns=feature_names)
        df["date"] = dates
        df["asset_id"] = asset_ids
        y = pd.Series(y_raw, name="ret")

        return df, y, feature_names

    def test_fit_predict(self, panel_data):
        df, y, feature_names = panel_data

        dh = DataHandler(cs_rank_standardize=True)
        X_proc, y_proc, _ = dh.fit_transform(df, y, time_col="date", entity_col="asset_id")

        engine = PanelTreeEngine(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=R2DiffCriterion(),
            max_depth=2,
            min_samples=50,
            verbose=0,
        )
        engine.fit(X_proc, y_proc, feature_names=dh.feature_names)

        assert engine.root_ is not None
        preds = engine.predict(X_proc)
        assert preds.shape == y_proc.shape

    def test_get_leaves(self, panel_data):
        df, y, feature_names = panel_data

        dh = DataHandler(cs_rank_standardize=True)
        X_proc, y_proc, _ = dh.fit_transform(df, y, time_col="date", entity_col="asset_id")

        engine = PanelTreeEngine(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=R2DiffCriterion(),
            max_depth=2,
            min_samples=50,
            verbose=0,
        )
        engine.fit(X_proc, y_proc, feature_names=dh.feature_names)

        leaves = engine.get_leaves()
        assert len(leaves) >= 1
        for leaf in leaves:
            assert leaf.is_leaf

    def test_node_reporter(self, panel_data):
        df, y, feature_names = panel_data

        dh = DataHandler(cs_rank_standardize=True)
        X_proc, y_proc, _ = dh.fit_transform(df, y, time_col="date", entity_col="asset_id")

        engine = PanelTreeEngine(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=R2DiffCriterion(),
            max_depth=2,
            min_samples=50,
            verbose=0,
        )
        engine.fit(X_proc, y_proc, feature_names=dh.feature_names)

        reporter = NodeReporter(engine)
        summary = reporter.summary()
        assert isinstance(summary, pd.DataFrame)
        assert "Node_ID" in summary.columns

        tree_str = reporter.print_tree()
        assert isinstance(tree_str, str)
        assert len(tree_str) > 0
