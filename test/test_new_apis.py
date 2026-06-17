"""Tests for the new public APIs introduced in v0.2.0.

Covers:
- ``PanelTreeEngine.predict_leaves`` / ``predict_node_path``
- ``PanelTreeEngine.evaluate`` (+ ``NodeEvalResult``)
- ``PanelTreeEngine.tune_ccp_alpha``
- ``RankICDiffCriterion``
- ``NodeReporter.print_tree(evaluation=..., show_child_diff=True)``
- ``NodeReporter.to_graphviz``
- ``PanelForest`` ``regime_metric`` / ``regime_aggregation`` options
- Pickling persistence of ``node.n_samples`` (the
  ``_n_samples_cached`` fix)
- ``WeightedR2DiffCriterion`` ``min_child_weight`` parameter
- ``MosaicVisualizer.plot_mosaic`` default cmap + ``center`` param
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from ptree import (
    DataHandler,
    NodeEvalResult,
    NodeReporter,
    PanelForest,
    PanelTreeEngine,
    R2DiffCriterion,
    RankICDiffCriterion,
    RidgeRegressor,
    WeightedR2DiffCriterion,
    __version__,
)


# ----------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def panel_data():
    """Mid-sized panel: 20 periods × 50 assets × 4 features."""
    rng = np.random.default_rng(2026)
    T, N, P = 20, 50, 4
    feature_names = [f"char_{i+1}" for i in range(P)]
    dates = np.repeat(np.arange(T), N)
    asset_ids = np.tile(np.arange(N), T)
    X_raw = rng.standard_normal((T * N, P))

    # Predictability concentrated on char_1 > 0.
    alpha = 0.6 * X_raw[:, 0] + 0.2 * X_raw[:, 1]
    y_raw = np.where(X_raw[:, 0] > 0, alpha, 0.05 * X_raw[:, 2])
    y_raw += rng.standard_normal(T * N) * 0.3

    df = pd.DataFrame(X_raw, columns=feature_names)
    df["date"] = dates
    df["asset_id"] = asset_ids
    y = pd.Series(y_raw, name="ret")

    dh = DataHandler(cs_rank_standardize=True)
    X_proc, y_proc, _ = dh.fit_transform(
        df, y, time_col="date", entity_col="asset_id"
    )
    return X_proc, y_proc, dh.feature_names


@pytest.fixture
def fitted_engine(panel_data):
    X, y, fnames = panel_data
    engine = PanelTreeEngine(
        predictor=RidgeRegressor(alpha=1.0),
        criterion=R2DiffCriterion(),
        max_depth=2,
        min_samples=80,
        random_state=0,
        verbose=0,
    )
    engine.fit(X, y, feature_names=fnames)
    return engine, X, y


# ----------------------------------------------------------------------
# Version + exports
# ----------------------------------------------------------------------


def test_version_is_v2():
    assert __version__ == "0.3.0"


def test_new_exports_present():
    """The v0.2.0 names must be importable from the top-level package."""
    from ptree import NodeEvalResult as _N
    from ptree import RankICDiffCriterion as _R

    assert _N is NodeEvalResult
    assert _R is RankICDiffCriterion


# ----------------------------------------------------------------------
# predict_leaves / predict_node_path
# ----------------------------------------------------------------------


class TestPredictLeaves:
    def test_predict_leaves_shape(self, fitted_engine):
        engine, X, _ = fitted_engine
        leaf_ids = engine.predict_leaves(X)
        assert isinstance(leaf_ids, np.ndarray)
        assert leaf_ids.shape == (len(X),)
        assert leaf_ids.dtype.kind in {"i", "u"}

    def test_predict_leaves_subset_of_leaves(self, fitted_engine):
        engine, X, _ = fitted_engine
        leaf_ids = engine.predict_leaves(X)
        expected = {lf.node_id for lf in engine.get_leaves()}
        assert set(int(x) for x in leaf_ids) <= expected

    def test_predict_node_path_starts_at_root(self, fitted_engine):
        engine, X, _ = fitted_engine
        paths = engine.predict_node_path(X.iloc[:5])
        assert len(paths) == 5
        for p in paths:
            assert p[0] == engine.root_.node_id
            assert p[-1] in {lf.node_id for lf in engine.get_leaves()}


# ----------------------------------------------------------------------
# evaluate() + NodeEvalResult
# ----------------------------------------------------------------------


class TestEvaluate:
    def test_returns_node_eval_result(self, fitted_engine):
        engine, X, y = fitted_engine
        result = engine.evaluate(X, y, time_col="date", metrics=("r2", "rank_ic"))
        assert isinstance(result, NodeEvalResult)
        assert isinstance(result.per_node_df, pd.DataFrame)
        assert "node_id" in result.per_node_df.columns
        assert "oos_r2" in result.per_node_df.columns
        assert "oos_rank_ic_mean" in result.per_node_df.columns

    def test_evaluate_without_time_col_drops_rank_ic(self, fitted_engine):
        engine, X, y = fitted_engine
        # X still has a "date" column but we pass time_col=None → rank_ic dropped
        X2 = X.drop(columns=["date"])
        result = engine.evaluate(X2, y, time_col=None, metrics=("r2", "rank_ic"))
        assert "r2" in result.metrics
        assert "rank_ic" not in result.metrics

    def test_evaluate_leaf_assignments(self, fitted_engine):
        engine, X, y = fitted_engine
        result = engine.evaluate(X, y, time_col="date", metrics=("r2",))
        assert result.leaf_assignments is not None
        assert len(result.leaf_assignments) == len(X)

    def test_evaluate_per_node_metrics_dict(self, fitted_engine):
        engine, X, y = fitted_engine
        result = engine.evaluate(X, y, time_col="date", metrics=("r2",))
        for node in engine.get_all_nodes():
            assert node.node_id in result.per_node_metrics


# ----------------------------------------------------------------------
# tune_ccp_alpha()
# ----------------------------------------------------------------------


class TestTuneCcpAlpha:
    def test_returns_alpha_and_curve(self, fitted_engine):
        engine, X, y = fitted_engine
        best_alpha, curve = engine.tune_ccp_alpha(X, y, metric="r2")
        assert isinstance(best_alpha, float)
        assert isinstance(curve, pd.DataFrame)
        assert "ccp_alpha" in curve.columns
        assert "n_leaves" in curve.columns
        assert len(curve) >= 1


# ----------------------------------------------------------------------
# RankICDiffCriterion
# ----------------------------------------------------------------------


class TestRankICDiffCriterion:
    def test_instantiation(self):
        crit = RankICDiffCriterion(balance=False, min_child_weight=0.1)
        assert crit.balance is False
        assert crit.min_child_weight == 0.1

    def test_min_child_weight_validation(self):
        with pytest.raises(ValueError):
            RankICDiffCriterion(min_child_weight=0.6)

    def test_engine_runs_with_rank_ic_criterion(self, panel_data):
        X, y, fnames = panel_data
        engine = PanelTreeEngine(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=RankICDiffCriterion(),
            max_depth=2,
            min_samples=80,
            random_state=0,
            verbose=0,
        )
        engine.fit(
            X, y, feature_names=fnames,
            time_index="date",
        )
        assert engine.root_ is not None
        preds = engine.predict(X)
        assert preds.shape == (len(X),)


# ----------------------------------------------------------------------
# NodeReporter.print_tree(evaluation=, show_child_diff=)
# ----------------------------------------------------------------------


class TestPrintTreeWithEvaluation:
    def test_print_tree_with_evaluation(self, fitted_engine):
        engine, X, y = fitted_engine
        result = engine.evaluate(X, y, time_col="date", metrics=("r2", "rank_ic"))
        reporter = NodeReporter(engine)
        text = reporter.print_tree(evaluation=result, show_child_diff=True)
        assert isinstance(text, str)
        assert len(text) > 0
        # Should mention OOS metric markers we added in the visualization
        # layer (case-insensitive sanity check).
        assert "OOS" in text or "oos" in text.lower()

    def test_print_tree_no_eval_still_works(self, fitted_engine):
        engine, _, _ = fitted_engine
        text = NodeReporter(engine).print_tree()
        assert isinstance(text, str) and len(text) > 0


# ----------------------------------------------------------------------
# NodeReporter.to_graphviz
# ----------------------------------------------------------------------


class TestToGraphviz:
    def test_to_graphviz_returns_dot_source(self, fitted_engine):
        engine, _, _ = fitted_engine
        dot = NodeReporter(engine).to_graphviz()
        assert isinstance(dot, str)
        assert "digraph" in dot
        # Should have one node per node in the tree.
        for node in engine.get_all_nodes():
            assert str(node.node_id) in dot


# ----------------------------------------------------------------------
# PanelForest regime_metric / regime_aggregation
# ----------------------------------------------------------------------


class TestPanelForestRegimeOptions:
    def test_default_regime_options(self, panel_data):
        X, y, fnames = panel_data
        forest = PanelForest(
            n_estimators=5,
            max_features="sqrt",
            block_size=4,
            base_params={"max_depth": 2, "min_samples": 80},
            random_state=0,
        )
        assert forest.regime_metric == "train_r2"
        assert forest.regime_aggregation == "train"
        forest.fit(X, y, feature_names=fnames, time_index="date")
        mem = forest.regime_membership(X)
        assert mem.shape == (len(X),)
        assert ((mem >= 0) & (mem <= 1)).all()

    def test_oof_regime_aggregation(self, panel_data):
        X, y, fnames = panel_data
        forest = PanelForest(
            n_estimators=5,
            max_features="sqrt",
            block_size=4,
            base_params={"max_depth": 2, "min_samples": 80},
            random_state=0,
            regime_metric="oof_r2",
            regime_aggregation="oof",
        )
        forest.fit(X, y, feature_names=fnames, time_index="date")
        mem = forest.regime_membership(X)
        assert mem.shape == (len(X),)
        assert ((mem >= 0) & (mem <= 1)).all()

    def test_rank_ic_regime_aggregation(self, panel_data):
        X, y, fnames = panel_data
        forest = PanelForest(
            n_estimators=5,
            max_features="sqrt",
            block_size=4,
            base_params={"max_depth": 2, "min_samples": 80},
            random_state=0,
            regime_metric="rank_ic",
            regime_aggregation="oof",
        )
        forest.fit(X, y, feature_names=fnames, time_index="date")
        mem = forest.regime_membership(X)
        assert mem.shape == (len(X),)

    def test_invalid_regime_metric(self):
        with pytest.raises(ValueError):
            PanelForest(regime_metric="bogus")

    def test_invalid_regime_aggregation(self):
        with pytest.raises(ValueError):
            PanelForest(regime_aggregation="bogus")


# ----------------------------------------------------------------------
# node.n_samples persistence across pickle
# ----------------------------------------------------------------------


class TestNodeNSamplesPersistence:
    def test_n_samples_survives_pickle(self, fitted_engine):
        engine, _, _ = fitted_engine
        before = {n.node_id: n.n_samples for n in engine.get_all_nodes()}
        # Every node should have a positive n_samples on the fitted engine.
        assert all(v > 0 for v in before.values()), before

        blob = pickle.dumps(engine)
        restored = pickle.loads(blob)
        after = {n.node_id: n.n_samples for n in restored.get_all_nodes()}
        assert before == after


# ----------------------------------------------------------------------
# WeightedR2DiffCriterion min_child_weight
# ----------------------------------------------------------------------


class TestWeightedR2MinChildWeight:
    def test_min_child_weight_accepted(self):
        crit = WeightedR2DiffCriterion(min_child_weight=0.1)
        assert crit.min_child_weight == 0.1

    def test_min_child_weight_validation(self):
        with pytest.raises(ValueError):
            WeightedR2DiffCriterion(min_child_weight=0.5)
        with pytest.raises(ValueError):
            WeightedR2DiffCriterion(min_child_weight=-0.01)


# ----------------------------------------------------------------------
# MosaicVisualizer default cmap + center kwarg
# ----------------------------------------------------------------------


class TestMosaicDefaults:
    def test_plot_mosaic_signature_has_center_and_rdbu(self):
        """The mosaic plot must keep a ``center`` knob and default to
        the diverging ``RdBu_r`` palette whenever the data contains
        negative values (auto-detected when ``cmap=None``).
        """
        import inspect
        import numpy as np

        from ptree import MosaicVisualizer
        from ptree.visualization import _UNSET

        sig = inspect.signature(MosaicVisualizer.plot_mosaic)
        params = sig.parameters
        assert "center" in params
        # ``cmap`` is now auto-selected when left unset (None).  The
        # divergent default for negative-containing data must still be
        # ``RdBu_r`` so existing callers behave the same way.
        assert params["cmap"].default is None
        cmap, center = MosaicVisualizer._auto_cmap_and_center(
            np.array([-0.1, 0.2, 0.3]),
            metric=None, cmap=None, center=_UNSET,
        )
        assert cmap == "RdBu_r"
        assert center == 0.0


