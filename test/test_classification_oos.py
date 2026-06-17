"""Regression tests for classification-criterion OOS pathway.

Pins the v0.2.x fix for the bug where ``PanelTreeEngine.evaluate``,
``tune_ccp_alpha`` and ``NodeReporter.plot_tree`` silently dropped OOS
metrics when the engine was fitted with a classification criterion such as
``ClassificationCriterion(metric="precision")``.

Covers:
* ``PanelTreeEngine.evaluate(metrics=("precision","f1","auc","logloss"))``
  populates ``oos_<m>`` / ``delta_oos_<m>`` columns on the per-node frame.
* ``NodeReporter.print_tree(evaluation=...)`` and ``to_graphviz(...)``
  surface the classification OOS strings.
* ``NodeReporter.plot_tree(evaluation=...)`` renders an "OOS Prec" /
  "ΔPrec" row instead of silently omitting it.
* ``tune_ccp_alpha(metric="precision" | "logloss")`` returns a finite
  ``best_alpha`` and a coherent curve (logloss is minimised internally).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402  — headless for CI
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
import pandas as pd
import pytest

from ptree import (
    DataHandler,
    NodeReporter,
    PanelTreeEngine,
)
from ptree.criteria import ClassificationCriterion
from ptree.predictors import RidgeLogitClassifier


# ----------------------------------------------------------------------
# Shared binary-classification panel fixture
# ----------------------------------------------------------------------


def _make_binary_panel(
    T: int = 30,
    N: int = 60,
    P: int = 4,
    seed: int = 13,
):
    """Build a binary-label panel with a clear ``char_1``-driven regime."""
    rng = np.random.default_rng(seed)
    feature_names = [f"char_{i + 1}" for i in range(P)]
    dates = np.repeat(np.arange(T), N)
    asset_ids = np.tile(np.arange(N), T)
    X_raw = rng.standard_normal((T * N, P))

    # Concentrate predictability on char_1 so the tree actually splits there.
    logit = 1.2 * X_raw[:, 0] + 0.4 * X_raw[:, 1] - 0.3 * X_raw[:, 2]
    p = 1.0 / (1.0 + np.exp(-logit))
    y_raw = (rng.uniform(size=T * N) < p).astype(np.float64)

    df = pd.DataFrame(X_raw, columns=feature_names)
    df["date"] = dates
    df["asset_id"] = asset_ids
    y = pd.Series(y_raw, name="up")

    dh = DataHandler(cs_rank_standardize=True)
    X_proc, y_proc, _ = dh.fit_transform(
        df, y, time_col="date", entity_col="asset_id"
    )
    return X_proc, y_proc, dh.feature_names


@pytest.fixture(scope="module")
def cls_engine_and_split():
    """Return a fitted classification engine plus an OOS slice."""
    X, y, fnames = _make_binary_panel()
    # Train / OOS split by date (last 8 periods held out).
    train_mask = X["date"] < 22
    X_train = X.loc[train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    X_oos = X.loc[~train_mask].reset_index(drop=True)
    y_oos = y.loc[~train_mask].reset_index(drop=True)

    engine = PanelTreeEngine(
        predictor=RidgeLogitClassifier(alpha=1.0),
        criterion=ClassificationCriterion(metric="precision"),
        max_depth=2,
        min_samples=80,
        random_state=0,
        verbose=0,
    )
    engine.fit(X_train, y_train, feature_names=fnames)
    return engine, X_train, y_train, X_oos, y_oos


# ----------------------------------------------------------------------
# evaluate() — classification metric columns must be populated
# ----------------------------------------------------------------------


class TestEvaluateClassificationMetrics:
    def test_precision_column_present_on_per_node_df(
        self, cls_engine_and_split
    ):
        engine, _, _, X_oos, y_oos = cls_engine_and_split
        result = engine.evaluate(X_oos, y_oos, metrics=("precision",))
        df = result.per_node_df
        assert "oos_precision" in df.columns
        # All leaf rows should have a finite precision (the OOS pool is small
        # but non-empty by construction).
        leaf_rows = df[df["is_leaf"]]
        assert leaf_rows["n_oos"].sum() == len(X_oos)
        assert np.isfinite(leaf_rows["oos_precision"]).any()

    def test_all_four_classification_metrics(self, cls_engine_and_split):
        engine, _, _, X_oos, y_oos = cls_engine_and_split
        result = engine.evaluate(
            X_oos, y_oos, metrics=("precision", "f1", "auc", "logloss")
        )
        for m in ("precision", "f1", "auc", "logloss"):
            assert f"oos_{m}" in result.per_node_df.columns, m
            assert m in result.metrics, m

    def test_delta_columns_on_internal_nodes(self, cls_engine_and_split):
        engine, _, _, X_oos, y_oos = cls_engine_and_split
        result = engine.evaluate(X_oos, y_oos, metrics=("precision", "f1"))
        df = result.per_node_df
        internals = df[~df["is_leaf"]]
        # The tree has at least one internal node (root) at max_depth=2.
        assert len(internals) >= 1
        for m in ("precision", "f1"):
            for col in (f"left_oos_{m}", f"right_oos_{m}", f"delta_oos_{m}"):
                assert col in df.columns, col
            # ``delta_oos_<m>`` should be finite on at least one internal node.
            assert np.isfinite(internals[f"delta_oos_{m}"]).any()


# ----------------------------------------------------------------------
# print_tree / to_graphviz — text views must include classification OOS
# ----------------------------------------------------------------------


class TestTreeTextViewsClassificationOOS:
    def test_print_tree_contains_oos_prec(self, cls_engine_and_split):
        engine, _, _, X_oos, y_oos = cls_engine_and_split
        result = engine.evaluate(X_oos, y_oos, metrics=("precision",))
        text = NodeReporter(engine).print_tree(
            evaluation=result, show_child_diff=True
        )
        assert "OOS Prec" in text, text

    def test_graphviz_source_contains_oos_prec(self, cls_engine_and_split):
        engine, _, _, X_oos, y_oos = cls_engine_and_split
        result = engine.evaluate(X_oos, y_oos, metrics=("precision",))
        dot = NodeReporter(engine).to_graphviz(evaluation=result)
        assert "OOS Prec" in dot, dot[:500]


# ----------------------------------------------------------------------
# plot_tree — the original bug: must now emit an OOS row for precision
# ----------------------------------------------------------------------


class TestPlotTreeClassificationOOS:
    def test_plot_tree_renders_oos_prec(self, cls_engine_and_split):
        """``plot_tree`` previously omitted OOS text for the precision
        criterion because ``_METRIC_AXIS["precision"]`` had ``None`` OOS
        columns; this regression test pins the fix."""
        engine, _, _, X_oos, y_oos = cls_engine_and_split
        result = engine.evaluate(X_oos, y_oos, metrics=("precision",))
        out = NodeReporter(engine).plot_tree(
            evaluation=result, show_child_diff=True
        )
        # ``plot_tree`` may return either a ``Figure`` or a ``(fig, ax)`` tuple
        # depending on implementation — accept both.
        fig = out[0] if isinstance(out, tuple) else out
        try:
            texts: list[str] = []
            for ax in fig.axes:
                for t in ax.texts:
                    texts.append(t.get_text())
            joined = "\n".join(texts)
            assert "OOS Prec" in joined, joined
        finally:
            plt.close(fig)


# ----------------------------------------------------------------------
# tune_ccp_alpha — classification metrics must work end-to-end
# ----------------------------------------------------------------------


class TestTuneCcpAlphaClassification:
    def test_tune_precision_returns_finite_alpha(self, cls_engine_and_split):
        engine, _, _, X_oos, y_oos = cls_engine_and_split
        best_alpha, curve = engine.tune_ccp_alpha(
            X_oos, y_oos, metric="precision"
        )
        assert np.isfinite(best_alpha)
        assert best_alpha >= 0.0
        assert "oos_precision" in curve.columns
        assert len(curve) >= 1

    def test_tune_logloss_minimises(self, cls_engine_and_split):
        """``logloss`` is "lower is better" — the picked alpha must
        correspond to the *minimum* observed OOS log-loss."""
        engine, _, _, X_oos, y_oos = cls_engine_and_split
        best_alpha, curve = engine.tune_ccp_alpha(
            X_oos, y_oos, metric="logloss"
        )
        assert np.isfinite(best_alpha)
        valid = curve.dropna(subset=["oos_logloss"])
        if not valid.empty:
            expected_alpha = float(
                valid.loc[valid["oos_logloss"].idxmin(), "ccp_alpha"]
            )
            assert best_alpha == pytest.approx(expected_alpha)
