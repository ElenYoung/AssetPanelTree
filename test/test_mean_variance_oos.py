"""Regression tests for the MeanVarianceCriterion OOS pathway.

Pins the M5 polish that promotes ``MeanVarianceCriterion`` to a
first-class citizen on a par with the regression / classification
criteria:

* ``metric_key()`` advertises ``"sharpe"``.
* Each fitted node carries a scalar ``sharpe`` (plus ``port_mean`` /
  ``port_vol``) on ``node.metrics`` alongside the original ``_port_ret``
  dict.
* ``PanelTreeEngine.evaluate(metrics=("sharpe",), time_col=...)`` populates
  ``oos_sharpe`` / ``left_oos_sharpe`` / ``right_oos_sharpe`` /
  ``delta_oos_sharpe`` columns on the per-node frame.
* ``NodeReporter.print_tree`` / ``to_graphviz`` surface the
  ``OOS Sharpe`` and ``ΔSharpe`` strings.
* ``NodeReporter.plot_tree`` renders an ``"OOS Sharpe"`` row instead of
  silently omitting it (the same class of bug that previously affected
  the ``precision`` criterion).
* ``tune_ccp_alpha(metric="sharpe", time_col=...)`` returns a finite
  ``best_alpha`` and a coherent curve; the metric is "higher is better"
  so the picked alpha must hit the argmax of the OOS Sharpe curve.
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
    MeanVarianceCriterion,
    NodeReporter,
    PanelTreeEngine,
    RidgeRegressor,
)


# ----------------------------------------------------------------------
# Shared cross-sectional return panel fixture
# ----------------------------------------------------------------------


def _make_panel(
    T: int = 36,
    N: int = 60,
    P: int = 5,
    seed: int = 11,
):
    """Build a panel where ``char_0`` switches the predictability regime.

    The right regime (``char_0 > 0``) is driven by ``char_1`` / ``char_2`` /
    ``char_3`` with strong loadings; the left regime is essentially noise.
    That structure makes ``MeanVarianceCriterion`` happy: splitting on
    ``char_0`` yields two child long-short portfolios with very different
    Sharpe profiles.
    """
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
    y = pd.Series(y_raw, name="ret")

    dh = DataHandler(cs_rank_standardize=True)
    X_proc, y_proc, _ = dh.fit_transform(
        df, y, time_col="date", entity_col="asset_id"
    )
    return X_proc, y_proc, dh.feature_names


@pytest.fixture(scope="module")
def mv_engine_and_split():
    """Return a fitted MV engine plus an OOS slice."""
    X, y, fnames = _make_panel()
    train_mask = X["date"] < 26
    X_train = X.loc[train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    X_oos = X.loc[~train_mask].reset_index(drop=True)
    y_oos = y.loc[~train_mask].reset_index(drop=True)

    engine = PanelTreeEngine(
        predictor=RidgeRegressor(alpha=1.0),
        criterion=MeanVarianceCriterion(),
        max_depth=2,
        min_samples=150,
        random_state=0,
        verbose=0,
    )
    engine.fit(X_train, y_train, feature_names=fnames, time_index="date")
    return engine, X_train, y_train, X_oos, y_oos


# ----------------------------------------------------------------------
# node.metrics — scalar sharpe must accompany the _port_ret dict
# ----------------------------------------------------------------------


class TestNodeMetricsCarryScalarSharpe:
    def test_root_has_scalar_sharpe(self, mv_engine_and_split):
        engine, *_ = mv_engine_and_split
        root_m = engine.root_.metrics
        # The original _port_ret dict must still be there...
        assert "_port_ret" in root_m
        assert isinstance(root_m["_port_ret"], dict)
        # ...and the new scalar Sharpe + mean + vol must coexist with it.
        assert "sharpe" in root_m
        assert np.isfinite(root_m["sharpe"])
        assert np.isfinite(root_m["port_mean"])
        assert np.isfinite(root_m["port_vol"])

    def test_every_node_has_scalar_sharpe(self, mv_engine_and_split):
        engine, *_ = mv_engine_and_split
        for node in engine.get_all_nodes():
            assert "sharpe" in node.metrics, node.node_id
            # Sharpe may legitimately be ±inf / nan on degenerate leaves;
            # we only require the key to be present and numeric.
            v = node.metrics["sharpe"]
            assert isinstance(v, (int, float))


# ----------------------------------------------------------------------
# evaluate(metrics=("sharpe",)) — OOS columns must be populated
# ----------------------------------------------------------------------


class TestEvaluateSharpeMetric:
    def test_oos_sharpe_column_present(self, mv_engine_and_split):
        engine, _, _, X_oos, y_oos = mv_engine_and_split
        result = engine.evaluate(
            X_oos, y_oos, metrics=("sharpe",), time_col="date"
        )
        df = result.per_node_df
        assert "oos_sharpe" in df.columns
        assert "sharpe" in result.metrics
        # At least one leaf row should have a finite OOS Sharpe given the
        # held-out window covers ≥2 dates with ≥2 assets per leaf.
        leaf_rows = df[df["is_leaf"]]
        assert np.isfinite(leaf_rows["oos_sharpe"]).any()

    def test_delta_oos_sharpe_on_internal_nodes(self, mv_engine_and_split):
        engine, _, _, X_oos, y_oos = mv_engine_and_split
        result = engine.evaluate(
            X_oos, y_oos, metrics=("sharpe",), time_col="date"
        )
        df = result.per_node_df
        internals = df[~df["is_leaf"]]
        assert len(internals) >= 1
        for col in ("left_oos_sharpe", "right_oos_sharpe", "delta_oos_sharpe"):
            assert col in df.columns, col
        # delta_oos_sharpe = left_oos_sharpe - right_oos_sharpe by definition.
        finite = internals.dropna(
            subset=["left_oos_sharpe", "right_oos_sharpe", "delta_oos_sharpe"]
        )
        if not finite.empty:
            recomputed = (
                finite["left_oos_sharpe"] - finite["right_oos_sharpe"]
            )
            np.testing.assert_allclose(
                recomputed.values,
                finite["delta_oos_sharpe"].values,
                rtol=1e-9, atol=1e-12,
            )

    def test_sharpe_without_time_col_is_silently_dropped(
        self, mv_engine_and_split
    ):
        """Matching the ``rank_ic`` policy, ``evaluate`` does not raise when
        ``sharpe`` is requested without ``time_col`` — it simply omits the
        ``oos_sharpe`` columns so callers can mix-and-match metrics freely.
        """
        engine, _, _, X_oos, y_oos = mv_engine_and_split
        result = engine.evaluate(X_oos, y_oos, metrics=("sharpe",))
        assert "oos_sharpe" not in result.per_node_df.columns
        assert "sharpe" not in result.metrics


# ----------------------------------------------------------------------
# print_tree / to_graphviz — text views must include OOS Sharpe
# ----------------------------------------------------------------------


class TestTreeTextViewsSharpeOOS:
    def test_print_tree_contains_oos_sharpe(self, mv_engine_and_split):
        engine, _, _, X_oos, y_oos = mv_engine_and_split
        result = engine.evaluate(
            X_oos, y_oos, metrics=("sharpe",), time_col="date"
        )
        text = NodeReporter(engine).print_tree(
            evaluation=result, show_child_diff=True
        )
        assert "OOS Sharpe" in text, text
        assert "ΔSharpe" in text, text

    def test_graphviz_source_contains_oos_sharpe(self, mv_engine_and_split):
        engine, _, _, X_oos, y_oos = mv_engine_and_split
        result = engine.evaluate(
            X_oos, y_oos, metrics=("sharpe",), time_col="date"
        )
        dot = NodeReporter(engine).to_graphviz(evaluation=result)
        assert "OOS Sharpe" in dot, dot[:500]


# ----------------------------------------------------------------------
# plot_tree — the analogue of the precision-criterion bug
# ----------------------------------------------------------------------


class TestPlotTreeSharpeOOS:
    def test_plot_tree_renders_oos_sharpe(self, mv_engine_and_split):
        """``plot_tree`` previously had no entry for ``"sharpe"`` in
        ``_METRIC_AXIS``, silently omitting the OOS row for
        ``MeanVarianceCriterion`` users.  This regression test pins the
        fix."""
        engine, _, _, X_oos, y_oos = mv_engine_and_split
        result = engine.evaluate(
            X_oos, y_oos, metrics=("sharpe",), time_col="date"
        )
        out = NodeReporter(engine).plot_tree(
            evaluation=result, show_child_diff=True
        )
        fig = out[0] if isinstance(out, tuple) else out
        try:
            texts: list[str] = []
            for ax in fig.axes:
                for t in ax.texts:
                    texts.append(t.get_text())
            joined = "\n".join(texts)
            assert "OOS Sharpe" in joined, joined
            assert "ΔSharpe" in joined, joined
        finally:
            plt.close(fig)


# ----------------------------------------------------------------------
# tune_ccp_alpha — Sharpe must work end-to-end
# ----------------------------------------------------------------------


class TestTuneCcpAlphaSharpe:
    def test_tune_sharpe_returns_finite_alpha(self, mv_engine_and_split):
        engine, _, _, X_oos, y_oos = mv_engine_and_split
        best_alpha, curve = engine.tune_ccp_alpha(
            X_oos, y_oos, metric="sharpe", time_col="date"
        )
        assert np.isfinite(best_alpha)
        assert best_alpha >= 0.0
        assert "oos_sharpe" in curve.columns
        assert len(curve) >= 1

    def test_tune_sharpe_maximises(self, mv_engine_and_split):
        """``sharpe`` is higher-is-better — the picked alpha must
        correspond to the argmax of the observed OOS Sharpe curve."""
        engine, _, _, X_oos, y_oos = mv_engine_and_split
        best_alpha, curve = engine.tune_ccp_alpha(
            X_oos, y_oos, metric="sharpe", time_col="date"
        )
        valid = curve.dropna(subset=["oos_sharpe"])
        if not valid.empty:
            expected_alpha = float(
                valid.loc[valid["oos_sharpe"].idxmax(), "ccp_alpha"]
            )
            assert best_alpha == pytest.approx(expected_alpha)

    def test_tune_sharpe_without_time_col_raises(self, mv_engine_and_split):
        engine, _, _, X_oos, y_oos = mv_engine_and_split
        with pytest.raises(ValueError, match="time_col"):
            engine.tune_ccp_alpha(X_oos, y_oos, metric="sharpe")
