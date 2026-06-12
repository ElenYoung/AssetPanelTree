"""
Golden baseline regression tests
================================

These tests pin down the numerical behaviour of ``PanelTreeEngine`` so that
subsequent A-phase refactors (incremental-update fix, AUC vectorisation,
Cholesky solver, parallelism) can be proven to *preserve results*.

Strategy
--------
We fit a fixed-seed engine on deterministic synthetic panel data and assert
against golden constants captured from the *pre-refactor* baseline
implementation (commit prior to the A-phase changes, 2026-06):

* ``BASELINE_OVERALL_R2`` — full-sample R² of the prediction mosaic;
* ``BASELINE_PRED_SUM``    — sum of the prediction vector (cheap fingerprint);
* ``BASELINE_STRUCTURE``   — per-node (id, depth, is_leaf, split_feature,
  split_threshold, n_samples) signature of the whole tree.

Because the A-phase refactors are **result-preserving by design**, these
constants must remain unchanged.  Any movement signals accidental drift and
must fail CI.  Update them ONLY when a behavioural change is *intended* and
reviewed (e.g. a new default criterion).
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

# ----------------------------------------------------------------------
# Golden constants — captured from the baseline implementation.
# DO NOT edit unless a behavioural change is intended and reviewed.
# ----------------------------------------------------------------------
BASELINE_OVERALL_R2 = 0.6872558754854161
BASELINE_PRED_SUM = 29.66100593702541
BASELINE_STRUCTURE = [
    (0, 0, False, "char_1", 0.5, 2400),
    (1, 1, False, "char_4", 0.7, 1170),
    (6, 1, False, "char_1", 0.7, 1230),
    (2, 2, False, "char_2", 0.3, 815),
    (5, 2, True, None, None, 355),
    (7, 2, False, "char_2", 0.5, 480),
    (10, 2, False, "char_2", 0.3, 750),
    (3, 3, True, None, None, 232),
    (4, 3, True, None, None, 583),
    (8, 3, True, None, None, 236),
    (9, 3, True, None, None, 244),
    (11, 3, True, None, None, 213),
    (12, 3, True, None, None, 537),
]


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


def _fit_engine(**overrides):
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


def _structure_signature(engine):
    """Return an order-stable description of the tree structure."""
    sig = []
    for node in engine.get_all_nodes():
        sig.append(
            (
                node.node_id,
                node.depth,
                bool(node.is_leaf),
                node.split_feature,
                None
                if node.split_threshold is None
                else round(float(node.split_threshold), 6),
                node.n_samples,
            )
        )
    return sig


class TestGoldenBaseline:
    """Pin current engine behaviour for refactor safety."""

    def test_predict_is_deterministic(self):
        """Two independent fits on identical data must match exactly."""
        engine_a, X_a, _ = _fit_engine()
        engine_b, X_b, _ = _fit_engine()
        preds_a = engine_a.predict(X_a)
        preds_b = engine_b.predict(X_b)
        assert np.array_equal(np.isnan(preds_a), np.isnan(preds_b))
        np.testing.assert_allclose(preds_a, preds_b, rtol=0, atol=0)

    def test_structure_matches_golden(self):
        engine, _, _ = _fit_engine()
        assert _structure_signature(engine) == BASELINE_STRUCTURE

    def test_predictions_finite_and_complete(self):
        engine, X, y = _fit_engine()
        preds = engine.predict(X)
        assert preds.shape == (len(y),)
        # Every sample is routed to some leaf → no NaN predictions.
        assert np.all(np.isfinite(preds))

    def test_leaf_partition_is_a_partition(self):
        """Leaf sample-index sets must be disjoint and cover all samples."""
        engine, X, y = _fit_engine()
        leaf_samples = engine.get_leaf_samples()
        all_idx = np.concatenate(list(leaf_samples.values()))
        # Disjoint + complete ⇒ sorted union equals arange(n).
        assert np.array_equal(np.sort(all_idx), np.arange(len(y)))

    def test_prediction_fingerprint_matches_golden(self):
        engine, X, _ = _fit_engine()
        preds = engine.predict(X)
        assert float(np.nansum(preds)) == pytest.approx(
            BASELINE_PRED_SUM, rel=0, abs=1e-6
        )

    def test_overall_r2_matches_golden(self):
        """Full-sample R² must match the pre-refactor baseline digest."""
        engine, X, y = _fit_engine()
        preds = engine.predict(X)
        ss_res = float(np.nansum((y.values - preds) ** 2))
        ss_tot = float(np.nansum((y.values - y.mean()) ** 2))
        overall_r2 = 1.0 - ss_res / ss_tot
        assert overall_r2 == pytest.approx(BASELINE_OVERALL_R2, rel=0, abs=1e-9)
