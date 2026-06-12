"""
M2 parallelism correctness tests
================================

The feature-dimension parallelism (A2) must produce a tree *bit-identical*
to the serial build, because results are reduced in deterministic
``eval_order``.  We verify equality of predictions and tree structure across
``n_jobs=1`` and ``n_jobs>1`` (threading backend).
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


def _data():
    rng = np.random.default_rng(2026)
    T, N, P = 25, 80, 6
    feats = [f"char_{i + 1}" for i in range(P)]
    X = rng.standard_normal((T * N, P))
    y = np.where(X[:, 0] > 0, 0.6 * X[:, 1] - 0.4 * X[:, 2], 0.05 * X[:, 3])
    y = y + rng.standard_normal(T * N) * 0.3
    df = pd.DataFrame(X, columns=feats)
    df["date"] = np.repeat(np.arange(T), N)
    df["asset_id"] = np.tile(np.arange(N), T)
    dh = DataHandler(cs_rank_standardize=True)
    X_proc, y_proc, _ = dh.fit_transform(
        df, pd.Series(y), time_col="date", entity_col="asset_id"
    )
    return X_proc, y_proc, dh.feature_names


def _fit(n_jobs):
    X, y, feats = _data()
    engine = PanelTreeEngine(
        predictor=RidgeRegressor(alpha=1.0),
        criterion=R2DiffCriterion(),
        split_thresholds=[0.3, 0.5, 0.7],
        max_depth=3,
        min_samples=200,
        n_jobs=n_jobs,
        verbose=0,
    )
    engine.fit(X, y, feature_names=feats)
    return engine, X


def _signature(engine):
    return [
        (n.node_id, n.depth, n.is_leaf, n.split_feature,
         None if n.split_threshold is None else round(float(n.split_threshold), 6),
         n.n_samples)
        for n in engine.get_all_nodes()
    ]


class TestParallelEquivalence:
    @pytest.mark.parametrize("n_jobs", [2, 4])
    def test_parallel_matches_serial(self, n_jobs):
        serial, Xs = _fit(1)
        par, Xp = _fit(n_jobs)
        # Identical structure.
        assert _signature(serial) == _signature(par)
        # Identical predictions.
        np.testing.assert_allclose(
            serial.predict(Xs), par.predict(Xp), rtol=0, atol=1e-12
        )
