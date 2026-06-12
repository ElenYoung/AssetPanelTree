"""
M1 acceleration correctness tests
=================================

Verify that the A-phase optimisations are *result-preserving*:

* A1 — incremental sufficient-statistics update: the "smaller-side + parent
  subtraction" trick must yield the same Ridge solution as fitting each child
  subset directly.
* A3 — vectorised Mann-Whitney AUC must match the brute-force double loop.
* A4 — Cholesky solver must match the previous LU solve.
"""

from __future__ import annotations

import numpy as np
import pytest

from ptree.criteria import _auc_mannwhitney
from ptree.predictors import (
    RidgeRegressor,
    _ridge_closed_form,
    compute_XtWX_XtWy,
)


def _auc_bruteforce(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """The original O(n_pos * n_neg) reference implementation."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    u = 0.0
    for p in pos:
        u += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(u / (len(pos) * len(neg)))


class TestAUCEquivalence:
    def test_auc_matches_bruteforce_random(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            n = rng.integers(10, 300)
            y_true = (rng.random(n) > 0.5).astype(int)
            y_score = rng.random(n)
            fast = _auc_mannwhitney(y_true, y_score)
            slow = _auc_bruteforce(y_true, y_score)
            assert fast == pytest.approx(slow, abs=1e-12)

    def test_auc_handles_ties(self):
        # Many tied scores → exercises the average-rank tie correction.
        rng = np.random.default_rng(1)
        y_true = (rng.random(200) > 0.5).astype(int)
        y_score = rng.integers(0, 5, size=200).astype(float)
        fast = _auc_mannwhitney(y_true, y_score)
        slow = _auc_bruteforce(y_true, y_score)
        assert fast == pytest.approx(slow, abs=1e-12)

    def test_auc_degenerate(self):
        assert _auc_mannwhitney(np.ones(5, int), np.random.rand(5)) == 0.5
        assert _auc_mannwhitney(np.zeros(5, int), np.random.rand(5)) == 0.5


class TestIncrementalEquivalence:
    """A1: incremental update == direct fit on the subset."""

    @pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
    @pytest.mark.parametrize("fit_intercept", [True, False])
    def test_incremental_beta_matches_direct(self, alpha, fit_intercept):
        rng = np.random.default_rng(42)
        n, p = 400, 6
        X = rng.standard_normal((n, p))
        y = X @ rng.standard_normal(p) + 0.1 * rng.standard_normal(n)
        w = np.abs(rng.standard_normal(n)) + 0.1

        # Parent statistics over the whole node.
        Xa = np.column_stack([np.ones(n), X]) if fit_intercept else X
        XtWX_parent, XtWy_parent = compute_XtWX_XtWy(Xa, y, w)

        # Arbitrary split into "small" / "large".
        mask = X[:, 0] < 0.3
        small = mask  # whichever; the identity holds for either side
        Xs = Xa[small]
        XtWX_small, XtWy_small = compute_XtWX_XtWy(Xs, y[small], w[small])
        XtWX_large = XtWX_parent - XtWX_small
        XtWy_large = XtWy_parent - XtWy_small

        # Direct computation for the large side.
        Xl = Xa[~small]
        XtWX_large_direct, XtWy_large_direct = compute_XtWX_XtWy(
            Xl, y[~small], w[~small]
        )

        np.testing.assert_allclose(XtWX_large, XtWX_large_direct, atol=1e-8)
        np.testing.assert_allclose(XtWy_large, XtWy_large_direct, atol=1e-8)

        beta_incr = _ridge_closed_form(XtWX_large, XtWy_large, alpha)
        beta_direct = _ridge_closed_form(
            XtWX_large_direct, XtWy_large_direct, alpha
        )
        np.testing.assert_allclose(beta_incr, beta_direct, atol=1e-8)

    def test_ridge_fit_with_supplied_stats_matches_plain(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((150, 4))
        y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + 0.05 * rng.standard_normal(150)

        plain = RidgeRegressor(alpha=1.0).fit(X, y)

        Xa = np.column_stack([np.ones(X.shape[0]), X])
        XtWX, XtWy = compute_XtWX_XtWy(Xa, y, None)
        supplied = RidgeRegressor(alpha=1.0).fit(X, y, XtWX=XtWX, XtWy=XtWy)

        np.testing.assert_allclose(
            plain.get_coefficients(), supplied.get_coefficients(), atol=1e-10
        )
        assert plain.get_intercept() == pytest.approx(
            supplied.get_intercept(), abs=1e-10
        )


class TestCholeskyEquivalence:
    """A4: Cholesky solve == general LU solve for SPD systems."""

    def test_solver_matches_numpy(self):
        rng = np.random.default_rng(3)
        for _ in range(10):
            p = rng.integers(2, 12)
            M = rng.standard_normal((p, p))
            XtWX = M @ M.T  # SPD
            XtWy = rng.standard_normal(p)
            alpha = 1.0
            beta = _ridge_closed_form(XtWX, XtWy, alpha)
            ref = np.linalg.solve(XtWX + alpha * np.eye(p), XtWy)
            np.testing.assert_allclose(beta, ref, atol=1e-8)
