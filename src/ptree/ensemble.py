"""
Ensemble Panel Trees (D1): :class:`PanelForest`.

A Panel Tree is a *high-variance* estimator — its greedy "pick the
``(feature, threshold)`` that maximises ``|R^2_L - R^2_R|``" step can flip to a
completely different partition under small data perturbations.  This is exactly
the regime where bagging helps.  :class:`PanelForest` grows many decorrelated
P-Trees and aggregates them at the *output* layer (predictions, regime
membership, co-association), while each tree's split criterion remains the
unchanged ``R2Diff`` rule.

Panel-specific design choices (cannot be copied blindly from a vanilla RF):

* **Sample perturbation = time-block bootstrap.**  Contiguous blocks of
  ``block_size`` time periods are resampled *with replacement*, preserving the
  serial autocorrelation of returns.  Never bootstrap individual ``(t, i)``
  cells (that would shatter the cross-section and leak look-ahead).
* **Feature perturbation = node-level random subset.**  Each tree restricts
  every node's split search to a random ``max_features`` subset (handled inside
  :class:`~ptree.engine.PanelTreeEngine`), forcing the trees apart.
* **OOB evaluation.**  Each tree's *unselected* time blocks form an out-of-bag
  sample used to estimate generalisation without a separate hold-out.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:  # Optional dependency: parallel tree fitting.
    from joblib import Parallel, delayed

    _HAS_JOBLIB = True
except ImportError:  # pragma: no cover - exercised only without joblib
    _HAS_JOBLIB = False

from ptree.criteria import R2DiffCriterion
from ptree.engine import PanelTreeEngine
from ptree.predictors import RidgeRegressor



logger = logging.getLogger("ptree")


def _block_bootstrap_times(
    unique_times: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample contiguous time blocks *with replacement*.

    The sorted unique time labels are partitioned into consecutive blocks of
    ``block_size`` periods; ``n_blocks`` blocks are then drawn with replacement
    (so a block may appear multiple times, up-weighting its periods, while some
    blocks are never drawn — these form the out-of-bag set).

    Returns
    -------
    ndarray
        The (possibly repeated) time labels of the sampled blocks, in sampling
        order.  Duplicates are intentional — they realise the bootstrap weight.
    """
    n_times = len(unique_times)
    blocks = [
        unique_times[i : i + block_size]
        for i in range(0, n_times, block_size)
    ]
    n_blocks = len(blocks)
    chosen = rng.integers(0, n_blocks, size=n_blocks)
    sampled = [blocks[b] for b in chosen]
    if not sampled:
        return np.asarray(unique_times)
    return np.concatenate(sampled)


def _fit_one_tree(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    weights: Optional[np.ndarray],
    time_arr: np.ndarray,
    unique_times: np.ndarray,
    block_size: int,
    base_params: Dict[str, Any],
    max_features: Optional[Union[str, int, float]],
    seed: int,
) -> Dict[str, Any]:
    """Fit a single bootstrapped P-Tree and return it with its OOB time set."""
    rng = np.random.default_rng(seed)
    sampled_times = _block_bootstrap_times(unique_times, block_size, rng)

    in_bag = set(t.item() if hasattr(t, "item") else t for t in sampled_times)
    oob_times = np.array(
        [t for t in unique_times if (t.item() if hasattr(t, "item") else t) not in in_bag]
    )

    # Build the bootstrapped training rows by concatenating, *with repetition*,
    # all rows belonging to each sampled time block.
    time_to_rows: Dict[Any, np.ndarray] = {}
    for t in unique_times:
        key = t.item() if hasattr(t, "item") else t
        time_to_rows[key] = np.flatnonzero(time_arr == t)
    parts = [
        time_to_rows[t.item() if hasattr(t, "item") else t]
        for t in sampled_times
    ]
    train_rows = np.concatenate(parts) if parts else np.arange(len(y))

    X_tr = X.iloc[train_rows].reset_index(drop=True)
    y_tr = y.iloc[train_rows].reset_index(drop=True)
    w_tr = None if weights is None else weights[train_rows]

    params = dict(base_params)
    params["max_features"] = max_features
    params["random_state"] = seed
    params.setdefault("verbose", 0)
    engine = PanelTreeEngine(**params)
    engine.fit(X_tr, y_tr, feature_names=feature_names, weights=w_tr)

    # Record the "high-predictability" leaves (R^2 above the tree's leaf median)
    # so the forest can later compute soft regime-membership probabilities.
    leaves = engine.get_leaves()
    leaf_r2 = [leaf.metrics.get("r2", 0.0) for leaf in leaves]
    median_r2 = float(np.median(leaf_r2)) if leaf_r2 else 0.0
    high_leaf_ids = {
        leaf.node_id
        for leaf, r2 in zip(leaves, leaf_r2)
        if r2 >= median_r2
    }

    return {
        "engine": engine,
        "oob_times": oob_times,
        "high_leaf_ids": high_leaf_ids,
    }


class PanelForest:
    """A bagged ensemble of Panel Trees (P-Forest, D1).

    Parameters
    ----------
    n_estimators : int, default 100
        Number of P-Trees to grow.
    max_features : {"sqrt", "log2"}, int, float or None, default "sqrt"
        Node-level random feature-subset size passed to each tree (see
        :meth:`PanelTreeEngine._resolve_max_features`).  ``None`` disables
        feature perturbation (trees then differ only through the bootstrap).
    block_size : int, default 5
        Number of consecutive time periods per bootstrap block.
    aggregate : {"mean", "consensus", "sdf"}, default "mean"
        Primary aggregation the forest is built for.  All output methods
        (:meth:`predict`, :meth:`regime_membership`, :meth:`coassociation_matrix`)
        remain available regardless; this only documents intent / picks the
        default of :meth:`output`.
    base_params : dict or None
        Extra keyword arguments forwarded to every :class:`PanelTreeEngine`
        (e.g. ``predictor``, ``criterion``, ``max_depth``, ``min_samples``).
        ``criterion`` defaults to :class:`R2DiffCriterion`.
    n_jobs : int, default 1
        Parallel workers for tree fitting (requires ``joblib``).  ``-1`` uses
        all cores.
    random_state : int or None
        Seed controlling both the block bootstrap and each tree's node-level
        feature subsetting (fully reproducible).
    verbose : int, default 0
        Verbosity passed through to each tree (and forest-level logging).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_features: Optional[Union[str, int, float]] = "sqrt",
        block_size: int = 5,
        aggregate: str = "mean",
        base_params: Optional[Dict[str, Any]] = None,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if block_size < 1:
            raise ValueError("block_size must be >= 1.")
        if aggregate not in {"mean", "consensus", "sdf"}:
            raise ValueError(
                f"aggregate must be 'mean', 'consensus' or 'sdf', got "
                f"{aggregate!r}."
            )
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.block_size = block_size
        self.aggregate = aggregate
        self.base_params = base_params or {}
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # Populated after fit.
        self.trees_: List[PanelTreeEngine] = []
        self._oob_times: List[np.ndarray] = []
        self._high_leaf_ids: List[set] = []
        self._feature_names: List[str] = []
        self._fit_X: Optional[pd.DataFrame] = None
        self._fit_y: Optional[np.ndarray] = None
        self._fit_time: Optional[np.ndarray] = None
        self.oob_score_: Optional[float] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
        weights: Optional[Union[np.ndarray, pd.Series]] = None,
        time_index: Optional[Union[np.ndarray, pd.Series, str]] = None,
    ) -> "PanelForest":
        """Grow the forest.

        Parameters
        ----------
        X : DataFrame
            Processed panel features (plus a time column if ``time_index`` is a
            column name).
        y : Series
            Target aligned with *X*.
        feature_names : list of str
            Feature columns used for splitting / prediction.
        weights : ndarray, Series or None
            Observation weights (e.g. inverse-volatility).
        time_index : ndarray, Series, str or None
            Per-observation time label (required — the block bootstrap operates
            on it).  A string is read as a column of *X*.
        """
        if time_index is None:
            raise ValueError(
                "PanelForest requires `time_index` (block bootstrap operates on "
                "time periods)."
            )
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        self._feature_names = list(feature_names)
        self._fit_X = X
        self._fit_y = y.values.astype(np.float64)

        if isinstance(time_index, str):
            time_arr = X[time_index].values
        elif isinstance(time_index, pd.Series):
            time_arr = time_index.values
        else:
            time_arr = np.asarray(time_index)
        self._fit_time = time_arr

        w_arr = weights.values if isinstance(weights, pd.Series) else weights

        unique_times = np.unique(time_arr)

        base_params = dict(self.base_params)
        base_params.setdefault("criterion", R2DiffCriterion())
        base_params.setdefault("predictor", RidgeRegressor(alpha=1.0))

        seed_rng = np.random.default_rng(self.random_state)
        seeds = [int(s) for s in seed_rng.integers(0, 2**31 - 1, size=self.n_estimators)]

        if self.n_jobs != 1 and _HAS_JOBLIB:

            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(_fit_one_tree)(
                    X, y, self._feature_names, w_arr, time_arr, unique_times,
                    self.block_size, base_params, self.max_features, seed,
                )
                for seed in seeds
            )
        else:
            results = [
                _fit_one_tree(
                    X, y, self._feature_names, w_arr, time_arr, unique_times,
                    self.block_size, base_params, self.max_features, seed,
                )
                for seed in seeds
            ]

        self.trees_ = [r["engine"] for r in results]
        self._oob_times = [r["oob_times"] for r in results]
        self._high_leaf_ids = [r["high_leaf_ids"] for r in results]

        self.oob_score_ = self._compute_oob_score()

        if self.verbose >= 1:
            logger.info(
                "PanelForest fitted: %d trees, oob_score=%.4f",
                self.n_estimators,
                self.oob_score_ if self.oob_score_ is not None else float("nan"),
            )
        return self

    # ------------------------------------------------------------------
    # Output methods
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the bagged prediction ``mean_b yhat_b(X)`` (variance reduction)."""
        assert self.trees_, "Call .fit() first."
        X = X.reset_index(drop=True)
        preds = np.zeros(len(X), dtype=np.float64)
        for tree in self.trees_:
            preds += tree.predict(X)
        return preds / len(self.trees_)

    def regime_membership(self, X: pd.DataFrame) -> np.ndarray:
        """Soft probability that each observation sits in a high-predictability regime.

        For every observation this is the fraction of trees that route it into
        a *high-R²* leaf (a leaf whose in-sample R² exceeds that tree's leaf
        median).  This upgrades the brittle 0/1 mosaic of a single tree into a
        smooth, robust regime probability in ``[0, 1]``.
        """
        assert self.trees_, "Call .fit() first."
        X = X.reset_index(drop=True)
        n = len(X)
        counts = np.zeros(n, dtype=np.float64)
        for tree, high_ids in zip(self.trees_, self._high_leaf_ids):
            leaf_ids = self._leaf_ids_for(tree, X)
            counts += np.array([lid in high_ids for lid in leaf_ids], dtype=np.float64)
        return counts / len(self.trees_)

    def coassociation_matrix(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Co-association (consensus) matrix ``C[i, j] ∈ [0, 1]``.

        ``C[i, j]`` is the fraction of trees in which observations *i* and *j*
        land in the *same* leaf.  It turns the forest's many fragile hard
        partitions into a single robust similarity, answering "which
        ``(time, asset)`` units *consistently* share a predictability regime".
        Suitable as a precomputed affinity for spectral clustering.

        Parameters
        ----------
        X : DataFrame or None
            Observations to relate.  Defaults to the training panel.  Note the
            result is ``O(n^2)`` in memory, so use a modest *n*.
        """
        assert self.trees_, "Call .fit() first."
        if X is None:
            X = self._fit_X
        X = X.reset_index(drop=True)
        n = len(X)
        C = np.zeros((n, n), dtype=np.float64)
        for tree in self.trees_:
            leaf_ids = self._leaf_ids_for(tree, X)
            # Same-leaf indicator via broadcasting.
            same = leaf_ids[:, None] == leaf_ids[None, :]
            C += same
        return C / len(self.trees_)

    def output(self, X: pd.DataFrame):
        """Return the forest's primary product, per the ``aggregate`` setting."""
        if self.aggregate == "mean":
            return self.predict(X)
        if self.aggregate == "consensus":
            return self.coassociation_matrix(X)
        # "sdf": averaged regime membership is the closest single-array proxy;
        # callers wanting full SDF series should use each tree's build_sdf_factor.
        return self.regime_membership(X)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _leaf_ids_for(self, tree: PanelTreeEngine, X: pd.DataFrame) -> np.ndarray:
        """Route every row of *X* through *tree* and return its leaf ids."""
        X_arr = X[self._feature_names].values.astype(np.float64)
        out = np.empty(X_arr.shape[0], dtype=int)
        tree._assign_leaf_ids(
            tree.root_, X_arr, np.arange(X_arr.shape[0]), out
        )
        return out

    def _compute_oob_score(self) -> Optional[float]:
        """Out-of-bag R²: average each tree's prediction over rows whose time
        was *not* in that tree's bootstrap, then score against ``y``."""
        if self._fit_X is None:
            return None
        n = len(self._fit_X)
        sum_pred = np.zeros(n, dtype=np.float64)
        n_oob = np.zeros(n, dtype=np.float64)
        time_arr = self._fit_time

        for tree, oob_times in zip(self.trees_, self._oob_times):
            if len(oob_times) == 0:
                continue
            oob_set = set(
                t.item() if hasattr(t, "item") else t for t in oob_times
            )
            mask = np.array(
                [(t.item() if hasattr(t, "item") else t) in oob_set for t in time_arr]
            )
            if not mask.any():
                continue
            preds = tree.predict(self._fit_X.iloc[mask])
            sum_pred[mask] += preds
            n_oob[mask] += 1.0

        scored = n_oob > 0
        if not scored.any():
            return None
        yhat = sum_pred[scored] / n_oob[scored]
        y_true = self._fit_y[scored]
        ss_res = float(np.sum((y_true - yhat) ** 2))
        y_mean = float(np.mean(y_true))
        ss_tot = float(np.sum((y_true - y_mean) ** 2))
        return 1.0 - ss_res / max(ss_tot, 1e-12)

    def __repr__(self) -> str:
        return (
            f"PanelForest(n_estimators={self.n_estimators}, "
            f"max_features={self.max_features!r}, block_size={self.block_size}, "
            f"aggregate={self.aggregate!r})"
        )
