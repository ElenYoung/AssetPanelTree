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
    regime_metric: str = "train_r2",
    regime_aggregation: str = "train",
) -> Dict[str, Any]:
    """Fit a single bootstrapped P-Tree and return it with its OOB time set.

    Parameters
    ----------
    regime_metric : {"train_r2", "oof_r2", "rank_ic"}, default "train_r2"
        Metric used to rank leaves when building the "high-predictability"
        regime set.  ``"train_r2"`` (legacy) uses the leaf's in-sample R²;
        ``"oof_r2"`` recomputes R² on the OOB rows routed to each leaf;
        ``"rank_ic"`` uses the cross-sectional rank-IC averaged over OOB
        time periods.
    regime_aggregation : {"train", "oof"}, default "train"
        ``"train"`` (legacy) ranks leaves on the bootstrap-train metric;
        ``"oof"`` ranks leaves on the OOB sample (more robust, but degrades
        gracefully to the train metric if a leaf has no OOB rows).
    """
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

    # Record the "high-predictability" leaves so the forest can later compute
    # soft regime-membership probabilities.  ``regime_aggregation`` controls
    # whether the ranking metric is computed in-sample (legacy) or on the OOB
    # rows (more robust, addresses the "train-R² doesn't generalise" issue).
    leaves = engine.get_leaves()
    high_leaf_ids = _select_high_leaves(
        engine=engine,
        leaves=leaves,
        X_full=X,
        y_full=y,
        time_full=time_arr,
        oob_times=oob_times,
        feature_names=feature_names,
        regime_metric=regime_metric,
        regime_aggregation=regime_aggregation,
    )

    return {
        "engine": engine,
        "oob_times": oob_times,
        "high_leaf_ids": high_leaf_ids,
    }


def _mean_rank_ic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    time_arr: np.ndarray,
) -> float:
    """Cross-sectional rank-IC averaged over time periods.

    For each unique time ``t``, compute Spearman-style correlation between
    rank-transformed ``y_true`` and ``y_pred`` over the cross-section, then
    average across times.  Times with fewer than 3 obs are skipped.
    """
    ics: List[float] = []
    for t in np.unique(time_arr):
        mask = time_arr == t
        if int(mask.sum()) < 3:
            continue
        yt = pd.Series(y_true[mask]).rank().values
        yp = pd.Series(y_pred[mask]).rank().values
        std_yt = yt.std(ddof=0)
        std_yp = yp.std(ddof=0)
        if std_yt < 1e-12 or std_yp < 1e-12:
            continue
        ic = float(np.corrcoef(yt, yp)[0, 1])
        if np.isfinite(ic):
            ics.append(ic)
    if not ics:
        return float("nan")
    return float(np.mean(ics))


def _select_high_leaves(
    engine: "PanelTreeEngine",
    leaves: List[Any],
    X_full: pd.DataFrame,
    y_full: pd.Series,
    time_full: np.ndarray,
    oob_times: np.ndarray,
    feature_names: List[str],
    regime_metric: str,
    regime_aggregation: str,
) -> set:
    """Return the set of leaf ids that count as "high-predictability".

    The default (``regime_aggregation="train"``) preserves the legacy behavior
    (leaves whose in-sample R² is above the tree's leaf median).  Setting
    ``regime_aggregation="oof"`` recomputes the chosen ``regime_metric``
    (``"oof_r2"`` or ``"rank_ic"``) on the OOB rows for each leaf and ranks by
    that — more robust on low-signal financial panels.
    """
    if not leaves:
        return set()

    # Legacy: rank by in-sample R^2.
    if regime_aggregation == "train" or regime_metric == "train_r2":
        leaf_r2 = [leaf.metrics.get("r2", 0.0) for leaf in leaves]
        median_r2 = float(np.median(leaf_r2)) if leaf_r2 else 0.0
        return {
            leaf.node_id
            for leaf, r2 in zip(leaves, leaf_r2)
            if r2 >= median_r2
        }

    # OOF-based scoring.  Route every OOB row to its leaf and aggregate.
    if oob_times is None or len(oob_times) == 0:
        # No OOB available: fall back to train R^2.
        leaf_r2 = [leaf.metrics.get("r2", 0.0) for leaf in leaves]
        median_r2 = float(np.median(leaf_r2)) if leaf_r2 else 0.0
        return {
            leaf.node_id
            for leaf, r2 in zip(leaves, leaf_r2)
            if r2 >= median_r2
        }

    oob_set = set(
        t.item() if hasattr(t, "item") else t for t in oob_times
    )
    oob_mask = np.array(
        [(t.item() if hasattr(t, "item") else t) in oob_set for t in time_full]
    )
    if not oob_mask.any():
        leaf_r2 = [leaf.metrics.get("r2", 0.0) for leaf in leaves]
        median_r2 = float(np.median(leaf_r2)) if leaf_r2 else 0.0
        return {
            leaf.node_id
            for leaf, r2 in zip(leaves, leaf_r2)
            if r2 >= median_r2
        }

    X_oob = X_full.iloc[oob_mask].reset_index(drop=True)
    y_oob = np.asarray(y_full)[oob_mask].astype(np.float64)
    t_oob = time_full[oob_mask]

    # Route OOB rows to leaves via the engine's public/internal API.
    try:
        leaf_ids = engine.predict_leaves(X_oob)
    except AttributeError:  # pragma: no cover - kept for safety on older engines
        X_arr = X_oob[feature_names].values.astype(np.float64)
        leaf_ids = np.empty(X_arr.shape[0], dtype=int)
        engine._assign_leaf_ids(
            engine.root_, X_arr, np.arange(X_arr.shape[0]), leaf_ids
        )

    yhat_oob = engine.predict(X_oob)

    scores: Dict[int, float] = {}
    for leaf in leaves:
        m = leaf_ids == leaf.node_id
        if int(m.sum()) < 3:
            scores[leaf.node_id] = float("-inf")
            continue
        y_l = y_oob[m]
        yhat_l = yhat_oob[m]
        if regime_metric == "rank_ic":
            scores[leaf.node_id] = _mean_rank_ic(y_l, yhat_l, t_oob[m])
        else:  # default OOF metric: R^2
            ss_res = float(np.sum((y_l - yhat_l) ** 2))
            y_mean = float(np.mean(y_l))
            ss_tot = float(np.sum((y_l - y_mean) ** 2))
            scores[leaf.node_id] = 1.0 - ss_res / max(ss_tot, 1e-12)

    finite = [s for s in scores.values() if np.isfinite(s)]
    if not finite:
        return set()
    median_s = float(np.median(finite))
    return {nid for nid, s in scores.items() if np.isfinite(s) and s >= median_s}


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
    regime_metric : {"train_r2", "oof_r2", "rank_ic"}, default "train_r2"
        Metric each tree uses to rank its leaves when building the
        "high-predictability" set surfaced by :meth:`regime_membership`.
        ``"train_r2"`` (default, legacy) ranks by in-sample R²; ``"oof_r2"``
        and ``"rank_ic"`` rank on the OOB sample (only effective when
        ``regime_aggregation="oof"``).
    regime_aggregation : {"train", "oof"}, default "train"
        ``"train"`` (default, legacy) preserves the original behavior.
        ``"oof"`` scores leaves on each tree's OOB rows — more robust on
        low-signal financial panels where train R² doesn't generalise.
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
        regime_metric: str = "train_r2",
        regime_aggregation: str = "train",
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
        if regime_metric not in {"train_r2", "oof_r2", "rank_ic"}:
            raise ValueError(
                f"regime_metric must be one of 'train_r2', 'oof_r2', "
                f"'rank_ic', got {regime_metric!r}."
            )
        if regime_aggregation not in {"train", "oof"}:
            raise ValueError(
                f"regime_aggregation must be 'train' or 'oof', got "
                f"{regime_aggregation!r}."
            )
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.block_size = block_size
        self.aggregate = aggregate
        self.base_params = base_params or {}
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.regime_metric = regime_metric
        self.regime_aggregation = regime_aggregation

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
                    self.regime_metric, self.regime_aggregation,
                )
                for seed in seeds
            )
        else:
            results = [
                _fit_one_tree(
                    X, y, self._feature_names, w_arr, time_arr, unique_times,
                    self.block_size, base_params, self.max_features, seed,
                    self.regime_metric, self.regime_aggregation,
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


class BoostedPanelTree:
    """Gradient-boosted Panel Trees (P-Boost, D2).

    P-Boost does **not** boost the ``|R^2_L - R^2_R|`` *criterion* (which is not
    an additive loss); instead it boosts the *target / residual*.  Each round
    strips the predictability already explained by the running ensemble and
    re-grows a fresh P-Tree on the residual, so successive trees uncover the
    *next, weaker* predictability regime that the greedy single tree would have
    masked::

        F_0(x) = 0
        for m = 1..M:
            r   = y - nu * F_{m-1}(x)          # residual (nu = learning_rate)
            T_m = PanelTreeEngine(R2Diff).fit(X, r)
            F_m(x) = F_{m-1}(x) + nu * T_m.predict(x)

    Each tree's split criterion stays ``R2Diff`` — only the *fit target*
    changes.  On single-feature-dominated data the method is **self-limiting**:
    once the first tree explains the dominant regime, the residual is near-noise
    and later trees add almost nothing (see :attr:`residual_norms_`).

    Parameters
    ----------
    n_estimators : int, default 50
        Number of boosting rounds (trees).
    learning_rate : float, default 0.1
        Shrinkage ``nu`` applied to each tree's contribution.
    max_depth : int, default 2
        Depth of each (weak-learner) P-Tree.  Boosting favours shallow trees.
    subsample : float, default 1.0
        Fraction of *time blocks* used to fit each tree (stochastic boosting).
        ``1.0`` uses all data every round.
    block_size : int, default 5
        Consecutive time periods per subsample block (only used when
        ``subsample < 1``).
    criterion : CriterionBase or None
        Split criterion for every tree (defaults to :class:`R2DiffCriterion`).
    base_params : dict or None
        Extra keyword arguments forwarded to each :class:`PanelTreeEngine`
        (e.g. ``predictor``, ``min_samples``).  ``max_depth`` / ``criterion``
        are controlled by the dedicated parameters above.
    random_state : int or None
        Seed for the (optional) subsampling.
    verbose : int, default 0
        Verbosity passed through to each tree.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 0.1,
        max_depth: int = 2,
        subsample: float = 1.0,
        block_size: int = 5,
        criterion: Optional[Any] = None,
        base_params: Optional[Dict[str, Any]] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError("learning_rate must lie in (0, 1].")
        if not (0.0 < subsample <= 1.0):
            raise ValueError("subsample must lie in (0, 1].")
        if block_size < 1:
            raise ValueError("block_size must be >= 1.")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.block_size = block_size
        self.criterion = criterion
        self.base_params = base_params or {}
        self.random_state = random_state
        self.verbose = verbose

        # Populated after fit.
        self.trees_: List[PanelTreeEngine] = []
        self.residual_norms_: List[float] = []
        self._feature_names: List[str] = []

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
    ) -> "BoostedPanelTree":
        """Fit the boosting sequence.

        Parameters
        ----------
        X : DataFrame
            Processed panel features.
        y : Series
            Target aligned with *X*.
        feature_names : list of str
            Feature columns used for splitting / prediction.
        weights : ndarray, Series or None
            Observation weights (e.g. inverse-volatility).
        time_index : ndarray, Series, str or None
            Per-observation time label.  Required only when ``subsample < 1``
            (the block subsample operates on it).  A string is read as a column
            of *X*.
        """
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        self._feature_names = list(feature_names)
        y_arr = y.values.astype(np.float64)
        n = len(y_arr)

        w_arr = weights.values if isinstance(weights, pd.Series) else weights

        if isinstance(time_index, str):
            time_arr = X[time_index].values
        elif isinstance(time_index, pd.Series):
            time_arr = time_index.values
        elif time_index is not None:
            time_arr = np.asarray(time_index)
        else:
            time_arr = None
        if self.subsample < 1.0 and time_arr is None:
            raise ValueError(
                "BoostedPanelTree requires `time_index` when subsample < 1 "
                "(the stochastic subsample operates on time blocks)."
            )

        nu = self.learning_rate
        rng = np.random.default_rng(self.random_state)

        F = np.zeros(n, dtype=np.float64)  # running ensemble prediction
        self.trees_ = []
        self.residual_norms_ = []

        for m in range(self.n_estimators):
            residual = y_arr - nu * F
            self.residual_norms_.append(float(np.sqrt(np.sum(residual ** 2))))

            rows = self._subsample_rows(time_arr, n, rng)

            params = dict(self.base_params)
            params["max_depth"] = self.max_depth
            params["criterion"] = (
                self.criterion if self.criterion is not None else R2DiffCriterion()
            )
            params.setdefault("predictor", RidgeRegressor(alpha=1.0))
            params.setdefault("verbose", 0)
            params["random_state"] = int(rng.integers(0, 2**31 - 1))

            tree = PanelTreeEngine(**params)
            X_tr = X.iloc[rows].reset_index(drop=True)
            r_tr = pd.Series(residual[rows], name="residual")
            w_tr = None if w_arr is None else w_arr[rows]
            tree.fit(X_tr, r_tr, feature_names=self._feature_names, weights=w_tr)

            # Update the running ensemble on the *full* sample.
            F = F + tree.predict(X)
            self.trees_.append(tree)

        # Final residual norm (after the last tree).
        final_resid = y_arr - nu * F
        self.residual_norms_.append(float(np.sqrt(np.sum(final_resid ** 2))))

        if self.verbose >= 1:
            logger.info(
                "BoostedPanelTree fitted: %d trees, residual %.4f -> %.4f",
                self.n_estimators,
                self.residual_norms_[0],
                self.residual_norms_[-1],
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return ``nu * sum_m T_m.predict(X)`` — the boosted prediction."""
        assert self.trees_, "Call .fit() first."
        X = X.reset_index(drop=True)
        F = np.zeros(len(X), dtype=np.float64)
        for tree in self.trees_:
            F += tree.predict(X)
        return self.learning_rate * F

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _subsample_rows(
        self,
        time_arr: Optional[np.ndarray],
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Pick the training rows for one boosting round.

        With ``subsample == 1`` (or no time labels) every row is used.
        Otherwise a random ``subsample`` fraction of contiguous time blocks is
        selected (stochastic boosting that respects the panel's time structure).
        """
        if self.subsample >= 1.0 or time_arr is None:
            return np.arange(n)
        unique_times = np.unique(time_arr)
        n_times = len(unique_times)
        blocks = [
            unique_times[i : i + self.block_size]
            for i in range(0, n_times, self.block_size)
        ]
        n_blocks = len(blocks)
        k = max(1, int(round(self.subsample * n_blocks)))
        chosen = rng.choice(n_blocks, size=k, replace=False)
        keep_times = set()
        for b in chosen:
            for t in blocks[b]:
                keep_times.add(t.item() if hasattr(t, "item") else t)
        mask = np.array(
            [(t.item() if hasattr(t, "item") else t) in keep_times for t in time_arr]
        )
        return np.flatnonzero(mask)

    def __repr__(self) -> str:
        return (
            f"BoostedPanelTree(n_estimators={self.n_estimators}, "
            f"learning_rate={self.learning_rate}, max_depth={self.max_depth}, "
            f"subsample={self.subsample})"
        )

