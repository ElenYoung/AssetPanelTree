"""
PanelTreeEngine: recursive splitting, pruning, feature-priority caching,
incremental matrix updates, and multiprocessing support.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)


import numpy as np
import pandas as pd
from multiprocessing import cpu_count

try:  # Optional dependency: real feature-dimension parallelism.
    from joblib import Parallel, delayed

    _HAS_JOBLIB = True
except ImportError:  # pragma: no cover - exercised only without joblib
    _HAS_JOBLIB = False


from ptree.node import PanelTreeNode
from ptree.predictors import (
    PredictorBase,
    RidgeRegressor,
    VolWeightedRidgeRegressor,
    RidgeLogitClassifier,
    compute_XtWX_XtWy,
)
from ptree.criteria import (
    CriterionBase,
    R2DiffCriterion,
    MeanVarianceCriterion,
    ClassificationCriterion,
    evaluate_regression,
    evaluate_classification,
)


logger = logging.getLogger("ptree")


# ======================================================================
# Engine
# ======================================================================

class PanelTreeEngine:
    """Build and query a Panel Tree.

    Parameters
    ----------
    predictor : PredictorBase or Type[PredictorBase]
        Instance or class of the leaf-node predictor.  If a class is given,
        it will be instantiated with ``predictor_params``.
    criterion : CriterionBase
        Splitting-quality criterion.
    split_thresholds : list of float or "adaptive", default [0.3, 0.5, 0.7]
        Candidate thresholds applied to (possibly rank-standardised) features.
        If the string ``"adaptive"`` is given, per-node, per-feature quantile
        thresholds (see ``adaptive_quantiles``) are used instead of a single
        global list — more robust for non-uniform feature distributions.
    adaptive_quantiles : list of float, default [0.25, 0.5, 0.75]
        Quantiles used to derive thresholds when ``split_thresholds="adaptive"``.
    max_depth : int, default 3
        Maximum tree depth.
    min_samples : int, default 100
        Minimum samples in a node for it to be considered for splitting.
        Also used to skip feature-threshold combos that produce too-small
        child nodes.
    min_impurity_decrease : float, default 0.0
        Minimum criterion score the best split must achieve for the node to be
        split; if the best score is below this value the node becomes a leaf.
        The default ``0.0`` preserves the original behaviour (any non-negative
        score splits).

    fast_mode : bool, default False
        Enable *Feature Persistence* – prioritise top-ranked features from
        the parent node in child-node searches.
    early_stopping_threshold : float or None
        If ``fast_mode`` is True and the best candidate so far exceeds this
        threshold, skip remaining features.  Ignored when ``fast_mode`` is
        False.
    n_jobs : int, default 1
        Number of parallel workers at the *node* level (``multiprocessing``).
        Set to ``-1`` to use all CPU cores.
    verbose : int, default 1
        Logging verbosity (0 = silent, 1 = per-level, 2 = per-candidate).
    """

    def __init__(
        self,
        predictor: Union[PredictorBase, Type[PredictorBase]] = RidgeRegressor,
        criterion: CriterionBase = R2DiffCriterion(),
        split_thresholds: Optional[Union[List[float], str]] = None,
        max_depth: int = 3,
        min_samples: int = 100,
        min_impurity_decrease: float = 0.0,
        adaptive_quantiles: Optional[List[float]] = None,
        honest: bool = False,
        honest_frac: float = 0.5,
        honest_refit_full: bool = True,
        random_state: Optional[int] = None,
        fast_mode: bool = False,
        early_stopping_threshold: Optional[float] = None,
        n_jobs: int = 1,
        verbose: int = 1,
        predictor_params: Optional[Dict] = None,
        keep_node_stats: bool = False,
        parallel_backend: str = "threads",
        max_features: Optional[Union[str, int, float]] = None,
        splitter: str = "best",
        n_random_splits: int = 1,
    ):





        # Predictor template
        if isinstance(predictor, type):
            params = predictor_params or {}
            self._predictor_template = predictor(**params)
        else:
            self._predictor_template = predictor

        self.criterion = criterion
        # ``split_thresholds`` may be a fixed list (default) or the string
        # "adaptive" (per-node quantile thresholds, see ``adaptive_quantiles``).
        if isinstance(split_thresholds, str):
            if split_thresholds != "adaptive":
                raise ValueError(
                    "split_thresholds must be a list of floats or the string "
                    f"'adaptive', got {split_thresholds!r}."
                )
            self.adaptive_thresholds = True
            self.split_thresholds = "adaptive"
        else:
            self.adaptive_thresholds = False
            self.split_thresholds = split_thresholds or [0.3, 0.5, 0.7]
        self.adaptive_quantiles = adaptive_quantiles or [0.25, 0.5, 0.75]
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_impurity_decrease = min_impurity_decrease
        self.fast_mode = fast_mode

        self.early_stopping_threshold = early_stopping_threshold
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.verbose = verbose
        self.keep_node_stats = keep_node_stats
        self.parallel_backend = parallel_backend
        # D1/P-Forest: node-level random feature subsetting.  ``None`` (default)
        # evaluates every feature (standard P-Tree); ``"sqrt"`` / ``"log2"`` /
        # an int / a float fraction restrict each node's search to a random
        # subset drawn from ``self._rng``, decorrelating trees in an ensemble.
        self.max_features = max_features
        # D3/Extra-Trees: ``splitter="random"`` replaces the exhaustive
        # ``split_thresholds`` sweep with ``n_random_splits`` random thresholds
        # drawn (per feature, per node) from the feature's in-node value range —
        # extra variance reduction + speed for ensembles.  ``"best"`` (default)
        # keeps the standard exhaustive search and so is fully backward
        # compatible (the golden baseline path is untouched).
        if splitter not in {"best", "random"}:
            raise ValueError(
                f"splitter must be 'best' or 'random', got {splitter!r}."
            )
        self.splitter = splitter
        if n_random_splits < 1:
            raise ValueError("n_random_splits must be >= 1.")
        self.n_random_splits = n_random_splits



        # B2: honest split — separate the samples used to *choose* a split from
        # those used to *evaluate* its quality, removing the in-sample
        # selection bias of greedy panel splitting.
        self.honest = honest
        if not (0.0 < honest_frac < 1.0):
            raise ValueError("honest_frac must lie strictly between 0 and 1.")
        self.honest_frac = honest_frac
        self.honest_refit_full = honest_refit_full
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)



        # State populated after fitting

        self.root_: Optional[PanelTreeNode] = None
        self._node_counter: int = 0
        self._total_samples: int = 0
        self._feature_names: List[str] = []
        self._is_classification: bool = isinstance(
            self.criterion, ClassificationCriterion
        )
        # When the mean-variance criterion is used, each child's metrics must
        # carry a per-time long-short portfolio return series (``_port_ret``).
        self._needs_port_ret: bool = isinstance(
            self.criterion, MeanVarianceCriterion
        )
        # Store processed data for predictions & cluster retrieval
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        self._time: Optional[np.ndarray] = None
        self._X_df: Optional[pd.DataFrame] = None


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
        weights: Optional[np.ndarray] = None,
        time_index: Optional[Union[np.ndarray, pd.Series, str]] = None,
    ) -> "PanelTreeEngine":
        """Build the Panel Tree.

        Parameters
        ----------
        X : DataFrame
            Processed panel data (with time/entity columns plus features).
        y : Series
            Target variable aligned with *X*.
        feature_names : list of str
            Names of the feature columns used for splitting and prediction.
        weights : ndarray or None
            Observation weights (e.g. inverse-volatility).
        time_index : ndarray, Series, str or None
            Per-observation time label aligned with *X*.  Required when the
            criterion is :class:`MeanVarianceCriterion` (used to aggregate each
            leaf's long-short portfolio into a return time series); ignored by
            the R²/classification criteria.  A string is interpreted as a
            column name in *X*.
        """
        self._feature_names = list(feature_names)
        X_arr = X[feature_names].values.astype(np.float64)
        y_arr = y.values.astype(np.float64)
        self._total_samples = len(y_arr)
        self._X = X_arr
        self._y = y_arr
        self._weights = weights.values if isinstance(weights, pd.Series) else weights
        self._X_df = X
        self._node_counter = 0

        # Resolve the optional per-observation time labels (mean-variance
        # criterion only).  Accept an array/Series or a column name in ``X``.
        if time_index is None:
            self._time = None
        elif isinstance(time_index, str):
            self._time = X[time_index].values
        elif isinstance(time_index, pd.Series):
            self._time = time_index.values
        else:
            self._time = np.asarray(time_index)
        if self._needs_port_ret and self._time is None:
            raise ValueError(
                "MeanVarianceCriterion requires `time_index` to be passed to "
                "fit() so leaf long-short portfolios can be aggregated by time."
            )


        all_indices = np.arange(self._total_samples)

        self.root_ = self._build_node(
            indices=all_indices,
            depth=0,
            rule="root",
            parent_id=None,
            parent_ranking=None,
        )

        if self.verbose >= 1:
            leaves = self.get_leaves()
            logger.info(
                "Tree built: %d nodes, %d leaves, max_depth=%d",
                self._node_counter,
                len(leaves),
                self.max_depth,
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using fitted leaf models.

        Parameters
        ----------
        X : DataFrame
            Must contain the same feature columns used during ``fit``.

        Returns
        -------
        preds : ndarray of shape (n,)
        """
        assert self.root_ is not None, "Call .fit() first."
        X_arr = X[self._feature_names].values.astype(np.float64)
        preds = np.full(X_arr.shape[0], np.nan)
        self._predict_recursive(self.root_, X_arr, np.arange(len(X_arr)), preds)
        return preds

    def get_leaves(self) -> List[PanelTreeNode]:
        """Return all leaf nodes."""
        assert self.root_ is not None
        leaves: List[PanelTreeNode] = []
        self._collect_leaves(self.root_, leaves)
        return leaves

    def get_all_nodes(self) -> List[PanelTreeNode]:
        """Return all nodes (BFS order)."""
        assert self.root_ is not None
        nodes: List[PanelTreeNode] = []
        queue = [self.root_]
        while queue:
            node = queue.pop(0)
            nodes.append(node)
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
        return nodes

    def get_node_report(self) -> pd.DataFrame:
        """Return a structured DataFrame summarising every node."""
        nodes = self.get_all_nodes()
        return pd.DataFrame([n.to_dict() for n in nodes])

    def get_leaf_samples(self) -> Dict[int, np.ndarray]:
        """Map leaf ``node_id`` → original sample indices."""
        return {leaf.node_id: leaf.sample_indices for leaf in self.get_leaves()}

    # ------------------------------------------------------------------
    # SDF / efficient-frontier aggregation (C3)
    # ------------------------------------------------------------------

    def build_sdf_factor(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        time_index: Optional[Union[np.ndarray, pd.Series, str]] = None,
        ridge: float = 1e-6,
    ) -> Dict[str, Any]:
        """Aggregate the leaf long-short portfolios into a tradeable SDF factor.

        Each leaf forms a per-time long-short portfolio (predicted-return
        weighted, cross-sectionally de-meaned).  The leaf portfolio return
        series are then combined by *mean-variance* (tangency) weights
        ``w ∝ Σ^{-1} μ`` to produce a single stochastic-discount-factor (SDF)
        return series — the in-sample maximum-Sharpe combination of the tree's
        regime portfolios, realising the "growing the efficient frontier" goal.

        Parameters
        ----------
        X, y, time_index : optional
            Data to evaluate the leaf portfolios on.  If omitted, the training
            data passed to ``fit`` is reused (requires that ``fit`` was given a
            ``time_index``).  When supplied, ``time_index`` may be an array,
            Series, or a column name in ``X``.
        ridge : float, default 1e-6
            Diagonal load added to the leaf covariance for a stable inverse.

        Returns
        -------
        dict with keys
            ``weights`` : ndarray, the tangency weight on each leaf portfolio;
            ``leaf_ids`` : list[int], leaf node ids aligned with ``weights``;
            ``times`` : ndarray, the (sorted) common time labels;
            ``sdf_returns`` : ndarray, the SDF factor return per time;
            ``sharpe`` : float, in-sample (non-annualised) Sharpe of the SDF.
        """
        assert self.root_ is not None, "Call .fit() first."

        # Resolve evaluation data: reuse training data unless new data is given.
        if X is None:
            X_arr = self._X
            y_arr = self._y
            time_arr = self._time
        else:
            X_arr = X[self._feature_names].values.astype(np.float64)
            if y is None:
                raise ValueError("`y` must be supplied when `X` is supplied.")
            y_arr = (
                y.values.astype(np.float64)
                if isinstance(y, pd.Series)
                else np.asarray(y, dtype=np.float64)
            )
            if isinstance(time_index, str):
                time_arr = X[time_index].values
            elif isinstance(time_index, pd.Series):
                time_arr = time_index.values
            elif time_index is not None:
                time_arr = np.asarray(time_index)
            else:
                time_arr = None

        if time_arr is None:
            raise ValueError(
                "build_sdf_factor requires time labels: pass `time_index` here "
                "or to fit()."
            )

        # Route every observation to its leaf, then build each leaf's per-time
        # long-short portfolio return series.
        leaf_ids_full = np.empty(X_arr.shape[0], dtype=int)
        self._assign_leaf_ids(self.root_, X_arr, np.arange(X_arr.shape[0]), leaf_ids_full)

        leaf_returns: Dict[int, Dict[Any, float]] = {}
        for leaf in self.get_leaves():
            mask = leaf_ids_full == leaf.node_id
            if not mask.any() or leaf.predictor is None:
                continue
            y_pred = leaf.predictor.predict(X_arr[mask])
            leaf_returns[leaf.node_id] = self._portfolio_returns(
                y_arr[mask], y_pred, time_arr[mask]
            )

        leaf_ids = [lid for lid, r in leaf_returns.items() if r]
        if not leaf_ids:
            raise ValueError("No leaf produced a non-empty portfolio series.")

        # Align leaf portfolio series on their common set of time periods.
        common = sorted(
            set.intersection(*(set(leaf_returns[lid]) for lid in leaf_ids))
        )
        if len(common) < 2:
            raise ValueError(
                "Leaf portfolios share fewer than 2 common periods; cannot "
                "estimate a mean-variance combination."
            )

        R = np.array(
            [[leaf_returns[lid][t] for lid in leaf_ids] for t in common],
            dtype=np.float64,
        )
        mu = R.mean(axis=0)
        Sigma = np.cov(R, rowvar=False)
        Sigma = np.atleast_2d(Sigma) + ridge * np.eye(len(leaf_ids))
        try:
            w = np.linalg.solve(Sigma, mu)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(Sigma, mu, rcond=None)[0]
        norm = np.abs(w).sum()
        if norm > 0:
            w = w / norm

        sdf_returns = R @ w
        sd = float(sdf_returns.std())
        sharpe = float(sdf_returns.mean() / sd) if sd > 1e-12 else 0.0

        return {
            "weights": w,
            "leaf_ids": leaf_ids,
            "times": np.asarray(common),
            "sdf_returns": sdf_returns,
            "sharpe": sharpe,
        }

    def _assign_leaf_ids(
        self,
        node: PanelTreeNode,
        X_arr: np.ndarray,
        row_indices: np.ndarray,
        out: np.ndarray,
    ) -> None:
        """Recursively record, for each row, the id of the leaf it falls into."""
        if node.is_leaf:
            out[row_indices] = node.node_id
            return
        feat_idx = node._split_feature_idx
        if feat_idx is None:
            feat_idx = self._feature_names.index(node.split_feature)
        mask_left = X_arr[row_indices, feat_idx] < node.split_threshold
        if node.left is not None:
            self._assign_leaf_ids(node.left, X_arr, row_indices[mask_left], out)
        if node.right is not None:
            self._assign_leaf_ids(node.right, X_arr, row_indices[~mask_left], out)


    # ------------------------------------------------------------------
    # Cost-complexity pruning (B3, post-fit)
    # ------------------------------------------------------------------

    @staticmethod
    def _subtree_stats(node: PanelTreeNode) -> Tuple[float, int]:
        """Return ``(total_split_score, n_leaves)`` for the subtree at *node*.

        ``total_split_score`` is the sum of ``split_score`` over the internal
        nodes of the subtree — the cumulative predictability gain the subtree
        contributes over collapsing it to a single leaf.
        """
        if node.is_leaf:
            return 0.0, 1
        score_l, leaves_l = PanelTreeEngine._subtree_stats(node.left)
        score_r, leaves_r = PanelTreeEngine._subtree_stats(node.right)
        own = node.split_score if node.split_score is not None else 0.0
        return own + score_l + score_r, leaves_l + leaves_r

    @classmethod
    def _collapse(cls, node: PanelTreeNode) -> None:
        """Turn an internal node into a leaf (keeps its already-fitted model)."""
        node.is_leaf = True
        node.left = None
        node.right = None
        node.split_feature = None
        node.split_threshold = None
        node.split_score = None
        node.feature_ranking = None
        node._split_feature_idx = None

    @classmethod
    def _prune_node(cls, node: PanelTreeNode, ccp_alpha: float) -> None:
        """Bottom-up collapse of subtrees whose gain does not justify their leaves.

        A subtree rooted at *node* is collapsed when its cumulative split-score
        gain does not exceed ``ccp_alpha * (n_leaves - 1)`` — i.e. its
        *effective alpha* ``gain / (n_leaves - 1)`` is ``<= ccp_alpha``.
        Children are pruned first so the decision uses the post-pruning subtree.
        """
        if node.is_leaf:
            return
        cls._prune_node(node.left, ccp_alpha)
        cls._prune_node(node.right, ccp_alpha)
        if node.is_leaf:  # children pruning may have already collapsed us
            return
        score, n_leaves = cls._subtree_stats(node)
        if n_leaves > 1 and score <= ccp_alpha * (n_leaves - 1):
            cls._collapse(node)

    def prune(self, ccp_alpha: float) -> "PanelTreeEngine":
        """Post-prune the fitted tree in place via cost-complexity pruning.

        Parameters
        ----------
        ccp_alpha : float
            Complexity penalty per leaf.  ``0`` leaves the tree unchanged;
            larger values collapse more (weaker) subtrees.  Leaf models are
            retained (each node was fitted during ``fit``), so ``predict`` keeps
            working after pruning.
        """
        assert self.root_ is not None, "Call .fit() first."
        if ccp_alpha < 0:
            raise ValueError("ccp_alpha must be non-negative.")
        if ccp_alpha > 0:
            self._prune_node(self.root_, ccp_alpha)
        return self

    def cost_complexity_pruning_path(self) -> Dict[str, np.ndarray]:
        """Compute the weakest-link cost-complexity pruning path.

        Repeatedly collapses the internal node with the smallest *effective
        alpha* (``subtree_gain / (n_leaves - 1)``) on a copy of the fitted
        tree, recording the alpha at which each collapse happens together with
        the resulting leaf count and total remaining split-score gain.

        Returns
        -------
        dict with keys ``ccp_alphas``, ``n_leaves`` and ``total_scores`` (all
        ``np.ndarray``), ordered by increasing alpha.  ``ccp_alphas[0] == 0``
        corresponds to the full, unpruned tree.
        """
        assert self.root_ is not None, "Call .fit() first."
        root = copy.deepcopy(self.root_)

        total_score, n_leaves = self._subtree_stats(root)
        alphas = [0.0]
        leaves_seq = [n_leaves]
        scores_seq = [total_score]

        while not root.is_leaf:
            weakest, weakest_alpha = self._find_weakest_link(root)
            if weakest is None:
                break
            self._collapse(weakest)
            total_score, n_leaves = self._subtree_stats(root)
            alphas.append(weakest_alpha)
            leaves_seq.append(n_leaves)
            scores_seq.append(total_score)

        return {
            "ccp_alphas": np.asarray(alphas, dtype=float),
            "n_leaves": np.asarray(leaves_seq, dtype=int),
            "total_scores": np.asarray(scores_seq, dtype=float),
        }

    @classmethod
    def _find_weakest_link(
        cls, root: PanelTreeNode
    ) -> Tuple[Optional[PanelTreeNode], float]:
        """Return the internal node with the smallest effective alpha."""
        best_node: Optional[PanelTreeNode] = None
        best_alpha = np.inf

        def _walk(node: PanelTreeNode) -> None:
            nonlocal best_node, best_alpha
            if node.is_leaf:
                return
            score, n_leaves = cls._subtree_stats(node)
            if n_leaves > 1:
                alpha = score / (n_leaves - 1)
                if alpha < best_alpha:
                    best_alpha = alpha
                    best_node = node
            _walk(node.left)
            _walk(node.right)

        _walk(root)
        return best_node, float(best_alpha)

    # ------------------------------------------------------------------
    # Tree construction (recursive)
    # ------------------------------------------------------------------


    def _next_id(self) -> int:
        nid = self._node_counter
        self._node_counter += 1
        return nid

    def _build_node(
        self,
        indices: np.ndarray,
        depth: int,
        rule: str,
        parent_id: Optional[int],
        parent_ranking: Optional[List[tuple]],
    ) -> PanelTreeNode:
        t0 = time.time()
        node_id = self._next_id()

        node = PanelTreeNode(
            node_id=node_id,
            depth=depth,
            sample_indices=indices,
            rule=rule,
            parent_id=parent_id,
            sample_ratio=len(indices) / max(self._total_samples, 1),
        )

        # Fit local predictor on this node
        X_node = self._X[indices]
        y_node = self._y[indices]
        w_node = self._weights[indices] if self._weights is not None else None
        time_node = self._time[indices] if self._time is not None else None

        predictor = self._make_predictor()
        if self.honest and not self.honest_refit_full:
            # Honest leaves fit their model on a fit-subset only (no full-sample
            # refit), keeping the leaf model strictly out-of-evaluation-sample.
            n = X_node.shape[0]
            n_fit = min(max(int(round(self.honest_frac * n)), 1), max(n - 1, 1))
            fit_sel = self._rng.permutation(n)[:n_fit]
            w_fit = None if w_node is None else w_node[fit_sel]
            predictor.fit(X_node[fit_sel], y_node[fit_sel], weights=w_fit)
        else:
            predictor.fit(X_node, y_node, weights=w_node)
        node.predictor = predictor
        node.metrics = self._evaluate(X_node, y_node, predictor, w_node, time_node)



        # Compute and cache sufficient statistics (for Ridge incremental updates)
        if isinstance(predictor, (RidgeRegressor, VolWeightedRidgeRegressor)):
            XtWX, XtWy = compute_XtWX_XtWy(
                np.column_stack([np.ones(X_node.shape[0]), X_node])
                if getattr(predictor, "fit_intercept", True)
                else X_node,
                y_node,
                w_node,
            )
            node._XtWX = XtWX
            node._XtWy = XtWy

        # Check stopping conditions
        if depth >= self.max_depth or len(indices) < 2 * self.min_samples:
            node.is_leaf = True
            node.elapsed_time = time.time() - t0
            return node

        # Search for the best split.  Honest mode (B2) uses a dedicated path
        # that fits children on a *fit* subset and scores them on a disjoint
        # *eval* subset, removing the in-sample selection bias of greedy panel
        # splitting.  It bypasses the A1 incremental optimisation by design.
        if self.honest:
            best = self._find_best_split_honest(
                X_node, y_node, w_node, depth, time_node,
            )
        else:
            best = self._find_best_split(
                indices, X_node, y_node, w_node, depth, parent_ranking, node,
                time_node,
            )



        if best is None:
            node.is_leaf = True
            node.elapsed_time = time.time() - t0
            return node

        feat_idx, threshold, score, ranking = best

        # B4: minimum-impurity-decrease stopping criterion.  If the best split
        # does not improve the criterion by at least ``min_impurity_decrease``,
        # keep the node as a leaf.  Default ``0.0`` preserves prior behaviour.
        if score < self.min_impurity_decrease:
            node.is_leaf = True
            node.elapsed_time = time.time() - t0
            return node

        feat_name = self._feature_names[feat_idx]
        node.is_leaf = False

        node.split_feature = feat_name
        node.split_threshold = threshold
        node.split_score = score
        node.feature_ranking = ranking
        node._split_feature_idx = feat_idx  # cache for O(1) prediction routing


        # Partition
        mask_left = X_node[:, feat_idx] < threshold
        left_idx = indices[mask_left]
        right_idx = indices[~mask_left]

        left_rule = f"{rule} & {feat_name} < {threshold}"
        right_rule = f"{rule} & {feat_name} >= {threshold}"

        if self.verbose >= 1:
            self._log_split(node, feat_name, threshold, score, len(left_idx), len(right_idx))

        node.left = self._build_node(
            left_idx, depth + 1, left_rule, node_id, ranking
        )
        node.right = self._build_node(
            right_idx, depth + 1, right_rule, node_id, ranking
        )

        # A5: release cached sufficient statistics once both children are built
        # (no longer needed for incremental updates).  Keeps deep trees lean.
        if not self.keep_node_stats:
            node._XtWX = None
            node._XtWy = None

        node.elapsed_time = time.time() - t0
        return node


    # ------------------------------------------------------------------
    # Split search
    # ------------------------------------------------------------------

    def _find_best_split(
        self,
        indices: np.ndarray,
        X_node: np.ndarray,
        y_node: np.ndarray,
        w_node: Optional[np.ndarray],
        depth: int,
        parent_ranking: Optional[List[tuple]],
        node: PanelTreeNode,
        time_node: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, float, float, List[tuple]]]:
        """Search all (feature, threshold) combinations and return the best.


        Returns ``None`` if no valid split is found.

        Returns
        -------
        (feature_index, threshold, criterion_score, ranking_list) or None
        """
        n_features = X_node.shape[1]
        # ``thresholds`` is either the fixed global list or, in adaptive mode,
        # computed per-feature inside ``_evaluate_feature`` (passed as None).
        thresholds = None if self.adaptive_thresholds else self.split_thresholds

        # Determine feature evaluation order (priority caching)

        if self.fast_mode and parent_ranking is not None:
            # Prioritise top 50% features from parent
            top_k = max(1, len(parent_ranking) // 2)
            priority_feats = [r[0] for r in parent_ranking[:top_k]]
            remaining = [i for i in range(n_features) if i not in priority_feats]
            eval_order = priority_feats + remaining
        else:
            eval_order = list(range(n_features))

        # D1/P-Forest: restrict the search to a random feature subset at this
        # node (decorrelates trees in an ensemble).  ``None`` keeps all
        # features (standard P-Tree).  The subset is drawn from ``self._rng``
        # so it is reproducible given ``random_state``.
        if self.max_features is not None:
            k = self._resolve_max_features(n_features)
            if k < n_features:
                chosen = self._rng.choice(eval_order, size=k, replace=False)
                eval_order = list(chosen)


        # A2: when parallelism is requested (and joblib is available) and we
        # are NOT in the inherently-serial early-stopping path, evaluate each
        # feature's candidate thresholds in parallel.  Results are reduced in
        # deterministic ``eval_order`` so the chosen split is bit-identical to
        # the serial path (ties broken by first-seen, matching ``>`` below).
        use_parallel = (
            self.n_jobs != 1
            and _HAS_JOBLIB
            and not (self.fast_mode and self.early_stopping_threshold is not None)
        )

        if use_parallel:
            per_feature = Parallel(
                n_jobs=self.n_jobs, backend="loky"
                if self.parallel_backend == "processes"
                else "threading",
            )(
                delayed(self._evaluate_feature)(
                    feat_idx, X_node, y_node, w_node, thresholds, node, time_node
                )
                for feat_idx in eval_order

            )
            ranking: List[tuple] = []
            best_score = -np.inf
            best_split: Optional[Tuple[int, float]] = None
            for feat_results in per_feature:  # already in eval_order
                for feat_idx, thr, score in feat_results:
                    ranking.append((feat_idx, thr, score))
                    if score > best_score:
                        best_score = score
                        best_split = (feat_idx, thr)
        else:
            ranking = []
            best_score = -np.inf
            best_split = None

            for feat_idx in eval_order:
                feat_results = self._evaluate_feature(
                    feat_idx, X_node, y_node, w_node, thresholds, node, time_node
                )

                for fidx, thr, score in feat_results:
                    ranking.append((fidx, thr, score))
                    if self.verbose >= 2:
                        logger.debug(
                            "  [depth=%d] %s < %.2f  score=%.6f",
                            depth, self._feature_names[fidx], thr, score,
                        )
                    if score > best_score:
                        best_score = score
                        best_split = (fidx, thr)

                # Early stopping (fast mode) — serial path only.
                if (
                    self.fast_mode
                    and self.early_stopping_threshold is not None
                    and best_score >= self.early_stopping_threshold
                    and feat_idx in (eval_order[: max(1, len(eval_order) // 2)])
                ):
                    if self.verbose >= 1:
                        logger.info(
                            "  [depth=%d] Early stopping: score %.4f >= threshold %.4f",
                            depth, best_score, self.early_stopping_threshold,
                        )
                    break

        # Sort ranking descending by score
        ranking.sort(key=lambda x: x[2], reverse=True)

        if best_split is None:
            return None

        return (best_split[0], best_split[1], best_score, ranking)

    def _find_best_split_honest(
        self,
        X_node: np.ndarray,
        y_node: np.ndarray,
        w_node: Optional[np.ndarray],
        depth: int,
        time_node: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, float, float, List[tuple]]]:
        """Honest split search (B2).


        The node sample is split once into a *fit* subset and a disjoint *eval*
        subset (``honest_frac`` controls the fit fraction).  For every candidate
        ``(feature, threshold)`` the left/right predictors are fitted on the fit
        subset, then scored on the eval subset.  Because the samples used to
        *choose* the split differ from those used to *evaluate* it, the reported
        ``|R^2_L - R^2_R|`` is free of the in-sample selection bias of the greedy
        path.  Incremental matrix updates are intentionally not used here.

        The candidate threshold validity check (``min_samples``) is applied on
        both the fit and eval subsets so neither side becomes degenerate.
        """
        n = X_node.shape[0]
        n_fit = int(round(self.honest_frac * n))
        # Guard against degenerate fit/eval partitions.
        n_fit = min(max(n_fit, 1), n - 1)
        perm = self._rng.permutation(n)
        fit_idx = perm[:n_fit]
        eval_idx = perm[n_fit:]

        X_fit, y_fit = X_node[fit_idx], y_node[fit_idx]
        X_eval, y_eval = X_node[eval_idx], y_node[eval_idx]
        w_fit = w_node[fit_idx] if w_node is not None else None
        w_eval = w_node[eval_idx] if w_node is not None else None
        t_eval = time_node[eval_idx] if time_node is not None else None


        n_features = X_node.shape[1]
        ranking: List[tuple] = []
        best_score = -np.inf
        best_split: Optional[Tuple[int, float]] = None

        for feat_idx in range(n_features):
            col_fit = X_fit[:, feat_idx]
            col_eval = X_eval[:, feat_idx]
            if self.adaptive_thresholds:
                qs = np.quantile(X_node[:, feat_idx], self.adaptive_quantiles)
                thresholds = sorted(set(float(q) for q in qs))
            else:
                thresholds = self.split_thresholds

            for thr in thresholds:
                fit_left = col_fit < thr
                eval_left = col_eval < thr
                n_fit_l = int(fit_left.sum())
                n_fit_r = fit_left.shape[0] - n_fit_l
                n_eval_l = int(eval_left.sum())
                n_eval_r = eval_left.shape[0] - n_eval_l
                # Require both fit and eval children to be non-degenerate.
                if min(n_fit_l, n_fit_r) < self.min_samples:
                    continue
                if n_eval_l == 0 or n_eval_r == 0:
                    continue

                left_pred = self._make_predictor()
                right_pred = self._make_predictor()
                left_pred.fit(
                    X_fit[fit_left], y_fit[fit_left],
                    weights=None if w_fit is None else w_fit[fit_left],
                )
                right_pred.fit(
                    X_fit[~fit_left], y_fit[~fit_left],
                    weights=None if w_fit is None else w_fit[~fit_left],
                )

                # Score on the held-out eval subset (honest evaluation).
                left_metrics = self._evaluate(
                    X_eval[eval_left], y_eval[eval_left], left_pred,
                    None if w_eval is None else w_eval[eval_left],
                    None if t_eval is None else t_eval[eval_left],
                )
                right_metrics = self._evaluate(
                    X_eval[~eval_left], y_eval[~eval_left], right_pred,
                    None if w_eval is None else w_eval[~eval_left],
                    None if t_eval is None else t_eval[~eval_left],
                )

                score = self.criterion.calculate_score(left_metrics, right_metrics)
                ranking.append((feat_idx, thr, score))
                if score > best_score:
                    best_score = score
                    best_split = (feat_idx, thr)

        ranking.sort(key=lambda x: x[2], reverse=True)
        if best_split is None:
            return None
        return (best_split[0], best_split[1], best_score, ranking)

    def _evaluate_feature(

        self,
        feat_idx: int,
        X_node: np.ndarray,
        y_node: np.ndarray,
        w_node: Optional[np.ndarray],
        thresholds: List[float],
        node: PanelTreeNode,
        time_node: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, float, float]]:
        """Evaluate all candidate thresholds for a single feature.

        Returns a list of ``(feat_idx, threshold, score)`` for thresholds that
        produce two sufficiently-large children.  Pure w.r.t. engine state, so
        it can be dispatched to a worker thread/process.

        When ``thresholds`` is ``None`` (adaptive mode), the candidate
        thresholds are this feature's empirical quantiles within the node
        (``adaptive_quantiles``), de-duplicated to avoid redundant work.
        """
        col = X_node[:, feat_idx]
        if self.splitter == "random":
            # D3/Extra-Trees: draw ``n_random_splits`` random thresholds from
            # this feature's in-node value range instead of the exhaustive
            # ``split_thresholds`` sweep.  A degenerate (constant) column yields
            # no valid split.
            lo = float(col.min())
            hi = float(col.max())
            if hi <= lo:
                return []
            thresholds = [
                float(t) for t in self._rng.uniform(lo, hi, size=self.n_random_splits)
            ]
        elif thresholds is None:
            qs = np.quantile(col, self.adaptive_quantiles)
            thresholds = sorted(set(float(q) for q in qs))
        results: List[Tuple[int, float, float]] = []

        for thr in thresholds:

            mask_left = col < thr
            n_left = int(mask_left.sum())
            n_right = mask_left.shape[0] - n_left
            if n_left < self.min_samples or n_right < self.min_samples:
                continue
            left_metrics, right_metrics = self._fit_and_evaluate_children(
                X_node, y_node, w_node, mask_left, parent_node=node,
                time_node=time_node,
            )

            score = self.criterion.calculate_score(left_metrics, right_metrics)
            results.append((feat_idx, thr, score))
        return results


    def _fit_and_evaluate_children(
        self,
        X_node: np.ndarray,
        y_node: np.ndarray,
        w_node: Optional[np.ndarray],
        mask_left: np.ndarray,
        parent_node: PanelTreeNode,
        time_node: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Fit *both* child predictors for one candidate split and evaluate.


        Incremental optimisation (A1): when the parent node has cached Ridge
        sufficient statistics, only the *smaller* child's ``XtWX`` / ``XtWy``
        are computed directly (one matmul); the *larger* child's statistics
        are obtained by subtracting the smaller from the parent's cached
        statistics (a cheap ``O(p^2)`` subtraction).  This halves the matmul
        work of the previous implementation, which redundantly computed both
        sides in full.

        Returns ``(left_metrics, right_metrics)``.
        """
        mask_right = ~mask_left
        X_left, y_left = X_node[mask_left], y_node[mask_left]
        X_right, y_right = X_node[mask_right], y_node[mask_right]
        w_left = w_node[mask_left] if w_node is not None else None
        w_right = w_node[mask_right] if w_node is not None else None
        t_left = time_node[mask_left] if time_node is not None else None
        t_right = time_node[mask_right] if time_node is not None else None


        left_predictor = self._make_predictor()
        right_predictor = self._make_predictor()

        use_incremental = (
            isinstance(left_predictor, (RidgeRegressor, VolWeightedRidgeRegressor))
            and parent_node._XtWX is not None
        )

        if use_incremental:
            fit_intercept = getattr(left_predictor, "fit_intercept", True)
            if isinstance(left_predictor, VolWeightedRidgeRegressor):
                fit_intercept = getattr(left_predictor._inner, "fit_intercept", True)

            # Compute sufficient statistics for the *smaller* side only.
            n_left = int(X_left.shape[0])
            n_right = int(X_right.shape[0])
            small_is_left = n_left <= n_right

            if small_is_left:
                Xs, ys, ws = X_left, y_left, w_left
            else:
                Xs, ys, ws = X_right, y_right, w_right

            Xs_aug = (
                np.column_stack([np.ones(Xs.shape[0]), Xs])
                if fit_intercept
                else Xs
            )
            XtWX_small, XtWy_small = compute_XtWX_XtWy(Xs_aug, ys, ws)
            XtWX_large = parent_node._XtWX - XtWX_small
            XtWy_large = parent_node._XtWy - XtWy_small

            if small_is_left:
                left_predictor.fit(
                    X_left, y_left, weights=w_left,
                    XtWX=XtWX_small, XtWy=XtWy_small,
                )
                right_predictor.fit(
                    X_right, y_right, weights=w_right,
                    XtWX=XtWX_large, XtWy=XtWy_large,
                )
            else:
                right_predictor.fit(
                    X_right, y_right, weights=w_right,
                    XtWX=XtWX_small, XtWy=XtWy_small,
                )
                left_predictor.fit(
                    X_left, y_left, weights=w_left,
                    XtWX=XtWX_large, XtWy=XtWy_large,
                )
        else:
            left_predictor.fit(X_left, y_left, weights=w_left)
            right_predictor.fit(X_right, y_right, weights=w_right)

        left_metrics = self._evaluate(X_left, y_left, left_predictor, w_left, t_left)
        right_metrics = self._evaluate(
            X_right, y_right, right_predictor, w_right, t_right
        )
        return left_metrics, right_metrics



    # ------------------------------------------------------------------
    # Evaluation dispatch
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        predictor: PredictorBase,
        weights: Optional[np.ndarray],
        time: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        if self._is_classification:
            if hasattr(predictor, "predict_proba"):
                proba = predictor.predict_proba(X)
            else:
                proba = predictor.predict(X)
            metrics = evaluate_classification(y, proba)
        else:
            y_pred = predictor.predict(X)
            metrics = evaluate_regression(y, y_pred, weights)
            # C3/B1: attach the per-time long-short portfolio return series so
            # the mean-variance criterion can evaluate the efficient-frontier
            # contribution of this (candidate) leaf.
            if self._needs_port_ret and time is not None:
                metrics["_port_ret"] = self._portfolio_returns(y, y_pred, time)
        return metrics

    @staticmethod
    def _portfolio_returns(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        time: np.ndarray,
    ) -> Dict[Any, float]:
        """Aggregate a leaf into a per-time long-short portfolio return.

        Within each time period the predicted returns are cross-sectionally
        de-meaned to form dollar-neutral long-short weights ``w_i = yhat_i -
        mean(yhat)``; the period's portfolio return is the weight-normalised
        realised return ``sum(w_i * y_i) / sum(|w_i|)``.  Periods with a single
        observation or zero net weight contribute ``0``.

        Returns
        -------
        dict mapping each time label to its portfolio return.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        time = np.asarray(time)
        out: Dict[Any, float] = {}
        # Group by time label.  ``np.unique`` keeps this O(n log n).
        uniq = np.unique(time)
        for t in uniq:
            mask = time == t
            if mask.sum() < 2:
                continue
            w = y_pred[mask] - y_pred[mask].mean()
            denom = np.abs(w).sum()
            if denom <= 1e-12:
                continue
            out[t.item() if hasattr(t, "item") else t] = float(
                (w * y_true[mask]).sum() / denom
            )
        return out


    # ------------------------------------------------------------------
    # Prediction (recursive)
    # ------------------------------------------------------------------

    def _predict_recursive(
        self,
        node: PanelTreeNode,
        X_arr: np.ndarray,
        row_indices: np.ndarray,
        out: np.ndarray,
    ) -> None:
        if node.is_leaf:
            if node.predictor is not None and len(row_indices) > 0:
                preds = node.predictor.predict(X_arr[row_indices])
                out[row_indices] = preds
            return

        # A6: use the cached integer index (falls back to a name lookup for
        # trees built before this field existed, e.g. unpickled old models).
        feat_idx = node._split_feature_idx
        if feat_idx is None:
            feat_idx = self._feature_names.index(node.split_feature)
        vals = X_arr[row_indices, feat_idx]

        mask_left = vals < node.split_threshold

        if node.left is not None:
            self._predict_recursive(node.left, X_arr, row_indices[mask_left], out)
        if node.right is not None:
            self._predict_recursive(node.right, X_arr, row_indices[~mask_left], out)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_predictor(self) -> PredictorBase:
        """Create a fresh copy of the predictor template."""
        return copy.deepcopy(self._predictor_template)

    def _resolve_max_features(self, n_features: int) -> int:
        """Resolve ``self.max_features`` to a concrete feature count (D1).

        Mirrors the scikit-learn convention used by random forests:

        * ``"sqrt"``  → ``max(1, floor(sqrt(p)))``
        * ``"log2"``  → ``max(1, floor(log2(p)))``
        * ``int``     → ``min(value, p)`` (absolute count)
        * ``float``   → ``max(1, floor(fraction * p))`` (fraction of features)
        * otherwise   → ``p`` (all features)
        """
        mf = self.max_features
        if mf is None:
            return n_features
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if mf == "log2":
                return max(1, int(np.log2(n_features)))
            raise ValueError(
                f"Unknown max_features string {mf!r}; expected 'sqrt' or 'log2'."
            )
        if isinstance(mf, (int, np.integer)) and not isinstance(mf, bool):
            return max(1, min(int(mf), n_features))
        if isinstance(mf, float):
            return max(1, min(int(mf * n_features), n_features))
        return n_features


    @staticmethod
    def _collect_leaves(node: PanelTreeNode, acc: List[PanelTreeNode]) -> None:
        if node.is_leaf:
            acc.append(node)
        else:
            if node.left is not None:
                PanelTreeEngine._collect_leaves(node.left, acc)
            if node.right is not None:
                PanelTreeEngine._collect_leaves(node.right, acc)

    def _log_split(
        self,
        node: PanelTreeNode,
        feat: str,
        thr: float,
        score: float,
        n_left: int,
        n_right: int,
    ) -> None:
        key = self.criterion.metric_key()
        m_left = node.metrics.get(key, float("nan"))
        logger.info(
            "[Level %d] Splitting Node %d...\n"
            "  - Best Split: '%s' at threshold %.4f\n"
            "  - Metric Delta: score = %.6f\n"
            "  - Left: %d samples | Right: %d samples",
            node.depth, node.node_id,
            feat, thr, score,
            n_left, n_right,
        )
