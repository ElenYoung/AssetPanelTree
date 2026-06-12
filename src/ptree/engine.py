"""
PanelTreeEngine: recursive splitting, pruning, feature-priority caching,
incremental matrix updates, and multiprocessing support.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import (
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
        fast_mode: bool = False,
        early_stopping_threshold: Optional[float] = None,
        n_jobs: int = 1,
        verbose: int = 1,
        predictor_params: Optional[Dict] = None,
        keep_node_stats: bool = False,
        parallel_backend: str = "threads",
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



        # State populated after fitting
        self.root_: Optional[PanelTreeNode] = None
        self._node_counter: int = 0
        self._total_samples: int = 0
        self._feature_names: List[str] = []
        self._is_classification: bool = isinstance(
            self.criterion, ClassificationCriterion
        )
        # Store processed data for predictions & cluster retrieval
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
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

        predictor = self._make_predictor()
        predictor.fit(X_node, y_node, weights=w_node)
        node.predictor = predictor
        node.metrics = self._evaluate(X_node, y_node, predictor, w_node)

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

        # Search for the best split
        best = self._find_best_split(
            indices, X_node, y_node, w_node, depth, parent_ranking, node,
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
                    feat_idx, X_node, y_node, w_node, thresholds, node
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
                    feat_idx, X_node, y_node, w_node, thresholds, node
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

    def _evaluate_feature(
        self,
        feat_idx: int,
        X_node: np.ndarray,
        y_node: np.ndarray,
        w_node: Optional[np.ndarray],
        thresholds: List[float],
        node: PanelTreeNode,
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
        if thresholds is None:
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

        left_metrics = self._evaluate(X_left, y_left, left_predictor, w_left)
        right_metrics = self._evaluate(X_right, y_right, right_predictor, w_right)
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
        return metrics

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
