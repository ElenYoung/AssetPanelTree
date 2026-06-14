"""
PanelTreeNode: container for a single node in the Panel Tree.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np

from ptree.predictors import PredictorBase


@dataclass
class PanelTreeNode:
    """Stores all structural and statistical information for one node.

    Attributes
    ----------
    node_id : int
        Unique identifier within the tree.
    depth : int
        Depth level (root = 0).
    sample_indices : ndarray
        Row indices (into the processed data) belonging to this node.
    predictor : PredictorBase or None
        Trained local model, populated after fitting.
    metrics : dict
        Evaluation metrics produced by the criterion (e.g. ``{"r2": 0.05}``).
    split_feature : str or None
        Name of the feature used for splitting (``None`` for leaves).
    split_threshold : float or None
        Threshold value used for the split.
    left : PanelTreeNode or None
        Left child (values < threshold).
    right : PanelTreeNode or None
        Right child (values >= threshold).
    is_leaf : bool
        ``True`` if the node is a leaf.
    split_score : float or None
        The criterion score achieved by the best split at this node.
    elapsed_time : float
        Seconds spent fitting / splitting this node.
    sample_ratio : float
        Fraction of total samples covered by this node.
    parent_id : int or None
        ``node_id`` of the parent (``None`` for root).
    rule : str
        Human-readable description of the path from root to this node.
    feature_ranking : list of tuple or None
        Ordered list of ``(feature, threshold, score)`` evaluated at this
        node, used to accelerate child splitting via priority caching.
    """

    node_id: int = 0
    depth: int = 0
    sample_indices: Optional[np.ndarray] = field(default=None, repr=False)
    predictor: Optional[PredictorBase] = field(default=None, repr=False)
    metrics: Dict[str, float] = field(default_factory=dict)
    split_feature: Optional[str] = None
    split_threshold: Optional[float] = None
    left: Optional["PanelTreeNode"] = None
    right: Optional["PanelTreeNode"] = None
    is_leaf: bool = True
    split_score: Optional[float] = None
    elapsed_time: float = 0.0
    sample_ratio: float = 0.0
    parent_id: Optional[int] = None
    rule: str = "root"
    feature_ranking: Optional[List[tuple]] = field(default=None, repr=False)

    # Cached sufficient statistics for incremental matrix updates
    _XtWX: Optional[np.ndarray] = field(default=None, repr=False)
    _XtWy: Optional[np.ndarray] = field(default=None, repr=False)
    # Cached integer column index of ``split_feature`` (avoids O(p) lookups
    # during prediction).  Populated by the engine after a split is chosen.
    _split_feature_idx: Optional[int] = field(default=None, repr=False)

    # Persisted sample-count cache.  ``sample_indices`` may be stripped before
    # pickling (or for memory reasons) — this field survives that strip so
    # downstream code can keep using ``node.n_samples`` as the single source
    # of truth rather than the ad-hoc ``metrics['n_samples']`` fallback.
    _n_samples_cached: Optional[int] = field(default=None, repr=False)
    # Honest-mode: when the engine was fitted with ``honest=True``, this is
    # the size of the *evaluation* (held-out) subset that scored each split.
    # ``None`` for non-honest trees or non-internal nodes that were not
    # involved in honest evaluation.
    honest_n_samples: Optional[int] = field(default=None, repr=False)


    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Number of training samples in this node.

        Resolution order (single source of truth):

        1. The current ``sample_indices`` length, if still attached.
        2. The cached ``_n_samples_cached`` value (survives index stripping).
        3. The legacy ``metrics['n_samples']`` value (older pickles).
        4. ``0`` if nothing is available.
        """
        if self.sample_indices is not None:
            return len(self.sample_indices)
        if self._n_samples_cached is not None:
            return int(self._n_samples_cached)
        legacy = self.metrics.get("n_samples")
        if legacy is not None:
            return int(legacy)
        return 0

    def get_model_weights(self) -> Optional[np.ndarray]:
        """Return the fitted model's coefficients, if available."""
        if self.predictor is not None:
            return self.predictor.get_coefficients()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise node metadata to a flat dictionary (for DataFrame)."""
        coef = self.get_model_weights()
        return {
            "Node_ID": self.node_id,
            "Depth": self.depth,
            "Rule": self.rule,
            "Is_Leaf": self.is_leaf,
            "N_Samples": self.n_samples,
            "Sample_Ratio": round(self.sample_ratio, 4),
            "Split_Feature": self.split_feature,
            "Split_Threshold": self.split_threshold,
            "Split_Score": self.split_score,
            "Predictability_Score": self.metrics.get(
                "r2", self.metrics.get("precision", None)
            ),
            "Metrics": self.metrics,
            "Model_Weights": coef.tolist() if coef is not None else None,
            "Elapsed_Time_s": round(self.elapsed_time, 4),
            "Parent_ID": self.parent_id,
        }

    def get_samples(self) -> Optional[np.ndarray]:
        """Return sample indices belonging to this node (for cluster retrieval)."""
        return self.sample_indices

    def __repr__(self) -> str:
        tag = "Leaf" if self.is_leaf else "Split"
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items() if isinstance(v, float))
        return (
            f"PanelTreeNode(id={self.node_id}, {tag}, depth={self.depth}, "
            f"n={self.n_samples}, {metric_str})"
        )
