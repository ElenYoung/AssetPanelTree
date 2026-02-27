"""
Visualization and reporting utilities for PanelTree.

* ``NodeReporter``    – structured DataFrame / text summaries.
* ``MosaicVisualizer`` – prediction mosaic heatmap.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ptree.engine import PanelTreeEngine

logger = logging.getLogger("ptree")


# ======================================================================
# Node Reporter
# ======================================================================

class NodeReporter:
    """Generate structured node-level reports from a fitted PanelTree.

    Parameters
    ----------
    engine : PanelTreeEngine
        A fitted engine instance.
    """

    def __init__(self, engine: "PanelTreeEngine"):
        self.engine = engine

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame with one row per node.

        Columns
        -------
        Node_ID, Depth, Rule, Is_Leaf, N_Samples, Sample_Ratio,
        Split_Feature, Split_Threshold, Split_Score,
        Predictability_Score, Metrics, Model_Weights, Elapsed_Time_s,
        Parent_ID.
        """
        return self.engine.get_node_report()

    def leaf_summary(self) -> pd.DataFrame:
        """Same as :meth:`summary` but only leaf nodes."""
        df = self.summary()
        return df[df["Is_Leaf"]].reset_index(drop=True)

    def print_tree(self, node=None) -> str:
        """Return a text representation of the tree structure.

        Uses box-drawing characters (``│``, ``├──``, ``└──``) with
        continuous vertical lines so that sibling relationships remain
        visible even in deep trees.  Both internal and leaf nodes display
        their evaluation metrics.
        """
        if node is None:
            node = self.engine.root_
        lines: List[str] = []
        self._print_recursive(node, "", "", lines)
        return "\n".join(lines)

    @staticmethod
    def _format_metrics(metrics: Dict, exclude: Optional[set] = None) -> str:
        """Format a metrics dict into a compact string."""
        skip = exclude or {"n_samples"}
        parts = []
        for k, v in metrics.items():
            if k in skip:
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            elif isinstance(v, int):
                parts.append(f"{k}={v}")
        return ", ".join(parts)

    def _print_recursive(
        self,
        node,
        prefix: str,
        connector: str,
        lines: List[str],
    ) -> None:
        """Recursively build tree-drawing lines.

        Parameters
        ----------
        prefix : str
            Accumulated prefix of ``│   `` or ``    `` segments that keep
            vertical lines aligned for ancestor branches.
        connector : str
            ``""`` for the root, ``"├── "`` for non-last children,
            ``"└── "`` for the last child.
        """
        metric_str = self._format_metrics(node.metrics)

        if node.is_leaf:
            label = (
                f"[Leaf {node.node_id}] "
                f"{metric_str}, n={node.n_samples}"
            )
        else:
            label = (
                f"[Node {node.node_id}] "
                f"{node.split_feature} < {node.split_threshold} | "
                f"{metric_str}, n={node.n_samples} "
                f"(Δ={node.split_score:.4f})"
            )

        lines.append(f"{prefix}{connector}{label}")

        if not node.is_leaf:
            # Build the new prefix for children.  If the current node was
            # connected with "├── " there is still a sibling below, so we
            # continue the vertical bar; otherwise we pad with spaces.
            if connector == "├── ":
                child_prefix = prefix + "│   "
            elif connector == "└── ":
                child_prefix = prefix + "    "
            else:
                # Root node – no extra prefix
                child_prefix = prefix

            children = []
            if node.left is not None:
                children.append(node.left)
            if node.right is not None:
                children.append(node.right)

            for i, child in enumerate(children):
                is_last = i == len(children) - 1
                child_connector = "└── " if is_last else "├── "
                self._print_recursive(
                    child, child_prefix, child_connector, lines,
                )


# ======================================================================
# Mosaic Visualizer
# ======================================================================

class MosaicVisualizer:
    """Generate prediction-mosaic heatmaps.

    The mosaic has:
    * **x-axis**: time periods.
    * **y-axis**: leaf nodes.
    * **colour**: predictability score (R² or precision) or contribution.

    Parameters
    ----------
    engine : PanelTreeEngine
        Fitted engine.
    """

    def __init__(self, engine: "PanelTreeEngine"):
        self.engine = engine

    def build_mosaic(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_col: str = "date",
        metric: str = "r2",
    ) -> pd.DataFrame:
        """Compute per-leaf, per-period metric values.

        Parameters
        ----------
        X : DataFrame
            Panel data (must include *time_col* and feature columns).
        y : Series
            Target variable aligned with *X*.
        time_col : str
            Column identifying the time dimension.
        metric : str
            ``"r2"`` for regression or ``"precision"`` / ``"f1"`` / ``"auc"``
            for classification.

        Returns
        -------
        mosaic : DataFrame
            Index = leaf node IDs, columns = time periods, values = metric.
        """
        engine = self.engine
        assert engine.root_ is not None, "Engine not fitted."

        feature_names = engine._feature_names
        X_arr = X[feature_names].values.astype(np.float64)
        y_arr = y.values.astype(np.float64)
        times = X[time_col].values

        # Assign each row to a leaf
        leaf_ids = np.full(len(X_arr), -1, dtype=int)
        self._assign_leaves(engine.root_, X_arr, np.arange(len(X_arr)), leaf_ids, feature_names)

        unique_times = np.sort(np.unique(times))
        leaves = engine.get_leaves()
        leaf_id_list = [l.node_id for l in leaves]

        mosaic_data: Dict[int, Dict] = {lid: {} for lid in leaf_id_list}

        for t in unique_times:
            time_mask = times == t
            for leaf in leaves:
                mask = time_mask & (leaf_ids == leaf.node_id)
                n = int(mask.sum())
                if n == 0:
                    mosaic_data[leaf.node_id][t] = np.nan
                    continue

                X_sub = X_arr[mask]
                y_sub = y_arr[mask]

                if metric == "r2":
                    y_pred = leaf.predictor.predict(X_sub)
                    ss_res = np.sum((y_sub - y_pred) ** 2)
                    ss_tot = np.sum((y_sub - y_sub.mean()) ** 2)
                    val = 1.0 - ss_res / max(ss_tot, 1e-12) if ss_tot > 0 else 0.0
                elif metric in ("precision", "f1", "auc"):
                    if hasattr(leaf.predictor, "predict_proba"):
                        proba = leaf.predictor.predict_proba(X_sub)
                    else:
                        proba = leaf.predictor.predict(X_sub)
                    from ptree.criteria import evaluate_classification
                    cls_metrics = evaluate_classification(y_sub, proba)
                    val = cls_metrics.get(metric, 0.0)
                else:
                    val = np.nan

                mosaic_data[leaf.node_id][t] = val

        mosaic = pd.DataFrame(mosaic_data).T
        mosaic.index.name = "Leaf_ID"
        mosaic.columns.name = time_col
        return mosaic

    def plot_mosaic(
        self,
        mosaic: pd.DataFrame,
        title: str = "Prediction Mosaic",
        cmap: str = "RdYlGn",
        figsize: tuple = (14, 6),
        save_path: Optional[str] = None,
    ):
        """Plot the mosaic as a heatmap.

        Requires ``matplotlib`` and ``seaborn``.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            mosaic.astype(float),
            cmap=cmap,
            center=0,
            ax=ax,
            linewidths=0.5,
            xticklabels=True,
            yticklabels=True,
        )
        ax.set_title(title)
        ax.set_xlabel(mosaic.columns.name or "Time")
        ax.set_ylabel("Leaf Node")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Mosaic saved to %s", save_path)

        return fig, ax

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_leaves(
        node,
        X_arr: np.ndarray,
        row_indices: np.ndarray,
        out: np.ndarray,
        feature_names: List[str],
    ) -> None:
        if node.is_leaf:
            out[row_indices] = node.node_id
            return
        feat_idx = feature_names.index(node.split_feature)
        vals = X_arr[row_indices, feat_idx]
        mask_left = vals < node.split_threshold
        if node.left is not None:
            MosaicVisualizer._assign_leaves(
                node.left, X_arr, row_indices[mask_left], out, feature_names
            )
        if node.right is not None:
            MosaicVisualizer._assign_leaves(
                node.right, X_arr, row_indices[~mask_left], out, feature_names
            )
