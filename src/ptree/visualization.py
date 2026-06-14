"""
Visualization and reporting utilities for PanelTree.

* ``NodeReporter``    – structured DataFrame / text summaries.
* ``MosaicVisualizer`` – prediction mosaic heatmap.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ptree.engine import NodeEvalResult, PanelTreeEngine


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

    def print_tree(
        self,
        node=None,
        evaluation: Optional["NodeEvalResult"] = None,
        show_child_diff: bool = False,
    ) -> str:
        """Return a text representation of the tree structure.

        Uses box-drawing characters (``│``, ``├──``, ``└──``) with
        continuous vertical lines so that sibling relationships remain
        visible even in deep trees.  Both internal and leaf nodes display
        their evaluation metrics.

        Parameters
        ----------
        node : PanelTreeNode, optional
            Subtree root to start from; defaults to the engine's root.
        evaluation : NodeEvalResult, optional
            Output of :meth:`PanelTreeEngine.evaluate`.  When supplied,
            each node's line is augmented with OOS metrics
            (``oos_r2``, ``oos_rank_ic_mean`` / ``IR``, ``n_oos``).
        show_child_diff : bool, default False
            When ``True`` *and* ``evaluation`` is supplied, an extra
            ``↳ L vs R | ΔR²=…  ΔIC=…`` line is inserted under every
            internal node, surfacing whether the split actually opened a
            predictability gap between its children.
        """
        if node is None:
            node = self.engine.root_

        eval_lookup: Optional[Dict[int, Dict[str, Any]]] = None
        if evaluation is not None:
            df = evaluation.per_node_df.set_index("node_id")
            eval_lookup = df.to_dict(orient="index")

        lines: List[str] = []
        self._print_recursive(
            node, "", "", lines,
            eval_lookup=eval_lookup,
            show_child_diff=show_child_diff,
        )
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

    @staticmethod
    def _format_eval_row(row: Dict[str, Any]) -> str:
        """Format the OOS columns of a NodeEvalResult row into one suffix."""
        bits: List[str] = []
        n_oos = row.get("n_oos")
        if n_oos is not None and not pd.isna(n_oos):
            bits.append(f"n_oos={int(n_oos)}")
        if "oos_r2" in row and not pd.isna(row["oos_r2"]):
            bits.append(f"oos_r2={row['oos_r2']:+.4f}")
        if "oos_rank_ic_mean" in row and not pd.isna(row["oos_rank_ic_mean"]):
            ir = row.get("oos_rank_ic_ir", float("nan"))
            if not pd.isna(ir):
                bits.append(
                    f"ic={row['oos_rank_ic_mean']:+.4f} (IR={ir:+.2f})"
                )
            else:
                bits.append(f"ic={row['oos_rank_ic_mean']:+.4f}")
        return " | ".join(bits)

    @staticmethod
    def _format_child_diff_row(row: Dict[str, Any]) -> str:
        """Format the ΔR² / ΔIC summary line for an internal node."""
        bits: List[str] = []
        if "delta_oos_r2" in row and not pd.isna(row["delta_oos_r2"]):
            l = row.get("left_oos_r2", float("nan"))
            r = row.get("right_oos_r2", float("nan"))
            bits.append(
                f"ΔR²={row['delta_oos_r2']:+.4f} ({l:+.4f} vs {r:+.4f})"
            )
        if "delta_oos_rank_ic" in row and not pd.isna(row["delta_oos_rank_ic"]):
            l = row.get("left_oos_rank_ic_mean", float("nan"))
            r = row.get("right_oos_rank_ic_mean", float("nan"))
            bits.append(
                f"ΔIC={row['delta_oos_rank_ic']:+.4f} ({l:+.4f} vs {r:+.4f})"
            )
        return " | ".join(bits)

    def _print_recursive(
        self,
        node,
        prefix: str,
        connector: str,
        lines: List[str],
        eval_lookup: Optional[Dict[int, Dict[str, Any]]] = None,
        show_child_diff: bool = False,
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
        eval_lookup, show_child_diff
            Optional OOS overlay; see :meth:`print_tree`.
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

        # OOS suffix (if an evaluation was supplied).
        if eval_lookup is not None and node.node_id in eval_lookup:
            suffix = self._format_eval_row(eval_lookup[node.node_id])
            if suffix:
                label = f"{label} | {suffix}"

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

            # Optional ΔR² / ΔIC summary line under the node.
            if (
                show_child_diff
                and eval_lookup is not None
                and node.node_id in eval_lookup
            ):
                diff_str = self._format_child_diff_row(eval_lookup[node.node_id])
                if diff_str:
                    lines.append(f"{child_prefix}↳ L vs R | {diff_str}")

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
                    eval_lookup=eval_lookup,
                    show_child_diff=show_child_diff,
                )

    # ------------------------------------------------------------------
    # Graphviz export
    # ------------------------------------------------------------------

    def to_graphviz(
        self,
        evaluation: Optional["NodeEvalResult"] = None,
        show_child_diff: bool = False,
        leaf_fill: str = "#cfe8d6",
        node_fill: str = "#dceaf7",
    ) -> str:
        """Return a Graphviz DOT representation of the fitted tree.

        Renderable with ``graphviz``::

            from graphviz import Source
            Source(NodeReporter(engine).to_graphviz()).render('tree')

        Parameters
        ----------
        evaluation : NodeEvalResult, optional
            When supplied, OOS metrics from :meth:`PanelTreeEngine.evaluate`
            are appended to each node's label (same conventions as
            :meth:`print_tree`).
        show_child_diff : bool, default False
            Add a ``ΔR² / ΔIC`` summary line under every internal node label.
        leaf_fill, node_fill : str
            Hex / named colour for leaf and internal node fill.

        Returns
        -------
        dot : str
            DOT source — no Graphviz binary or python ``graphviz``
            package is required just to obtain the source string.
        """
        assert self.engine.root_ is not None, "Engine not fitted."

        eval_lookup: Optional[Dict[int, Dict[str, Any]]] = None
        if evaluation is not None:
            df = evaluation.per_node_df.set_index("node_id")
            eval_lookup = df.to_dict(orient="index")

        lines: List[str] = [
            "digraph PanelTree {",
            "  graph [rankdir=TB, fontname=\"Helvetica\"];",
            "  node  [shape=box, style=\"rounded,filled\", "
            "fontname=\"Helvetica\"];",
            "  edge  [fontname=\"Helvetica\"];",
        ]

        def _node_label(node) -> str:
            metric_str = self._format_metrics(node.metrics)
            if node.is_leaf:
                head = f"Leaf {node.node_id}\\nn={node.n_samples}"
            else:
                head = (
                    f"Node {node.node_id}\\n"
                    f"{node.split_feature} < {node.split_threshold}\\n"
                    f"n={node.n_samples}, Δ={node.split_score:.4f}"
                )
            label = f"{head}\\n{metric_str}" if metric_str else head
            if eval_lookup is not None and node.node_id in eval_lookup:
                suffix = self._format_eval_row(eval_lookup[node.node_id])
                if suffix:
                    label = f"{label}\\n{suffix}"
                if show_child_diff and not node.is_leaf:
                    diff = self._format_child_diff_row(
                        eval_lookup[node.node_id]
                    )
                    if diff:
                        label = f"{label}\\nL vs R: {diff}"
            return label.replace('"', r"\"")

        def _walk(node) -> None:
            fill = leaf_fill if node.is_leaf else node_fill
            lines.append(
                f"  n{node.node_id} ["
                f"label=\"{_node_label(node)}\", fillcolor=\"{fill}\"];"
            )
            if node.left is not None:
                lines.append(
                    f"  n{node.node_id} -> n{node.left.node_id} "
                    f"[label=\"< {node.split_threshold}\"];"
                )
                _walk(node.left)
            if node.right is not None:
                lines.append(
                    f"  n{node.node_id} -> n{node.right.node_id} "
                    f"[label=\">= {node.split_threshold}\"];"
                )
                _walk(node.right)

        _walk(self.engine.root_)
        lines.append("}")
        return "\n".join(lines)



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
        cmap: str = "RdBu_r",
        center: Optional[float] = 0.0,
        figsize: tuple = (14, 6),
        save_path: Optional[str] = None,
    ):
        """Plot the mosaic as a heatmap.

        Requires ``matplotlib`` and ``seaborn``.

        Parameters
        ----------
        mosaic : DataFrame
            Output of :meth:`build_mosaic`.
        title : str
            Figure title.
        cmap : str, default ``"RdBu_r"``
            Matplotlib / seaborn colormap.  The default ``"RdBu_r"`` is
            colour-blind safe and uses the *finance-positive-is-red*
            convention (replaces the previous ``"RdYlGn"`` default).
        center : float or None, default 0.0
            Value at which the colormap centres.  Pass ``None`` to disable
            centering (useful for one-sided metrics like raw |IC|).
        figsize : tuple
            Matplotlib figure size in inches.
        save_path : str, optional
            If given, the figure is also written to this path.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            mosaic.astype(float),
            cmap=cmap,
            center=center,
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
