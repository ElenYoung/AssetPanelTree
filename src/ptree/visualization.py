"""
Visualization and reporting utilities for PanelTree.

Public classes
--------------
* :class:`NodeReporter`     – structured DataFrame / text / Graphviz /
  matplotlib tree summaries.
* :class:`MosaicVisualizer` – prediction mosaic heatmap.

Design principles
-----------------
1. **Library defaults are publication-clean English.** Titles, axis labels
   and colour-bar labels never embed parameter values, version strings or
   Chinese text.  Callers may override every label.
2. **Metric strings are curated, not dumped.** Only domain-meaningful keys
   are rendered (``r2``, ``rank_ic*``, ``precision``, ``f1``, ``auc``,
   ``ir``) and they are translated to short, conventional symbols
   (``R²``, ``IC``, ``IR``, …).  Internal bookkeeping fields like
   ``n_features`` and ``n_samples`` are filtered out.
3. **Charts respect this server's font setup.** A single helper applies the
   project-wide matplotlib rc settings (``Noto Sans CJK SC`` + DejaVu) so
   user-supplied Chinese titles still render correctly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ptree.engine import NodeEvalResult, PanelTreeEngine


logger = logging.getLogger("ptree")


# ======================================================================
# Shared helpers
# ======================================================================

# Pretty names for known metric keys.  Anything not listed here is hidden
# by default to keep node labels compact and professional.
_METRIC_DISPLAY = {
    "r2": "R²",
    "mse": "MSE",
    "rank_ic": "IC",
    "rank_ic_mean": "IC",
    "rank_ic_ir": "IR",
    "ir": "IR",
    "precision": "Prec",
    "f1": "F1",
    "auc": "AUC",
    "logloss": "LogLoss",
}

# Keys included by default in node labels (text tree, graphviz, plot_tree).
_DEFAULT_VISIBLE_METRICS = ("r2", "rank_ic_mean", "ir", "precision", "f1", "auc")

# Keys explicitly hidden by default (bookkeeping rather than diagnostic).
_DEFAULT_HIDDEN_METRICS = {"n_samples", "n_features", "mse", "logloss"}


def _apply_mpl_style() -> None:
    """Apply the project-wide matplotlib defaults.

    Sets the CJK-capable font stack and disables the Unicode-minus glyph
    so that negative numbers and any Chinese strings supplied by the user
    render correctly.  Safe to call repeatedly.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 110,
        }
    )


# All numeric values rendered by this module are clipped to three decimal
# places — finer precision is misleading on noisy financial panels and the
# extra digits add visual clutter to node boxes / heatmap annotations.
_DEC = 3


def _format_float(v: float, digits: int = _DEC, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if signed:
        return f"{v:+.{digits}f}"
    return f"{v:.{digits}f}"


def _crit_metric_key(engine) -> Optional[str]:
    """Return the criterion's primary metric key (e.g. ``"r2"``).

    Defensive against engines fitted with a custom criterion that doesn't
    expose ``metric_key()`` – we simply fall back to ``"r2"``.
    """
    crit = getattr(engine, "criterion", None)
    if crit is None:
        return "r2"
    try:
        key = crit.metric_key()
    except Exception:  # pragma: no cover - defensive
        return "r2"
    return key or "r2"


def _crit_metric_display(engine) -> str:
    """Return a human-readable symbol for the criterion's primary metric."""
    key = _crit_metric_key(engine)
    return _METRIC_DISPLAY.get(key, key.upper() if key else "metric")



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

    # ------------------------------------------------------------------
    # DataFrame summaries
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Text tree
    # ------------------------------------------------------------------

    def print_tree(
        self,
        node=None,
        evaluation: Optional["NodeEvalResult"] = None,
        show_child_diff: bool = False,
        metric_keys: Optional[Sequence[str]] = None,
    ) -> str:
        """Return a text representation of the tree structure.

        Uses box-drawing characters (``│``, ``├──``, ``└──``) so sibling
        relationships stay visible even in deep trees.  Both internal and
        leaf nodes show curated evaluation metrics.

        Parameters
        ----------
        node : PanelTreeNode, optional
            Subtree root; defaults to the engine's root.
        evaluation : NodeEvalResult, optional
            Output of :meth:`PanelTreeEngine.evaluate`.  When supplied,
            each node's line is augmented with OOS metrics
            (``n_oos``, ``oos_r2``, ``oos_rank_ic_mean`` / ``IR``).
        show_child_diff : bool, default False
            When ``True`` *and* ``evaluation`` is supplied, an extra
            ``↳ L vs R | ΔR²=…  ΔIC=…`` line is inserted under every
            internal node, surfacing whether the split actually opened a
            predictability gap between its children.
        metric_keys : sequence of str, optional
            Override which training-metric keys to display.  Defaults to
            the conventional subset (``r2``, ``rank_ic_mean``, …).
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
            metric_keys=metric_keys,
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Metric formatting (curated, not dumped)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_metrics(
        metrics: Dict,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[set] = None,
    ) -> str:
        """Format a metrics dict into a compact, curated string.

        Only keys appearing in ``include`` (defaults to
        :data:`_DEFAULT_VISIBLE_METRICS`) are rendered, minus any key in
        ``exclude`` and minus the universally hidden bookkeeping fields.
        Floats are printed with 4 significant digits; ints with no
        decimals.
        """
        include = tuple(include) if include is not None else _DEFAULT_VISIBLE_METRICS
        skip = set(_DEFAULT_HIDDEN_METRICS)
        if exclude:
            skip |= set(exclude)

        parts: List[str] = []
        for k in include:
            if k in skip or k not in metrics:
                continue
            v = metrics[k]
            if isinstance(v, bool) or v is None:
                continue
            if isinstance(v, float):
                if not np.isfinite(v):
                    continue
                parts.append(f"{_METRIC_DISPLAY.get(k, k)}={v:.{_DEC}f}")
            elif isinstance(v, (int, np.integer)):

                parts.append(f"{_METRIC_DISPLAY.get(k, k)}={int(v)}")
        return ", ".join(parts)

    @staticmethod
    def _format_eval_row(row: Dict[str, Any]) -> str:
        """Format the OOS columns of a NodeEvalResult row into a suffix."""
        bits: List[str] = []
        n_oos = row.get("n_oos")
        if n_oos is not None and not pd.isna(n_oos):
            bits.append(f"n_oos={int(n_oos)}")
        if "oos_r2" in row and not pd.isna(row["oos_r2"]):
            bits.append(f"OOS R²={row['oos_r2']:+.{_DEC}f}")
        if "oos_rank_ic_mean" in row and not pd.isna(row["oos_rank_ic_mean"]):
            ir = row.get("oos_rank_ic_ir", float("nan"))
            if not pd.isna(ir):
                bits.append(
                    f"OOS IC={row['oos_rank_ic_mean']:+.{_DEC}f} "
                    f"(IR={ir:+.{_DEC}f})"
                )
            else:
                bits.append(f"OOS IC={row['oos_rank_ic_mean']:+.{_DEC}f}")
        return " | ".join(bits)

    @staticmethod
    def _format_child_diff_row(row: Dict[str, Any]) -> str:
        """Format the ΔR² / ΔIC summary line for an internal node."""
        bits: List[str] = []
        if "delta_oos_r2" in row and not pd.isna(row["delta_oos_r2"]):
            l = row.get("left_oos_r2", float("nan"))
            r = row.get("right_oos_r2", float("nan"))
            bits.append(
                f"ΔR²={row['delta_oos_r2']:+.{_DEC}f} "
                f"(L={l:+.{_DEC}f} vs R={r:+.{_DEC}f})"
            )
        if "delta_oos_rank_ic" in row and not pd.isna(row["delta_oos_rank_ic"]):
            l = row.get("left_oos_rank_ic_mean", float("nan"))
            r = row.get("right_oos_rank_ic_mean", float("nan"))
            bits.append(
                f"ΔIC={row['delta_oos_rank_ic']:+.{_DEC}f} "
                f"(L={l:+.{_DEC}f} vs R={r:+.{_DEC}f})"
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
        metric_keys: Optional[Sequence[str]] = None,
    ) -> None:
        """Recursively build tree-drawing lines."""
        metric_str = self._format_metrics(node.metrics, include=metric_keys)

        if node.is_leaf:
            label = f"[Leaf {node.node_id}] n={node.n_samples}"
            if metric_str:
                label = f"{label} | {metric_str}"
        else:
            label = (
                f"[Node {node.node_id}] "
                f"{node.split_feature} < {node.split_threshold:g} | "
                f"n={node.n_samples}, gain={node.split_score:.{_DEC}f}"
            )
            if metric_str:
                label = f"{label} | {metric_str}"

        if eval_lookup is not None and node.node_id in eval_lookup:
            suffix = self._format_eval_row(eval_lookup[node.node_id])
            if suffix:
                label = f"{label} | {suffix}"

        lines.append(f"{prefix}{connector}{label}")

        if not node.is_leaf:
            if connector == "├── ":
                child_prefix = prefix + "│   "
            elif connector == "└── ":
                child_prefix = prefix + "    "
            else:
                child_prefix = prefix

            if (
                show_child_diff
                and eval_lookup is not None
                and node.node_id in eval_lookup
            ):
                diff_str = self._format_child_diff_row(eval_lookup[node.node_id])
                if diff_str:
                    lines.append(f"{child_prefix}↳ split gain | {diff_str}")

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
                    metric_keys=metric_keys,
                )

    # ------------------------------------------------------------------
    # Graphviz export
    # ------------------------------------------------------------------

    def to_graphviz(
        self,
        evaluation: Optional["NodeEvalResult"] = None,
        show_child_diff: bool = False,
        leaf_fill: str = "#E8F1FB",
        node_fill: str = "#FFF4DC",
        edge_color: str = "#5B6B7B",
        font_name: str = "Helvetica",
        metric_keys: Optional[Sequence[str]] = None,
    ) -> str:
        """Return a Graphviz DOT representation of the fitted tree.

        Renderable with the ``graphviz`` Python package::

            from graphviz import Source
            Source(NodeReporter(engine).to_graphviz()).render('tree')

        Each node is laid out as a multi-row HTML-like table so the split
        rule, training metrics and (optionally) OOS metrics live in their
        own visual rows instead of being smashed into one cell.

        Parameters
        ----------
        evaluation : NodeEvalResult, optional
            OOS metrics from :meth:`PanelTreeEngine.evaluate` to overlay.
        show_child_diff : bool, default False
            Append a ``ΔR² / ΔIC`` row to every internal node.
        leaf_fill, node_fill, edge_color : str
            Colour overrides (any DOT-recognised colour spec).
        font_name : str
            Font for all text in the DOT output.
        metric_keys : sequence of str, optional
            Override which training-metric keys to render.

        Returns
        -------
        dot : str
            DOT source — no Graphviz binary is required to obtain it.
        """
        assert self.engine.root_ is not None, "Engine not fitted."

        eval_lookup: Optional[Dict[int, Dict[str, Any]]] = None
        if evaluation is not None:
            df = evaluation.per_node_df.set_index("node_id")
            eval_lookup = df.to_dict(orient="index")

        def _esc(s: str) -> str:
            return (
                s.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

        def _label_html(node) -> str:
            head_bg = leaf_fill if node.is_leaf else node_fill
            title = f"Leaf {node.node_id}" if node.is_leaf else f"Node {node.node_id}"
            rows = [
                f'<TR><TD BGCOLOR="{head_bg}"><B>{title}</B></TD></TR>'
            ]
            if not node.is_leaf:
                rule = (
                    f"{node.split_feature} &lt; {node.split_threshold:g}"
                )
                rows.append(f"<TR><TD>{rule}</TD></TR>")
                rows.append(
                    f"<TR><TD>n={node.n_samples}, "
                    f"gain={node.split_score:.{_DEC}f}</TD></TR>"
                )
            else:
                rows.append(f"<TR><TD>n={node.n_samples}</TD></TR>")

            metric_str = self._format_metrics(node.metrics, include=metric_keys)
            if metric_str:
                rows.append(f"<TR><TD>{_esc(metric_str)}</TD></TR>")

            if eval_lookup is not None and node.node_id in eval_lookup:
                row = eval_lookup[node.node_id]
                oos = self._format_eval_row(row)
                if oos:
                    rows.append(f"<TR><TD>{_esc(oos)}</TD></TR>")
                if show_child_diff and not node.is_leaf:
                    diff = self._format_child_diff_row(row)
                    if diff:
                        rows.append(f"<TR><TD>{_esc(diff)}</TD></TR>")

            return (
                '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2" '
                'CELLPADDING="2">'
                + "".join(rows)
                + "</TABLE>>"
            )

        lines: List[str] = [
            "digraph PanelTree {",
            f'  graph [rankdir=TB, fontname="{font_name}", '
            'bgcolor="white", pad=0.3, nodesep=0.35, ranksep=0.45];',
            f'  node  [shape=box, style="rounded,filled", '
            f'fontname="{font_name}", fontsize=10, '
            f'fillcolor="white", color="#A0A8B5"];',
            f'  edge  [fontname="{font_name}", fontsize=9, '
            f'color="{edge_color}", arrowsize=0.7];',
        ]

        def _walk(node) -> None:
            fill = leaf_fill if node.is_leaf else node_fill
            lines.append(
                f"  n{node.node_id} ["
                f"label={_label_html(node)}, "
                f'fillcolor="{fill}"];'
            )
            if node.left is not None:
                lines.append(
                    f"  n{node.node_id} -> n{node.left.node_id} "
                    f'[label="< {node.split_threshold:g}"];'
                )
                _walk(node.left)
            if node.right is not None:
                lines.append(
                    f"  n{node.node_id} -> n{node.right.node_id} "
                    f'[label="≥ {node.split_threshold:g}"];'
                )
                _walk(node.right)

        _walk(self.engine.root_)
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # matplotlib tree plot
    # ------------------------------------------------------------------

    def plot_tree(
        self,
        evaluation: Optional["NodeEvalResult"] = None,
        show_child_diff: bool = False,
        title: str = "PanelTree structure",
        leaf_color: str = "#3F8AC1",
        node_color: str = "#E0A85A",
        text_color: str = "#1F2A36",
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        metric_keys: Optional[Sequence[str]] = None,
    ):
        """Plot the tree as a pure-matplotlib diagram.

        This is a self-contained alternative to :meth:`to_graphviz` for
        users who don't have the Graphviz binary installed.  Each node is
        drawn as a rounded box with up to four short text rows (header,
        split rule, training metrics, OOS metrics).

        Parameters
        ----------
        evaluation : NodeEvalResult, optional
            OOS metrics to overlay (same conventions as :meth:`print_tree`).
        show_child_diff : bool, default False
            Append a ``ΔR² / ΔIC`` row under every internal node.
        title : str
            Figure title.  Defaults to a neutral English string.
        leaf_color, node_color : str
            Box fill colours for leaves and internal nodes.
        text_color : str
            Colour for the text inside boxes.
        figsize : tuple, optional
            Matplotlib figure size in inches; auto-derived when ``None``.
        save_path : str, optional
            If given, the figure is also written to this path (PNG/PDF/…).
        metric_keys : sequence of str, optional
            Override which training-metric keys to render.

        Returns
        -------
        fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        _apply_mpl_style()
        assert self.engine.root_ is not None, "Engine not fitted."

        eval_lookup: Optional[Dict[int, Dict[str, Any]]] = None
        if evaluation is not None:
            df = evaluation.per_node_df.set_index("node_id")
            eval_lookup = df.to_dict(orient="index")

        # Resolve the criterion's primary metric so the boxes emphasise
        # *that* metric (R², IC, precision, …) rather than always R².
        crit_key = _crit_metric_key(self.engine)
        crit_disp = _crit_metric_display(self.engine)

        # Map crit_key → (IS key in node.metrics, OOS col, Δ col).
        # The engine currently emits OOS columns only for r2 / rank_ic;
        # for other criteria we fall back to the IS metric alone.
        _METRIC_AXIS = {
            "r2": ("r2", "oos_r2", "delta_oos_r2"),
            "rank_ic": ("rank_ic_mean", "oos_rank_ic_mean",
                        "delta_oos_rank_ic"),
            "rank_ic_mean": ("rank_ic_mean", "oos_rank_ic_mean",
                             "delta_oos_rank_ic"),
            "precision": ("precision", None, None),
            "f1": ("f1", None, None),
            "auc": ("auc", None, None),
            "logloss": ("logloss", None, None),
        }
        is_key, oos_col, delta_col = _METRIC_AXIS.get(
            crit_key, ("r2", "oos_r2", "delta_oos_r2")
        )

        # ---- 1. Layout: compute (x, y) for every node ---------------
        #
        # ``row_h`` is the vertical distance (in data coords) between two
        # successive depth levels.  Boxes occupy at most ``box_h`` of that
        # distance, so the remaining ``row_h - box_h`` is *breathing room*
        # between parent and child – important once a tree gets a few
        # levels deep, otherwise the rendered nodes touch.
        row_h = 2.35



        positions: Dict[int, Tuple[float, float]] = {}
        depths: Dict[int, int] = {}
        leaf_counter = [0]

        def _layout(node, depth: int) -> float:
            depths[node.node_id] = depth
            if node.is_leaf or (node.left is None and node.right is None):
                x = float(leaf_counter[0])
                leaf_counter[0] += 1
                positions[node.node_id] = (x, -float(depth) * row_h)
                return x
            xs = []
            if node.left is not None:
                xs.append(_layout(node.left, depth + 1))
            if node.right is not None:
                xs.append(_layout(node.right, depth + 1))
            x = sum(xs) / len(xs)
            positions[node.node_id] = (x, -float(depth) * row_h)
            return x

        _layout(self.engine.root_, 0)
        max_depth = max(depths.values()) if depths else 0
        n_leaves = max(leaf_counter[0], 1)


        # ---- 2. Compose multi-row label text -------------------------
        # Each node box renders as
        #   header (bold)
        #   [secondary]   (split rule – internal only)
        #   primary 1     (n = …)
        #   primary 2     (IS criterion metric)
        #   primary 3     (OOS criterion metric  OR  Δcriterion)
        # Primary lines get a larger, bolder font so the eye sees the
        # *important* numbers first; secondary lines (split rule) stay
        # small and grey.
        def _box_payload(node):
            is_leaf = node.is_leaf
            header = (
                f"Leaf {node.node_id}" if is_leaf
                else f"Node {node.node_id}"
            )
            secondary: List[str] = []
            primary: List[str] = []

            if not is_leaf:
                secondary.append(
                    f"{node.split_feature} < {node.split_threshold:g}"
                )

            # n is always primary.
            primary.append(f"n = {node.n_samples}")

            # IS criterion metric.
            is_val = node.metrics.get(is_key)
            # ``rank_ic_mean`` sometimes lives under ``rank_ic``.
            if is_val is None and is_key == "rank_ic_mean":
                is_val = node.metrics.get("rank_ic")
            if (
                isinstance(is_val, (int, float))
                and np.isfinite(float(is_val))
            ):
                primary.append(
                    f"IS {crit_disp} = {float(is_val):+.{_DEC}f}"
                )

            # OOS column for leaves; Δ for internals (if available).
            if eval_lookup is not None and node.node_id in eval_lookup:
                row = eval_lookup[node.node_id]
                if is_leaf:
                    if oos_col is not None and oos_col in row:
                        v = row[oos_col]
                        if v is not None and not pd.isna(v):
                            primary.append(
                                f"OOS {crit_disp} = {float(v):+.{_DEC}f}"
                            )
                else:
                    # OOS metric of the node itself.
                    if oos_col is not None and oos_col in row:
                        v = row[oos_col]
                        if v is not None and not pd.isna(v):
                            primary.append(
                                f"OOS {crit_disp} = {float(v):+.{_DEC}f}"
                            )
                    # ΔR² / ΔIC across children — the key driver of the split.
                    if (
                        show_child_diff
                        and delta_col is not None
                        and delta_col in row
                    ):
                        d = row[delta_col]
                        if d is not None and not pd.isna(d):
                            primary.append(
                                f"Δ{crit_disp} = {float(d):+.{_DEC}f}"
                            )

            return header, secondary, primary

        node_payloads = {
            n: _box_payload(node) for n, node in self._iter_nodes()
        }

        # ---- 3. Figure size heuristics ------------------------------
        # Width grows with the leaf count; height grows with depth × row_h
        # so each successive level keeps the same vertical breathing room
        # regardless of how many leaves the tree has.  The 1.05× factor
        # on the per-row inch budget produces ~1.6 in per depth level –
        # comfortable for the 0.95-tall boxes plus ≥0.55 of edge space.
        if figsize is None:
            width = max(8.0, 1.7 * n_leaves)
            # Each depth level needs ~``row_h`` data-units → multiply by
            # an inches-per-data-unit factor; add a small margin top/bot.
            inches_per_row = 1.05
            height = max(
                4.5,
                inches_per_row * row_h * (max_depth + 1) + 1.2,
            )
            figsize = (width, height)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_axis_off()
        ax.set_xlim(-1.0, n_leaves)
        # y ranges from 0 (root) down to -max_depth * row_h (deepest level);
        # pad each end by ~0.7 data-units for box halves + the legend strip.
        ax.set_ylim(-(max_depth * row_h + 0.9), 0.9)

        # Box dimensions in *data coords* (one leaf = 1 x-unit).  Taller
        # box than the default so each primary metric row gets more
        # vertical room (i.e. more visible line spacing) — the inter-row
        # ``step`` below is ``(box_h - …) / n_primary``.
        box_w = 0.94
        box_h = 1.30




        # ---- 4. Draw edges first so boxes overlay them --------------
        for node_id, node in self._iter_nodes():
            if node.is_leaf:
                continue
            x_p, y_p = positions[node_id]
            for child in (node.left, node.right):
                if child is None:
                    continue
                x_c, y_c = positions[child.node_id]
                ax.plot(
                    [x_p, x_c],
                    [y_p - box_h / 2, y_c + box_h / 2],
                    color="#A0A8B5",
                    linewidth=1.0,
                    zorder=1,
                )
                # Edge label: "< τ" on left, "≥ τ" on right.
                is_left = child is node.left
                edge_lbl = (
                    f"< {node.split_threshold:g}" if is_left
                    else f"≥ {node.split_threshold:g}"
                )
                ax.text(
                    (x_p + x_c) / 2,
                    (y_p - box_h / 2 + y_c + box_h / 2) / 2,
                    edge_lbl,
                    fontsize=8,
                    color="#5B6B7B",
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        pad=1.0,
                        alpha=0.85,
                    ),
                    zorder=2,
                )

        # ---- 5. Draw boxes + text ----------------------------------
        # Vertical layout inside each box (in box-relative offsets from top):
        #   header       ~0.13          (bold, 11pt, in fill colour)
        #   secondary    ~0.30          (8pt, grey, italic) – split rule
        #   primary 0..k evenly spaced  (10.5pt, semibold, dark)
        # Internal nodes carry an extra "secondary" row (split rule); leaves
        # don't, so primary content gets more vertical room.
        for node_id, node in self._iter_nodes():
            x, y = positions[node_id]
            fill = leaf_color if node.is_leaf else node_color
            patch = FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2),
                box_w, box_h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.4,
                edgecolor=fill,
                facecolor=fill + "33",  # 20% alpha hex
                zorder=3,
            )
            ax.add_patch(patch)

            header, secondary, primary = node_payloads[node_id]
            top_y = y + box_h / 2

            # Header (bold, fill colour).
            ax.text(
                x, top_y - 0.10,
                header,
                fontsize=11,
                fontweight="bold",
                color=fill,
                ha="center",
                va="top",
                zorder=4,
            )

            # Secondary line (split rule).
            sec_consumed = 0.0
            if secondary:
                ax.text(
                    x, top_y - 0.28,
                    secondary[0],
                    fontsize=8,
                    style="italic",
                    color="#5B6B7B",
                    ha="center",
                    va="top",
                    zorder=4,
                )
                sec_consumed = 0.18

            # Primary rows: take whatever vertical room remains.
            primary_top = top_y - 0.32 - sec_consumed
            primary_bottom = y - box_h / 2 + 0.08
            avail = primary_top - primary_bottom
            n_primary = max(len(primary), 1)
            step = avail / n_primary
            # Internal nodes carry one extra primary row (Δcriterion), so
            # their text gets a slightly smaller font to avoid crowding the
            # box vertically.  Leaves keep the larger size.
            primary_fontsize = 9.5 if not node.is_leaf else 10.5
            for i, txt in enumerate(primary):
                ax.text(
                    x,
                    primary_top - step * (i + 0.5),
                    txt,
                    fontsize=primary_fontsize,
                    fontweight="semibold",
                    color=text_color,
                    ha="center",
                    va="center",
                    zorder=4,
                )



        # ---- 6. Title + legend --------------------------------------
        ax.set_title(title)

        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=node_color + "33", edgecolor=node_color,
                  label="Internal node"),
            Patch(facecolor=leaf_color + "33", edgecolor=leaf_color,
                  label="Leaf"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            frameon=False,
            fontsize=9,
        )

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Tree plot saved to %s", save_path)

        return fig, ax

    # ------------------------------------------------------------------
    # Internal: iterate (node_id, node) over the whole tree
    # ------------------------------------------------------------------

    def _iter_nodes(self):
        """Yield ``(node_id, node)`` for every node in the tree."""
        def _walk(node):
            yield node.node_id, node
            if node.left is not None:
                yield from _walk(node.left)
            if node.right is not None:
                yield from _walk(node.right)

        if self.engine.root_ is not None:
            yield from _walk(self.engine.root_)


# ======================================================================
# Mosaic Visualizer
# ======================================================================

class MosaicVisualizer:
    """Generate prediction-mosaic heatmaps.

    The mosaic has:

    * **x-axis**: time periods
    * **y-axis**: leaf nodes (sorted by Node_ID)
    * **colour**: one metric per ``(leaf, period)`` cell

    Parameters
    ----------
    engine : PanelTreeEngine
        Fitted engine.
    """

    # All metrics this visualizer knows about.
    _SUPPORTED_METRICS = (
        "r2", "mean", "median", "std", "ic",
        "precision", "f1", "auc",
    )

    def __init__(self, engine: "PanelTreeEngine"):
        self.engine = engine

    # ------------------------------------------------------------------
    # Mosaic construction
    # ------------------------------------------------------------------

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
            One of:

            * ``"r2"`` – out-of-sample R² of the leaf's predictor.
            * ``"mean"`` – cross-sectional mean of ``y`` within the cell.
            * ``"median"``, ``"std"`` – analogous summary statistics.
            * ``"ic"`` – cross-sectional Pearson correlation between
              prediction and ``y`` within the cell (rank-IC proxy).
            * ``"precision"`` / ``"f1"`` / ``"auc"`` – classification
              metrics computed against the leaf's classifier.

        Returns
        -------
        mosaic : DataFrame
            Index = leaf node IDs (sorted), columns = time periods,
            values = metric.  Empty cells become NaN.
        """
        if metric not in self._SUPPORTED_METRICS:
            raise ValueError(
                f"Unknown metric={metric!r}.  Supported: "
                f"{', '.join(self._SUPPORTED_METRICS)}."
            )

        engine = self.engine
        assert engine.root_ is not None, "Engine not fitted."

        feature_names = engine._feature_names
        X_arr = X[feature_names].values.astype(np.float64)
        y_arr = y.values.astype(np.float64)
        times = X[time_col].values

        # Assign each row to a leaf.
        leaf_ids = np.full(len(X_arr), -1, dtype=int)
        self._assign_leaves(
            engine.root_, X_arr, np.arange(len(X_arr)),
            leaf_ids, feature_names,
        )

        unique_times = np.sort(np.unique(times))
        leaves = sorted(engine.get_leaves(), key=lambda n: n.node_id)
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
                    ss_res = float(np.sum((y_sub - y_pred) ** 2))
                    ss_tot = float(np.sum((y_sub - y_sub.mean()) ** 2))
                    val = (
                        1.0 - ss_res / max(ss_tot, 1e-12)
                        if ss_tot > 0 else 0.0
                    )
                elif metric == "mean":
                    val = float(np.mean(y_sub))
                elif metric == "median":
                    val = float(np.median(y_sub))
                elif metric == "std":
                    val = float(np.std(y_sub))
                elif metric == "ic":
                    if n < 2:
                        val = np.nan
                    else:
                        y_pred = leaf.predictor.predict(X_sub)
                        sd_p = float(np.std(y_pred))
                        sd_y = float(np.std(y_sub))
                        if sd_p == 0.0 or sd_y == 0.0:
                            val = np.nan
                        else:
                            val = float(np.corrcoef(y_pred, y_sub)[0, 1])
                elif metric in ("precision", "f1", "auc"):
                    if hasattr(leaf.predictor, "predict_proba"):
                        proba = leaf.predictor.predict_proba(X_sub)
                    else:
                        proba = leaf.predictor.predict(X_sub)
                    from ptree.criteria import evaluate_classification
                    cls_metrics = evaluate_classification(y_sub, proba)
                    val = cls_metrics.get(metric, np.nan)
                else:  # pragma: no cover - guarded above
                    val = np.nan

                mosaic_data[leaf.node_id][t] = val

        mosaic = pd.DataFrame(mosaic_data).T
        mosaic.index.name = "Leaf_ID"
        mosaic.columns.name = time_col
        return mosaic

    # ------------------------------------------------------------------
    # Mosaic plot
    # ------------------------------------------------------------------

    # Recommended labels per metric (used when caller doesn't override).
    _METRIC_LABELS = {
        "r2": "R²",
        "mean": "Mean return",
        "median": "Median return",
        "std": "Cross-section std",
        "ic": "Information coefficient",
        "precision": "Precision",
        "f1": "F1",
        "auc": "AUC",
    }

    @classmethod
    def _auto_cmap_and_center(
        cls,
        values: np.ndarray,
        metric: Optional[str],
        cmap: Optional[str],
        center: Optional[float],
    ) -> Tuple[str, Optional[float]]:
        """Pick a sensible cmap / centering when caller leaves them blank.

        * If the data is strictly non-negative (e.g. raw R² ≥ 0,
          precision, AUC), default to a perceptually uniform sequential
          map (``viridis``) and *no* centering.
        * Otherwise default to ``RdBu_r`` centred at zero – matches the
          convention used for IC / mean-return mosaics.
        """
        if cmap is not None and center is not _UNSET:
            return cmap, center

        finite = values[np.isfinite(values)]
        if finite.size == 0:
            chosen_cmap, chosen_center = "RdBu_r", 0.0
        elif np.all(finite >= 0):
            chosen_cmap, chosen_center = "viridis", None
        else:
            chosen_cmap, chosen_center = "RdBu_r", 0.0

        if cmap is not None:
            chosen_cmap = cmap
        if center is not _UNSET:
            chosen_center = center
        return chosen_cmap, chosen_center

    def plot_mosaic(
        self,
        mosaic: pd.DataFrame,
        title: Optional[str] = None,
        metric: Optional[str] = None,
        cmap: Optional[str] = None,
        center: Any = None,  # sentinel handled below
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        max_xticks: int = 12,
        annotate: bool = False,
        cbar_label: Optional[str] = None,
    ):
        """Plot the mosaic as a heatmap.

        Requires ``matplotlib`` and ``seaborn``.

        Parameters
        ----------
        mosaic : DataFrame
            Output of :meth:`build_mosaic`.
        title : str, optional
            Figure title.  Defaults to a neutral English string derived
            from *metric* (``"Prediction mosaic – R²"`` etc.).  The
            library never embeds parameter values into titles; callers
            wanting to do so should pass an explicit string.
        metric : str, optional
            Metric name (e.g. ``"r2"``, ``"ic"``).  Used to pick a default
            title, colorbar label and colour map when those aren't given.
        cmap : str, optional
            Matplotlib / seaborn colormap.  Auto-selected when ``None``:
            sequential ``viridis`` for non-negative data, diverging
            ``RdBu_r`` otherwise.
        center : float or None, default ``None``
            Value at which a diverging colormap centres.  Pass the
            sentinel ``_UNSET`` (or use the default keyword) to let the
            method decide based on the data range.  Pass an explicit
            ``None`` to disable centering.
        figsize : tuple, optional
            Matplotlib figure size in inches; auto-derived when missing.
        save_path : str, optional
            If given, the figure is also written to this path.
        max_xticks : int, default 12
            When ``mosaic`` has many time columns, only ~``max_xticks``
            evenly spaced labels are shown to keep the axis readable.
        annotate : bool, default False
            Annotate each cell with its numeric value.  Only useful for
            small mosaics.
        cbar_label : str, optional
            Colour-bar label.  Defaults to the metric's human name.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        _apply_mpl_style()

        # Auto cmap / center selection (uses the user-supplied values
        # when they are not the sentinel "leave-me-alone" marker).
        values = mosaic.values.astype(float)
        # ``center`` defaults to None at the signature level so we can
        # tell apart "user wants no centering" from "user did not ask".
        # The convention: if user explicitly passes ``center=None`` we
        # honour it.  Otherwise we treat any other value as explicit.
        # To avoid breaking the previous keyword default we pick
        # _UNSET via the sentinel below.
        if center is None and cmap is None:
            cmap_used, center_used = self._auto_cmap_and_center(
                values, metric, cmap=None, center=_UNSET,
            )
        else:
            cmap_used = cmap if cmap is not None else "RdBu_r"
            center_used = center  # could be None (no centering) or 0.0
            if cmap is None:
                # User picked center but not cmap → still auto-pick cmap.
                cmap_used, _ = self._auto_cmap_and_center(
                    values, metric, cmap=None, center=center,
                )

        # Resolve display strings.
        metric_label = self._METRIC_LABELS.get(metric or "", metric or "Value")
        if title is None:
            if metric is not None:
                title = f"Prediction mosaic — {metric_label}"
            else:
                title = "Prediction mosaic"
        if cbar_label is None:
            cbar_label = metric_label

        # Figure size: scale with #periods so things don't squish.
        n_rows, n_cols = mosaic.shape
        if figsize is None:
            width = max(8.0, min(0.18 * n_cols + 4.0, 22.0))
            height = max(3.0, min(0.45 * n_rows + 2.0, 12.0))
            figsize = (width, height)

        fig, ax = plt.subplots(figsize=figsize)

        # Sparse tick labels along the x-axis.
        if n_cols > max_xticks:
            step = max(1, n_cols // max_xticks)
            xticklabels = [
                str(c) if i % step == 0 else "" for i, c in enumerate(mosaic.columns)
            ]
        else:
            xticklabels = [str(c) for c in mosaic.columns]

        sns.heatmap(
            mosaic.astype(float),
            cmap=cmap_used,
            center=center_used,
            ax=ax,
            linewidths=0.4,
            linecolor="white",
            xticklabels=xticklabels,
            yticklabels=[f"Leaf {lid}" for lid in mosaic.index],
            cbar_kws={"label": cbar_label, "shrink": 0.85, "pad": 0.02},
            annot=annotate,
            fmt=f".{_DEC}f" if annotate else "",
        )

        ax.set_title(title)
        ax.set_xlabel(self._axis_label(mosaic.columns.name) or "Period")
        ax.set_ylabel("Leaf node")
        plt.setp(
            ax.get_xticklabels(),
            rotation=45, ha="right", rotation_mode="anchor",
        )
        plt.setp(ax.get_yticklabels(), rotation=0)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Mosaic saved to %s", save_path)

        return fig, ax

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _axis_label(name: Optional[str]) -> Optional[str]:
        """Map a (possibly None) column-name into a readable axis label."""
        if not name:
            return None
        # Common cases: keep them lower-case as supplied; capitalise once.
        return str(name).replace("_", " ").strip().capitalize()

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


# ======================================================================
# Sentinel for "user did not pass anything" (kept private)
# ======================================================================

class _Unset:
    """Sentinel used to distinguish *no value supplied* from ``None``."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover
        return "<UNSET>"


_UNSET = _Unset()
