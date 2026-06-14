"""
aptree visualization demo (v0.2)
================================

聚焦 aptree v0.2 引入的可视化能力。所有产物统一写入 ``example/outputs/``。

包含 5 个可视化产物：

1. **文本树（含 OOS）**：``NodeReporter.print_tree(evaluation=...,
   show_child_diff=True)`` 在每个节点尾部追加样本外 R² / IC / IR，并显示
   父节点对子节点的相对增益。指标按白名单输出，不再被 ``mse=…,
   n_features=0`` 等噪音字段污染。

2. **matplotlib 树形图**：``NodeReporter.plot_tree`` 纯 matplotlib 实现，
   不依赖 graphviz 二进制，PNG 直接可看。

3. **Graphviz DOT 导出**：``NodeReporter.to_graphviz`` 现在用 HTML-like
   ``<TABLE>`` 标签结构化（节点名 / 分裂规则 / 训练 / OOS 分行排版）。

4. **ccp_alpha 调参曲线**：``PanelTreeEngine.tune_ccp_alpha`` 在剪枝路径
   上搜索验证集 Rank-IC 最高的剪枝强度。

5. **预测马赛克热力图**：``MosaicVisualizer`` 现在内置自动配色（非负 →
   ``viridis``，含负 → ``RdBu_r`` 且自动居中），不再把参数值写进标题；
   新增 ``metric="mean"`` / ``"ic"`` 等真正可用的指标。
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

# 让示例同时在源码仓 & ``pip install aptree`` 后均可运行
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from ptree import (  # noqa: E402  (path tweak above)
    DataHandler,
    MosaicVisualizer,
    NodeReporter,
    PanelTreeEngine,
    R2DiffCriterion,
    RankICDiffCriterion,
    RidgeRegressor,
    WeightedR2DiffCriterion,
)

logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")

OUT_DIR = os.path.join(_HERE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Synthetic panel: predictability concentrated in a small region of (x1, x2)
# ---------------------------------------------------------------------------
print("=" * 64)
print("1. 生成合成面板（信号集中在 char_1 高位 + char_2 高位的区域）")
print("=" * 64)

rng = np.random.default_rng(20260614)
T, N, P = 80, 150, 5
feature_names = [f"char_{i + 1}" for i in range(P)]

dates = np.repeat(np.arange(T), N)
asset_ids = np.tile(np.arange(N), T)
X_raw = rng.standard_normal((T * N, P))

alpha_strong = 0.55 * X_raw[:, 2] - 0.40 * X_raw[:, 3] + 0.20 * X_raw[:, 4]
alpha_weak = 0.02 * X_raw[:, 2]
in_regime = (X_raw[:, 0] > 0.3) & (X_raw[:, 1] > 0.0)
alpha = np.where(in_regime, alpha_strong, alpha_weak)
y_raw = alpha + rng.standard_normal(T * N) * 0.7

df = pd.DataFrame(X_raw, columns=feature_names)
df["date"] = dates
df["asset_id"] = asset_ids
y_series = pd.Series(y_raw, name="ret")

dh = DataHandler(cs_rank_standardize=True, fillna_method="ffill")
X_proc, y_proc, _ = dh.fit_transform(
    df, y_series, time_col="date", entity_col="asset_id"
)
X_proc["date"] = df["date"].values

split_t = int(T * 0.7)
train_mask = X_proc["date"].values < split_t
X_train = X_proc[train_mask].reset_index(drop=True)
y_train = y_proc[train_mask].reset_index(drop=True)
X_val = X_proc[~train_mask].reset_index(drop=True)
y_val = y_proc[~train_mask].reset_index(drop=True)
print(f"  训练样本: {len(X_train)},  验证样本: {len(X_val)}")
print(f"  产物输出目录: {OUT_DIR}")
print()


# ---------------------------------------------------------------------------
# 2. Fit a tree and evaluate every node OOS
# ---------------------------------------------------------------------------
print("=" * 64)
print("2. 训练一棵 P-Tree，并在验证集上做逐节点 OOS 评估")
print("=" * 64)

engine = PanelTreeEngine(
    predictor=RidgeRegressor(alpha=1.0),
    criterion=WeightedR2DiffCriterion(min_child_weight=0.05),
    split_thresholds=[0.3, 0.5, 0.7],
    max_depth=3,
    min_samples=200,
    verbose=0,
)
engine.fit(X_train, y_train, feature_names=dh.feature_names)

evaluation = engine.evaluate(
    X_val, y_val,
    time_col=X_val["date"].values,
    metrics=("r2", "rank_ic"),
)
print(f"  评估到 {len(evaluation.per_node_df)} 个节点（含中间节点 + 叶子）")

eval_csv = os.path.join(OUT_DIR, "node_eval.csv")
evaluation.per_node_df.to_csv(eval_csv, index=False)
print(f"  逐节点 OOS 指标已写入: {eval_csv}")
print()
print("  per_node_df 预览:")
print(evaluation.per_node_df.head(10).to_string(index=False))
print()


# ---------------------------------------------------------------------------
# 3. Text tree with OOS overlay (and child-vs-parent diff)
# ---------------------------------------------------------------------------
print("=" * 64)
print("3. 文本树 + OOS 覆盖：NodeReporter.print_tree(...)")
print("=" * 64)

reporter = NodeReporter(engine)
text_tree = reporter.print_tree(evaluation=evaluation, show_child_diff=True)
print(text_tree)

text_path = os.path.join(OUT_DIR, "tree_overlay.txt")
with open(text_path, "w", encoding="utf-8") as f:
    f.write(text_tree)
print(f"  已写入: {text_path}")
print()


# ---------------------------------------------------------------------------
# 4. matplotlib tree plot (no Graphviz binary needed)
# ---------------------------------------------------------------------------
print("=" * 64)
print("4. 纯 matplotlib 树形图：NodeReporter.plot_tree(...)")
print("=" * 64)

try:
    import matplotlib
    matplotlib.use("Agg")

    tree_png = os.path.join(OUT_DIR, "tree.png")
    reporter.plot_tree(
        evaluation=evaluation,
        show_child_diff=True,
        save_path=tree_png,
    )
    print(f"  树形图已保存: {tree_png}")
except ImportError:
    print("  [提示] 安装 matplotlib 可生成树形图")
print()


# ---------------------------------------------------------------------------
# 5. Graphviz DOT export + optional SVG render
# ---------------------------------------------------------------------------
print("=" * 64)
print("5. Graphviz DOT 导出：NodeReporter.to_graphviz(...)")
print("=" * 64)

dot_source = reporter.to_graphviz(
    evaluation=evaluation,
    show_child_diff=True,
)
dot_path = os.path.join(OUT_DIR, "tree.dot")
with open(dot_path, "w", encoding="utf-8") as f:
    f.write(dot_source)
print(f"  DOT 源码已写入: {dot_path}")

try:
    import graphviz as _graphviz  # type: ignore

    src = _graphviz.Source(dot_source)
    svg_path_no_ext = os.path.join(OUT_DIR, "tree")
    rendered = src.render(svg_path_no_ext, format="svg", cleanup=True)
    print(f"  已自动渲染 SVG: {rendered}")
except Exception as exc:  # noqa: BLE001
    print(
        "  未渲染 SVG（可选）。如需，可执行:\n"
        f"      dot -Tsvg {dot_path} -o {OUT_DIR}/tree.svg\n"
        f"  原因: {type(exc).__name__}: {exc}"
    )
print()


# ---------------------------------------------------------------------------
# 6. ccp_alpha pruning sweep on the validation set
# ---------------------------------------------------------------------------
print("=" * 64)
print("6. ccp_alpha 剪枝路径上的 Rank-IC 扫描")
print("=" * 64)

deep_engine = PanelTreeEngine(
    predictor=RidgeRegressor(alpha=1.0),
    criterion=R2DiffCriterion(),
    split_thresholds=[0.3, 0.5, 0.7],
    max_depth=4,
    min_samples=100,
    verbose=0,
)
deep_engine.fit(X_train, y_train, feature_names=dh.feature_names)

best_alpha, sweep = deep_engine.tune_ccp_alpha(
    X_val, y_val,
    time_col=X_val["date"].values,
    metric="rank_ic",
)
print(sweep.to_string(index=False))
print(f"\n  最佳 ccp_alpha (按验证集 Rank-IC): {best_alpha:.6f}")

sweep_csv = os.path.join(OUT_DIR, "ccp_alpha_sweep.csv")
sweep.to_csv(sweep_csv, index=False)
print(f"  曲线已写入: {sweep_csv}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # Brand-ish blue / red, suitable for B&W print as well.
    ic_color = "#1f4e79"
    leaf_color = "#c0504d"

    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    ax1.plot(
        sweep["ccp_alpha"], sweep["oos_rank_ic"],
        marker="o", linewidth=1.8,
        color=ic_color, label="Validation Rank-IC",
    )
    ax1.axvline(
        best_alpha, color=ic_color, linestyle=":", linewidth=1.2,
        label=f"best α = {best_alpha:.4f}",
    )
    ax1.set_xlabel("Cost-complexity parameter α")
    ax1.set_ylabel("Validation Rank-IC", color=ic_color)
    ax1.tick_params(axis="y", labelcolor=ic_color)
    ax1.grid(True, linewidth=0.4, alpha=0.4)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(
        sweep["ccp_alpha"], sweep["n_leaves"],
        marker="s", linewidth=1.4, linestyle="--",
        color=leaf_color, label="Number of leaves",
    )
    ax2.set_ylabel("Number of leaves", color=leaf_color)
    ax2.tick_params(axis="y", labelcolor=leaf_color)

    ax1.set_title("Cost-complexity pruning path")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2,
        loc="best", frameon=False, fontsize=9,
    )

    fig.tight_layout()
    sweep_png = os.path.join(OUT_DIR, "ccp_alpha_sweep.png")
    fig.savefig(sweep_png, dpi=150)
    plt.close(fig)
    print(f"  曲线图已保存: {sweep_png}")
except ImportError:
    print("  [提示] 安装 matplotlib 可生成扫描曲线图")
print()


# ---------------------------------------------------------------------------
# 7. Mosaic heatmaps – defaults are now clean, metric-aware
# ---------------------------------------------------------------------------
print("=" * 64)
print("7. 预测马赛克：MosaicVisualizer.plot_mosaic(...)")
print("=" * 64)

viz = MosaicVisualizer(engine)
mosaic_r2 = viz.build_mosaic(X_val, y_val, time_col="date", metric="r2")
mosaic_mean = viz.build_mosaic(X_val, y_val, time_col="date", metric="mean")
mosaic_ic = viz.build_mosaic(X_val, y_val, time_col="date", metric="ic")
print(f"  R² 马赛克形状:   {mosaic_r2.shape}  (叶子 × 时间)")
print(f"  Mean 马赛克形状: {mosaic_mean.shape}")
print(f"  IC 马赛克形状:   {mosaic_ic.shape}")

try:
    import matplotlib
    matplotlib.use("Agg")

    mean_path = os.path.join(OUT_DIR, "mosaic_leaf_mean.png")
    viz.plot_mosaic(
        mosaic_mean,
        metric="mean",
        save_path=mean_path,
    )
    print(f"  叶子均值马赛克: {mean_path}")

    r2_path = os.path.join(OUT_DIR, "mosaic_r2.png")
    viz.plot_mosaic(
        mosaic_r2,
        metric="r2",
        save_path=r2_path,
    )
    print(f"  叶子 R² 马赛克: {r2_path}")

    ic_path = os.path.join(OUT_DIR, "mosaic_ic.png")
    viz.plot_mosaic(
        mosaic_ic,
        metric="ic",
        save_path=ic_path,
    )
    print(f"  叶子 IC 马赛克: {ic_path}")
except ImportError:
    print("  [提示] 安装 matplotlib + seaborn 可生成马赛克热力图")
print()


# ---------------------------------------------------------------------------
# 8. RankIC criterion side-by-side (sanity check)
# ---------------------------------------------------------------------------
print("=" * 64)
print("8. RankICDiffCriterion：用同一份数据训练一棵 IC 驱动的树并打印 OOS")
print("=" * 64)

ic_engine = PanelTreeEngine(
    predictor=RidgeRegressor(alpha=1.0),
    criterion=RankICDiffCriterion(min_child_weight=0.05),
    split_thresholds=[0.3, 0.5, 0.7],
    max_depth=3,
    min_samples=200,
    verbose=0,
)
ic_engine.fit(X_train, y_train, feature_names=dh.feature_names)
ic_eval = ic_engine.evaluate(
    X_val, y_val,
    time_col=X_val["date"].values,
    metrics=("r2", "rank_ic"),
)
ic_text = NodeReporter(ic_engine).print_tree(
    evaluation=ic_eval, show_child_diff=True
)
print(ic_text)

ic_path = os.path.join(OUT_DIR, "tree_rank_ic_overlay.txt")
with open(ic_path, "w", encoding="utf-8") as f:
    f.write(ic_text)
print(f"  已写入: {ic_path}")
print()


print("=" * 64)
print(f"示例运行完毕！所有产物位于: {OUT_DIR}")
print("=" * 64)
