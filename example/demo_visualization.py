"""
P-Tree Visualization 完整示例（v0.2 新功能）
=============================================

本示例聚焦 aptree v0.2 引入的可视化能力，所有产物统一写到
``example/outputs/`` 目录下，运行后可以直接查看。

包含 4 个可视化产物：

1. **OOS 评估覆盖的文本树**：
   ``NodeReporter.print_tree(evaluation=..., show_child_diff=True)``
   把每个节点的样本外 R² / Rank-IC / Rank-IC IR 嵌入打印结果，
   并显示父节点 vs 两个子节点的相对增益。

2. **Graphviz DOT 导出**：
   ``NodeReporter.to_graphviz(evaluation=..., show_child_diff=True,
   leaf_fill=..., node_fill=...)`` 返回纯 DOT 字符串（不依赖
   graphviz 二进制）。如果系统装了 Graphviz 与 ``python-graphviz``
   包，会自动渲染为 SVG。

3. **OOS 评估的 ccp_alpha 调参曲线**：
   ``PanelTreeEngine.tune_ccp_alpha`` 在剪枝路径上扫描，
   挑选验证集 Rank-IC 最高的剪枝强度。

4. **预测马赛克热力图**：
   ``MosaicVisualizer.plot_mosaic`` v0.2 默认改为发散色板
   ``RdBu_r`` 并以 ``center=0.0`` 对齐零线，分别按 R² 与按 leaf 均值
   做两个视角；同时演示自定义 ``cmap`` / ``center`` 的覆盖能力。
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

# OOS evaluation: pass the date column so Rank-IC can be aggregated cross-sectionally
# ``metrics=("r2","rank_ic")`` 同时产出 oos_r2 / oos_rank_ic_mean / oos_rank_ic_ir
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
print("3. NodeReporter.print_tree(evaluation=..., show_child_diff=True)")
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
# 4. Graphviz DOT export + optional SVG render
# ---------------------------------------------------------------------------
print("=" * 64)
print("4. NodeReporter.to_graphviz(evaluation=..., show_child_diff=True)")
print("=" * 64)

dot_source = reporter.to_graphviz(
    evaluation=evaluation,
    show_child_diff=True,
    leaf_fill="#fff7e6",
    node_fill="#eef5ff",
)
dot_path = os.path.join(OUT_DIR, "tree.dot")
with open(dot_path, "w", encoding="utf-8") as f:
    f.write(dot_source)
print(f"  DOT 源码已写入: {dot_path}")

# Render to SVG only when both python-graphviz and the dot binary are present.
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
# 5. ccp_alpha pruning sweep on the validation set
# ---------------------------------------------------------------------------
print("=" * 64)
print("5. PanelTreeEngine.tune_ccp_alpha — 剪枝路径上的 Rank-IC 扫描")
print("=" * 64)

# Re-fit a deeper tree so the pruning path has something to sweep
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

# Plot the sweep if matplotlib is installed
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(sweep["ccp_alpha"], sweep["oos_rank_ic"], marker="o",
             color="#1f77b4", label="验证集 Rank-IC")
    ax1.axvline(best_alpha, color="#1f77b4", linestyle=":",
                label=f"best α={best_alpha:.4f}")
    ax1.set_xlabel("ccp_alpha")
    ax1.set_ylabel("Rank-IC", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(sweep["ccp_alpha"], sweep["n_leaves"], marker="s",
             color="#d62728", label="叶子数")
    ax2.set_ylabel("叶子数", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_title("ccp_alpha sweep（Rank-IC vs 叶子数）")
    ax1.legend(loc="lower left")
    fig.tight_layout()
    sweep_png = os.path.join(OUT_DIR, "ccp_alpha_sweep.png")
    fig.savefig(sweep_png, dpi=150)
    plt.close(fig)
    print(f"  曲线图已保存: {sweep_png}")
except ImportError:
    print("  [提示] 安装 matplotlib 可生成扫描曲线图")
print()


# ---------------------------------------------------------------------------
# 6. Mosaic heatmaps (default RdBu_r + center=0.0; and a custom variant)
# ---------------------------------------------------------------------------
print("=" * 64)
print("6. MosaicVisualizer.plot_mosaic — 默认 RdBu_r + center=0.0")
print("=" * 64)

viz = MosaicVisualizer(engine)
mosaic_r2 = viz.build_mosaic(X_val, y_val, time_col="date", metric="r2")
mosaic_mean = viz.build_mosaic(X_val, y_val, time_col="date", metric="mean")
print(f"  R² 马赛克形状:   {mosaic_r2.shape}  (叶子 × 时间)")
print(f"  Mean 马赛克形状: {mosaic_mean.shape}")

try:
    import matplotlib

    matplotlib.use("Agg")

    # 6a: leaf-mean mosaic — divergent palette centered at 0 is meaningful
    mean_path = os.path.join(OUT_DIR, "mosaic_leaf_mean.png")
    viz.plot_mosaic(
        mosaic_mean,
        title="叶子均值马赛克（v0.2 默认 RdBu_r，center=0.0）",
        save_path=mean_path,
    )
    print(f"  叶子均值马赛克: {mean_path}")

    # 6b: R² mosaic — override center because R² is non-negative
    r2_path = os.path.join(OUT_DIR, "mosaic_r2.png")
    viz.plot_mosaic(
        mosaic_r2,
        title="叶子 R² 马赛克（cmap='viridis', center=None）",
        save_path=r2_path,
        cmap="viridis",
        center=None,
    )
    print(f"  叶子 R² 马赛克: {r2_path}")
except ImportError:
    print("  [提示] 安装 matplotlib + seaborn 可生成马赛克热力图")
print()


# ---------------------------------------------------------------------------
# 7. RankIC criterion side-by-side (sanity check)
# ---------------------------------------------------------------------------
print("=" * 64)
print("7. RankICDiffCriterion：用同一份数据训练一棵 IC 驱动的树并打印 OOS")
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
