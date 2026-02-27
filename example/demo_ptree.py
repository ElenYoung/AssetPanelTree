"""
Panel Tree (P-Tree) 完整使用示例
================================

本示例使用合成面板数据演示 P-Tree 的完整工作流程：
1. 数据生成与预处理（DataHandler）
2. 回归任务：Ridge / VolWeightedRidge + R²Diff 准则
3. 分类任务：RidgeLogit + ClassificationCriterion
4. 快速模式（Feature Priority Caching + Early Stopping）
5. 节点报告与预测马赛克可视化
6. 叶子节点样本提取
"""

import sys, os, logging
import numpy as np
import pandas as pd

# 将 src/ 加入搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ptree import (
    DataHandler,
    RidgeRegressor,
    VolWeightedRidgeRegressor,
    RidgeLogitClassifier,
    R2DiffCriterion,
    ClassificationCriterion,
    PanelTreeEngine,
    NodeReporter,
    MosaicVisualizer,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# =====================================================================
# 1. 生成合成面板数据
# =====================================================================
print("=" * 60)
print("1. 生成合成面板数据")
print("=" * 60)

np.random.seed(2026)

T = 60          # 时间期数
N = 200         # 每期资产数
P = 5           # 特征数量
feature_names = [f"char_{i+1}" for i in range(P)]

dates = np.repeat(np.arange(T), N)
asset_ids = np.tile(np.arange(N), T)

# 特征矩阵
X_raw = np.random.randn(T * N, P)

# 构造非线性收益率：
#   - char_1 > 0 时 alpha 显著 (char_2 系数大)
#   - char_1 <= 0 时 alpha 弱 (噪声主导)
# 这模拟了"预测能力在特定区间集中"的面板树场景
alpha_strong = 0.6 * X_raw[:, 1] - 0.4 * X_raw[:, 2] + 0.3 * X_raw[:, 3]
alpha_weak = 0.05 * X_raw[:, 1] + 0.02 * X_raw[:, 4]
mask_strong = X_raw[:, 0] > 0

y_raw = np.where(mask_strong, alpha_strong, alpha_weak)
# 加入异方差噪声（波动率与 char_3 相关）
vol = 0.3 + 0.5 * np.abs(X_raw[:, 2])
y_raw += np.random.randn(T * N) * vol

# 组装 DataFrame
df = pd.DataFrame(X_raw, columns=feature_names)
df["date"] = dates
df["asset_id"] = asset_ids
y_series = pd.Series(y_raw, name="ret")
ret_series = y_series.copy()  # 用于波动率计算

print(f"  样本总量: {len(df)}")
print(f"  时间期数: {T},  每期资产: {N},  特征数: {P}")
print(f"  特征列: {feature_names}")
print()

# =====================================================================
# 2. 数据预处理
# =====================================================================
print("=" * 60)
print("2. 数据预处理 (DataHandler)")
print("=" * 60)

dh = DataHandler(
    cs_rank_standardize=True,  # 截面秩标准化 → 特征映射至 [0,1]
    vol_window=20,
    fillna_method="ffill",
)
X_proc, y_proc, vol_weights = dh.fit_transform(
    df, y_series,
    time_col="date",
    entity_col="asset_id",
    ret_series_for_vol=ret_series,
)
print(f"  处理后 X 形状: {X_proc.shape}")
print(f"  特征名称: {dh.feature_names}")
print(f"  波动率权重是否可用: {vol_weights is not None}")
print()

# =====================================================================
# 3. 回归任务 — 普通岭回归 + R² 差异准则
# =====================================================================
print("=" * 60)
print("3. 回归任务: RidgeRegressor + R2DiffCriterion")
print("=" * 60)

engine_reg = PanelTreeEngine(
    predictor=RidgeRegressor(alpha=1.0),
    criterion=R2DiffCriterion(),
    split_thresholds=[0.3, 0.5, 0.7],
    max_depth=3,
    min_samples=200,
    verbose=1,
)
engine_reg.fit(X_proc, y_proc, feature_names=dh.feature_names)

reporter_reg = NodeReporter(engine_reg)
print("\n--- 树结构 ---")
print(reporter_reg.print_tree())

print("\n--- 叶子节点报告 ---")
leaf_df = reporter_reg.leaf_summary()
print(leaf_df[["Node_ID", "Rule", "Predictability_Score", "N_Samples", "Sample_Ratio"]].to_string(index=False))

# 预测
preds_reg = engine_reg.predict(X_proc)
ss_res = np.nansum((y_proc.values - preds_reg) ** 2)
ss_tot = np.nansum((y_proc.values - y_proc.mean()) ** 2)
overall_r2 = 1 - ss_res / ss_tot
print(f"\n  全样本 R² (Panel Tree): {overall_r2:.4f}")
print()

# =====================================================================
# 4. 回归任务 — 波动率加权岭回归
# =====================================================================
print("=" * 60)
print("4. 回归任务: VolWeightedRidgeRegressor")
print("=" * 60)

engine_vol = PanelTreeEngine(
    predictor=VolWeightedRidgeRegressor(alpha=1.0),
    criterion=R2DiffCriterion(),
    split_thresholds=[0.3, 0.5, 0.7],
    max_depth=2,
    min_samples=300,
    verbose=1,
)
engine_vol.fit(X_proc, y_proc, feature_names=dh.feature_names, weights=vol_weights)

reporter_vol = NodeReporter(engine_vol)
print("\n--- 树结构 (VolWeighted) ---")
print(reporter_vol.print_tree())
print()

# =====================================================================
# 5. 分类任务 — RidgeLogit + Precision 差异准则
# =====================================================================
print("=" * 60)
print("5. 分类任务: RidgeLogitClassifier + ClassificationCriterion")
print("=" * 60)

# 将收益率二值化：正收益 → 1, 否则 → 0
y_binary = (y_proc > 0).astype(float)

engine_cls = PanelTreeEngine(
    predictor=RidgeLogitClassifier(alpha=1.0, max_iter=30),
    criterion=ClassificationCriterion(metric="precision"),
    split_thresholds=[0.3, 0.5, 0.7],
    max_depth=2,
    min_samples=300,
    verbose=1,
)
engine_cls.fit(X_proc, y_binary, feature_names=dh.feature_names)

reporter_cls = NodeReporter(engine_cls)
print("\n--- 树结构 (Classification) ---")
print(reporter_cls.print_tree())

leaf_cls = reporter_cls.leaf_summary()
print("\n--- 叶子节点 Precision ---")
for _, row in leaf_cls.iterrows():
    metrics = row["Metrics"]
    print(f"  Leaf {row['Node_ID']}: precision={metrics.get('precision', 0):.4f}, "
          f"f1={metrics.get('f1', 0):.4f}, n={row['N_Samples']}")
print()

# =====================================================================
# 6. 快速模式 (Feature Priority Caching + Early Stopping)
# =====================================================================
print("=" * 60)
print("6. 快速模式 (fast_mode + early_stopping)")
print("=" * 60)

engine_fast = PanelTreeEngine(
    predictor=RidgeRegressor(alpha=1.0),
    criterion=R2DiffCriterion(),
    split_thresholds=[0.3, 0.5, 0.7],
    max_depth=3,
    min_samples=200,
    fast_mode=True,
    early_stopping_threshold=0.02,
    verbose=1,
)
engine_fast.fit(X_proc, y_proc, feature_names=dh.feature_names)

reporter_fast = NodeReporter(engine_fast)
print("\n--- 树结构 (Fast Mode) ---")
print(reporter_fast.print_tree())
print()

# =====================================================================
# 7. 预测马赛克 (Mosaic Visualization)
# =====================================================================
print("=" * 60)
print("7. 预测马赛克")
print("=" * 60)

viz = MosaicVisualizer(engine_reg)
mosaic = viz.build_mosaic(X_proc, y_proc, time_col="date", metric="r2")
print(f"  马赛克矩阵形状: {mosaic.shape}  (叶子数 × 时间期数)")
print(f"  前 5 期预览:")
print(mosaic.iloc[:, :5].to_string())

# 如果有 matplotlib，保存图片
try:
    import matplotlib
    matplotlib.use("Agg")  # 无头模式
    save_path = os.path.join(os.path.dirname(__file__), "mosaic.png")
    fig, ax = viz.plot_mosaic(mosaic, title="P-Tree Prediction Mosaic (R²)", save_path=save_path)
    print(f"\n  马赛克图已保存至: {save_path}")
except ImportError:
    print("\n  [提示] 安装 matplotlib + seaborn 可生成马赛克热力图")
print()

# =====================================================================
# 8. 叶子节点样本提取（Cluster Retrieval）
# =====================================================================
print("=" * 60)
print("8. 叶子节点样本提取")
print("=" * 60)

leaf_samples = engine_reg.get_leaf_samples()
for leaf_id, indices in leaf_samples.items():
    subset = df.iloc[indices]
    mean_ret = y_raw[indices].mean()
    std_ret = y_raw[indices].std()
    print(f"  Leaf {leaf_id}: {len(indices):>5d} 样本, "
          f"平均收益={mean_ret:+.4f}, 波动率={std_ret:.4f}, "
          f"覆盖资产数={subset['asset_id'].nunique()}")
print()

# =====================================================================
# 9. 完整节点报告 (DataFrame)
# =====================================================================
print("=" * 60)
print("9. 完整节点报告")
print("=" * 60)

full_report = reporter_reg.summary()
print(full_report[[
    "Node_ID", "Depth", "Is_Leaf", "Split_Feature",
    "Split_Threshold", "Predictability_Score", "N_Samples"
]].to_string(index=False))

print("\n" + "=" * 60)
print("示例运行完毕！")
print("=" * 60)
