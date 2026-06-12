"""
P-Forest / P-Boost（衍生集成算法）完整使用示例
==============================================

本示例使用 *多特征驱动* 的合成面板数据演示 P-Tree 的两个衍生集成算法：
1. 数据生成与预处理（DataHandler）
2. 单棵 P-Tree 基线（PanelTreeEngine）
3. P-Forest（PanelForest）：时间块自助 + 节点级随机特征子集
     - 袋装预测、软 regime 隶属概率、共识/协同关联矩阵、OOB R²
4. P-Boost（BoostedPanelTree）：残差提升，逐层剥离可预测性
     - 残差范数随轮数下降、Extra-Trees 随机阈值变体
5. 单树 vs 森林 vs 提升 的样本外 R² 对比

设计要点（见 docs/OPTIMIZATION_PLAN.md §D）：
- 集成作用在"预测/目标"层；每棵树的切分准则仍是 R²Diff，保持不变。
- P-Forest 的收益主要来自"可预测性由多个弱识别特征共同驱动"的场景，
  因此本示例特意构造多特征驱动数据（而非 demo_ptree 的单特征主导）。
"""

import sys, os, logging
import numpy as np
import pandas as pd

# 将 src/ 加入搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ptree import (
    DataHandler,
    RidgeRegressor,
    R2DiffCriterion,
    PanelTreeEngine,
    PanelForest,
    BoostedPanelTree,
    NodeReporter,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _overall_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """样本外/样本内全样本 R²。"""
    ss_res = np.nansum((y_true - y_pred) ** 2)
    ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-12)


# =====================================================================
# 1. 生成 *多特征驱动* 的合成面板数据
# =====================================================================
print("=" * 60)
print("1. 生成多特征驱动的合成面板数据")
print("=" * 60)

rng = np.random.default_rng(2026)

T = 80          # 时间期数
N = 150         # 每期资产数
P = 8           # 特征数量
feature_names = [f"char_{i+1}" for i in range(P)]

dates = np.repeat(np.arange(T), N)
asset_ids = np.tile(np.arange(N), T)

X_raw = rng.standard_normal((T * N, P))

# 多个弱 regime 共同驱动可预测性（没有单一主导特征）：
#   - char_1 高位时 char_2 提供 alpha
#   - char_3 高位时 char_4 提供 alpha
#   - char_5 高位时 char_6 提供 alpha
# 每个 regime 都较弱、需要不同特征识别 —— 这正是 P-Forest 受益的场景。
alpha = np.zeros(T * N)
alpha += np.where(X_raw[:, 0] > 0.5, 0.5 * X_raw[:, 1], 0.0)
alpha += np.where(X_raw[:, 2] > 0.5, 0.5 * X_raw[:, 3], 0.0)
alpha += np.where(X_raw[:, 4] > 0.5, 0.5 * X_raw[:, 5], 0.0)

y_raw = alpha + rng.standard_normal(T * N) * 0.8

df = pd.DataFrame(X_raw, columns=feature_names)
df["date"] = dates
df["asset_id"] = asset_ids
y_series = pd.Series(y_raw, name="ret")

print(f"  样本总量: {len(df)}")
print(f"  时间期数: {T},  每期资产: {N},  特征数: {P}")
print(f"  驱动特征: char_1/2, char_3/4, char_5/6 (多个弱 regime)")
print()

# =====================================================================
# 2. 数据预处理 + 训练/测试时间切分
# =====================================================================
print("=" * 60)
print("2. 数据预处理 (DataHandler) 与时间切分")
print("=" * 60)

dh = DataHandler(cs_rank_standardize=True, fillna_method="ffill")
X_proc, y_proc, _ = dh.fit_transform(
    df, y_series, time_col="date", entity_col="asset_id",
)
# 处理后保留 date 列用于时间块自助
X_proc["date"] = df["date"].values

# 按时间前 70% 训练，后 30% 测试（避免前视）
split_t = int(T * 0.7)
train_mask = X_proc["date"].values < split_t
test_mask = ~train_mask

X_train = X_proc[train_mask].reset_index(drop=True)
y_train = y_proc[train_mask].reset_index(drop=True)
X_test = X_proc[test_mask].reset_index(drop=True)
y_test = y_proc[test_mask].reset_index(drop=True)

print(f"  训练样本: {len(X_train)} (t < {split_t})")
print(f"  测试样本: {len(X_test)} (t >= {split_t})")
print(f"  特征名称: {dh.feature_names}")
print()

# =====================================================================
# 3. 单棵 P-Tree 基线
# =====================================================================
print("=" * 60)
print("3. 单棵 P-Tree 基线 (PanelTreeEngine)")
print("=" * 60)

single = PanelTreeEngine(
    predictor=RidgeRegressor(alpha=1.0),
    criterion=R2DiffCriterion(),
    split_thresholds=[0.3, 0.5, 0.7],
    max_depth=3,
    min_samples=150,
    verbose=0,
)
single.fit(X_train, y_train, feature_names=dh.feature_names)

reporter = NodeReporter(single)
print("\n--- 单树结构 ---")
print(reporter.print_tree())

single_test_pred = single.predict(X_test)
single_r2 = _overall_r2(y_test.values, single_test_pred)
print(f"\n  单树样本外 R²: {single_r2:.4f}")
print()

# =====================================================================
# 4. P-Forest（PanelForest）
# =====================================================================
print("=" * 60)
print("4. P-Forest (PanelForest) — 袋装集成")
print("=" * 60)

forest = PanelForest(
    n_estimators=60,
    max_features="sqrt",     # 节点级随机特征子集，去相关
    block_size=5,            # 时间块自助，保留时序结构
    base_params={
        "predictor": RidgeRegressor(alpha=1.0),
        "criterion": R2DiffCriterion(),
        "max_depth": 3,
        "min_samples": 100,
    },
    n_jobs=1,
    random_state=0,
    verbose=1,
)
forest.fit(
    X_train, y_train,
    feature_names=dh.feature_names,
    time_index="date",       # 时间块自助操作在 date 上
)

print(f"\n  OOB R² (袋外，训练集时间块): {forest.oob_score_:.4f}")

forest_test_pred = forest.predict(X_test)
forest_r2 = _overall_r2(y_test.values, forest_test_pred)
print(f"  森林袋装预测样本外 R²:     {forest_r2:.4f}")

# 软 regime 隶属概率
membership = forest.regime_membership(X_test)
print(f"\n  Regime 隶属概率 (落入高 R² 叶子的树占比):")
print(f"    范围 [{membership.min():.3f}, {membership.max():.3f}], 均值 {membership.mean():.3f}")

# 共识/协同关联矩阵（取测试集前 100 个观测，控制 O(n²) 内存）
sub = X_test.iloc[:100].reset_index(drop=True)
C = forest.coassociation_matrix(sub)
off_diag = C[~np.eye(C.shape[0], dtype=bool)]
print(f"\n  协同关联矩阵 (前 100 观测): 形状 {C.shape}")
print(f"    非对角共识度均值: {off_diag.mean():.3f} (两观测落入同一叶子的频率)")
print()

# =====================================================================
# 5. P-Boost（BoostedPanelTree）
# =====================================================================
print("=" * 60)
print("5. P-Boost (BoostedPanelTree) — 残差提升")
print("=" * 60)

booster = BoostedPanelTree(
    n_estimators=20,
    learning_rate=0.3,
    max_depth=2,             # 提升偏好浅树（弱学习器）
    subsample=1.0,
    base_params={
        "predictor": RidgeRegressor(alpha=1.0),
        "min_samples": 100,
    },
    random_state=0,
    verbose=1,
)
booster.fit(X_train, y_train, feature_names=dh.feature_names)

print(f"\n  残差范数序列 (随轮数应单调下降):")
norms = booster.residual_norms_
for i in range(0, len(norms), max(1, len(norms) // 8)):
    print(f"    round {i:>2d}: ||residual|| = {norms[i]:.3f}")
print(f"    final  : ||residual|| = {norms[-1]:.3f}")

boost_test_pred = booster.predict(X_test)
boost_r2 = _overall_r2(y_test.values, boost_test_pred)
print(f"\n  提升预测样本外 R²: {boost_r2:.4f}")
print()

# =====================================================================
# 6. Extra-Trees 风格随机阈值森林
# =====================================================================
print("=" * 60)
print("6. Extra-Trees 风格 (splitter='random') 随机阈值森林")
print("=" * 60)

extra_forest = PanelForest(
    n_estimators=60,
    max_features="sqrt",
    block_size=5,
    base_params={
        "predictor": RidgeRegressor(alpha=1.0),
        "criterion": R2DiffCriterion(),
        "max_depth": 3,
        "min_samples": 100,
        "splitter": "random",        # 随机阈值，额外注入多样性
        "n_random_splits": 3,
    },
    n_jobs=1,
    random_state=0,
    verbose=0,
)
extra_forest.fit(X_train, y_train, feature_names=dh.feature_names, time_index="date")
extra_test_pred = extra_forest.predict(X_test)
extra_r2 = _overall_r2(y_test.values, extra_test_pred)
print(f"  Extra-Trees 森林 OOB R²:     {extra_forest.oob_score_:.4f}")
print(f"  Extra-Trees 森林样本外 R²:   {extra_r2:.4f}")
print()

# =====================================================================
# 7. 样本外 R² 对比汇总
# =====================================================================
print("=" * 60)
print("7. 样本外 R² 对比汇总")
print("=" * 60)

summary = pd.DataFrame(
    {
        "Model": [
            "Single P-Tree",
            "P-Forest (bagging)",
            "P-Forest (extra-trees)",
            "P-Boost (boosting)",
        ],
        "OOS_R2": [single_r2, forest_r2, extra_r2, boost_r2],
    }
)
print(summary.to_string(index=False))

print("\n  说明：在多特征驱动数据上，集成（尤其 P-Forest）通常优于单树；")
print("  若改用单特征主导数据，森林增益会显著收窄（见 docs §D1 的诚实局限）。")

print("\n" + "=" * 60)
print("示例运行完毕！")
print("=" * 60)
