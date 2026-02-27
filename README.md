# Panel Tree (P-Tree)

A supervised clustering algorithm designed for **panel data**, commonly used in quantitative finance to identify time-varying, cross-sectional predictability regimes.

## Core Idea

P-Tree recursively splits the full sample into disjoint leaf nodes using asset characteristics or macro states as thresholds. Unlike standard decision trees that minimise residual MSE, P-Tree **maximises the difference in predictive performance across child nodes**, producing a *prediction mosaic* — a map showing where and when alpha is concentrated.

## Project Structure

```
src/ptree/
├── __init__.py          # Package exports
├── data_handler.py      # DataHandler – alignment, missing-value fill, rank standardisation, volatility
├── predictors.py        # PredictorBase, RidgeRegressor, VolWeightedRidgeRegressor, RidgeLogitClassifier, SelfDefinedPredictor
├── criteria.py          # CriterionBase, R2DiffCriterion, ClassificationCriterion, evaluation helpers
├── node.py              # PanelTreeNode – per-node metadata container
├── engine.py            # PanelTreeEngine – recursive splitting, incremental matrix updates, feature-priority caching
└── visualization.py     # NodeReporter (text/DataFrame reports), MosaicVisualizer (heatmap)
```

## Quick Start

```python
import numpy as np
import pandas as pd
from ptree import DataHandler, RidgeRegressor, R2DiffCriterion, PanelTreeEngine
from ptree import NodeReporter, MosaicVisualizer

# 1. Prepare panel data (DataFrame with date, asset_id, and feature columns)
dh = DataHandler(cs_rank_standardize=True)
X, y, vol_weights = dh.fit_transform(
    df, y_series,
    time_col="date", entity_col="asset_id",
    ret_series_for_vol=ret_series,       # optional, for VolWeightedRidge
)

# 2. Build the tree
engine = PanelTreeEngine(
    predictor=RidgeRegressor(alpha=1.0),
    criterion=R2DiffCriterion(),
    split_thresholds=[0.3, 0.5, 0.7],
    max_depth=3,
    min_samples=100,
    fast_mode=False,
    verbose=1,
)
engine.fit(X, y, feature_names=dh.feature_names, weights=vol_weights)

# 3. Inspect results
reporter = NodeReporter(engine)
print(reporter.print_tree())           # text tree
print(reporter.leaf_summary())         # DataFrame

# 4. Prediction mosaic
viz = MosaicVisualizer(engine)
mosaic = viz.build_mosaic(X, y, time_col="date", metric="r2")
fig, ax = viz.plot_mosaic(mosaic)       # requires matplotlib & seaborn

# 5. Retrieve leaf-node samples
for leaf_id, indices in engine.get_leaf_samples().items():
    print(f"Leaf {leaf_id}: {len(indices)} observations")
```

## Module Overview

### DataHandler
| Parameter | Default | Description |
|---|---|---|
| `cs_rank_standardize` | `True` | Cross-sectional rank normalisation to [0, 1] |
| `vol_window` | `60` | Rolling window for volatility computation |
| `fillna_method` | `"ffill"` | Missing-value strategy (`ffill`, `bfill`, `zero`, `mean`, `None`) |

### Predictors
| Class | Use Case |
|---|---|
| `RidgeRegressor` | Standard Ridge regression (closed-form) |
| `VolWeightedRidgeRegressor` | Inverse-volatility weighted Ridge (handles heteroscedasticity) |
| `RidgeLogitClassifier` | Ridge logistic regression via IRLS |
| `SelfDefinedPredictor` | User-defined model base class |

### Criteria
| Class | Description |
|---|---|
| `R2DiffCriterion` | Maximise \|R²_L − R²_R\| (regression) |
| `ClassificationCriterion` | Maximise difference in Precision / F1 / AUC (classification) |

### PanelTreeEngine
| Parameter | Default | Description |
|---|---|---|
| `split_thresholds` | `[0.3, 0.5, 0.7]` | Candidate split points on (rank-standardised) feature values |
| `max_depth` | `3` | Maximum tree depth |
| `min_samples` | `100` | Minimum observations per node |
| `fast_mode` | `False` | Enable feature-priority caching from parent nodes |
| `early_stopping_threshold` | `None` | Stop searching if criterion exceeds this value (requires `fast_mode`) |
| `n_jobs` | `1` | Parallel workers (`-1` = all cores) |

## Output & Query API Reference

P-Tree 提供了丰富的输出与查询接口，分布在 `PanelTreeEngine`、`PanelTreeNode`、`NodeReporter` 和 `MosaicVisualizer` 四个类中。

---

### PanelTreeEngine 输出方法

#### `engine.predict(X) → np.ndarray`

对新数据生成逐样本预测。每条观测值沿树结构下行至对应叶子节点，由该叶子的局部模型给出预测值。

```python
preds = engine.predict(X_proc)  # shape: (n_samples,)
```

#### `engine.get_leaves() → List[PanelTreeNode]`

返回所有叶子节点对象的列表。

```python
for leaf in engine.get_leaves():
    print(f"Leaf {leaf.node_id}: R²={leaf.metrics.get('r2', None):.4f}, n={leaf.n_samples}")
```

#### `engine.get_all_nodes() → List[PanelTreeNode]`

返回整棵树的所有节点（BFS 顺序），包括内部节点和叶子节点。

```python
all_nodes = engine.get_all_nodes()
print(f"总节点数: {len(all_nodes)}")
```

#### `engine.get_node_report() → pd.DataFrame`

返回一个结构化 DataFrame，每行对应一个节点，包含以下列：

| 列名 | 说明 |
|---|---|
| `Node_ID` | 唯一节点标识 |
| `Depth` | 节点深度（root = 0） |
| `Rule` | 从根到该节点的完整路径规则，如 `root & char_1 >= 0.5 & char_3 < 0.7` |
| `Is_Leaf` | 是否为叶子节点 |
| `N_Samples` | 节点包含的样本数 |
| `Sample_Ratio` | 样本数占总样本的比例 |
| `Split_Feature` | 分裂所用特征名（叶子为 NaN） |
| `Split_Threshold` | 分裂阈值（叶子为 NaN） |
| `Split_Score` | 分裂时准则得分 |
| `Predictability_Score` | 该节点的可预测性强度（回归为 R²，分类为 Precision） |
| `Metrics` | 完整指标字典（如 `{"r2": 0.63, "mse": 0.22, "n_samples": 2429}`） |
| `Model_Weights` | 叶子节点模型的特征系数列表 |
| `Elapsed_Time_s` | 该节点构建耗时（秒） |
| `Parent_ID` | 父节点 ID |

```python
report = engine.get_node_report()
print(report[["Node_ID", "Depth", "Rule", "Predictability_Score", "N_Samples"]])
```

#### `engine.get_leaf_samples() → Dict[int, np.ndarray]`

返回一个字典，key 为叶子节点 `node_id`，value 为该叶子覆盖的原始样本行索引数组。用于提取各个 cluster 对应的原始数据。

```python
leaf_samples = engine.get_leaf_samples()
for leaf_id, indices in leaf_samples.items():
    subset = original_df.iloc[indices]
    print(f"Leaf {leaf_id}: {len(indices)} 样本, "
          f"平均收益={subset['ret'].mean():.4f}, "
          f"覆盖资产数={subset['asset_id'].nunique()}")
```

---

### PanelTreeNode 输出方法

每个节点对象可通过 `engine.get_leaves()` 或 `engine.get_all_nodes()` 获取。

#### `node.n_samples → int`

该节点包含的样本数量（只读属性）。

#### `node.metrics → Dict[str, float]`

节点的评估指标字典。回归任务包含 `r2`, `mse`, `n_samples`；分类任务包含 `precision`, `f1`, `auc`, `n_samples`。

```python
leaf = engine.get_leaves()[0]
print(leaf.metrics)  # {"r2": 0.63, "mse": 0.22, "n_samples": 2429}
```

#### `node.get_model_weights() → np.ndarray | None`

返回叶子节点局部模型的特征系数向量。可直接查看在特定环境下哪些因子在起作用。

```python
for leaf in engine.get_leaves():
    coef = leaf.get_model_weights()
    if coef is not None:
        for name, w in zip(dh.feature_names, coef):
            print(f"  {name}: {w:+.4f}")
```

#### `node.get_samples() → np.ndarray | None`

返回该节点覆盖的样本行索引。功能与 `engine.get_leaf_samples()` 类似，但可用于任意节点（包括内部节点）。

```python
node = engine.get_all_nodes()[1]  # 第二个节点
indices = node.get_samples()
print(f"Node {node.node_id} 包含 {len(indices)} 个样本")
```

#### `node.to_dict() → Dict[str, Any]`

将节点的所有元数据序列化为平坦字典，方便构建 DataFrame 或导出 JSON。

```python
import json
leaf = engine.get_leaves()[0]
print(json.dumps(leaf.to_dict(), indent=2, default=str))
```

#### 常用只读属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `node.node_id` | `int` | 唯一标识 |
| `node.depth` | `int` | 深度层级 |
| `node.rule` | `str` | 路径描述，如 `root & char_1 < 0.5 & char_3 >= 0.7` |
| `node.split_feature` | `str \| None` | 分裂特征名 |
| `node.split_threshold` | `float \| None` | 分裂阈值 |
| `node.split_score` | `float \| None` | 分裂时准则得分 |
| `node.is_leaf` | `bool` | 是否为叶子 |
| `node.sample_ratio` | `float` | 样本覆盖比例 |
| `node.elapsed_time` | `float` | 构建耗时（秒） |
| `node.predictor` | `PredictorBase` | 已训练的局部模型实例 |

---

### NodeReporter 输出方法

`NodeReporter` 封装了面向用户的报告功能，需传入已拟合的 `PanelTreeEngine`。

```python
from ptree import NodeReporter
reporter = NodeReporter(engine)
```

#### `reporter.summary() → pd.DataFrame`

返回完整节点报告 DataFrame（所有节点，包括内部节点和叶子），列定义同 `engine.get_node_report()`。

```python
full = reporter.summary()
print(full[["Node_ID", "Depth", "Is_Leaf", "Split_Feature", "Predictability_Score"]])
```

#### `reporter.leaf_summary() → pd.DataFrame`

仅返回叶子节点的报告，结构同 `summary()`，适合快速查看最终聚类结果。

```python
leaves = reporter.leaf_summary()
print(leaves[["Node_ID", "Rule", "Predictability_Score", "N_Samples", "Model_Weights"]])
```

输出示例：

```
 Node_ID                                              Rule  Predictability_Score  N_Samples
       3   root & char_1 < 0.5 & char_1 < 0.3 & char_3 < 0.7              0.0147       2438
       4  root & char_1 < 0.5 & char_1 < 0.3 & char_3 >= 0.7              0.0018       1102
      13  root & char_1 >= 0.5 & char_3 >= 0.3 & char_3 < 0.7              0.6323       2429
```

#### `reporter.print_tree() → str`

返回一个格式化的树结构文本字符串，使用缩进和 `├─` / `└─` 表示层级关系。

```python
print(reporter.print_tree())
```

输出示例：

```
[Node 0] char_1 < 0.5 (score=0.4569)
  ├─ Left (< 0.5):
    [Node 1] char_1 < 0.3 (score=0.0140)
      ├─ Left (< 0.3):
        [Leaf 3] n=2438 | r2=0.0147, mse=0.4769
      └─ Right (>= 0.3):
        [Leaf 4] n=1102 | r2=0.0018, mse=0.8028
  └─ Right (>= 0.5):
    [Leaf 5] n=6060 | r2=0.4640, mse=0.5483
```

---

### MosaicVisualizer 输出方法

`MosaicVisualizer` 用于生成"预测马赛克"——一张二维热力图，直观展示模型在不同时间点、不同资产群体中的预测能力。

```python
from ptree import MosaicVisualizer
viz = MosaicVisualizer(engine)
```

#### `viz.build_mosaic(X, y, time_col, metric) → pd.DataFrame`

计算每个叶子节点在每个时间期的预测表现，返回一个 DataFrame。

| 参数 | 说明 |
|---|---|
| `X` | 处理后的面板 DataFrame（需包含 `time_col` 和特征列） |
| `y` | 目标变量 |
| `time_col` | 时间列名，默认 `"date"` |
| `metric` | 评估指标：回归用 `"r2"`，分类用 `"precision"` / `"f1"` / `"auc"` |

返回值结构：
- **行索引**：叶子节点 ID（`Leaf_ID`）
- **列**：时间期（由 `time_col` 决定）
- **值**：该叶子在该时间期的指标值

```python
mosaic = viz.build_mosaic(X_proc, y_proc, time_col="date", metric="r2")
print(mosaic.shape)       # (n_leaves, n_periods)
print(mosaic.iloc[:, :5]) # 预览前 5 期

# 分析哪些叶子在哪些时段表现最好
best_leaf_per_period = mosaic.idxmax(axis=0)
print(best_leaf_per_period)
```

输出示例：

```
         0         1         2         3         4
Leaf_ID
3     0.016    -0.042     0.006    -0.089     0.036
13    0.621     0.782     0.599     0.687     0.605
14    0.502     0.465     0.350     0.462     0.289
```

#### `viz.plot_mosaic(mosaic, title, cmap, figsize, save_path) → (fig, ax)`

将马赛克矩阵绘制为 seaborn 热力图。需要 `matplotlib` 和 `seaborn`。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `mosaic` | — | `build_mosaic()` 返回的 DataFrame |
| `title` | `"Prediction Mosaic"` | 图表标题 |
| `cmap` | `"RdYlGn"` | 色彩映射（红=差，绿=好） |
| `figsize` | `(14, 6)` | 图表尺寸 |
| `save_path` | `None` | 若指定路径，自动保存为 PNG |

```python
# 交互查看
fig, ax = viz.plot_mosaic(mosaic, title="P-Tree R² Mosaic")

# 保存到文件
fig, ax = viz.plot_mosaic(mosaic, save_path="output/mosaic.png", cmap="coolwarm")
```

热力图含义：
- **横轴**：时间期 $t$
- **纵轴**：各叶子节点
- **颜色**：该叶子在该时间期的预测精度（R² 或 Precision）
- 可一眼看出模型在哪些时间点、哪些资产群体中"失效"或"爆发"

---

### 日志输出 (Verbose Logging)

`PanelTreeEngine` 在 `verbose >= 1` 时会通过 Python `logging` 模块输出详细的分裂过程日志：

```python
import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

engine = PanelTreeEngine(..., verbose=1)
engine.fit(X, y, feature_names=...)
```

日志示例：

```
[INFO] [Level 0] Splitting Node 0...
  - Best Split: 'char_1' at threshold 0.5000
  - Metric Delta: score = 0.456896
  - Left: 5940 samples | Right: 6060 samples
[INFO] [Level 1] Splitting Node 1...
  - Best Split: 'char_3' at threshold 0.3000
  - Metric Delta: score = 0.179045
  - Left: 1808 samples | Right: 4252 samples
[INFO] Tree built: 15 nodes, 8 leaves, max_depth=3
```

设置 `verbose=2` 可查看每个 (feature, threshold) 候选组合的逐条评估结果。

---

## Performance Optimisations

1. **Incremental matrix updates** – For Ridge models, $X^TWX$ and $X^TWy$ are cached at each node. Child-node statistics are obtained by subtraction, avoiding redundant matrix multiplications.
2. **Feature-priority caching** – When `fast_mode=True`, child nodes first evaluate the top-50% features from the parent, with optional early stopping.
3. **Multiprocessing** – Node-level parallelism via `n_jobs` for high-dimensional feature sets.

## Requirements

- Python ≥ 3.10
- `numpy`, `pandas`
- `matplotlib`, `seaborn` (optional, for visualisation)

## License

MIT
