# P-Tree API Reference / API 参考手册

> **Scope / 范围.** This document is a *flat, dictionary-style reference* for every
> public class, method, parameter and return value exposed by the `ptree`
> package (`v0.3.x`). Use it as a lookup table; for conceptual background
> see [`README.md`](../README.md) and for the algorithmic specification
> see [`ALGORITHM.md`](ALGORITHM.md).
>
> 本文档是 `ptree` 包（`v0.3.x`）所有公开类 / 方法 / 参数 / 返回值的**字典式查询手册**。
> 概念背景见 [`README.md`](../README.md)；算法规范见 [`ALGORITHM.md`](ALGORITHM.md)。
>
> Every entry follows the same template:
> *Signature / 签名 → Purpose / 作用 → Parameters / 参数 → Returns / 返回 → Example / 示例.*
> 每条 API 条目格式统一：*签名 → 作用 → 参数 → 返回 → 示例*。

---

## Table of Contents / 目录

- [0. Overview & Cheat Sheet / 总览与速查表](#0-overview--cheat-sheet--总览与速查表)
- [1. `DataHandler` / 数据预处理](#1-datahandler--数据预处理)
- [2. Predictors / 叶子模型](#2-predictors--叶子模型)
- [3. Criteria / 切分准则](#3-criteria--切分准则)
- [4. `PanelTreeEngine` / 主引擎](#4-paneltreeengine--主引擎)
- [5. `PanelTreeNode` / 节点对象](#5-paneltreenode--节点对象)
- [6. `NodeEvalResult` / OOS 评估容器](#6-nodeevalresult--oos-评估容器)
- [7. Ensembles / 集成方法](#7-ensembles--集成方法)
- [8. Visualization / 可视化](#8-visualization--可视化)
- [9. Logging / 日志](#9-logging--日志)
- [10. Version Migration / 版本迁移](#10-version-migration--版本迁移)

---

## 0. Overview & Cheat Sheet / 总览与速查表

### 0.1 Top-level imports / 顶层导入

```python
from ptree import (
    # Data
    DataHandler,
    # Predictors
    PredictorBase, RidgeRegressor, VolWeightedRidgeRegressor,
    RidgeLogitClassifier, ElasticNetRegressor, PLSRegressor,
    SelfDefinedPredictor,
    # Criteria
    CriterionBase, R2DiffCriterion, WeightedR2DiffCriterion,
    RankICDiffCriterion, MeanVarianceCriterion, ClassificationCriterion,
    # Core
    PanelTreeNode, PanelTreeEngine, NodeEvalResult,
    # Ensembles
    PanelForest, BoostedPanelTree,
    # Visualization
    NodeReporter, MosaicVisualizer,
)
```

| Name / 名字 | One-liner / 一句话说明 |
|---|---|
| `DataHandler` | Panel preprocessing: alignment, fillna, cross-sectional rank, vol weights / 面板预处理：对齐、缺失填充、截面秩标准化、波动率权重 |
| `RidgeRegressor` | Closed-form L2 leaf regressor / 闭式 L2 岭回归 |
| `VolWeightedRidgeRegressor` | Inverse-vol weighted Ridge / 逆波动率加权岭回归 |
| `RidgeLogitClassifier` | IRLS-based Ridge logit (with `predict_proba`) / IRLS 岭逻辑回归 |
| `ElasticNetRegressor` | L1+L2 coordinate descent / 弹性网络 |
| `PLSRegressor` | NIPALS partial least squares / 偏最小二乘 |
| `SelfDefinedPredictor` | User-defined leaf model template / 自定义叶子模型模板 |
| `R2DiffCriterion` | `\|R²_L − R²_R\|` (default) / R² 差（默认） |
| `WeightedR2DiffCriterion` | Balance + shrinkage stabilised R² diff / 稳健化 R² 差 |
| `RankICDiffCriterion` | `\|IC_L − IC_R\|` (scale-free) / Rank-IC 差 |
| `MeanVarianceCriterion` | Tangency-portfolio Sharpe of two children / 切线组合最大夏普 |
| `ClassificationCriterion` | Classification metric diff / 分类指标差 |
| `PanelTreeEngine` | The fit / predict / prune / evaluate engine / 主引擎 |
| `PanelTreeNode` | Per-node metadata container / 节点元数据容器 |
| `NodeEvalResult` | Per-node OOS metrics container / 逐节点 OOS 评估容器 |
| `PanelForest` | Bagged ensemble (P-Forest) / 装袋集成 |
| `BoostedPanelTree` | Residual-boosted ensemble (P-Boost) / 残差提升集成 |
| `NodeReporter` | Text / DataFrame / Graphviz / mpl tree reports / 文本/表/图树形报告 |
| `MosaicVisualizer` | Prediction-mosaic heatmap / 预测马赛克热力图 |

### 0.2 Task → API cheat sheet / 任务 → API 速查

| I want to … / 我想 … | Call / 调用 |
|---|---|
| Preprocess a raw `(date, asset_id, …)` DataFrame / 预处理原始面板 DataFrame | `DataHandler().fit_transform(...)` |
| Build the tree / 训练一棵 P-Tree | `PanelTreeEngine(...).fit(X, y, feature_names=...)` |
| Predict on new data / 在新数据上预测 | `engine.predict(X)` |
| Route rows to leaf ids / 把每行路由到叶子 id | `engine.predict_leaves(X)` |
| Get the root-to-leaf node id path / 取根→叶完整节点路径 | `engine.predict_node_path(X)` |
| Compute per-node OOS R² / Rank-IC / Sharpe / 逐节点 OOS 指标 | `engine.evaluate(X_oos, y_oos, time_col=...)` |
| Inspect every node as a DataFrame / 表格式查看所有节点 | `engine.get_node_report()` or `NodeReporter(engine).summary()` |
| Inspect only leaves / 仅看叶子 | `NodeReporter(engine).leaf_summary()` |
| Get the original sample indices in each leaf / 每个叶子原始样本 idx | `engine.get_leaf_samples()` |
| Post-prune the tree / 剪枝 | `engine.prune(ccp_alpha)` / `engine.tune_ccp_alpha(...)` |
| Build a tradeable SDF factor from leaves / 用叶子组合构造 SDF 因子 | `engine.build_sdf_factor(...)` |
| Print a pretty text tree / 文本格式打印树 | `NodeReporter(engine).print_tree()` |
| Export a Graphviz DOT diagram / 导出 Graphviz DOT | `NodeReporter(engine).to_graphviz()` |
| Plot a matplotlib tree / matplotlib 树图 | `NodeReporter(engine).plot_tree()` |
| Prediction mosaic / 预测马赛克 | `MosaicVisualizer(engine).build_mosaic(...)` → `plot_mosaic(...)` |
| Soft regime probability via forest / 森林软 regime 概率 | `PanelForest(...).regime_membership(X)` |
| Consensus similarity for clustering / 共识相似度矩阵 | `PanelForest(...).coassociation_matrix(X)` |
| OOB validation R² / 袋外 R² | `PanelForest(...).oob_score_` |
| Boosted (multi-layer) regime discovery / 分层弱 regime 揭示 | `BoostedPanelTree(...).fit(...).predict(...)` |

---

## 1. `DataHandler` / 数据预处理

Module: `ptree.data_handler`

### 1.1 `DataHandler.__init__`

**Signature / 签名**

```python
DataHandler(
    cs_rank_standardize: bool = True,
    vol_window: int = 60,
    min_obs: int = 20,
    fillna_method: Optional[str] = "ffill",
)
```

**Purpose / 作用.**
Panel preprocessing: align `(X, y)`, fill missing values, optionally
cross-sectionally rank-standardise features to `[0, 1]`, and (optionally)
compute inverse-volatility weights from a return series.
面板预处理：对齐 `(X, y)`、填充缺失、可选地把特征按截面秩归一到 `[0, 1]`、并可基于
收益序列计算逆波动率权重。

| Param / 参数 | Type / 类型 | Default / 默认 | Meaning / 含义 |
|---|---|---|---|
| `cs_rank_standardize` | `bool` | `True` | Apply per-cross-section rank → `[0,1]` mapping / 每个时间截面内做秩归一 |
| `vol_window` | `int` | `60` | Rolling window for realised volatility / 波动率滚动窗口 |
| `min_obs` | `int` | `20` | Min non-NaN obs to produce a vol value / 计算波动率所需最小观测数 |
| `fillna_method` | `str` or `None` | `"ffill"` | One of `"ffill" / "bfill" / "zero" / "mean" / None` |

### 1.2 `DataHandler.fit`

```python
fit(
    X: pd.DataFrame,
    y: pd.Series,
    time_col: str = "date",
    entity_col: str = "asset_id",
) -> "DataHandler"
```

Learn feature-column metadata. `X` must contain `time_col` and `entity_col`
either as columns or as named MultiIndex levels.
学习特征列元数据。`X` 必须包含 `time_col` 与 `entity_col`（列或具名 MultiIndex 层均可）。

### 1.3 `DataHandler.transform`

```python
transform(
    X: pd.DataFrame,
    y: pd.Series,
    ret_series_for_vol: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]
```

Apply the pipeline.
应用预处理流水线。

**Returns / 返回**
- `X_processed`: cleaned (and optionally rank-standardised) feature panel,
  retains `time_col`/`entity_col` columns. / 清洗后的特征面板。
- `y_processed`: target aligned with `X_processed`. / 与 X 对齐后的目标。
- `vol_weights`: inverse-vol weights (`1/σ`) or `None`. / 逆波动率权重或 `None`。

### 1.4 `DataHandler.fit_transform`

```python
fit_transform(X, y, time_col="date", entity_col="asset_id",
              ret_series_for_vol=None) -> (X_proc, y_proc, vol_weights)
```

Convenience wrapper: `fit` then `transform`. / `fit` + `transform` 的快捷调用。

### 1.5 Property `feature_names`

```python
dh.feature_names  # -> List[str]
```

Feature column names learnt during `fit` (excludes `time_col`/`entity_col`).
`fit` 学到的特征列名（不含时间 / 实体列）。

### 1.6 Example / 示例

```python
from ptree import DataHandler

dh = DataHandler(cs_rank_standardize=True, vol_window=60, fillna_method="ffill")
X_proc, y_proc, w = dh.fit_transform(
    raw_df, y_series,
    time_col="date", entity_col="asset_id",
    ret_series_for_vol=ret_series,
)
print(dh.feature_names)
```

---

## 2. Predictors / 叶子模型

Module: `ptree.predictors`. All predictors inherit from `PredictorBase`.
所有 predictor 均继承自 `PredictorBase`。

### 2.1 `PredictorBase` (abstract / 抽象基类)

```python
class PredictorBase(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray,
            weights: Optional[np.ndarray] = None) -> "PredictorBase": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def get_coefficients(self) -> Optional[np.ndarray]: ...
    def get_intercept(self)    -> Optional[float]: ...
    def get_params(self)       -> Dict[str, Any]: ...
```

| Method / 方法 | Returns / 返回 | Notes / 备注 |
|---|---|---|
| `fit(X, y, weights=None)` | `self` | Weights are optional; closed-form predictors also accept cached `XtWX`/`XtWy` via keyword arguments. / 权重可选；闭式 predictor 还接受 `XtWX`/`XtWy` 关键字以做增量更新。 |
| `predict(X)` | `ndarray` shape `(n,)` | — |
| `get_coefficients()` | coefficient vector or `None` | — |
| `get_intercept()` | intercept or `None` | — |
| `get_params()` | hyperparameter dict | — |

### 2.2 `RidgeRegressor`

```python
RidgeRegressor(alpha: float = 1.0, fit_intercept: bool = True)
```

Closed-form L2-regularised linear regression: β = (XᵀWX + αI)⁻¹ XᵀWy.
Uses SciPy Cholesky when available, falls back to NumPy `solve`/`lstsq`.
闭式岭回归；优先用 SciPy Cholesky，回退 NumPy。

| Param | Type | Default | Meaning |
|---|---|---|---|
| `alpha` | `float` | `1.0` | L2 regularisation / L2 强度 |
| `fit_intercept` | `bool` | `True` | Prepend a constant column / 是否拟合截距 |

### 2.3 `VolWeightedRidgeRegressor`

```python
VolWeightedRidgeRegressor(alpha: float = 1.0, fit_intercept: bool = True)
```

Same closed form as `RidgeRegressor`, expected to be called with
inverse-vol weights (`1/σ`). The `weights` argument is what carries the
heteroscedasticity correction.
与 `RidgeRegressor` 公式一致，但期望以 `1/σ` 作为 `weights`，以缓解异方差。

### 2.4 `RidgeLogitClassifier`

```python
RidgeLogitClassifier(
    alpha: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-6,
    fit_intercept: bool = True,
)
```

L2-regularised logistic regression via Iteratively Reweighted Least Squares
(IRLS). Exposes:
- `predict_proba(X) -> ndarray` — `P(y=1|X)`
- `predict(X) -> ndarray` — hard 0/1 labels at threshold `0.5`.

IRLS 岭逻辑回归。除常规 `fit/predict` 外，额外提供 `predict_proba`（`P(y=1|X)`）。

### 2.5 `ElasticNetRegressor`

```python
ElasticNetRegressor(
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-4,
    fit_intercept: bool = True,
)
```

Coordinate-descent L1+L2 regression. Useful for sparse factor selection on
highly correlated characteristics. Does **not** participate in the engine's
incremental sufficient-statistic fast path.
坐标下降弹性网络。不参与引擎的增量统计量加速路径。

### 2.6 `PLSRegressor`

```python
PLSRegressor(n_components: int = 2, fit_intercept: bool = True)
```

NIPALS single-response partial least squares. Effective when characteristics
are strongly collinear. Also bypasses the incremental fast path.
NIPALS 偏最小二乘（单响应）。对强共线性因子有效，同样不参与增量加速。

### 2.7 `SelfDefinedPredictor`

Template base class — subclass it, implement `fit` and `predict`, and pass an
instance to `PanelTreeEngine(predictor=...)`.
用户自定义模板：继承后实现 `fit` 与 `predict`，把实例传给 `PanelTreeEngine`。

```python
from ptree import SelfDefinedPredictor

class MyLGBPredictor(SelfDefinedPredictor):
    def fit(self, X, y, weights=None):
        import lightgbm as lgb
        self.model = lgb.LGBMRegressor().fit(X, y, sample_weight=weights)
        return self
    def predict(self, X):
        return self.model.predict(X)
```

---

## 3. Criteria / 切分准则

Module: `ptree.criteria`. All criteria implement `calculate_score` and
`metric_key`.
所有准则实现 `calculate_score` 与 `metric_key`。

### 3.1 `CriterionBase` (abstract / 抽象)

```python
class CriterionBase(ABC):
    @abstractmethod
    def calculate_score(self, left_metrics: dict, right_metrics: dict) -> float: ...
    @abstractmethod
    def metric_key(self) -> str: ...
```

`metric_key()` is the *primary metric* the engine, `NodeReporter` and
`PanelForest` use to summarise / rank leaves. Returns one of
`"r2" / "rank_ic" / "sharpe" / "precision" / "f1" / "auc" / "logloss"`.
`metric_key()` 决定 engine / 报告 / 森林如何汇总叶子，取值见上。

### 3.2 `R2DiffCriterion`

```python
R2DiffCriterion(weight_by_size: bool = False)
```

Score: `score = |R²_L − R²_R|` (default), optionally multiplied by
`min(n_L, n_R) / (n_L + n_R)`.
打分 = `|R²_L − R²_R|`，可选乘平衡项。

- `metric_key() → "r2"`.

### 3.3 `WeightedR2DiffCriterion`

```python
WeightedR2DiffCriterion(
    balance: bool = True,
    shrinkage_k: float = 0.0,
    use_adjusted_r2: bool = False,
    min_child_weight: float = 0.0,    # ∈ [0, 0.5)
)
```

Stabilised R² diff for noisy panels.
为低信噪比面板设计的稳健 R² 差。

`score = |R²_L − R²_R| · balance · shrinkage`, with
balance = `min(n_L,n_R)/(n_L+n_R)`, shrinkage = `(n_min−1)/(n_min−1+k)`.
`min_child_weight` is a *hard floor* on the smaller child's sample share —
sliver splits score `0`.

| Param | Default | Meaning |
|---|---|---|
| `balance` | `True` | Apply the balance penalty / 是否乘平衡项 |
| `shrinkage_k` | `0.0` | Sample-size shrinkage strength / 样本量收缩强度 |
| `use_adjusted_r2` | `False` | Use adjusted-R² with `n_features` / 用调整 R² |
| `min_child_weight` | `0.0` | Hard sample-share floor on smaller child / 较小子节点的样本占比硬下限 |

- `metric_key() → "r2"`.

### 3.4 `RankICDiffCriterion`

```python
RankICDiffCriterion(
    balance: bool = False,
    min_child_weight: float = 0.0,
    min_periods: int = 3,
)
```

`score = |mean cross-sectional rank-IC_L − mean rank-IC_R|`. Scale-free,
robust on low-SNR / non-Gaussian targets.
**Requires `fit(..., time_index=...)`** so the engine can build the per-time
series payload `"_rank_ic_series"` attached to each child's metrics.

低信噪比时推荐使用。需在 `fit` 中传 `time_index`。

- `metric_key() → "r2"` (kept for node logging; the criterion actually
  operates on the per-time series payload). / 节点显示沿用 `"r2"`，但实际打分作用于时序载荷。

### 3.5 `MeanVarianceCriterion`

```python
MeanVarianceCriterion(
    annualization: float = 12.0,
    ridge: float = 1e-6,
    min_periods: int = 3,
)
```

Score = annualised tangency Sharpe of the two children's long-short portfolio
return series, `score = √(μᵀΣ⁻¹μ) · √A`. Aligns the splitting objective with
the *efficient-frontier* goal.
**Requires `fit(..., time_index=...)`** so the engine can construct the
`"_port_ret"` payload.
打分 = 两侧多空组合切线组合年化夏普；需 `time_index`。

- `metric_key() → "sharpe"` (each leaf's long-short Sharpe is stored in
  `node.metrics["sharpe"]`). / 每个叶子的多空夏普写入 `node.metrics["sharpe"]`。

### 3.6 `ClassificationCriterion`

```python
ClassificationCriterion(
    metric: str = "precision",        # ∈ {"precision","f1","auc","logloss"}
    weight_by_size: bool = False,
)
```

`score = |metric_L − metric_R|`. `metric_key()` returns the chosen `metric`.

### 3.7 Helper functions / 辅助函数

| Function / 函数 | Returns / 返回 |
|---|---|
| `evaluate_regression(y_true, y_pred, weights=None)` | `{"r2", "mse", "n_samples", "n_features"}` |
| `evaluate_classification(y_true, y_proba, threshold=0.5)` | `{"precision", "f1", "auc", "logloss", "n_samples"}` |

These are usually called by the engine internally; you can also call them
directly for ad-hoc metric computation.
一般由 engine 内部调用，亦可独立用作即席指标计算。

---

## 4. `PanelTreeEngine` / 主引擎

Module: `ptree.engine`.

### 4.1 Constructor / 构造函数

```python
PanelTreeEngine(
    predictor: Union[PredictorBase, Type[PredictorBase]] = RidgeRegressor,
    criterion: CriterionBase = R2DiffCriterion(),
    split_thresholds: Union[List[float], str, None] = None,   # default [0.3, 0.5, 0.7]
    max_depth: int = 3,
    min_samples: int = 100,
    min_impurity_decrease: float = 0.0,
    adaptive_quantiles: Optional[List[float]] = None,         # default [0.25, 0.5, 0.75]
    honest: bool = False,
    honest_frac: float = 0.5,
    honest_refit_full: bool = True,
    random_state: Optional[int] = None,
    fast_mode: bool = False,
    early_stopping_threshold: Optional[float] = None,
    n_jobs: int = 1,
    verbose: int = 1,
    predictor_params: Optional[Dict] = None,
    keep_node_stats: bool = False,
    parallel_backend: str = "threads",                        # {"threads","processes"}
    max_features: Optional[Union[str, int, float]] = None,    # None / "sqrt" / "log2" / int / float
    splitter: str = "best",                                   # {"best","random"}
    n_random_splits: int = 1,
)
```

Parameters grouped by role / 按用途分组：

**Core / 核心**

| Param | Default | Meaning |
|---|---|---|
| `predictor` | `RidgeRegressor` | Leaf-model class or instance. If a class, instantiated with `predictor_params`. / 叶子模型类或实例 |
| `criterion` | `R2DiffCriterion()` | Splitting criterion / 切分准则 |
| `max_depth` | `3` | Max tree depth / 最大深度 |
| `min_samples` | `100` | Min samples for a node to be splittable / 节点可分裂所需的最小样本数 |
| `min_impurity_decrease` | `0.0` | Best-split score floor / 最优切分分阈值；不达则成叶 |

**Threshold search / 阈值搜索**

| Param | Default | Meaning |
|---|---|---|
| `split_thresholds` | `[0.3, 0.5, 0.7]` | Global fixed list, *or* `"adaptive"` for per-node quantile thresholds / 全局固定列表，或 `"adaptive"` 使用节点分位 |
| `adaptive_quantiles` | `[0.25, 0.5, 0.75]` | Quantiles used when `split_thresholds="adaptive"` |
| `splitter` | `"best"` | `"best"` = exhaustive; `"random"` = Extra-Trees random thresholds |
| `n_random_splits` | `1` | Random thresholds per feature when `splitter="random"` |
| `max_features` | `None` | Random feature subset per node: `None / "sqrt" / "log2" / int / float`. Used by ensembles to decorrelate trees |

**Honest splits / 诚实切分**

| Param | Default | Meaning |
|---|---|---|
| `honest` | `False` | Enable honest split (fit-set + eval-set partition inside each node) / 启用诚实切分 |
| `honest_frac` | `0.5` | Fraction held out as eval-set in `(0, 1)` |
| `honest_refit_full` | `True` | After choosing the split, refit leaf models on the full sample |
| `random_state` | `None` | Seed for honest split, random splitter and feature subset / 随机种子（控制诚实切分、随机阈值、随机特征子集） |

**Performance / 性能**

| Param | Default | Meaning |
|---|---|---|
| `fast_mode` | `False` | Feature-priority caching from parent / 父节点特征优先级缓存 |
| `early_stopping_threshold` | `None` | Early-stop feature loop when best ≥ threshold (requires `fast_mode`) |
| `n_jobs` | `1` | Feature-dim parallel workers (`-1` = all cores) / 特征维并行 |
| `parallel_backend` | `"threads"` | joblib backend: `"threads"` or `"processes"` |
| `keep_node_stats` | `False` | Keep per-node cached `XtWX/XtWy` after splitting / 是否保留节点统计量 |

**Logging / 日志**

| Param | Default | Meaning |
|---|---|---|
| `verbose` | `1` | `0` = silent, `1` = per-level, `2` = per-candidate |
| `predictor_params` | `None` | Used only when `predictor` is a class, to instantiate it / 仅当 `predictor` 为类时使用 |

### 4.2 `fit`

```python
fit(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    weights: Optional[np.ndarray] = None,
    time_index: Optional[Union[np.ndarray, pd.Series, str]] = None,
) -> "PanelTreeEngine"
```

Build the tree. **`time_index` is required** when `criterion` is
`MeanVarianceCriterion` (long-short portfolios are aggregated by time);
it is also necessary for `RankICDiffCriterion` to produce the per-time
series payload. A `str` is treated as a column name in `X`.

构建树。当准则为 `MeanVarianceCriterion` / `RankICDiffCriterion` 时
**必须**传入 `time_index`。

**Returns / 返回**: `self` (chaining).

### 4.3 Prediction APIs / 预测系列

#### `engine.predict(X)`

```python
predict(X: pd.DataFrame) -> np.ndarray  # shape (n,)
```

Route each row to its leaf and apply the leaf's predictor.
对每行路由到叶子并调用叶子的 predictor。

#### `engine.predict_leaves(X)`

```python
predict_leaves(X: pd.DataFrame) -> np.ndarray[int]  # shape (n,)
```

Return the leaf `node_id` each row falls into.
返回每行所在叶子的 `node_id`。

#### `engine.predict_node_path(X)`

```python
predict_node_path(X: pd.DataFrame) -> List[List[int]]
```

For each row, the ordered list of `node_id`s from root to leaf — useful for
explaining a specific prediction.
每行返回根到叶完整的 `node_id` 序列，用于解释单个预测。

```python
paths = engine.predict_node_path(X.head(3))
# paths[0] == [0, 1, 4]  → root → node 1 → leaf 4
```

### 4.4 `engine.evaluate` — per-node OOS diagnostics / 逐节点 OOS 评估

```python
evaluate(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    time_col: Optional[Union[str, np.ndarray, pd.Series]] = None,
    metrics: Sequence[str] = ("r2", "rank_ic"),
    weights: Optional[Union[pd.Series, np.ndarray]] = None,
) -> NodeEvalResult
```

Compute OOS metrics for *every* node (internal + leaf). Internal-node pools
are built by merging the children's pools so leaf and internal rows are
directly comparable.

对每个节点（含内部节点与叶子）计算 OOS 指标。内部节点的样本池由左右孩子样本合并而来。

| Param | Notes |
|---|---|
| `time_col` | Required for `"rank_ic"` and `"sharpe"`; silently dropped otherwise. `str` → column name in `X`. |
| `metrics` | Subset of `{"r2","rank_ic","sharpe","precision","f1","auc","logloss"}`. |
| `weights` | Forwarded to weighted R². |

**Returns / 返回**: a [`NodeEvalResult`](#6-nodeevalresult--oos-评估容器).

```python
result = engine.evaluate(X_oos, y_oos, time_col="date",
                         metrics=("r2", "rank_ic"))
print(result.per_node_df.head())
print(reporter.print_tree(evaluation=result, show_child_diff=True))
```

### 4.5 Node inspection / 节点查询

| Method | Returns |
|---|---|
| `engine.get_leaves()` | `List[PanelTreeNode]` — all leaves / 所有叶子 |
| `engine.get_all_nodes()` | `List[PanelTreeNode]` — BFS order / BFS 顺序所有节点 |
| `engine.get_node_report()` | `pd.DataFrame` — one row per node (see schema below) / 每节点一行的 DataFrame |
| `engine.get_leaf_samples()` | `Dict[int, np.ndarray]` — leaf id → row indices / 叶子 id → 原始行索引 |

`get_node_report()` columns / 列：

| Column / 列 | Description / 含义 |
|---|---|
| `Node_ID` | Node id / 节点 id |
| `Depth` | Depth, root = 0 / 深度，根为 0 |
| `Rule` | Path rule string `"root & x1 < 0.5 & x3 >= 0.7"` / 路径规则字符串 |
| `Is_Leaf` | `bool` |
| `N_Samples` | In-node sample count / 节点样本数 |
| `Sample_Ratio` | `N_Samples / total` |
| `Split_Feature` | Split feature name (NaN for leaves) |
| `Split_Threshold` | Split threshold value |
| `Split_Score` | Criterion score at this split |
| `Predictability_Score` | `r2` (regression) or `precision` (classification) |
| `Metrics` | Raw metrics dict |
| `Model_Weights` | Leaf-model coefficient list |
| `Elapsed_Time_s` | Seconds spent building this node |
| `Parent_ID` | Parent node id, `None` for root |

### 4.6 Pruning / 剪枝

#### `engine.prune(ccp_alpha)`

```python
prune(ccp_alpha: float) -> "PanelTreeEngine"
```

Bottom-up cost-complexity pruning *in place*. A subtree at node $v$ is
collapsed when its cumulative split-score gain does not exceed
`ccp_alpha · (n_leaves(v) − 1)`. Leaf predictors are retained, so
`predict` keeps working.
原位剪枝。

#### `engine.cost_complexity_pruning_path()`

```python
cost_complexity_pruning_path() -> Dict[str, np.ndarray]
# keys: "ccp_alphas", "n_leaves", "total_scores"
```

Weakest-link path. `ccp_alphas[0] == 0` corresponds to the full tree;
each subsequent entry collapses the next weakest internal node.

#### `engine.tune_ccp_alpha`

```python
tune_ccp_alpha(
    X_val: pd.DataFrame,
    y_val: Union[pd.Series, np.ndarray],
    metric: str = "r2",                          # also "rank_ic","sharpe","precision","f1","auc","logloss"
    time_col: Optional[...] = None,
    weights: Optional[...] = None,
) -> Tuple[float, pd.DataFrame]
```

Sweep `ccp_alpha` along the weakest-link path, evaluate each pruned tree on
`(X_val, y_val)` via `evaluate`, and return the alpha maximising the root
OOS metric. `logloss` is treated as "smaller is better" internally.
沿弱链路径扫 `ccp_alpha`，挑使根节点 OOS 指标最优的 α。`logloss` 自动反号。

**Returns / 返回**: `(best_alpha, curve_df)` where `curve_df` has columns
`ccp_alpha`, `n_leaves`, `oos_<metric>`.

### 4.7 `engine.build_sdf_factor` — SDF factor construction / SDF 因子构造

```python
build_sdf_factor(
    X: Optional[pd.DataFrame] = None,
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    time_index: Optional[Union[np.ndarray, pd.Series, str]] = None,
    ridge: float = 1e-6,
) -> Dict[str, Any]
```

Aggregate every leaf's long-short portfolio into a single mean-variance
(tangency) combination — the in-sample maximum-Sharpe Stochastic Discount
Factor portfolio realising the "growing the efficient frontier" goal.

If `X` is `None`, the training data passed to `fit` is reused (`fit` must
have been called with `time_index`).

把所有叶子多空组合按切线权重 `w ∝ Σ⁻¹μ` 组合成单一 SDF 因子。
当 `X=None` 时复用训练数据。

**Returns / 返回** (dict):

| Key | Type | Meaning |
|---|---|---|
| `weights` | `ndarray` | Tangency weight on each leaf portfolio (L1-normalised) |
| `leaf_ids` | `List[int]` | Leaf node ids aligned with `weights` |
| `times` | `ndarray` | Sorted common time labels |
| `sdf_returns` | `ndarray` | SDF return per time |
| `sharpe` | `float` | In-sample, **non-annualised** Sharpe of the SDF |

```python
sdf = engine.build_sdf_factor()
print(sdf["sharpe"], sdf["weights"])
```

---

## 5. `PanelTreeNode` / 节点对象

Module: `ptree.node`. Returned by `engine.get_leaves()` / `get_all_nodes()`.

### 5.1 Attributes / 属性

| Attribute / 属性 | Type | Meaning |
|---|---|---|
| `node_id` | `int` | Unique id within the tree |
| `depth` | `int` | Depth (root = 0) |
| `parent_id` | `int` or `None` | Parent node id |
| `rule` | `str` | Path description from root |
| `is_leaf` | `bool` | — |
| `split_feature` | `str` or `None` | Split feature (leaves: `None`) |
| `split_threshold` | `float` or `None` | Split threshold |
| `split_score` | `float` or `None` | Criterion score at this split |
| `left`, `right` | `PanelTreeNode` or `None` | Child references |
| `metrics` | `dict` | E.g. `{"r2":..., "mse":..., "n_samples":...}` |
| `predictor` | `PredictorBase` or `None` | Trained leaf model |
| `sample_ratio` | `float` | Coverage of total samples |
| `elapsed_time` | `float` | Seconds spent building this node |
| `n_samples` (property) | `int` | Samples in this node — *survives* pickling without `sample_indices`. / 持久化后仍可用 |
| `feature_ranking` | `list[tuple]` or `None` | `(feature, threshold, score)` for child priority caching |
| `honest_n_samples` | `int` or `None` | Eval-set size when `honest=True` |

### 5.2 Methods / 方法

```python
node.get_model_weights()  -> Optional[np.ndarray]   # leaf model coefficients
node.get_samples()        -> Optional[np.ndarray]   # row indices belonging to this node
node.to_dict()            -> Dict[str, Any]         # flat serialisation
```

`to_dict()` uses the same column names as `engine.get_node_report()`.

```python
leaf = engine.get_leaves()[0]
print(leaf.n_samples, leaf.metrics)
for name, w in zip(dh.feature_names, leaf.get_model_weights()):
    print(f"  {name}: {w:+.4f}")
```

---

## 6. `NodeEvalResult` / OOS 评估容器

Module: `ptree.engine`. The return type of `engine.evaluate(...)`.

```python
@dataclass
class NodeEvalResult:
    per_node_df:        pd.DataFrame                  # one row per node (BFS order)
    per_node_metrics:   Dict[int, Dict[str, float]]   # node_id → raw metrics dict
    leaf_assignments:   Optional[np.ndarray]          # leaf node_id per input row
    metrics:            Tuple[str, ...]               # actually computed metrics
```

### 6.1 `per_node_df` schema / 表结构

| Column / 列 | Always present? / 是否始终存在 | Meaning |
|---|---|---|
| `node_id`, `depth`, `is_leaf`, `split_feature`, `split_threshold`, `n_oos`, `train_r2` | Yes | Structural & training-side fields |
| `oos_r2` | when `"r2"` requested | Pooled OOS R² over this node |
| `oos_rank_ic_mean`, `oos_rank_ic_ir` | when `"rank_ic"` requested *and* `time_col` provided | Mean / IR of per-period cross-sectional rank-IC |
| `oos_sharpe` | when `"sharpe"` requested *and* `time_col` provided | Period (or criterion-annualised) Sharpe of the node's L/S portfolio |
| `oos_precision`, `oos_f1`, `oos_auc`, `oos_logloss` | when the corresponding metric is in `metrics` | Classification OOS metrics |
| `left_oos_<m>`, `right_oos_<m>`, `delta_oos_<m>` | internal nodes only | Child-side OOS metric and L−R difference, for every requested metric `<m>` |

### 6.2 Typical use / 典型用法

```python
result = engine.evaluate(X_oos, y_oos, time_col="date",
                         metrics=("r2", "rank_ic"))
result.metrics                          # ("r2", "rank_ic")
result.per_node_df.set_index("node_id")
result.per_node_metrics[3]              # {"n_oos": ..., "oos_r2": ..., ...}
result.leaf_assignments                 # ndarray (n,) of leaf ids per row

# Plug into the reporter:
print(NodeReporter(engine).print_tree(evaluation=result, show_child_diff=True))
dot = NodeReporter(engine).to_graphviz(evaluation=result, show_child_diff=True)
```

---

## 7. Ensembles / 集成方法

Module: `ptree.ensemble`.

### 7.1 `PanelForest`

```python
PanelForest(
    n_estimators: int = 100,
    max_features: Optional[Union[str,int,float]] = "sqrt",
    block_size: int = 5,
    aggregate: str = "mean",                # {"mean","consensus","sdf"}
    base_params: Optional[Dict[str, Any]] = None,
    n_jobs: int = 1,
    random_state: Optional[int] = None,
    verbose: int = 0,
    regime_metric: str = "train_r2",        # {"train_r2","auto","oof_r2","rank_ic","precision","f1","auc","logloss"}
    regime_aggregation: str = "train",      # {"train","oof"}
)
```

Bagged ensemble of P-Trees with **time-block bootstrap** + **node-level
random feature subset**. Task-agnostic: a regression criterion in
`base_params` gives a regression forest with Ridge leaves; a
classification criterion auto-switches the default leaf predictor to
`RidgeLogitClassifier`.
基于时间块自助和节点随机特征子集的 P-Forest。
回归 / 分类自动适配（按 `base_params["criterion"]` 探测）。

| Param | Meaning |
|---|---|
| `n_estimators` | Number of trees / 树数 |
| `max_features` | Random feature subset size per node (decorrelation) |
| `block_size` | Consecutive time periods per bootstrap block — should bracket the target's autocorrelation horizon / 时间块长度 |
| `aggregate` | Default `output()` mode: `"mean"` → bagged prediction, `"consensus"` → co-association matrix, `"sdf"` → regime membership |
| `base_params` | Forwarded to each `PanelTreeEngine`. `criterion` defaults to `R2DiffCriterion()`; `predictor` auto-set per task. |
| `regime_metric` | How leaves get ranked into the "high-predictability" set for `regime_membership`. `"train_r2"` / `"auto"` → `criterion.metric_key()`. |
| `regime_aggregation` | `"train"` → leaf metric on bootstrap-train (legacy); `"oof"` → recompute on OOB rows (recommended on noisy panels). |

**Attributes after `fit` / 拟合后属性**

| Attribute | Type | Meaning |
|---|---|---|
| `trees_` | `List[PanelTreeEngine]` | The fitted ensemble |
| `oob_score_` | `float` or `None` | OOB R² (regression) or `criterion.metric_key()` (classification; `logloss` is negated) |
| `is_classification_` | `bool` | Auto-detected task type |

#### `forest.fit`

```python
fit(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    weights: Optional[Union[np.ndarray, pd.Series]] = None,
    time_index: Required[Union[np.ndarray, pd.Series, str]] = ...,
) -> "PanelForest"
```

**`time_index` is mandatory** — the block bootstrap operates on time
periods. A `str` is read as a column of `X`.
**`time_index` 必传**。

#### `forest.predict(X)`

```python
predict(X: pd.DataFrame) -> np.ndarray   # shape (n,)
```

Bagged mean prediction. For classification this is the bagged `P(y=1|X)`
(averages `predict_proba` when available, else 0/1 labels).
装袋平均预测；分类情况下为 `P(y=1|X)`。

#### `forest.predict_proba(X)`

```python
predict_proba(X: pd.DataFrame) -> np.ndarray
```

Classification-only alias of `predict`. Raises on a regression forest.
仅分类森林可用。

#### `forest.regime_membership(X)`

```python
regime_membership(X: pd.DataFrame) -> np.ndarray  # shape (n,), values in [0,1]
```

For each row, the *fraction of trees* that route it into a "high-metric"
leaf (leaf metric ≥ that tree's leaf-median, ranked by
`criterion.metric_key()` and the `regime_metric / regime_aggregation`
config).

This is the **soft mosaic** — the forest-level upgrade of a single tree's
brittle 0/1 high-R² indicator.

返回每个样本被森林路由进高指标叶子的树占比，是单树硬规则的「软马赛克」对应物。

```python
p = forest.regime_membership(X)
X_high = X[p > 0.7]                       # samples deemed "high-predictability"
```

#### `forest.coassociation_matrix(X=None)`

```python
coassociation_matrix(X: Optional[pd.DataFrame] = None) -> np.ndarray   # (n, n)
```

`C[i, j]` = fraction of trees in which rows `i, j` land in the same leaf.
A robust task-agnostic similarity, usable as a precomputed affinity for
spectral clustering. **Memory is O(n²)** — keep `n` modest.
样本两两共叶频率；O(n²) 内存。

#### `forest.output(X)`

Shorthand chosen by `aggregate`: returns `predict` / `coassociation_matrix`
/ `regime_membership`.

### 7.2 `BoostedPanelTree`

```python
BoostedPanelTree(
    n_estimators: int = 50,
    learning_rate: float = 0.1,           # shrinkage ν ∈ (0, 1]
    max_depth: int = 2,
    subsample: float = 1.0,               # ∈ (0, 1]; stochastic boosting on time blocks when <1
    block_size: int = 5,                  # only used when subsample < 1
    criterion: Optional[CriterionBase] = None,
    base_params: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
    verbose: int = 0,
)
```

Forward-stagewise residual boosting: each round refits a fresh shallow
P-Tree on the *residual* of the running ensemble. The split criterion of
each tree is unchanged (defaults to `R2DiffCriterion`).
前向分步残差提升；每轮在残差上重新训练浅 P-Tree。

> **Regression only / 仅限回归任务.** Applying it to 0/1 labels with a
> classification criterion does *not* yield GBDT-classification. For
> classification use `PanelForest`.

#### `boost.fit`

```python
fit(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    weights: Optional[...] = None,
    time_index: Optional[...] = None,   # required only when subsample < 1
) -> "BoostedPanelTree"
```

#### `boost.predict(X)`

```python
predict(X: pd.DataFrame) -> np.ndarray
```

Returns `ν · Σ_m T_m.predict(X)`.

**Attributes / 属性**

| Attribute | Type | Meaning |
|---|---|---|
| `trees_` | `List[PanelTreeEngine]` | Layer-by-layer trees / 逐层弱学习器 |
| `residual_norms_` | `List[float]` | `‖residual‖₂` per round; monotone ↓ then plateaus on self-limited data / 每轮残差 L2 范数 |

---

## 8. Visualization / 可视化

Module: `ptree.visualization`. Two classes — `NodeReporter` (tree summaries)
and `MosaicVisualizer` (prediction mosaic).

### 8.1 `NodeReporter`

```python
NodeReporter(engine: PanelTreeEngine)
```

#### `reporter.summary()`

Returns `engine.get_node_report()` — the full per-node DataFrame.

#### `reporter.leaf_summary()`

Same schema but only the leaf rows.

#### `reporter.print_tree`

```python
print_tree(
    node: Optional[PanelTreeNode] = None,           # default = root
    evaluation: Optional[NodeEvalResult] = None,
    show_child_diff: bool = False,
    metric_keys: Optional[Sequence[str]] = None,
) -> str
```

Box-drawing text tree (`│ ├── └──`). When `evaluation` is supplied, each
node is augmented with OOS metrics (`n_oos`, `OOS R²`, `OOS IC (IR=...)`,
`OOS Sharpe`, `OOS Prec/F1/AUC/LogLoss`). When `show_child_diff=True`, an
extra `↳ split gain | ΔR²=...` line is inserted under each internal node.
文本树。`evaluation` 注入 OOS 指标，`show_child_diff` 在内部节点下增加 Δ 行。

```python
print(reporter.print_tree(evaluation=result, show_child_diff=True))
```

Sample output / 样例输出:

```
[Node 0] char_1 < 0.5 | n=12000, gain=0.457 | R²=0.123 | n_oos=4000 | OOS R²=+0.115
├── [Node 1] char_3 < 0.3 | n=5940, gain=0.179 | R²=0.052 | n_oos=2010 | OOS R²=+0.041
│   ↳ split gain | ΔR²=+0.014 (L=-0.001 vs R=-0.015)
│   ├── [Leaf 3] n=1808 | R²=0.015
│   └── [Leaf 4] n=4132 | R²=0.002
└── [Leaf 5] n=6060 | R²=0.464
```

#### `reporter.to_graphviz`

```python
to_graphviz(
    evaluation: Optional[NodeEvalResult] = None,
    show_child_diff: bool = False,
    leaf_fill: str = "#E8F1FB",
    node_fill: str = "#FFF4DC",
    edge_color: str = "#5B6B7B",
    font_name: str = "Helvetica",
    metric_keys: Optional[Sequence[str]] = None,
) -> str
```

Return Graphviz DOT source. The Graphviz Python package is *not* required
to obtain the source string, only to render it.
返回 Graphviz DOT 源码字符串。

```python
dot = reporter.to_graphviz()
from graphviz import Source
Source(dot).render("tree")
```

#### `reporter.plot_tree`

```python
plot_tree(
    evaluation: Optional[NodeEvalResult] = None,
    show_child_diff: bool = False,
    title: str = "PanelTree structure",
    leaf_color: str = "#3F8AC1",
    node_color: str = "#E0A85A",
    text_color: str = "#1F2A36",
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    metric_keys: Optional[Sequence[str]] = None,
) -> Tuple[Figure, Axes]
```

Pure-matplotlib tree diagram, no Graphviz binary needed. Each node box
shows up to four rows: header, split rule (internal nodes only),
`n = …`, IS criterion metric, OOS / Δ criterion metric.
纯 matplotlib 树图，无需 Graphviz。

### 8.2 `MosaicVisualizer`

```python
MosaicVisualizer(engine: PanelTreeEngine)
```

#### `viz.build_mosaic`

```python
build_mosaic(
    X: pd.DataFrame,
    y: pd.Series,
    time_col: str = "date",
    metric: str = "r2",            # ∈ {"r2","mean","median","std","ic","precision","f1","auc"}
) -> pd.DataFrame
```

| Metric / 指标 | Meaning |
|---|---|
| `"r2"` | Per-cell OOS R² of the leaf's predictor |
| `"mean"` / `"median"` / `"std"` | Per-cell summary statistic of `y` |
| `"ic"` | Per-cell Pearson correlation between prediction and `y` |
| `"precision"` / `"f1"` / `"auc"` | Per-cell classification metric (via `predict_proba` if available) |

**Returns / 返回**: DataFrame with row index = leaf `node_id` (sorted),
columns = sorted unique time periods, values = metric. Empty `(leaf, time)`
cells are `NaN`.

```python
mosaic = viz.build_mosaic(X_proc, y_proc, time_col="date", metric="r2")
print(mosaic.shape)                # (n_leaves, n_periods)
best_leaf_per_period = mosaic.idxmax(axis=0)
```

#### `viz.plot_mosaic`

```python
plot_mosaic(
    mosaic: pd.DataFrame,
    title: Optional[str] = None,
    metric: Optional[str] = None,
    cmap: Optional[str] = None,
    center: Any = None,             # `None` = let the helper decide; pass e.g. `0.0` explicitly
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    max_xticks: int = 12,
    annotate: bool = False,
    cbar_label: Optional[str] = None,
) -> Tuple[Figure, Axes]
```

Render the mosaic via seaborn heatmap. When `cmap` is left `None`, a
sensible default is chosen — sequential `viridis` for non-negative data
(`r2`, `precision`, `auc`), diverging `RdBu_r` centred at `0` otherwise.
当 `cmap=None` 时按数据范围自动选 cmap：非负数据用 `viridis`，否则用 `RdBu_r` 居中 0。

```python
fig, ax = viz.plot_mosaic(mosaic, metric="r2", save_path="mosaic.png")
```

---

## 9. Logging / 日志

The engine, forest and boost classes log through the standard
`logging` module under the logger name `"ptree"`. Enable verbose tracing
with the corresponding `verbose` constructor argument (`0` silent,
`1` per-level / per-fit summary, `2` per-candidate (engine only)).
引擎 / 森林 / 提升均使用 `logging`，logger 名为 `"ptree"`。`verbose` 控制详细级别。

```python
import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

engine = PanelTreeEngine(..., verbose=1)
engine.fit(X, y, feature_names=...)
```

Typical output / 典型输出:

```
[INFO] [Level 0] Splitting Node 0...
  - Best Split: 'char_1' at threshold 0.5000
  - Metric Delta: score = 0.456896
  - Left: 5940 samples | Right: 6060 samples
[INFO] Tree built: 15 nodes, 8 leaves, max_depth=3
```

---

## 10. Version Migration / 版本迁移

The API additions you most often need to know about. (v0.1 defaults remain
bit-for-bit reproducible — anything *new* is opt-in.)
以下功能均为新增，启用前 v0.1 默认行为保持逐位可复现。

| Since / 起始版本 | API |
|---|---|
| **v0.2** | `engine.evaluate(...) -> NodeEvalResult` |
| v0.2 | `engine.predict_leaves(X)`, `engine.predict_node_path(X)` |
| v0.2 | `engine.tune_ccp_alpha(X_val, y_val, metric="r2")` |
| v0.2 | `RankICDiffCriterion(...)`, `WeightedR2DiffCriterion(min_child_weight=...)` |
| v0.2 | `PanelForest(regime_metric="rank_ic", regime_aggregation="oof")` |
| v0.2 | `NodeReporter.print_tree(evaluation=..., show_child_diff=True)`, `NodeReporter.to_graphviz(...)` |
| v0.2 | `MosaicVisualizer.plot_mosaic(... center=...)`, default cmap → `"RdBu_r"` (diverging data) / `"viridis"` (non-negative) |
| v0.2 | `node.n_samples` persists across `pickle` (no longer depends on `metrics["n_samples"]`) |
| v0.3 | `engine.build_sdf_factor(...)` — tangency-weighted SDF return series from leaf portfolios |
| v0.3 | `MeanVarianceCriterion(metric_key() → "sharpe")` exposes per-leaf Sharpe in `node.metrics["sharpe"]` and in `engine.evaluate(metrics=("sharpe",))` |
| v0.3 | `NodeReporter.plot_tree(...)` — pure-matplotlib tree diagram |
| v0.3 | `PanelForest` task auto-detection + `predict_proba`, classification leaves auto-set to `RidgeLogitClassifier` |
| v0.3 | `engine.evaluate(metrics=...)` now accepts `"sharpe", "precision", "f1", "auc", "logloss"` |

---

> If a public symbol is missing from this manual, please open an issue or
> a PR adding it. Internal helpers (functions or attributes starting with
> `_`) are intentionally undocumented and may change without notice.
> 若发现公共 API 缺失，欢迎提交 issue 或 PR。下划线开头的内部成员不在本手册范围内，
> 可能随版本变动。
