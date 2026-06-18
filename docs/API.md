# P-Tree API Reference

> **Scope.** This document is a *flat, dictionary-style reference* for every
> public class, method, parameter and return value exposed by the `ptree`
> package (`v0.3.x`). Use it as a lookup table; for conceptual background
> see [`README.md`](../README.md) and for the algorithmic specification
> see [`ALGORITHM.md`](ALGORITHM.md).
>
> Every entry follows the same template:
> *Signature → Purpose → Parameters → Returns → Example.*

---

## Table of Contents

- [0. Overview & Cheat Sheet](#0-overview--cheat-sheet)
- [1. `DataHandler`](#1-datahandler)
- [2. Predictors](#2-predictors)
- [3. Criteria](#3-criteria)
- [4. `PanelTreeEngine`](#4-paneltreeengine)
- [5. `PanelTreeNode`](#5-paneltreenode)
- [6. `NodeEvalResult`](#6-nodeevalresult)
- [7. Ensembles](#7-ensembles)
- [8. Visualization](#8-visualization)
- [9. Logging](#9-logging)
- [10. Version Migration](#10-version-migration)

---

## 0. Overview & Cheat Sheet

### 0.1 Top-level imports

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

| Name | Description |
|---|---|
| `DataHandler` | Panel preprocessing: alignment, fillna, cross-sectional rank, vol weights |
| `RidgeRegressor` | Closed-form L2 leaf regressor |
| `VolWeightedRidgeRegressor` | Inverse-vol weighted Ridge |
| `RidgeLogitClassifier` | IRLS-based Ridge logit (with `predict_proba`) |
| `ElasticNetRegressor` | L1+L2 coordinate descent |
| `PLSRegressor` | NIPALS partial least squares |
| `SelfDefinedPredictor` | User-defined leaf model template |
| `R2DiffCriterion` | `\|R²_L − R²_R\|` (default) |
| `WeightedR2DiffCriterion` | Balance + shrinkage stabilised R² diff |
| `RankICDiffCriterion` | `\|IC_L − IC_R\|` (scale-free) |
| `MeanVarianceCriterion` | Tangency-portfolio Sharpe of two children |
| `ClassificationCriterion` | Classification metric diff |
| `PanelTreeEngine` | The fit / predict / prune / evaluate engine |
| `PanelTreeNode` | Per-node metadata container |
| `NodeEvalResult` | Per-node OOS metrics container |
| `PanelForest` | Bagged ensemble (P-Forest) |
| `BoostedPanelTree` | Residual-boosted ensemble (P-Boost) |
| `NodeReporter` | Text / DataFrame / Graphviz / mpl tree reports |
| `MosaicVisualizer` | Prediction-mosaic heatmap |

### 0.2 Task → API cheat sheet

| I want to … | Call |
|---|---|
| Preprocess a raw `(date, asset_id, …)` DataFrame | `DataHandler().fit_transform(...)` |
| Build the tree | `PanelTreeEngine(...).fit(X, y, feature_names=...)` |
| Predict on new data | `engine.predict(X)` |
| Route rows to leaf ids | `engine.predict_leaves(X)` |
| Get the root-to-leaf node id path | `engine.predict_node_path(X)` |
| Compute per-node OOS R² / Rank-IC / Sharpe | `engine.evaluate(X_oos, y_oos, time_col=...)` |
| Inspect every node as a DataFrame | `engine.get_node_report()` or `NodeReporter(engine).summary()` |
| Inspect only leaves | `NodeReporter(engine).leaf_summary()` |
| Get the original sample indices in each leaf | `engine.get_leaf_samples()` |
| Post-prune the tree | `engine.prune(ccp_alpha)` / `engine.tune_ccp_alpha(...)` |
| Build a tradeable SDF factor from leaves | `engine.build_sdf_factor(...)` |
| Print a pretty text tree | `NodeReporter(engine).print_tree()` |
| Export a Graphviz DOT diagram | `NodeReporter(engine).to_graphviz()` |
| Plot a matplotlib tree | `NodeReporter(engine).plot_tree()` |
| Prediction mosaic | `MosaicVisualizer(engine).build_mosaic(...)` → `plot_mosaic(...)` |
| Soft regime probability via forest | `PanelForest(...).regime_membership(X)` |
| Consensus similarity for clustering | `PanelForest(...).coassociation_matrix(X)` |
| OOB validation R² | `PanelForest(...).oob_score_` |
| Boosted (multi-layer) regime discovery | `BoostedPanelTree(...).fit(...).predict(...)` |

---

## 1. `DataHandler`

Module: `ptree.data_handler`

### 1.1 `DataHandler.__init__`

**Signature**

```python
DataHandler(
    cs_rank_standardize: bool = True,
    vol_window: int = 60,
    min_obs: int = 20,
    fillna_method: Optional[str] = "ffill",
)
```

**Purpose.**
Panel preprocessing: align `(X, y)`, fill missing values, optionally
cross-sectionally rank-standardise features to `[0, 1]`, and (optionally)
compute inverse-volatility weights from a return series.

| Param | Type | Default | Meaning |
|---|---|---|---|
| `cs_rank_standardize` | `bool` | `True` | Apply per-cross-section rank → `[0,1]` mapping |
| `vol_window` | `int` | `60` | Rolling window for realised volatility |
| `min_obs` | `int` | `20` | Min non-NaN obs to produce a vol value |
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

### 1.3 `DataHandler.transform`

```python
transform(
    X: pd.DataFrame,
    y: pd.Series,
    ret_series_for_vol: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]
```

Apply the pipeline.

**Returns**
- `X_processed`: cleaned (and optionally rank-standardised) feature panel,
  retains `time_col`/`entity_col` columns.
- `y_processed`: target aligned with `X_processed`.
- `vol_weights`: inverse-vol weights (`1/σ`) or `None`.

### 1.4 `DataHandler.fit_transform`

```python
fit_transform(X, y, time_col="date", entity_col="asset_id",
              ret_series_for_vol=None) -> (X_proc, y_proc, vol_weights)
```

Convenience wrapper: `fit` then `transform`.

### 1.5 Property `feature_names`

```python
dh.feature_names  # -> List[str]
```

Feature column names learnt during `fit` (excludes `time_col`/`entity_col`).

### 1.6 Example

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

## 2. Predictors

Module: `ptree.predictors`. All predictors inherit from `PredictorBase`.

### 2.1 `PredictorBase` (abstract)

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

| Method | Returns | Notes |
|---|---|---|
| `fit(X, y, weights=None)` | `self` | Weights are optional; closed-form predictors also accept cached `XtWX`/`XtWy` via keyword arguments for incremental updates. |
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

| Param | Type | Default | Meaning |
|---|---|---|---|
| `alpha` | `float` | `1.0` | L2 regularisation strength |
| `fit_intercept` | `bool` | `True` | Prepend a constant column |

### 2.3 `VolWeightedRidgeRegressor`

```python
VolWeightedRidgeRegressor(alpha: float = 1.0, fit_intercept: bool = True)
```

Same closed form as `RidgeRegressor`, expected to be called with
inverse-vol weights (`1/σ`). The `weights` argument is what carries the
heteroscedasticity correction.

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

### 2.6 `PLSRegressor`

```python
PLSRegressor(n_components: int = 2, fit_intercept: bool = True)
```

NIPALS single-response partial least squares. Effective when characteristics
are strongly collinear. Also bypasses the incremental fast path.

### 2.7 `SelfDefinedPredictor`

Template base class — subclass it, implement `fit` and `predict`, and pass an
instance to `PanelTreeEngine(predictor=...)`.

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

## 3. Criteria

Module: `ptree.criteria`. All criteria implement `calculate_score` and
`metric_key`.

### 3.1 `CriterionBase` (abstract)

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

### 3.2 `R2DiffCriterion`

```python
R2DiffCriterion(weight_by_size: bool = False)
```

Score: `score = |R²_L − R²_R|` (default), optionally multiplied by
`min(n_L, n_R) / (n_L + n_R)`.

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

`score = |R²_L − R²_R| · balance · shrinkage`, with
balance = `min(n_L,n_R)/(n_L+n_R)`, shrinkage = `(n_min−1)/(n_min−1+k)`.
`min_child_weight` is a *hard floor* on the smaller child's sample share —
sliver splits score `0`.

| Param | Default | Meaning |
|---|---|---|
| `balance` | `True` | Apply the balance penalty |
| `shrinkage_k` | `0.0` | Sample-size shrinkage strength |
| `use_adjusted_r2` | `False` | Use adjusted-R² with `n_features` |
| `min_child_weight` | `0.0` | Hard sample-share floor on smaller child |

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

- `metric_key() → "r2"` (kept for node logging; the criterion actually
  operates on the per-time series payload).

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

- `metric_key() → "sharpe"` (each leaf's long-short Sharpe is stored in
  `node.metrics["sharpe"]`).

### 3.6 `ClassificationCriterion`

```python
ClassificationCriterion(
    metric: str = "precision",        # ∈ {"precision","f1","auc","logloss"}
    weight_by_size: bool = False,
)
```

`score = |metric_L − metric_R|`. `metric_key()` returns the chosen `metric`.

### 3.7 Helper functions

| Function | Returns |
|---|---|
| `evaluate_regression(y_true, y_pred, weights=None)` | `{"r2", "mse", "n_samples", "n_features"}` |
| `evaluate_classification(y_true, y_proba, threshold=0.5)` | `{"precision", "f1", "auc", "logloss", "n_samples"}` |

These are usually called by the engine internally; you can also call them
directly for ad-hoc metric computation.

---

## 4. `PanelTreeEngine`

Module: `ptree.engine`.

### 4.1 Constructor

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

Parameters grouped by role:

**Core**

| Param | Default | Meaning |
|---|---|---|
| `predictor` | `RidgeRegressor` | Leaf-model class or instance. If a class, instantiated with `predictor_params`. |
| `criterion` | `R2DiffCriterion()` | Splitting criterion |
| `max_depth` | `3` | Max tree depth |
| `min_samples` | `100` | Min samples for a node to be splittable |
| `min_impurity_decrease` | `0.0` | Best-split score floor; below this the node becomes a leaf |

**Threshold search**

| Param | Default | Meaning |
|---|---|---|
| `split_thresholds` | `[0.3, 0.5, 0.7]` | Global fixed list, *or* `"adaptive"` for per-node quantile thresholds |
| `adaptive_quantiles` | `[0.25, 0.5, 0.75]` | Quantiles used when `split_thresholds="adaptive"` |
| `splitter` | `"best"` | `"best"` = exhaustive; `"random"` = Extra-Trees random thresholds |
| `n_random_splits` | `1` | Random thresholds per feature when `splitter="random"` |
| `max_features` | `None` | Random feature subset per node: `None / "sqrt" / "log2" / int / float`. Used by ensembles to decorrelate trees |

**Honest splits**

| Param | Default | Meaning |
|---|---|---|
| `honest` | `False` | Enable honest split (fit-set + eval-set partition inside each node) |
| `honest_frac` | `0.5` | Fraction held out as eval-set in `(0, 1)` |
| `honest_refit_full` | `True` | After choosing the split, refit leaf models on the full sample |
| `random_state` | `None` | Seed for honest split, random splitter and feature subset |

**Performance**

| Param | Default | Meaning |
|---|---|---|
| `fast_mode` | `False` | Feature-priority caching from parent |
| `early_stopping_threshold` | `None` | Early-stop feature loop when best ≥ threshold (requires `fast_mode`) |
| `n_jobs` | `1` | Feature-dim parallel workers (`-1` = all cores) |
| `parallel_backend` | `"threads"` | joblib backend: `"threads"` or `"processes"` |
| `keep_node_stats` | `False` | Keep per-node cached `XtWX/XtWy` after splitting |

**Logging**

| Param | Default | Meaning |
|---|---|---|
| `verbose` | `1` | `0` = silent, `1` = per-level, `2` = per-candidate |
| `predictor_params` | `None` | Used only when `predictor` is a class, to instantiate it |

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

**Returns**: `self` (chaining).

### 4.3 Prediction APIs

#### `engine.predict(X)`

```python
predict(X: pd.DataFrame) -> np.ndarray  # shape (n,)
```

Route each row to its leaf and apply the leaf's predictor.

#### `engine.predict_leaves(X)`

```python
predict_leaves(X: pd.DataFrame) -> np.ndarray[int]  # shape (n,)
```

Return the leaf `node_id` each row falls into.

#### `engine.predict_node_path(X)`

```python
predict_node_path(X: pd.DataFrame) -> List[List[int]]
```

For each row, the ordered list of `node_id`s from root to leaf — useful for
explaining a specific prediction.

```python
paths = engine.predict_node_path(X.head(3))
# paths[0] == [0, 1, 4]  → root → node 1 → leaf 4
```

### 4.4 `engine.evaluate` — per-node OOS diagnostics

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

| Param | Notes |
|---|---|
| `time_col` | Required for `"rank_ic"` and `"sharpe"`; silently dropped otherwise. `str` → column name in `X`. |
| `metrics` | Subset of `{"r2","rank_ic","sharpe","precision","f1","auc","logloss"}`. |
| `weights` | Forwarded to weighted R². |

**Returns**: a [`NodeEvalResult`](#6-nodeevalresult).

```python
result = engine.evaluate(X_oos, y_oos, time_col="date",
                         metrics=("r2", "rank_ic"))
print(result.per_node_df.head())
print(reporter.print_tree(evaluation=result, show_child_diff=True))
```

### 4.5 Node inspection

| Method | Returns |
|---|---|
| `engine.get_leaves()` | `List[PanelTreeNode]` — all leaves |
| `engine.get_all_nodes()` | `List[PanelTreeNode]` — BFS order |
| `engine.get_node_report()` | `pd.DataFrame` — one row per node (see schema below) |
| `engine.get_leaf_samples()` | `Dict[int, np.ndarray]` — leaf id → row indices |

`get_node_report()` columns:

| Column | Description |
|---|---|
| `Node_ID` | Node id |
| `Depth` | Depth, root = 0 |
| `Rule` | Path rule string `"root & x1 < 0.5 & x3 >= 0.7"` |
| `Is_Leaf` | `bool` |
| `N_Samples` | In-node sample count |
| `Sample_Ratio` | `N_Samples / total` |
| `Split_Feature` | Split feature name (NaN for leaves) |
| `Split_Threshold` | Split threshold value |
| `Split_Score` | Criterion score at this split |
| `Predictability_Score` | `r2` (regression) or `precision` (classification) |
| `Metrics` | Raw metrics dict |
| `Model_Weights` | Leaf-model coefficient list |
| `Elapsed_Time_s` | Seconds spent building this node |
| `Parent_ID` | Parent node id, `None` for root |

### 4.6 Pruning

#### `engine.prune(ccp_alpha)`

```python
prune(ccp_alpha: float) -> "PanelTreeEngine"
```

Bottom-up cost-complexity pruning *in place*. A subtree at node $v$ is
collapsed when its cumulative split-score gain does not exceed
`ccp_alpha · (n_leaves(v) − 1)`. Leaf predictors are retained, so
`predict` keeps working.

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

**Returns**: `(best_alpha, curve_df)` where `curve_df` has columns
`ccp_alpha`, `n_leaves`, `oos_<metric>`.

### 4.7 `engine.build_sdf_factor` — SDF factor construction

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

**Returns** (dict):

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

## 5. `PanelTreeNode`

Module: `ptree.node`. Returned by `engine.get_leaves()` / `get_all_nodes()`.

### 5.1 Attributes

| Attribute | Type | Meaning |
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
| `n_samples` (property) | `int` | Samples in this node — *survives* pickling without `sample_indices`. |
| `feature_ranking` | `list[tuple]` or `None` | `(feature, threshold, score)` for child priority caching |
| `honest_n_samples` | `int` or `None` | Eval-set size when `honest=True` |

### 5.2 Methods

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

## 6. `NodeEvalResult`

Module: `ptree.engine`. The return type of `engine.evaluate(...)`.

```python
@dataclass
class NodeEvalResult:
    per_node_df:        pd.DataFrame                  # one row per node (BFS order)
    per_node_metrics:   Dict[int, Dict[str, float]]   # node_id → raw metrics dict
    leaf_assignments:   Optional[np.ndarray]          # leaf node_id per input row
    metrics:            Tuple[str, ...]               # actually computed metrics
```

### 6.1 `per_node_df` schema

| Column | Always present? | Meaning |
|---|---|---|
| `node_id`, `depth`, `is_leaf`, `split_feature`, `split_threshold`, `n_oos`, `train_r2` | Yes | Structural & training-side fields |
| `oos_r2` | when `"r2"` requested | Pooled OOS R² over this node |
| `oos_rank_ic_mean`, `oos_rank_ic_ir` | when `"rank_ic"` requested *and* `time_col` provided | Mean / IR of per-period cross-sectional rank-IC |
| `oos_sharpe` | when `"sharpe"` requested *and* `time_col` provided | Period (or criterion-annualised) Sharpe of the node's L/S portfolio |
| `oos_precision`, `oos_f1`, `oos_auc`, `oos_logloss` | when the corresponding metric is in `metrics` | Classification OOS metrics |
| `left_oos_<m>`, `right_oos_<m>`, `delta_oos_<m>` | internal nodes only | Child-side OOS metric and L−R difference, for every requested metric `<m>` |

### 6.2 Typical use

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

## 7. Ensembles

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

| Param | Meaning |
|---|---|
| `n_estimators` | Number of trees |
| `max_features` | Random feature subset size per node (decorrelation) |
| `block_size` | Consecutive time periods per bootstrap block — should bracket the target's autocorrelation horizon |
| `aggregate` | Default `output()` mode: `"mean"` → bagged prediction, `"consensus"` → co-association matrix, `"sdf"` → regime membership |
| `base_params` | Forwarded to each `PanelTreeEngine`. `criterion` defaults to `R2DiffCriterion()`; `predictor` auto-set per task. |
| `regime_metric` | How leaves get ranked into the "high-predictability" set for `regime_membership`. `"train_r2"` / `"auto"` → `criterion.metric_key()`. |
| `regime_aggregation` | `"train"` → leaf metric on bootstrap-train (legacy); `"oof"` → recompute on OOB rows (recommended on noisy panels). |

**Attributes after `fit`**

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

#### `forest.predict(X)`

```python
predict(X: pd.DataFrame) -> np.ndarray   # shape (n,)
```

Bagged mean prediction. For classification this is the bagged `P(y=1|X)`
(averages `predict_proba` when available, else 0/1 labels).

#### `forest.predict_proba(X)`

```python
predict_proba(X: pd.DataFrame) -> np.ndarray
```

Classification-only alias of `predict`. Raises on a regression forest.

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

> **Regression only.** Applying it to 0/1 labels with a classification
> criterion does *not* yield GBDT-classification. For classification use
> `PanelForest`.

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

**Attributes**

| Attribute | Type | Meaning |
|---|---|---|
| `trees_` | `List[PanelTreeEngine]` | Layer-by-layer trees |
| `residual_norms_` | `List[float]` | `‖residual‖₂` per round; monotone ↓ then plateaus on self-limited data |

---

## 8. Visualization

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

```python
print(reporter.print_tree(evaluation=result, show_child_diff=True))
```

Sample output:

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

| Metric | Meaning |
|---|---|
| `"r2"` | Per-cell OOS R² of the leaf's predictor |
| `"mean"` / `"median"` / `"std"` | Per-cell summary statistic of `y` |
| `"ic"` | Per-cell Pearson correlation between prediction and `y` |
| `"precision"` / `"f1"` / `"auc"` | Per-cell classification metric (via `predict_proba` if available) |

**Returns**: DataFrame with row index = leaf `node_id` (sorted),
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

```python
fig, ax = viz.plot_mosaic(mosaic, metric="r2", save_path="mosaic.png")
```

---

## 9. Logging

The engine, forest and boost classes log through the standard
`logging` module under the logger name `"ptree"`. Enable verbose tracing
with the corresponding `verbose` constructor argument (`0` silent,
`1` per-level / per-fit summary, `2` per-candidate (engine only)).

```python
import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

engine = PanelTreeEngine(..., verbose=1)
engine.fit(X, y, feature_names=...)
```

Typical output:

```
[INFO] [Level 0] Splitting Node 0...
  - Best Split: 'char_1' at threshold 0.5000
  - Metric Delta: score = 0.456896
  - Left: 5940 samples | Right: 6060 samples
[INFO] Tree built: 15 nodes, 8 leaves, max_depth=3
```

---

## 10. Version Migration

The API additions you most often need to know about. (v0.1 defaults remain
bit-for-bit reproducible — anything *new* is opt-in.)

| Since | API |
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
