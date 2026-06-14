# Panel Tree (P-Tree)

[![PyPI version](https://badge.fury.io/py/aptree.svg)](https://pypi.org/project/aptree/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **supervised clustering algorithm** designed for **panel data**, commonly used in quantitative finance to identify time-varying, cross-sectional predictability regimes.

> **What's new in v0.2** — Major usability + diagnostics upgrade:
> - `engine.evaluate(X_oos, y_oos, time_col, metrics=("r2", "rank_ic"))` →
>   per-node OOS R² / Rank-IC in a single call (new `NodeEvalResult` container).
> - `NodeReporter.print_tree(evaluation=..., show_child_diff=True)` and
>   `NodeReporter.to_graphviz(...)` for richer tree visualisation.
> - `engine.predict_leaves(X)` / `engine.predict_node_path(X)` — public
>   routing API (no more re-implementing private helpers).
> - `engine.tune_ccp_alpha(X_val, y_val, metric="r2")` — pruning-curve
>   tuning in one line.
> - New `RankICDiffCriterion` (scale-free, robust on low-SNR panels);
>   `WeightedR2DiffCriterion` now exposes `min_child_weight`.
> - `PanelForest(regime_metric="rank_ic", regime_aggregation="oof")` —
>   OOB-driven regime scoring instead of brittle train-R².
> - `MosaicVisualizer.plot_mosaic` default cmap → `"RdBu_r"` (colour-blind
>   safe) with explicit `center` kwarg.
> - `node.n_samples` now persists across `pickle` (no more dependency on
>   `metrics["n_samples"]` fallback).
>
> All v0.1 defaults are preserved — existing code keeps producing identical
> trees. 
## Installation

```bash
pip install aptree

# With visualization support (matplotlib, seaborn)
pip install aptree[viz]

# For development
pip install aptree[dev]
```

> **Note:** The PyPI distribution name is `aptree`, while the import name remains `ptree` (e.g. `from ptree import PanelTreeEngine`).

## Core Idea

P-Tree recursively splits the full sample into disjoint leaf nodes using asset characteristics or macro states as thresholds. Unlike standard decision trees that minimise residual MSE, P-Tree **maximises the difference in predictive performance across child nodes**, producing a *prediction mosaic* — a map showing where and when alpha is concentrated.

### Key Differentiators

| Feature | Standard Decision Tree | P-Tree |
|---------|----------------------|--------|
| **Objective** | Minimise residual MSE/Gini | Maximise predictability difference |
| **Leaf Model** | Constant (mean) | Ridge regression / Logit |
| **Use Case** | Point prediction | Regime identification |
| **Output** | Single prediction | Prediction mosaic |

### Algorithm Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Full Sample                              │
│                     (all time × assets)                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         │     For each (feature, threshold): │
         │     1. Split into Left & Right     │
         │     2. Fit Ridge on each subset    │
         │     3. Compute R² for each         │
         │     4. Score = |R²_L - R²_R|       │
         └─────────────────┬─────────────────┘
                           │
              Select split with max score
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
   ┌──────────┐                       ┌──────────┐
   │ Left Node│                       │Right Node│
   │ (low val)│                       │(high val)│
   └────┬─────┘                       └────┬─────┘
        │                                  │
        ▼                                  ▼
   Recurse or                         Recurse or
   become Leaf                        become Leaf
```

## Project Structure

```
src/ptree/
├── __init__.py          # Package exports
├── data_handler.py      # DataHandler – alignment, missing-value fill, rank standardisation, volatility
├── predictors.py        # PredictorBase, RidgeRegressor, VolWeightedRidgeRegressor, RidgeLogitClassifier, ElasticNetRegressor, PLSRegressor, SelfDefinedPredictor
├── criteria.py          # CriterionBase, R2DiffCriterion, WeightedR2DiffCriterion, MeanVarianceCriterion, ClassificationCriterion, evaluation helpers
├── node.py              # PanelTreeNode – per-node metadata container
├── engine.py            # PanelTreeEngine – recursive splitting, cost-complexity pruning, honest splits, incremental matrix updates, feature-priority caching, joblib parallelism
├── ensemble.py          # PanelForest (P-Forest bagging), BoostedPanelTree (P-Boost residual boosting)
└── visualization.py     # NodeReporter (text/DataFrame reports), MosaicVisualizer (heatmap)
```


## Quick Start

```python
import numpy as np
import pandas as pd
from ptree import (
    DataHandler, RidgeRegressor,
    R2DiffCriterion, WeightedR2DiffCriterion, RankICDiffCriterion,
    PanelTreeEngine, NodeReporter, MosaicVisualizer,
)

# 1. Prepare panel data (DataFrame with date, asset_id, and feature columns)
dh = DataHandler(cs_rank_standardize=True)
X, y, vol_weights = dh.fit_transform(
    df, y_series,
    time_col="date", entity_col="asset_id",
    ret_series_for_vol=ret_series,       # optional, for VolWeightedRidge
)

# 2. Build the tree.
#    Recommended for low-SNR financial panels:
#      WeightedR2DiffCriterion(balance=True, shrinkage_k=200,
#                              min_child_weight=0.05)
#    plus ``split_thresholds="adaptive"`` (per-node quantile thresholds).
#    The default R2DiffCriterion is preserved for v0.1 reproducibility.
engine = PanelTreeEngine(
    predictor=RidgeRegressor(alpha=1.0),
    criterion=WeightedR2DiffCriterion(balance=True, shrinkage_k=200),
    split_thresholds="adaptive",
    adaptive_quantiles=[0.25, 0.5, 0.75],
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

# 4. NEW in v0.2: per-node OOS evaluation in a single call.
result = engine.evaluate(
    X_oos, y_oos,
    time_col="date",
    metrics=("r2", "rank_ic"),
)
print(result.per_node_df.head())
# Overlay OOS metrics on the text tree + show child-difference rows
print(reporter.print_tree(evaluation=result, show_child_diff=True))
# Or export a Graphviz DOT source for prettier rendering
dot = reporter.to_graphviz(evaluation=result, show_child_diff=True)

# 5. NEW in v0.2: public leaf-routing API (no more private helpers).
leaf_ids = engine.predict_leaves(X_new)
paths    = engine.predict_node_path(X_new.iloc[:5])

# 6. NEW in v0.2: cost-complexity pruning tuned by OOS R² in one line.
best_alpha, curve = engine.tune_ccp_alpha(X_val, y_val, metric="r2")
engine.prune(best_alpha)

# 7. Prediction mosaic. v0.2 changes the default cmap to ``"RdBu_r"``
#    (colour-blind safe) and exposes an explicit ``center`` kwarg.
viz = MosaicVisualizer(engine)
mosaic = viz.build_mosaic(X, y, time_col="date", metric="r2")
fig, ax = viz.plot_mosaic(mosaic)       # requires matplotlib & seaborn

# 8. Retrieve leaf-node samples
for leaf_id, indices in engine.get_leaf_samples().items():
    print(f"Leaf {leaf_id}: {len(indices)} observations")
```

## Module Overview

### DataHandler

Handles panel data preprocessing including alignment, missing value imputation, cross-sectional rank standardisation, and volatility computation.

| Parameter | Default | Description |
|---|---|---|
| `cs_rank_standardize` | `True` | Cross-sectional rank normalisation to [0, 1] |
| `vol_window` | `60` | Rolling window for volatility computation |
| `min_obs` | `20` | Minimum observations for volatility calculation |
| `fillna_method` | `"ffill"` | Missing-value strategy (`ffill`, `bfill`, `zero`, `mean`, `None`) |

### Predictors

All predictors inherit from `PredictorBase` and implement `fit()` / `predict()`.

| Class | Use Case |
|---|---|
| `RidgeRegressor` | Standard Ridge regression (closed-form) |
| `VolWeightedRidgeRegressor` | Inverse-volatility weighted Ridge (handles heteroscedasticity) |
| `RidgeLogitClassifier` | Ridge logistic regression via IRLS |
| `ElasticNetRegressor` | L1+L2 coordinate-descent regression (sparse factor selection) |
| `PLSRegressor` | Partial least squares (handles highly-correlated factors) |
| `SelfDefinedPredictor` | User-defined model base class |


**Custom Predictor Example:**

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

### Criteria

Split-quality criteria evaluate whether a candidate split produces child nodes with meaningfully different predictability.

| Class | Description |
|---|---|
| `R2DiffCriterion` | Maximise \|R²_L − R²_R\| (regression, default — fast & minimal; preserved for backward compatibility) |
| `WeightedR2DiffCriterion` | **Recommended for real financial panels.** \|R²_L − R²_R\| with balance + sample-size shrinkage + adjusted-R² penalties + `min_child_weight` floor — directly addresses the "small-n child wins on inflated R²" failure mode of `R2DiffCriterion` |
| `RankICDiffCriterion` (**new in v0.2**) | \|mean cross-sectional rank-IC_L − mean rank-IC_R\| — scale-free, robust on low-SNR / non-Gaussian targets (requires `fit(..., time_index=...)`) |
| `MeanVarianceCriterion` | Tangency (max) Sharpe of the two child long-short portfolios — aligns splits with the SDF / efficient-frontier objective (requires `fit(..., time_index=...)`) |
| `ClassificationCriterion` | Maximise difference in Precision / F1 / AUC / LogLoss (classification) |

> **Recommendation for quant panels:**
> use `WeightedR2DiffCriterion(balance=True, shrinkage_k=200, min_child_weight=0.05)`
> (or `RankICDiffCriterion()` for very low-SNR targets) together with
> `split_thresholds="adaptive"`. The default `R2DiffCriterion` is preserved
> to keep v0.1.x golden baselines reproducible bit-for-bit.


### PanelTreeEngine

The main engine for building and querying Panel Trees.

| Parameter | Default | Description |
|---|---|---|
| `predictor` | `RidgeRegressor` | Leaf-node predictor (instance or class) |
| `criterion` | `R2DiffCriterion()` | Split-quality criterion |
| `split_thresholds` | `[0.3, 0.5, 0.7]` | Candidate split points on (rank-standardised) feature values. Pass `"adaptive"` to use per-node, per-feature quantile thresholds instead |
| `adaptive_quantiles` | `[0.25, 0.5, 0.75]` | Quantiles used when `split_thresholds="adaptive"` |
| `max_depth` | `3` | Maximum tree depth |
| `min_samples` | `100` | Minimum observations per node |
| `min_impurity_decrease` | `0.0` | Minimum criterion score the best split must reach; below it the node becomes a leaf |
| `honest` | `False` | Honest splits — fit leaf models on one in-node subset and evaluate split quality on a disjoint subset, removing the selection bias of fitting and scoring on the same data |
| `honest_frac` | `0.5` | Fraction of in-node samples held out as the honest evaluation set |
| `honest_refit_full` | `True` | Refit each final leaf model on the full in-node sample after honest split selection |
| `random_state` | `None` | Seed for honest splitting, random feature subsetting and the random splitter (reproducibility) |
| `fast_mode` | `False` | Enable feature-priority caching from parent nodes |
| `early_stopping_threshold` | `None` | Stop searching if criterion exceeds this value (requires `fast_mode`) |
| `n_jobs` | `1` | Parallel workers (`-1` = all cores), used for feature-dimension parallelism (requires `joblib`) |
| `parallel_backend` | `"threads"` | joblib backend for parallel feature evaluation (`"threads"` or `"processes"`) |
| `max_features` | `None` | Node-level random feature-subset size for splits (`"sqrt"`, `"log2"`, int, float or `None`); used by ensembles to decorrelate trees |
| `splitter` | `"best"` | `"best"` exhaustively scans `split_thresholds`; `"random"` draws random thresholds (Extra-Trees style) |
| `n_random_splits` | `1` | Number of random thresholds drawn per feature when `splitter="random"` |
| `keep_node_stats` | `False` | Retain per-node cached matrices after splitting (uses more memory; useful for debugging) |
| `verbose` | `1` | Logging verbosity (0=silent, 1=per-level, 2=per-candidate) |

**Pruning / honest helpers**

| Method | Description |
|---|---|
| `engine.prune(ccp_alpha)` | Cost-complexity post-pruning: bottom-up, collapse subtrees whose score gain does not justify their leaf-count penalty `ccp_alpha` |
| `engine.cost_complexity_pruning_path()` | Return `(ccp_alphas, n_leaves, scores)` to help select `ccp_alpha` |


## Output & Query API Reference

P-Tree provides rich output and query interfaces across four main classes: `PanelTreeEngine`, `PanelTreeNode`, `NodeReporter`, and `MosaicVisualizer`.

---

### PanelTreeEngine Methods

#### `engine.predict(X) → np.ndarray`

Generate per-sample predictions on new data. Each observation traverses down the tree to its corresponding leaf node, which provides the prediction using its local model.

```python
preds = engine.predict(X_proc)  # shape: (n_samples,)
```

#### `engine.get_leaves() → List[PanelTreeNode]`

Return a list of all leaf node objects.

```python
for leaf in engine.get_leaves():
    print(f"Leaf {leaf.node_id}: R²={leaf.metrics.get('r2', None):.4f}, n={leaf.n_samples}")
```

#### `engine.get_all_nodes() → List[PanelTreeNode]`

Return all nodes in the tree (BFS order), including both internal nodes and leaves.

```python
all_nodes = engine.get_all_nodes()
print(f"Total nodes: {len(all_nodes)}")
```

#### `engine.get_node_report() → pd.DataFrame`

Return a structured DataFrame with one row per node containing the following columns:

| Column | Description |
|---|---|
| `Node_ID` | Unique node identifier |
| `Depth` | Node depth (root = 0) |
| `Rule` | Full path rule from root, e.g., `root & char_1 >= 0.5 & char_3 < 0.7` |
| `Is_Leaf` | Whether the node is a leaf |
| `N_Samples` | Number of samples in the node |
| `Sample_Ratio` | Ratio of samples relative to total |
| `Split_Feature` | Feature used for splitting (NaN for leaves) |
| `Split_Threshold` | Split threshold value (NaN for leaves) |
| `Split_Score` | Criterion score at split |
| `Predictability_Score` | Predictability strength (R² for regression, Precision for classification) |
| `Metrics` | Full metrics dictionary, e.g., `{"r2": 0.63, "mse": 0.22, "n_samples": 2429}` |
| `Model_Weights` | Feature coefficients of the leaf model |
| `Elapsed_Time_s` | Time spent building the node (seconds) |
| `Parent_ID` | Parent node ID |

```python
report = engine.get_node_report()
print(report[["Node_ID", "Depth", "Rule", "Predictability_Score", "N_Samples"]])
```

#### `engine.get_leaf_samples() → Dict[int, np.ndarray]`

Return a dictionary mapping leaf `node_id` to an array of original sample row indices. Useful for extracting the raw data corresponding to each cluster.

```python
leaf_samples = engine.get_leaf_samples()
for leaf_id, indices in leaf_samples.items():
    subset = original_df.iloc[indices]
    print(f"Leaf {leaf_id}: {len(indices)} samples, "
          f"mean_return={subset['ret'].mean():.4f}, "
          f"unique_assets={subset['asset_id'].nunique()}")
```

---

### PanelTreeNode Methods

Node objects can be obtained via `engine.get_leaves()` or `engine.get_all_nodes()`.

#### `node.n_samples → int`

Number of samples contained in this node (read-only property).

#### `node.metrics → Dict[str, float]`

Evaluation metrics dictionary. For regression: `r2`, `mse`, `n_samples`. For classification: `precision`, `f1`, `auc`, `n_samples`.

```python
leaf = engine.get_leaves()[0]
print(leaf.metrics)  # {"r2": 0.63, "mse": 0.22, "n_samples": 2429}
```

#### `node.get_model_weights() → np.ndarray | None`

Return the feature coefficient vector of the leaf node's local model. Useful for inspecting which factors are active in a specific regime.

```python
for leaf in engine.get_leaves():
    coef = leaf.get_model_weights()
    if coef is not None:
        for name, w in zip(dh.feature_names, coef):
            print(f"  {name}: {w:+.4f}")
```

#### `node.get_samples() → np.ndarray | None`

Return sample row indices belonging to this node. Similar to `engine.get_leaf_samples()`, but can be used for any node (including internal nodes).

```python
node = engine.get_all_nodes()[1]  # Second node
indices = node.get_samples()
print(f"Node {node.node_id} contains {len(indices)} samples")
```

#### `node.to_dict() → Dict[str, Any]`

Serialise all node metadata to a flat dictionary, convenient for building DataFrames or exporting to JSON.

```python
import json
leaf = engine.get_leaves()[0]
print(json.dumps(leaf.to_dict(), indent=2, default=str))
```

#### Common Read-Only Attributes

| Attribute | Type | Description |
|---|---|---|
| `node.node_id` | `int` | Unique identifier |
| `node.depth` | `int` | Depth level |
| `node.rule` | `str` | Path description, e.g., `root & char_1 < 0.5 & char_3 >= 0.7` |
| `node.split_feature` | `str \| None` | Split feature name |
| `node.split_threshold` | `float \| None` | Split threshold |
| `node.split_score` | `float \| None` | Criterion score at split |
| `node.is_leaf` | `bool` | Whether this is a leaf |
| `node.sample_ratio` | `float` | Sample coverage ratio |
| `node.elapsed_time` | `float` | Build time (seconds) |
| `node.predictor` | `PredictorBase` | Trained local model instance |

---

### NodeReporter Methods

`NodeReporter` encapsulates user-facing reporting functionality. It requires a fitted `PanelTreeEngine`.

```python
from ptree import NodeReporter
reporter = NodeReporter(engine)
```

#### `reporter.summary() → pd.DataFrame`

Return a complete node report DataFrame (all nodes, including internal nodes and leaves). Column definitions are the same as `engine.get_node_report()`.

```python
full = reporter.summary()
print(full[["Node_ID", "Depth", "Is_Leaf", "Split_Feature", "Predictability_Score"]])
```

#### `reporter.leaf_summary() → pd.DataFrame`

Return only the leaf nodes report. Structure is the same as `summary()`, suitable for quickly viewing final clustering results.

```python
leaves = reporter.leaf_summary()
print(leaves[["Node_ID", "Rule", "Predictability_Score", "N_Samples", "Model_Weights"]])
```

**Example Output:**

```
 Node_ID                                              Rule  Predictability_Score  N_Samples
       3   root & char_1 < 0.5 & char_1 < 0.3 & char_3 < 0.7              0.0147       2438
       4  root & char_1 < 0.5 & char_1 < 0.3 & char_3 >= 0.7              0.0018       1102
      13  root & char_1 >= 0.5 & char_3 >= 0.3 & char_3 < 0.7              0.6323       2429
```

#### `reporter.print_tree() → str`

Return a formatted tree structure text string using indentation and `├─` / `└─` to represent hierarchical relationships.

```python
print(reporter.print_tree())
```

**Example Output:**

```
[Node 0] char_1 < 0.5 | r2=0.1234, n=12000 (Δ=0.4569)
├── [Node 1] char_1 < 0.3 | r2=0.0523, n=5940 (Δ=0.0140)
│   ├── [Leaf 3] r2=0.0147, mse=0.4769, n=2438
│   └── [Leaf 4] r2=0.0018, mse=0.8028, n=1102
└── [Leaf 5] r2=0.4640, mse=0.5483, n=6060
```

---

### MosaicVisualizer Methods

`MosaicVisualizer` generates "prediction mosaics" — 2D heatmaps that visually display the model's predictive power across different time periods and asset clusters.

```python
from ptree import MosaicVisualizer
viz = MosaicVisualizer(engine)
```

#### `viz.build_mosaic(X, y, time_col, metric) → pd.DataFrame`

Compute per-leaf, per-period metric values and return a DataFrame.

| Parameter | Description |
|---|---|
| `X` | Processed panel DataFrame (must include `time_col` and feature columns) |
| `y` | Target variable |
| `time_col` | Time column name, default `"date"` |
| `metric` | Evaluation metric: `"r2"` for regression, `"precision"` / `"f1"` / `"auc"` for classification |

**Return Structure:**
- **Row index**: Leaf node IDs (`Leaf_ID`)
- **Columns**: Time periods (determined by `time_col`)
- **Values**: Metric value for that leaf in that period

```python
mosaic = viz.build_mosaic(X_proc, y_proc, time_col="date", metric="r2")
print(mosaic.shape)       # (n_leaves, n_periods)
print(mosaic.iloc[:, :5]) # Preview first 5 periods

# Analyse which leaves perform best in which periods
best_leaf_per_period = mosaic.idxmax(axis=0)
print(best_leaf_per_period)
```

**Example Output:**

```
         0         1         2         3         4
Leaf_ID
3     0.016    -0.042     0.006    -0.089     0.036
13    0.621     0.782     0.599     0.687     0.605
14    0.502     0.465     0.350     0.462     0.289
```

#### `viz.plot_mosaic(mosaic, title, cmap, figsize, save_path) → (fig, ax)`

Render the mosaic matrix as a seaborn heatmap. Requires `matplotlib` and `seaborn`.

| Parameter | Default | Description |
|---|---|---|
| `mosaic` | — | DataFrame returned by `build_mosaic()` |
| `title` | `"Prediction Mosaic"` | Chart title |
| `cmap` | `"RdYlGn"` | Colour map (red=poor, green=good) |
| `figsize` | `(14, 6)` | Figure size |
| `save_path` | `None` | If specified, automatically save as PNG |

```python
# Interactive viewing
fig, ax = viz.plot_mosaic(mosaic, title="P-Tree R² Mosaic")

# Save to file
fig, ax = viz.plot_mosaic(mosaic, save_path="output/mosaic.png", cmap="coolwarm")
```

**Heatmap Interpretation:**
- **X-axis**: Time period $t$
- **Y-axis**: Leaf nodes
- **Colour**: Predictive accuracy for that leaf in that period (R² or Precision)
- Instantly reveals when and where the model "fails" or "excels"

---

### Verbose Logging

`PanelTreeEngine` outputs detailed splitting process logs via Python's `logging` module when `verbose >= 1`:

```python
import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

engine = PanelTreeEngine(..., verbose=1)
engine.fit(X, y, feature_names=...)
```

**Example Log Output:**

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

Set `verbose=2` to view per-candidate (feature, threshold) evaluation results.

---

## Performance Optimisations

1. **Incremental matrix updates** – For Ridge models, $X^TWX$ and $X^TWy$ are cached at each node. Only the *smaller* child's statistics are computed directly (a single matmul); the larger child is obtained by subtracting it from the cached parent, halving the matrix-multiplication work per candidate split.
2. **Feature-priority caching** – When `fast_mode=True`, child nodes first evaluate the top-50% features from the parent, with optional early stopping.
3. **Feature-dimension parallelism** – When `n_jobs != 1` (and `joblib` is installed), candidate features are evaluated in parallel. The default `"threads"` backend keeps NumPy/BLAS matmuls GIL-free with zero data copying; switch to `parallel_backend="processes"` for heavy pure-Python custom predictors.
4. **Cost-complexity pruning & honest splits** – `engine.prune(ccp_alpha)` removes over-fit subtrees post-hoc, while `honest=True` evaluates split quality on a held-out in-node subset to remove the selection bias of fitting and scoring on the same data.
5. **Vectorised AUC** – Classification AUC uses an `O(n log n)` rank-based Mann–Whitney statistic instead of the former `O(n⁺·n⁻)` double loop.

## Ensembles (P-Forest & P-Boost)

A single Panel Tree is a *high-variance* estimator — small data perturbations can flip the greedily-chosen split and produce a completely different partition. The `ensemble` module provides two derived algorithms (mirroring the decision-tree → random-forest / gradient-boosting relationship) that act on the **prediction / target layer** while each tree's split criterion stays the unchanged `R2Diff` rule.

### PanelForest (P-Forest — bagging)

Grows many decorrelated P-Trees via **time-block bootstrap** (contiguous blocks of `block_size` periods resampled with replacement, preserving serial autocorrelation) plus **node-level random feature subsets** (`max_features`), then aggregates at the output layer.

```python
from ptree import PanelForest

forest = PanelForest(
    n_estimators=100, max_features="sqrt", block_size=5,
    base_params={"max_depth": 3, "min_samples": 100},
    n_jobs=-1, random_state=0,
    # NEW in v0.2: score "high-predictability" leaves on OOB samples
    # instead of brittle train-R². Use "rank_ic" for low-SNR panels.
    regime_metric="rank_ic",        # default "train_r2"
    regime_aggregation="oof",       # default "train"
)
forest.fit(X, y, feature_names=dh.feature_names, time_index="date")

forest.predict(X)                  # bagged mean ŷ (variance reduction)
forest.regime_membership(X)        # soft P(obs ∈ high-predictability regime) ∈ [0,1]
forest.coassociation_matrix(X)     # consensus similarity C[i,j] ∈ [0,1] (same-leaf frequency)
forest.oob_score_                  # out-of-bag R² (unselected time blocks)
```

| Output | Description |
|---|---|
| `.predict(X)` | Bagged mean prediction across trees (lower variance than a single tree) |
| `.regime_membership(X)` | Fraction of trees routing each observation into a *high-R²* leaf — a smooth, robust upgrade of the 0/1 mosaic |
| `.coassociation_matrix(X)` | Consensus / co-association matrix; fraction of trees in which two observations share a leaf (a precomputed affinity for spectral clustering) |
| `.oob_score_` | Out-of-bag R² estimated on each tree's unselected time blocks |

> **Note:** P-Forest's gains are largest when predictability is driven by *several weakly-identified features*. If a single strong feature dominates, all trees split on it, become highly correlated, and the ensemble adds little.

### BoostedPanelTree (P-Boost — residual boosting)

Boosts the **target/residual** (not the criterion): each round strips the predictability already explained by the running ensemble and re-grows a fresh P-Tree on the residual, uncovering the *next, weaker* regime the greedy single tree would have masked.

```python
from ptree import BoostedPanelTree

booster = BoostedPanelTree(
    n_estimators=50, learning_rate=0.1, max_depth=2,
    subsample=1.0, random_state=0,
)
booster.fit(X, y, feature_names=dh.feature_names)

booster.predict(X)            # ν · Σ_m tree_m.predict(X)
booster.residual_norms_       # residual L2-norm per round (monotone ↓; flat ⇒ self-limited)
```

On single-feature-dominated data P-Boost is **self-limiting**: once the first tree explains the dominant regime, the residual is near-noise and later trees add almost nothing (visible in `residual_norms_`). Set `splitter="random"` via `base_params` for an Extra-Trees-style variant that injects extra diversity.

## Requirements

- Python ≥ 3.10
- `numpy`, `pandas`
- `matplotlib`, `seaborn` (optional, for visualisation)


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

```bash
# Clone the repository
git clone https://github.com/ElenYoung/AssetPanelTree.git
cd AssetPanelTree

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest test/ -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use P-Tree in your research, please consider citing:

```bibtex
@software{ptree2026,
  author = {ElenYoung},
  title = {P-Tree: Panel Tree for Supervised Clustering},
  year = {2026},
  url = {https://github.com/ElenYoung/AssetPanelTree}
}
```
