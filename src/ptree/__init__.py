"""
Panel Tree (P-Tree): A supervised clustering algorithm for panel data.

A supervised clustering algorithm designed for panel data, commonly used in
quantitative finance to identify time-varying, cross-sectional predictability
regimes.

Modules:
    - data_handler: Data preprocessing, alignment, and cross-sectional rank standardization.
    - predictors: Predictor models (Ridge, VolWeightedRidge, RidgeLogit, custom).
    - criteria: Split quality criteria (R2Diff, Classification metrics).
    - node: PanelTreeNode container class.
    - engine: PanelTreeEngine – recursive splitting, pruning, and parallel execution.
    - visualization: Logging, node reports, and mosaic visualization.

Example
-------
>>> from ptree import DataHandler, RidgeRegressor, R2DiffCriterion, PanelTreeEngine
>>> dh = DataHandler(cs_rank_standardize=True)
>>> X, y, vol_weights = dh.fit_transform(df, y_series, time_col="date", entity_col="asset_id")
>>> engine = PanelTreeEngine(
...     predictor=RidgeRegressor(alpha=1.0),
...     criterion=R2DiffCriterion(),
...     max_depth=3,
... )
>>> engine.fit(X, y, feature_names=dh.feature_names)
"""

from __future__ import annotations

__version__ = "0.2.0"
__author__ = "ElenYoung"

from ptree.data_handler import DataHandler
from ptree.predictors import (
    PredictorBase,
    RidgeRegressor,
    VolWeightedRidgeRegressor,
    RidgeLogitClassifier,
    ElasticNetRegressor,
    PLSRegressor,
    SelfDefinedPredictor,
)

from ptree.criteria import (
    CriterionBase,
    R2DiffCriterion,
    WeightedR2DiffCriterion,
    RankICDiffCriterion,
    MeanVarianceCriterion,
    ClassificationCriterion,
)


from ptree.node import PanelTreeNode
from ptree.engine import PanelTreeEngine, NodeEvalResult
from ptree.ensemble import PanelForest, BoostedPanelTree

from ptree.visualization import MosaicVisualizer, NodeReporter



__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Data handling
    "DataHandler",
    # Predictors
    "PredictorBase",
    "RidgeRegressor",
    "VolWeightedRidgeRegressor",
    "RidgeLogitClassifier",
    "ElasticNetRegressor",
    "PLSRegressor",
    "SelfDefinedPredictor",
    # Criteria

    "CriterionBase",
    "R2DiffCriterion",
    "WeightedR2DiffCriterion",
    "RankICDiffCriterion",
    "MeanVarianceCriterion",
    "ClassificationCriterion",


    # Core classes
    "PanelTreeNode",
    "PanelTreeEngine",
    "NodeEvalResult",
    # Ensembles
    "PanelForest",
    "BoostedPanelTree",
    # Visualization


    "MosaicVisualizer",
    "NodeReporter",
]
