"""
Split-quality criteria for PanelTree.

Each criterion evaluates whether a candidate split produces child nodes
whose predictability differs meaningfully.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class CriterionBase(ABC):
    """Abstract base for split-quality criteria.

    Subclasses must implement :meth:`calculate_score`, which receives
    metrics from both child nodes and returns a scalar indicating the
    quality of the split (higher is better).
    """

    @abstractmethod
    def calculate_score(
        self,
        left_metrics: Dict[str, float],
        right_metrics: Dict[str, float],
    ) -> float:
        """Return split quality score.

        Parameters
        ----------
        left_metrics, right_metrics : dict
            Dictionaries produced by the predictor's evaluation routine.
            Expected keys depend on the concrete criterion (e.g. ``"r2"``,
            ``"precision"``, ``"f1"``, ``"auc"``).
        """
        ...

    @abstractmethod
    def metric_key(self) -> str:
        """Return the name of the primary metric this criterion optimises."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ======================================================================
# Regression: R² difference
# ======================================================================

class R2DiffCriterion(CriterionBase):
    """Maximise the absolute R² difference between left and right nodes.

    .. math::
        \\text{score} = |R^2_L - R^2_R|

    Optionally weight by sample counts so that larger nodes matter more.

    Parameters
    ----------
    weight_by_size : bool, default False
        If True, multiply the R² difference by
        ``min(n_L, n_R) / (n_L + n_R)`` to penalise highly imbalanced
        splits.
    """

    def __init__(self, weight_by_size: bool = False):
        self.weight_by_size = weight_by_size

    def calculate_score(
        self,
        left_metrics: Dict[str, float],
        right_metrics: Dict[str, float],
    ) -> float:
        r2_l = left_metrics.get("r2", 0.0)
        r2_r = right_metrics.get("r2", 0.0)
        diff = abs(r2_l - r2_r)

        if self.weight_by_size:
            n_l = left_metrics.get("n_samples", 1)
            n_r = right_metrics.get("n_samples", 1)
            balance = min(n_l, n_r) / max(n_l + n_r, 1)
            diff *= balance

        return diff

    def metric_key(self) -> str:
        return "r2"

    def __repr__(self) -> str:
        return f"R2DiffCriterion(weight_by_size={self.weight_by_size})"


# ======================================================================
# Classification: Precision / F1 / AUC difference
# ======================================================================

class ClassificationCriterion(CriterionBase):
    """Maximise the difference in a classification metric across child nodes.

    Parameters
    ----------
    metric : str, default "precision"
        One of ``"precision"``, ``"f1"``, ``"auc"``.
    weight_by_size : bool, default False
        Same balance penalty as :class:`R2DiffCriterion`.
    """

    _VALID_METRICS = {"precision", "f1", "auc"}

    def __init__(
        self,
        metric: str = "precision",
        weight_by_size: bool = False,
    ):
        if metric not in self._VALID_METRICS:
            raise ValueError(
                f"Unknown metric '{metric}'. Choose from {self._VALID_METRICS}."
            )
        self.metric = metric
        self.weight_by_size = weight_by_size

    def calculate_score(
        self,
        left_metrics: Dict[str, float],
        right_metrics: Dict[str, float],
    ) -> float:
        m_l = left_metrics.get(self.metric, 0.0)
        m_r = right_metrics.get(self.metric, 0.0)
        diff = abs(m_l - m_r)

        if self.weight_by_size:
            n_l = left_metrics.get("n_samples", 1)
            n_r = right_metrics.get("n_samples", 1)
            balance = min(n_l, n_r) / max(n_l + n_r, 1)
            diff *= balance

        return diff

    def metric_key(self) -> str:
        return self.metric

    def __repr__(self) -> str:
        return (
            f"ClassificationCriterion(metric='{self.metric}', "
            f"weight_by_size={self.weight_by_size})"
        )


# ======================================================================
# Evaluation helpers (used by the engine to build metric dicts)
# ======================================================================

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute regression metrics.

    Returns
    -------
    dict with keys: ``r2``, ``mse``, ``n_samples``.
    """
    n = len(y_true)
    if n == 0:
        return {"r2": -np.inf, "mse": np.inf, "n_samples": 0}

    if weights is None:
        weights = np.ones(n)

    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    y_mean = np.average(y_true, weights=weights)
    ss_tot = np.sum(weights * (y_true - y_mean) ** 2)

    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    mse = ss_res / max(np.sum(weights), 1e-12)
    return {"r2": float(r2), "mse": float(mse), "n_samples": n}


def evaluate_classification(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics.

    Returns
    -------
    dict with keys: ``precision``, ``f1``, ``auc``, ``n_samples``.
    """
    n = len(y_true)
    if n == 0:
        return {"precision": 0.0, "f1": 0.0, "auc": 0.5, "n_samples": 0}

    y_pred = (y_proba >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    # Simple AUC via Mann–Whitney U statistic
    auc = _auc_mannwhitney(y_true, y_proba)

    return {
        "precision": float(precision),
        "f1": float(f1),
        "auc": float(auc),
        "n_samples": n,
    }


def _auc_mannwhitney(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC via the Mann-Whitney U statistic (no sklearn dependency)."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    u = 0.0
    for p in pos:
        u += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(u / (len(pos) * len(neg)))
