"""
Split-quality criteria for PanelTree.

Each criterion evaluates whether a candidate split produces child nodes
whose predictability differs meaningfully.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional



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


class WeightedR2DiffCriterion(CriterionBase):
    """Academically-weighted variant of :class:`R2DiffCriterion`.

    Keeps the ``|R^2_L - R^2_R|`` core of the default criterion but multiplies
    it by stabilising factors that temper spuriously-high R² on small or
    imbalanced child nodes (a common source of over-fitting in greedy panel
    splits).  The default criterion's behaviour is intentionally *not* changed;
    this is an opt-in, stricter alternative.

    .. math::
        \\text{score} = |R^2_L - R^2_R|
            \\cdot \\underbrace{\\frac{\\min(n_L, n_R)}{n_L + n_R}}_{\\text{balance}}
            \\cdot \\underbrace{\\frac{n_{\\min} - 1}{n_{\\min} - 1 + k}}_{\\text{shrinkage}}

    Parameters
    ----------
    balance : bool, default True
        Apply the ``min(n_L, n_R) / (n_L + n_R)`` balance penalty.
    shrinkage_k : float, default 0.0
        Sample-size shrinkage strength applied via the smaller node's count.
        ``0`` disables shrinkage; larger values penalise small nodes harder.
    use_adjusted_r2 : bool, default False
        If True, degrade each child's R² toward an adjusted-R² using the model
        dimensionality ``p`` recorded in the metrics (``n_features``), guarding
        against high R² merely from many regressors on few samples.
    """

    def __init__(
        self,
        balance: bool = True,
        shrinkage_k: float = 0.0,
        use_adjusted_r2: bool = False,
        min_child_weight: float = 0.0,
    ):
        if not (0.0 <= min_child_weight < 0.5):
            raise ValueError(
                "min_child_weight must lie in [0, 0.5); got "
                f"{min_child_weight!r}."
            )
        self.balance = balance
        self.shrinkage_k = shrinkage_k
        self.use_adjusted_r2 = use_adjusted_r2
        self.min_child_weight = min_child_weight

    @staticmethod
    def _adjusted_r2(r2: float, n: int, p: int) -> float:
        denom = n - p - 1
        if denom <= 0:
            return r2
        return 1.0 - (1.0 - r2) * (n - 1) / denom

    def calculate_score(
        self,
        left_metrics: Dict[str, float],
        right_metrics: Dict[str, float],
    ) -> float:
        r2_l = left_metrics.get("r2", 0.0)
        r2_r = right_metrics.get("r2", 0.0)
        n_l = int(left_metrics.get("n_samples", 1))
        n_r = int(right_metrics.get("n_samples", 1))

        # Hard floor on the smaller child's sample share — splits that produce
        # a tiny sliver on either side score 0 and so are never selected.
        if self.min_child_weight > 0.0:
            total = max(n_l + n_r, 1)
            if min(n_l, n_r) / total < self.min_child_weight:
                return 0.0

        if self.use_adjusted_r2:
            p_l = int(left_metrics.get("n_features", 0))
            p_r = int(right_metrics.get("n_features", 0))
            r2_l = self._adjusted_r2(r2_l, n_l, p_l)
            r2_r = self._adjusted_r2(r2_r, n_r, p_r)

        diff = abs(r2_l - r2_r)

        if self.balance:
            diff *= min(n_l, n_r) / max(n_l + n_r, 1)

        if self.shrinkage_k > 0.0:
            n_min = min(n_l, n_r)
            diff *= max(n_min - 1, 0) / max(n_min - 1 + self.shrinkage_k, 1e-12)

        return diff

    def metric_key(self) -> str:
        return "r2"

    def __repr__(self) -> str:
        return (
            f"WeightedR2DiffCriterion(balance={self.balance}, "
            f"shrinkage_k={self.shrinkage_k}, "
            f"use_adjusted_r2={self.use_adjusted_r2}, "
            f"min_child_weight={self.min_child_weight})"
        )


# ======================================================================
# Regression: Rank-IC difference (cross-sectional)
# ======================================================================

class RankICDiffCriterion(CriterionBase):
    """Maximise the absolute difference in cross-sectional Rank-IC.

    On low-signal-to-noise panels (e.g. monthly stock returns) ``|R^2_L -
    R^2_R|`` is dominated by sum-of-squares variance and tends to over-reward
    small high-variance child nodes.  The cross-sectional **Spearman rank IC**
    — i.e. the mean per-time correlation between predicted and realised
    rankings — is far more robust because it ignores scale and only measures
    *ordering*.  This criterion therefore scores splits by

    .. math::
        \\text{score} = |\\overline{IC}_L - \\overline{IC}_R|,

    optionally rescaled by the sample-balance factor used by
    :class:`R2DiffCriterion`.

    The engine attaches the per-time predicted/realised pairs to each child's
    metrics under the non-scalar key ``"_rank_ic_series"`` (only populated
    when ``MeanVarianceCriterion``-style time labels are available, i.e. a
    ``time_index`` was passed to ``fit()``).  When the series is missing the
    criterion falls back to a pre-computed scalar ``rank_ic`` value if
    present, else returns ``0`` — i.e. it never raises, so it can be used
    safely outside a time-aware build.

    Parameters
    ----------
    balance : bool, default False
        Multiply the IC difference by ``min(n_L, n_R) / (n_L + n_R)``.
    min_child_weight : float, default 0.0
        Sample-share floor on the smaller side; splits that produce a sliver
        smaller than this fraction of ``n_L + n_R`` score ``0``.
    min_periods : int, default 3
        Minimum number of overlapping time periods required to estimate a
        per-side IC; below this the side's IC defaults to ``0``.
    """

    def __init__(
        self,
        balance: bool = False,
        min_child_weight: float = 0.0,
        min_periods: int = 3,
    ):
        if not (0.0 <= min_child_weight < 0.5):
            raise ValueError(
                "min_child_weight must lie in [0, 0.5); got "
                f"{min_child_weight!r}."
            )
        if min_periods < 1:
            raise ValueError("min_periods must be >= 1.")
        self.balance = balance
        self.min_child_weight = min_child_weight
        self.min_periods = min_periods

    @staticmethod
    def _ic_mean_from_series(series: Optional[Dict[Any, tuple]], min_periods: int) -> float:
        """Compute mean cross-sectional Spearman IC over the per-time series.

        ``series`` is the engine-attached payload: a dict ``{time: (y_true,
        y_pred)}`` (both ndarray).  Returns ``0`` when fewer than
        ``min_periods`` valid periods are available.
        """
        if not series:
            return 0.0
        ics: List[float] = []
        for _, (yt, yp) in series.items():
            yt = np.asarray(yt, dtype=np.float64)
            yp = np.asarray(yp, dtype=np.float64)
            if yt.size < 2:
                continue
            # Spearman = Pearson of ranks.  Average-rank tie handling.
            rt = _rank_average(yt)
            rp = _rank_average(yp)
            if rt.std() < 1e-12 or rp.std() < 1e-12:
                continue
            ic = float(np.corrcoef(rt, rp)[0, 1])
            if np.isfinite(ic):
                ics.append(ic)
        if len(ics) < min_periods:
            return 0.0
        return float(np.mean(ics))

    def calculate_score(
        self,
        left_metrics: Dict[str, float],
        right_metrics: Dict[str, float],
    ) -> float:
        n_l = int(left_metrics.get("n_samples", 1))
        n_r = int(right_metrics.get("n_samples", 1))
        if self.min_child_weight > 0.0:
            total = max(n_l + n_r, 1)
            if min(n_l, n_r) / total < self.min_child_weight:
                return 0.0

        # Prefer the per-time series payload; fall back to a pre-computed
        # scalar ``rank_ic`` if a caller has supplied one.
        sl = left_metrics.get("_rank_ic_series")
        sr = right_metrics.get("_rank_ic_series")
        if sl is not None or sr is not None:
            ic_l = self._ic_mean_from_series(sl, self.min_periods)
            ic_r = self._ic_mean_from_series(sr, self.min_periods)
        else:
            ic_l = float(left_metrics.get("rank_ic", 0.0))
            ic_r = float(right_metrics.get("rank_ic", 0.0))

        diff = abs(ic_l - ic_r)
        if self.balance:
            diff *= min(n_l, n_r) / max(n_l + n_r, 1)
        return diff

    def metric_key(self) -> str:
        # Keep ``r2`` for node-level logging; the criterion operates on the
        # non-scalar ``_rank_ic_series`` payload.
        return "r2"

    def __repr__(self) -> str:
        return (
            f"RankICDiffCriterion(balance={self.balance}, "
            f"min_child_weight={self.min_child_weight}, "
            f"min_periods={self.min_periods})"
        )


def _rank_average(x: np.ndarray) -> np.ndarray:
    """Return average ranks (1-based) of *x* with tie-handling."""
    order = np.argsort(x, kind="mergesort")
    n = x.shape[0]
    sorted_x = x[order]
    is_group_start = np.empty(n, dtype=bool)
    is_group_start[0] = True
    np.not_equal(sorted_x[1:], sorted_x[:-1], out=is_group_start[1:])
    group_id = np.cumsum(is_group_start) - 1
    group_starts = np.flatnonzero(is_group_start)
    group_ends = np.append(group_starts[1:], n)
    group_avg_rank = 0.5 * (group_starts + group_ends - 1) + 1.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = group_avg_rank[group_id]
    return ranks



# ======================================================================
# Regression: Mean-Variance / SDF Sharpe increment
# ======================================================================

class MeanVarianceCriterion(CriterionBase):
    """Split quality = tangency Sharpe of the two child long-short portfolios.

    Aligns the split objective with the academic Panel Tree goal of *growing
    the efficient frontier*: instead of comparing single-node R², each child
    leaf forms a cross-sectionally de-meaned, predicted-return-weighted
    long-short portfolio, producing a return *time series*.  The split score is
    the in-sample tangency (maximum) Sharpe ratio obtainable by mean-variance
    combining the left and right portfolio series,

    .. math::
        \\text{score} = \\sqrt{\\mu^\\top \\Sigma^{-1} \\mu}\\,\\sqrt{A},

    where :math:`\\mu, \\Sigma` are the mean vector / covariance of the two
    child portfolio returns and :math:`A` is the annualisation factor.  A high
    score means the two children span *complementary* predictability and thus
    push the efficient frontier out further than either alone.

    Unlike the R²-based criteria this requires the engine to attach a per-time
    portfolio return series to each child's metrics under the non-scalar key
    ``"_port_ret"`` (the engine does this automatically when this criterion is
    used and a ``time_index`` is supplied to ``fit``).

    Parameters
    ----------
    annualization : float, default 12.0
        Periods per year used to annualise the Sharpe ratio (e.g. 12 monthly,
        252 daily).  Only rescales the score monotonically.
    ridge : float, default 1e-6
        Diagonal load added to the 2x2 covariance for numerical stability.
    min_periods : int, default 3
        Minimum number of overlapping time periods required; below this the
        split scores ``0`` (insufficient data to estimate a Sharpe).
    """

    def __init__(
        self,
        annualization: float = 12.0,
        ridge: float = 1e-6,
        min_periods: int = 3,
    ):
        self.annualization = annualization
        self.ridge = ridge
        self.min_periods = min_periods

    def calculate_score(
        self,
        left_metrics: Dict[str, float],
        right_metrics: Dict[str, float],
    ) -> float:
        rl = left_metrics.get("_port_ret")
        rr = right_metrics.get("_port_ret")
        if not rl or not rr:
            return 0.0

        common = sorted(set(rl).intersection(rr))
        if len(common) < self.min_periods:
            return 0.0

        R = np.array([[rl[t], rr[t]] for t in common], dtype=np.float64)
        mu = R.mean(axis=0)
        # ``np.cov`` needs >=2 observations; guarded by ``min_periods``.
        Sigma = np.cov(R, rowvar=False)
        Sigma = np.atleast_2d(Sigma) + self.ridge * np.eye(2)
        try:
            inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            return 0.0
        sr_sq = float(mu @ inv @ mu)
        if sr_sq <= 0.0:
            return 0.0
        return float(np.sqrt(sr_sq) * np.sqrt(self.annualization))

    def metric_key(self) -> str:
        # The criterion's natural per-node summary is the leaf's long-short
        # portfolio Sharpe ratio.  The engine attaches a scalar ``sharpe``
        # field to ``node.metrics`` when this criterion is active so that
        # logging, node labels and ``NodeReporter`` can display it directly.
        return "sharpe"

    def __repr__(self) -> str:
        return (
            f"MeanVarianceCriterion(annualization={self.annualization}, "
            f"ridge={self.ridge}, min_periods={self.min_periods})"
        )


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

    _VALID_METRICS = {"precision", "f1", "auc", "logloss"}


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
    n_features = int(y_pred.shape[1]) if y_pred.ndim > 1 else 0
    return {
        "r2": float(r2),
        "mse": float(mse),
        "n_samples": n,
        "n_features": n_features,
    }



def evaluate_classification(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics.

    Returns
    -------
    dict with keys: ``precision``, ``f1``, ``auc``, ``logloss``, ``n_samples``.
    """
    n = len(y_true)
    if n == 0:
        return {
            "precision": 0.0, "f1": 0.0, "auc": 0.5,
            "logloss": np.inf, "n_samples": 0,
        }

    y_pred = (y_proba >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    # Simple AUC via Mann–Whitney U statistic
    auc = _auc_mannwhitney(y_true, y_proba)

    # Binary cross-entropy (log loss); lower is better.  Probabilities are
    # clipped to avoid ``log(0)``.  The split criterion compares the *absolute
    # difference* between children, so the sign convention is immaterial.
    eps = 1e-12
    p_clip = np.clip(np.asarray(y_proba, dtype=np.float64), eps, 1.0 - eps)
    yt = np.asarray(y_true, dtype=np.float64)
    logloss = float(-np.mean(yt * np.log(p_clip) + (1.0 - yt) * np.log(1.0 - p_clip)))

    return {
        "precision": float(precision),
        "f1": float(f1),
        "auc": float(auc),
        "logloss": logloss,
        "n_samples": n,
    }



def _auc_mannwhitney(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUC via the Mann-Whitney U statistic (no sklearn dependency).

    Vectorised ``O(n log n)`` rank-based implementation (replaces the former
    ``O(n_pos * n_neg)`` double loop).  Ties in ``y_score`` are handled by
    average ranks, which is exactly equivalent to the ``+0.5`` tie correction
    of the brute-force Mann-Whitney U.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(y_true.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Average ranks (1-based) with fully-vectorised tie handling.
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    n = y_score.shape[0]

    # Group boundaries: a new tie-group starts wherever the score changes.
    is_group_start = np.empty(n, dtype=bool)
    is_group_start[0] = True
    np.not_equal(sorted_scores[1:], sorted_scores[:-1], out=is_group_start[1:])
    group_id = np.cumsum(is_group_start) - 1  # 0-based group index per position

    # For each group, the average 1-based rank is the mean of its positions+1.
    group_starts = np.flatnonzero(is_group_start)  # first position of each group
    group_ends = np.append(group_starts[1:], n)    # one-past-last position
    # Average of 1-based ranks over [start, end): 0.5*(start + end - 1) + 1.
    group_avg_rank = 0.5 * (group_starts + group_ends - 1) + 1.0

    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = group_avg_rank[group_id]

    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))

    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))

