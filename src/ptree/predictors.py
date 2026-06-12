"""
Predictor models for PanelTree leaf nodes.

All predictors inherit from ``PredictorBase`` and implement ``fit`` /
``predict``.  The closed-form Ridge solver is shared so that the engine can
exploit incremental matrix updates (Redundancy Saving).
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

try:  # SciPy provides a faster, more stable SPD solve (Cholesky).
    from scipy.linalg import cho_factor, cho_solve

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - exercised only without SciPy
    _HAS_SCIPY = False



# ======================================================================
# Base class
# ======================================================================

class PredictorBase(ABC):
    """Abstract base class for leaf-node predictors.

    Subclasses must implement :meth:`fit` and :meth:`predict`.
    """

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> "PredictorBase":
        """Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n, p)
        y : ndarray of shape (n,)
        weights : ndarray of shape (n,) or None
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for *X*."""
        ...

    def get_coefficients(self) -> Optional[np.ndarray]:
        """Return model coefficients if available."""
        return None

    def get_intercept(self) -> Optional[float]:
        """Return model intercept if available."""
        return None

    def get_params(self) -> Dict[str, Any]:
        """Return a dict of hyper-parameters."""
        return {}


# ======================================================================
# Ridge helpers (closed-form, also used for incremental updates)
# ======================================================================

def _ridge_closed_form(
    XtWX: np.ndarray,
    XtWy: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Solve the weighted Ridge normal equation.

    .. math::
        \\beta = (X^T W X + \\alpha I)^{-1} X^T W y

    Parameters
    ----------
    XtWX : (p, p)  —  X^T W X  (accumulated)
    XtWy : (p,)    —  X^T W y  (accumulated)
    alpha : float   —  regularization strength

    Returns
    -------
    beta : (p,)

    Notes
    -----
    ``A = XtWX + alpha I`` is symmetric positive-definite for ``alpha > 0``,
    so a Cholesky factorisation is both faster and more numerically stable
    than a general LU solve.  We prefer SciPy's ``cho_factor`` / ``cho_solve``
    when available and fall back to ``np.linalg.solve`` → ``lstsq`` otherwise.
    The result is numerically equivalent to the previous LU solve.
    """
    p = XtWX.shape[0]
    A = XtWX + alpha * np.eye(p)

    if _HAS_SCIPY:
        try:
            c, low = cho_factor(A, check_finite=False)
            return cho_solve((c, low), XtWy, check_finite=False)
        except (np.linalg.LinAlgError, ValueError):
            # Not SPD (e.g. alpha == 0 with rank deficiency) → fall through.
            pass

    try:
        return np.linalg.solve(A, XtWy)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, XtWy, rcond=None)[0]



def compute_XtWX_XtWy(
    X: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute sufficient statistics for Ridge.

    Parameters
    ----------
    X : (n, p)
    y : (n,)
    weights : (n,) or None — diagonal weight vector

    Returns
    -------
    XtWX : (p, p)
    XtWy : (p,)
    """
    if weights is not None:
        W_sqrt = np.sqrt(weights)
        Xw = X * W_sqrt[:, None]
        yw = y * W_sqrt
    else:
        Xw = X
        yw = y
    XtWX = Xw.T @ Xw
    XtWy = Xw.T @ yw
    return XtWX, XtWy


# ======================================================================
# Concrete predictors
# ======================================================================

class RidgeRegressor(PredictorBase):
    """Ordinary Ridge regression (closed-form).

    Parameters
    ----------
    alpha : float, default 1.0
        L2 regularization strength.
    fit_intercept : bool, default True
        Whether to fit an intercept term.
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self._coef: Optional[np.ndarray] = None
        self._intercept: float = 0.0
        # Cached sufficient statistics (for incremental updates)
        self._XtWX: Optional[np.ndarray] = None
        self._XtWy: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        XtWX: Optional[np.ndarray] = None,
        XtWy: Optional[np.ndarray] = None,
    ) -> "RidgeRegressor":
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        if XtWX is not None and XtWy is not None:
            self._XtWX, self._XtWy = XtWX, XtWy
        else:
            self._XtWX, self._XtWy = compute_XtWX_XtWy(X, y, weights)

        beta = _ridge_closed_form(self._XtWX, self._XtWy, self.alpha)

        if self.fit_intercept:
            self._intercept = float(beta[0])
            self._coef = beta[1:]
        else:
            self._intercept = 0.0
            self._coef = beta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._coef is not None, "Model not fitted."
        return X @ self._coef + self._intercept

    def get_coefficients(self) -> Optional[np.ndarray]:
        return self._coef

    def get_intercept(self) -> Optional[float]:
        return self._intercept

    def get_params(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "fit_intercept": self.fit_intercept}


class VolWeightedRidgeRegressor(PredictorBase):
    """Inverse-volatility-weighted Ridge regression.

    Uses ``1 / sigma_i`` as observation weights to address return
    heteroscedasticity, as described in the Panel Tree paper.

    Parameters
    ----------
    alpha : float, default 1.0
    fit_intercept : bool, default True
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self._inner = RidgeRegressor(alpha=alpha, fit_intercept=fit_intercept)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        XtWX: Optional[np.ndarray] = None,
        XtWy: Optional[np.ndarray] = None,
    ) -> "VolWeightedRidgeRegressor":
        """Fit with inverse-vol weights.

        The *weights* argument should already contain inverse-volatility
        weights (``1 / sigma``).  If ``None``, falls back to unweighted Ridge.
        """
        self._inner.fit(X, y, weights=weights, XtWX=XtWX, XtWy=XtWy)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._inner.predict(X)

    def get_coefficients(self) -> Optional[np.ndarray]:
        return self._inner.get_coefficients()

    def get_intercept(self) -> Optional[float]:
        return self._inner.get_intercept()

    def get_params(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "fit_intercept": self.fit_intercept}


class RidgeLogitClassifier(PredictorBase):
    """Ridge logistic regression for classification tasks.

    Uses iteratively reweighted least-squares (IRLS) with an L2 penalty.

    Parameters
    ----------
    alpha : float, default 1.0
    max_iter : int, default 50
    tol : float, default 1e-6
    fit_intercept : bool, default True
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 50,
        tol: float = 1e-6,
        fit_intercept: bool = True,
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self._coef: Optional[np.ndarray] = None
        self._intercept: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
        **kwargs,
    ) -> "RidgeLogitClassifier":
        n, p = X.shape
        if self.fit_intercept:
            X_aug = np.column_stack([np.ones(n), X])
        else:
            X_aug = X
        d = X_aug.shape[1]

        if weights is None:
            weights = np.ones(n)

        beta = np.zeros(d)

        for _ in range(self.max_iter):
            z = X_aug @ beta
            mu = self._sigmoid(z)
            s = mu * (1.0 - mu)
            s = np.clip(s, 1e-8, None)

            W = weights * s
            residual = y - mu

            grad = X_aug.T @ (weights * residual) - self.alpha * beta
            H = (X_aug * W[:, None]).T @ X_aug + self.alpha * np.eye(d)

            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H, grad, rcond=None)[0]

            beta += delta
            if np.max(np.abs(delta)) < self.tol:
                break

        if self.fit_intercept:
            self._intercept = float(beta[0])
            self._coef = beta[1:]
        else:
            self._intercept = 0.0
            self._coef = beta
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(y=1 | X)."""
        assert self._coef is not None, "Model not fitted."
        return self._sigmoid(X @ self._coef + self._intercept)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions (threshold = 0.5)."""
        return (self.predict_proba(X) >= 0.5).astype(float)

    def get_coefficients(self) -> Optional[np.ndarray]:
        return self._coef

    def get_intercept(self) -> Optional[float]:
        return self._intercept

    def get_params(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "fit_intercept": self.fit_intercept,
        }


class SelfDefinedPredictor(PredictorBase):
    """Template for user-defined predictors.

    Users should subclass this and implement ``fit`` / ``predict``.
    Optionally wrap an external model (e.g. LightGBM, simple NN).

    Example
    -------
    >>> class MyLGBPredictor(SelfDefinedPredictor):
    ...     def fit(self, X, y, weights=None):
    ...         import lightgbm as lgb
    ...         self.model = lgb.LGBMRegressor().fit(X, y, sample_weight=weights)
    ...         return self
    ...     def predict(self, X):
    ...         return self.model.predict(X)
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> "SelfDefinedPredictor":
        raise NotImplementedError("Subclass must implement fit().")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclass must implement predict().")
