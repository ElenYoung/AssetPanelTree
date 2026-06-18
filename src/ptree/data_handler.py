"""
DataHandler: Panel data preprocessing, alignment, missing-value handling,
cross-sectional rank standardization, and rolling volatility computation.

Low-quality feature filtering (added 2026-06)
---------------------------------------------
``DataHandler`` now also screens out columns that are very unlikely to
produce a useful split:

* ``max_nan_frac`` — drop a feature when the fraction of missing values in
  its raw (pre-fill) column exceeds this threshold.  Default ``0.5``.
* ``min_unique_frac`` — drop a feature when its number of distinct
  non-NaN values divided by the total row count is below this threshold,
  i.e. the column is "almost a single value".  Default ``0.15``.

Filtering is performed inside :meth:`DataHandler.fit` on the **raw**
input (before NaN filling and cross-sectional rank standardisation), so
the metrics reflect the true information content of each feature.  The
list of dropped columns is exposed on :attr:`DataHandler.dropped_features_`
for downstream inspection / logging.

Set either parameter to ``None`` to disable the corresponding check.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List


logger = logging.getLogger("ptree")


class DataHandler:
    """Process and prepare panel data for PanelTree.

    Parameters
    ----------
    cs_rank_standardize : bool, default True
        If True, apply cross-sectional rank normalization mapping features
        to [0, 1] within each time period.
    vol_window : int, default 60
        Rolling window size for computing realised volatility (used by
        VolWeightedRidgeRegressor).
    min_obs : int, default 20
        Minimum number of non-NaN observations required in the volatility
        rolling window to produce a value.
    fillna_method : str or None, default "ffill"
        Method for filling missing values across the time dimension.
        Accepts ``"ffill"``, ``"bfill"``, ``"zero"``, ``"mean"`` or ``None``.
    max_nan_frac : float or None, default 0.5
        Drop a feature when its raw missing-value fraction strictly exceeds
        this threshold.  ``None`` disables the check.  Computed on the raw
        ``X`` passed to :meth:`fit` *before* any fill / standardisation, so
        the metric reflects information actually present in the input.
    min_unique_frac : float or None, default 0.15
        Drop a feature when ``n_unique_non_nan / n_rows`` is strictly below
        this threshold, i.e. the column is dominated by a small handful of
        values (a hidden "near-constant").  ``None`` disables the check.

        .. note::
            The unique fraction is measured on **raw**, pre-standardised
            values.  Cross-sectional rank standardisation can otherwise
            artificially collapse unique counts (continuous features get
            mapped to at most ``cross-section size`` distinct ranks per
            period), making post-transform unique-fraction misleading.
            Binary / coarsely-categorical features can fail the default
            threshold; lower ``min_unique_frac`` or set it to ``None`` if
            you intend to use them.
    verbose : int, default 0
        ``>= 1`` logs the list of dropped features.

    Attributes
    ----------
    dropped_features_ : dict
        ``{feature_name: reason}`` mapping for every feature filtered out
        by the low-quality screen.  Populated after :meth:`fit`.
    """

    def __init__(
        self,
        cs_rank_standardize: bool = True,
        vol_window: int = 60,
        min_obs: int = 20,
        fillna_method: Optional[str] = "ffill",
        max_nan_frac: Optional[float] = 0.5,
        min_unique_frac: Optional[float] = 0.15,
        verbose: int = 0,
    ):
        if max_nan_frac is not None and not (0.0 <= max_nan_frac <= 1.0):
            raise ValueError(
                f"max_nan_frac must lie in [0, 1] or be None; got {max_nan_frac!r}."
            )
        if min_unique_frac is not None and not (0.0 <= min_unique_frac <= 1.0):
            raise ValueError(
                f"min_unique_frac must lie in [0, 1] or be None; got "
                f"{min_unique_frac!r}."
            )
        self.cs_rank_standardize = cs_rank_standardize
        self.vol_window = vol_window
        self.min_obs = min_obs
        self.fillna_method = fillna_method
        self.max_nan_frac = max_nan_frac
        self.min_unique_frac = min_unique_frac
        self.verbose = verbose

        # state populated by .fit()
        self._feature_names: Optional[List[str]] = None
        self._is_fitted: bool = False
        self.dropped_features_: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_col: str = "date",
        entity_col: str = "asset_id",
    ) -> "DataHandler":
        """Learn metadata from the panel.

        Parameters
        ----------
        X : DataFrame
            Panel of features.  Must contain *time_col* and *entity_col* as
            columns or as a MultiIndex with levels named accordingly.
        y : Series
            Target variable aligned with ``X``.
        time_col, entity_col : str
            Column names (or index level names) identifying time and entity.
        """
        self._time_col = time_col
        self._entity_col = entity_col
        X, y = self._align_index(X, y)
        feature_cols = [
            c for c in X.columns if c not in (time_col, entity_col)
        ]
        # Screen out low-quality columns on the *raw* values, before any fill
        # or standardisation skews the unique-count / NaN-rate statistics.
        feature_cols, dropped = self._screen_low_quality(X, feature_cols)
        self.dropped_features_ = dropped
        if dropped and self.verbose >= 1:
            logger.info(
                "DataHandler: dropped %d low-quality feature(s): %s",
                len(dropped),
                {k: v for k, v in dropped.items()},
            )
        self._feature_names = feature_cols
        self._is_fitted = True
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ret_series_for_vol: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply all preprocessing steps.

        Parameters
        ----------
        X : DataFrame
            Raw feature panel.
        y : Series
            Target variable.
        ret_series_for_vol : Series or None
            Return series used to compute rolling volatility weights.
            If ``None`` and volatility weights are later required, the engine
            will fall back to ``y``.

        Returns
        -------
        X_processed : DataFrame
            Cleaned and (optionally) rank-standardized feature panel.  Columns
            include ``self._time_col``, ``self._entity_col``, plus features.
        y_processed : Series
            Target aligned with ``X_processed``.
        vol_weights : Series or None
            Inverse-volatility weights (``1 / sigma``), or ``None`` if
            *ret_series_for_vol* was not provided.
        """
        assert self._is_fitted, "Call .fit() before .transform()."
        X, y = self._align_index(X, y)
        # Restrict to the (kept) feature set + time/entity housekeeping
        # columns; this also guarantees ``transform`` is robust to extra
        # columns the caller may have left in the input frame.
        keep_cols = [self._time_col, self._entity_col] + list(
            self._feature_names or []
        )
        X = X[[c for c in keep_cols if c in X.columns]].copy()
        X = self._fill_missing(X)

        vol_weights = None
        if ret_series_for_vol is not None:
            vol_weights = self._compute_vol_weights(ret_series_for_vol, X)

        if self.cs_rank_standardize:
            X = self._cross_sectional_rank(X)

        return X, y, vol_weights

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_col: str = "date",
        entity_col: str = "asset_id",
        ret_series_for_vol: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Convenience wrapper: ``fit`` then ``transform``."""
        self.fit(X, y, time_col=time_col, entity_col=entity_col)
        return self.transform(X, y, ret_series_for_vol=ret_series_for_vol)

    @property
    def feature_names(self) -> List[str]:
        assert self._is_fitted, "Call .fit() first."
        return list(self._feature_names)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _screen_low_quality(
        self, X: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[List[str], Dict[str, str]]:
        """Return ``(kept, dropped)`` partition of *feature_cols*.

        A column is dropped when *either* (i) its raw NaN fraction exceeds
        ``max_nan_frac`` or (ii) its ``n_unique_non_nan / n_rows`` falls
        below ``min_unique_frac``.  Either check can be disabled by setting
        the corresponding parameter to ``None``.

        ``dropped`` records the human-readable reason per column for
        downstream inspection and logging.
        """
        if not feature_cols:
            return feature_cols, {}
        # Both checks disabled → fast-path the original behaviour.
        if self.max_nan_frac is None and self.min_unique_frac is None:
            return feature_cols, {}

        n_rows = int(len(X))
        if n_rows == 0:
            return feature_cols, {}

        kept: List[str] = []
        dropped: Dict[str, str] = {}
        for col in feature_cols:
            s = X[col]
            nan_frac = float(s.isna().mean()) if n_rows > 0 else 0.0
            if self.max_nan_frac is not None and nan_frac > self.max_nan_frac:
                dropped[col] = (
                    f"nan_frac={nan_frac:.3f} > max_nan_frac={self.max_nan_frac}"
                )
                continue
            if self.min_unique_frac is not None:
                # ``nunique(dropna=True)`` ignores missing observations so a
                # column that is "constant on the present rows" is still
                # caught even if it has a few NaNs.
                n_uniq = int(s.nunique(dropna=True))
                uniq_frac = n_uniq / n_rows
                if uniq_frac < self.min_unique_frac:
                    dropped[col] = (
                        f"unique_frac={uniq_frac:.4f} < min_unique_frac="
                        f"{self.min_unique_frac}"
                    )
                    continue
            kept.append(col)
        return kept, dropped

    def _align_index(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Ensure *X* has time and entity as regular columns and align *y*."""
        tc, ec = self._time_col, self._entity_col

        # Promote MultiIndex levels to columns if needed
        if isinstance(X.index, pd.MultiIndex):
            if tc in X.index.names or ec in X.index.names:
                X = X.reset_index()
        for col in (tc, ec):
            if col not in X.columns:
                raise ValueError(
                    f"Column '{col}' not found in X. "
                    "Pass correct *time_col* / *entity_col*."
                )

        # Align y to X on the same index
        common = X.index.intersection(y.index)
        X = X.loc[common]
        y = y.loc[common]
        return X, y

    def _fill_missing(self, X: pd.DataFrame) -> pd.DataFrame:
        feature_cols = self._feature_names
        if self.fillna_method is None or feature_cols is None:
            return X
        Xf = X.copy()
        if self.fillna_method == "ffill":
            Xf[feature_cols] = (
                Xf.groupby(self._entity_col)[feature_cols]
                .ffill()
            )
        elif self.fillna_method == "bfill":
            Xf[feature_cols] = (
                Xf.groupby(self._entity_col)[feature_cols]
                .bfill()
            )
        elif self.fillna_method == "zero":
            Xf[feature_cols] = Xf[feature_cols].fillna(0.0)
        elif self.fillna_method == "mean":
            means = Xf.groupby(self._time_col)[feature_cols].transform("mean")
            Xf[feature_cols] = Xf[feature_cols].fillna(means)
        # After group-level fill, fill remaining NaNs with 0
        Xf[feature_cols] = Xf[feature_cols].fillna(0.0)
        return Xf

    def _cross_sectional_rank(self, X: pd.DataFrame) -> pd.DataFrame:
        """Rank-standardize features within each cross-section to [0, 1]."""
        feature_cols = self._feature_names
        if feature_cols is None:
            return X
        Xr = X.copy()
        ranked = Xr.groupby(self._time_col)[feature_cols].rank(pct=True)
        Xr[feature_cols] = ranked
        return Xr

    def _compute_vol_weights(
        self, ret_series: pd.Series, X: pd.DataFrame
    ) -> pd.Series:
        """Compute inverse rolling-volatility weights aligned with X."""
        # Build a frame with entity, time, and returns
        df = X[[self._time_col, self._entity_col]].copy()
        df["__ret__"] = ret_series.reindex(df.index).values

        vol = (
            df.groupby(self._entity_col)["__ret__"]
            .rolling(window=self.vol_window, min_periods=self.min_obs)
            .std()
            .reset_index(level=0, drop=True)
        )
        vol = vol.reindex(X.index)
        # Clip to avoid division by zero
        vol = vol.clip(lower=1e-8)
        return 1.0 / vol
