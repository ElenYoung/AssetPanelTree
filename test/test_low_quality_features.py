"""
Low-quality feature filtering — DataHandler + PanelTreeEngine
=============================================================

Covers the two parameterised guards added in 2026-06:

* ``DataHandler.max_nan_frac`` and ``DataHandler.min_unique_frac`` — drop
  high-NaN-rate and near-constant feature columns at preprocessing time.
* ``PanelTreeEngine.min_unique_per_split`` and ``min_feature_variance`` —
  per-node split-time guard that rejects degenerate columns even when
  they survived the DataHandler screen (defensive depth, plus mid-tree
  protection on samples that the parent split happened to thin out).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ptree import (
    DataHandler,
    PanelTreeEngine,
    R2DiffCriterion,
    RidgeRegressor,
)


def _make_synth_panel(seed: int = 0):
    """Synthetic panel:

    * char_1 .. char_3 : usable Gaussian features
    * char_nan         : ~80% NaN (above default ``max_nan_frac=0.5``)
    * char_const       : single unique value (below ``min_unique_frac=0.15``)
    * char_binary      : binary 0/1 (only 2 unique → below 0.15 in 2400 rows)
    """
    rng = np.random.default_rng(seed)
    T, N = 30, 80
    n = T * N
    df = pd.DataFrame({
        "date": np.repeat(np.arange(T), N),
        "asset_id": np.tile(np.arange(N), T),
        "char_1": rng.standard_normal(n),
        "char_2": rng.standard_normal(n),
        "char_3": rng.standard_normal(n),
        "char_nan": rng.standard_normal(n),
        "char_const": np.full(n, 0.5),
        "char_binary": rng.integers(0, 2, size=n).astype(float),
    })
    nan_mask = rng.random(n) < 0.8
    df.loc[nan_mask, "char_nan"] = np.nan
    y = pd.Series(
        0.5 * df["char_1"].values
        - 0.3 * df["char_2"].values
        + 0.2 * df["char_3"].values
        + rng.standard_normal(n) * 0.3,
        name="ret",
    )
    return df, y


# ----------------------------------------------------------------------
# DataHandler
# ----------------------------------------------------------------------


class TestDataHandlerLowQualityFilters:
    def test_default_drops_nan_and_constant_and_binary(self):
        df, y = _make_synth_panel()
        dh = DataHandler(cs_rank_standardize=True)  # defaults: 0.5 / 0.15
        dh.fit(df, y, time_col="date", entity_col="asset_id")
        kept = set(dh.feature_names)
        dropped = set(dh.dropped_features_)

        assert "char_1" in kept
        assert "char_2" in kept
        assert "char_3" in kept
        # NaN-rate filter
        assert "char_nan" in dropped
        assert "nan_frac" in dh.dropped_features_["char_nan"]
        # Single-value column → unique_frac = 1/n < 0.15
        assert "char_const" in dropped
        # Binary column → unique_frac = 2/n << 0.15
        assert "char_binary" in dropped

    def test_disabled_filters_keep_everything(self):
        df, y = _make_synth_panel()
        dh = DataHandler(
            cs_rank_standardize=True,
            max_nan_frac=None,
            min_unique_frac=None,
        )
        dh.fit(df, y, time_col="date", entity_col="asset_id")
        kept = set(dh.feature_names)
        assert {"char_1", "char_2", "char_3",
                "char_nan", "char_const", "char_binary"} <= kept
        assert dh.dropped_features_ == {}

    def test_threshold_overrides_keep_binary(self):
        """Lowering ``min_unique_frac`` rescues a binary feature."""
        df, y = _make_synth_panel()
        dh = DataHandler(
            cs_rank_standardize=True,
            max_nan_frac=0.5,
            min_unique_frac=1e-4,  # 2 unique in 2400 rows = 8e-4 > threshold
        )
        dh.fit(df, y, time_col="date", entity_col="asset_id")
        kept = set(dh.feature_names)
        assert "char_binary" in kept
        # NaN filter still active
        assert "char_nan" in dh.dropped_features_

    def test_transform_drops_columns_consistently(self):
        df, y = _make_synth_panel()
        dh = DataHandler(cs_rank_standardize=True)
        X_proc, y_proc, _ = dh.fit_transform(
            df, y, time_col="date", entity_col="asset_id"
        )
        # Dropped columns must not appear in the processed frame.
        for c in dh.dropped_features_:
            assert c not in X_proc.columns
        # Kept features must all be present.
        for c in dh.feature_names:
            assert c in X_proc.columns

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            DataHandler(max_nan_frac=1.5)
        with pytest.raises(ValueError):
            DataHandler(min_unique_frac=-0.1)


# ----------------------------------------------------------------------
# PanelTreeEngine
# ----------------------------------------------------------------------


def _fit_with_constant_column(min_unique_per_split: int = 2,
                              min_feature_variance: float = 0.0):
    """Inject a constant column into a clean processed panel and fit a tree."""
    rng = np.random.default_rng(0)
    T, N = 20, 60
    n = T * N
    feature_names = ["char_1", "char_2", "char_const"]
    df = pd.DataFrame({
        "date": np.repeat(np.arange(T), N),
        "asset_id": np.tile(np.arange(N), T),
        "char_1": rng.uniform(0, 1, n),
        "char_2": rng.uniform(0, 1, n),
        "char_const": np.full(n, 0.5),  # truly constant
    })
    y = pd.Series(
        0.5 * df["char_1"].values - 0.3 * df["char_2"].values
        + rng.standard_normal(n) * 0.2,
        name="ret",
    )
    engine = PanelTreeEngine(
        predictor=RidgeRegressor(alpha=1.0),
        criterion=R2DiffCriterion(),
        split_thresholds=[0.3, 0.5, 0.7],
        max_depth=2,
        min_samples=100,
        min_unique_per_split=min_unique_per_split,
        min_feature_variance=min_feature_variance,
        verbose=0,
    )
    engine.fit(df, y, feature_names=feature_names)
    return engine


class TestEngineLowQualityFeatureGuards:
    def test_constant_column_never_chosen(self):
        engine = _fit_with_constant_column()
        chosen = {
            node.split_feature
            for node in engine.get_all_nodes()
            if node.split_feature is not None
        }
        assert "char_const" not in chosen

    def test_min_feature_variance_zero_does_not_change_existing_behaviour(self):
        # Just smoke-test that variance=0 (default) is identical to "off".
        eng_default = _fit_with_constant_column(min_feature_variance=0.0)
        eng_explicit = _fit_with_constant_column(min_feature_variance=1e-12)
        sig_default = [
            (n.node_id, n.split_feature, n.split_threshold)
            for n in eng_default.get_all_nodes()
        ]
        sig_explicit = [
            (n.node_id, n.split_feature, n.split_threshold)
            for n in eng_explicit.get_all_nodes()
        ]
        assert sig_default == sig_explicit

    def test_invalid_min_unique_per_split_raises(self):
        with pytest.raises(ValueError):
            PanelTreeEngine(min_unique_per_split=1)
        with pytest.raises(ValueError):
            PanelTreeEngine(min_feature_variance=-0.1)

    def test_higher_min_unique_per_split_rejects_low_cardinality_columns(self):
        """A column with only 3 distinct values is filtered out at threshold 5."""
        rng = np.random.default_rng(1)
        T, N = 20, 60
        n = T * N
        feature_names = ["char_useful", "char_lowcard"]
        df = pd.DataFrame({
            "date": np.repeat(np.arange(T), N),
            "asset_id": np.tile(np.arange(N), T),
            "char_useful": rng.uniform(0, 1, n),
            "char_lowcard": rng.choice([0.1, 0.5, 0.9], size=n),
        })
        y = pd.Series(
            0.5 * df["char_useful"].values + rng.standard_normal(n) * 0.2,
            name="ret",
        )
        engine = PanelTreeEngine(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=R2DiffCriterion(),
            split_thresholds=[0.3, 0.5, 0.7],
            max_depth=2,
            min_samples=100,
            min_unique_per_split=5,  # >3 → rejects char_lowcard
            verbose=0,
        )
        engine.fit(df, y, feature_names=feature_names)
        chosen = {
            n.split_feature for n in engine.get_all_nodes()
            if n.split_feature is not None
        }
        assert "char_lowcard" not in chosen

    def test_honest_path_also_filters_constants(self):
        """Same guard applies in the honest split path."""
        rng = np.random.default_rng(2)
        T, N = 20, 80
        n = T * N
        feature_names = ["char_useful", "char_const"]
        df = pd.DataFrame({
            "date": np.repeat(np.arange(T), N),
            "asset_id": np.tile(np.arange(N), T),
            "char_useful": rng.uniform(0, 1, n),
            "char_const": np.full(n, 0.42),
        })
        y = pd.Series(
            0.5 * df["char_useful"].values + rng.standard_normal(n) * 0.2,
            name="ret",
        )
        engine = PanelTreeEngine(
            predictor=RidgeRegressor(alpha=1.0),
            criterion=R2DiffCriterion(),
            split_thresholds=[0.3, 0.5, 0.7],
            max_depth=2,
            min_samples=100,
            honest=True,
            honest_frac=0.5,
            random_state=42,
            verbose=0,
        )
        engine.fit(df, y, feature_names=feature_names)
        chosen = {
            n.split_feature for n in engine.get_all_nodes()
            if n.split_feature is not None
        }
        assert "char_const" not in chosen
