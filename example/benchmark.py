"""
P-Tree 性能基准脚本（benchmark）
================================

用于在重构前后测量 ``PanelTreeEngine`` 的端到端耗时，验证 A 部分
（增量更新修复 / AUC 向量化 / 常数优化 / 真并行）的加速效果，并确保
``predict`` 输出在重构前后保持数值一致（配合 ``test_golden_baseline``）。

用法::

    PYTHONPATH=src ~/.venv/bin/python example/benchmark.py
    PYTHONPATH=src ~/.venv/bin/python example/benchmark.py --T 120 --N 500 --P 30 --repeat 3

输出：每次 fit 的耗时统计（min / mean）、树规模、全样本 R²。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from statistics import mean

import numpy as np
import pandas as pd

# 将 src/ 加入搜索路径（与 demo 一致，便于直接运行）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ptree import (  # noqa: E402
    DataHandler,
    PanelTreeEngine,
    RidgeRegressor,
    R2DiffCriterion,
)


def make_panel_data(T: int, N: int, P: int, seed: int = 2026):
    """生成可复现的合成面板数据。

    构造"可预测性集中在 char_1 > 0 区间"的结构，便于 P-Tree 产生
    有意义的分裂。返回 (df, y_series, feature_names)。
    """
    rng = np.random.default_rng(seed)
    feature_names = [f"char_{i + 1}" for i in range(P)]

    dates = np.repeat(np.arange(T), N)
    asset_ids = np.tile(np.arange(N), T)
    X_raw = rng.standard_normal((T * N, P))

    # 强可预测 regime：char_1 > 0 时多个特征共同驱动 alpha
    alpha_strong = (
        0.6 * X_raw[:, 1 % P]
        - 0.4 * X_raw[:, 2 % P]
        + 0.3 * X_raw[:, 3 % P]
    )
    alpha_weak = 0.05 * X_raw[:, 1 % P]
    mask_strong = X_raw[:, 0] > 0
    y_raw = np.where(mask_strong, alpha_strong, alpha_weak)
    vol = 0.3 + 0.5 * np.abs(X_raw[:, 2 % P])
    y_raw = y_raw + rng.standard_normal(T * N) * vol

    df = pd.DataFrame(X_raw, columns=feature_names)
    df["date"] = dates
    df["asset_id"] = asset_ids
    y_series = pd.Series(y_raw, name="ret")
    return df, y_series, feature_names


def build_engine(**overrides) -> PanelTreeEngine:
    params = dict(
        predictor=RidgeRegressor(alpha=1.0),
        criterion=R2DiffCriterion(),
        split_thresholds=[0.3, 0.5, 0.7],
        max_depth=4,
        min_samples=200,
        verbose=0,
    )
    params.update(overrides)
    return PanelTreeEngine(**params)


def run_benchmark(T: int, N: int, P: int, repeat: int, n_jobs: int) -> None:
    print("=" * 64)
    print(f"P-Tree benchmark  (T={T}, N={N}, P={P}, repeat={repeat}, n_jobs={n_jobs})")
    print("=" * 64)

    df, y_series, feature_names = make_panel_data(T, N, P)
    print(f"  样本总量: {len(df):,}  特征数: {P}")

    dh = DataHandler(cs_rank_standardize=True)
    t0 = time.perf_counter()
    X_proc, y_proc, _ = dh.fit_transform(
        df, y_series, time_col="date", entity_col="asset_id"
    )
    print(f"  预处理耗时: {time.perf_counter() - t0:.3f}s")

    fit_times = []
    leaves = 0
    nodes = 0
    overall_r2 = float("nan")
    for i in range(repeat):
        engine = build_engine(n_jobs=n_jobs)
        t0 = time.perf_counter()
        engine.fit(X_proc, y_proc, feature_names=dh.feature_names)
        dt = time.perf_counter() - t0
        fit_times.append(dt)

        leaves = len(engine.get_leaves())
        nodes = len(engine.get_all_nodes())
        preds = engine.predict(X_proc)
        ss_res = np.nansum((y_proc.values - preds) ** 2)
        ss_tot = np.nansum((y_proc.values - y_proc.mean()) ** 2)
        overall_r2 = 1 - ss_res / ss_tot
        print(f"  [run {i + 1}/{repeat}] fit={dt:.3f}s  nodes={nodes} leaves={leaves}")

    print("-" * 64)
    print(f"  fit 耗时: min={min(fit_times):.3f}s  mean={mean(fit_times):.3f}s")
    print(f"  树规模:   nodes={nodes}  leaves={leaves}")
    print(f"  全样本 R²: {overall_r2:.6f}")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="P-Tree benchmark")
    parser.add_argument("--T", type=int, default=120, help="时间期数")
    parser.add_argument("--N", type=int, default=500, help="每期资产数")
    parser.add_argument("--P", type=int, default=30, help="特征数")
    parser.add_argument("--repeat", type=int, default=3, help="重复次数")
    parser.add_argument("--n_jobs", type=int, default=1, help="并行 worker 数")
    args = parser.parse_args()
    run_benchmark(args.T, args.N, args.P, args.repeat, args.n_jobs)


if __name__ == "__main__":
    main()
