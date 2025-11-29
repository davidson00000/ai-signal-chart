"""strategy_lab/optimizers.py

MA Cross などの戦略パラメータを探索するための
シンプルな最適化ユーティリティ。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import pandas as pd

from backend.backtester import BacktestEngine
from backend.strategies.ma_cross import MACrossStrategy


@dataclass
class OptimizationResult:
    params: Dict[str, Any]
    metrics: Dict[str, Any]


def _run_single_ma_backtest(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    initial_capital: float = 1_000_000.0,
    commission: float = 0.0005,
    position_size: float = 1.0,
) -> OptimizationResult:
    """内部用: 特定パラメータで1回バックテスト。"""
    strategy = MACrossStrategy(
        short_window=short_window,
        long_window=long_window,
    )
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        position_size=position_size,
    )
    results = engine.run(df, strategy)

    return OptimizationResult(
        params={"short_window": short_window, "long_window": long_window},
        metrics=results["metrics"],
    )


def grid_search_ma_cross(
    df: pd.DataFrame,
    short_window_range: Iterable[int],
    long_window_range: Iterable[int],
    initial_capital: float = 1_000_000.0,
    commission: float = 0.0005,
    position_size: float = 1.0,
    sort_key: str = "return_pct",
    descending: bool = True,
) -> List[OptimizationResult]:
    """MA クロスのグリッドサーチ。"""
    results: List[OptimizationResult] = []

    for short_w in short_window_range:
        for long_w in long_window_range:
            if short_w >= long_w:
                continue
            res = _run_single_ma_backtest(
                df,
                short_window=short_w,
                long_window=long_w,
                initial_capital=initial_capital,
                commission=commission,
                position_size=position_size,
            )
            results.append(res)

    results.sort(
        key=lambda r: r.metrics.get(sort_key, float("-inf")),
        reverse=descending,
    )
    return results


def best_n_results(results: List[OptimizationResult], n: int = 10) -> List[OptimizationResult]:
    """上位 n 件だけ返すショートカット。"""
    return results[:n]
