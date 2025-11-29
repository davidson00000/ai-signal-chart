"""strategy_lab/evaluator.py

Strategy DSL と BacktestEngine をつなぐ評価モジュール。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from backend.backtester import BacktestEngine
from .auto_generator import load_strategy_from_dsl, load_dsl


def run_backtest_from_dsl(
    dsl_path: str | Path,
    price_df: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    commission: float = 0.0005,
    position_size: float = 1.0,
) -> Dict[str, Any]:
    """DSL ファイルと価格データから1発でバックテストを実行する。"""
    spec = load_dsl(dsl_path)
    strategy = load_strategy_from_dsl(dsl_path)

    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        position_size=position_size,
    )

    results = engine.run(price_df, strategy)

    results["strategy_meta"] = {
        "dsl_path": str(Path(dsl_path).resolve()),
        "name": spec.name,
        "type": spec.type,
        "params": spec.params,
    }

    return results
