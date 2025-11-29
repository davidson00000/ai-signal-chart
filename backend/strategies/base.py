# backend/strategies/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


class StrategyError(Exception):
    """Base exception for strategy-related errors."""


@dataclass
class BaseStrategy(ABC):
    """
    Base class for all trading strategies.

    - name: 名前（デフォルトはクラス名）
    - params: パラメータ辞書（テストで参照される）
    """

    name: str = "BaseStrategy"
    params: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, name: Optional[str] = None, **params: Any) -> None:
        # テストで strategy.params を見るのでここで必ず設定
        self.name = name or self.__class__.__name__
        self.params = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Return 1D Series of trading signals aligned with df.index.
        Values are typically -1, 0, 1.
        """
        raise NotImplementedError

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        共通の DataFrame バリデーション。

        tests/test_strategies.py では以下を期待している：
        - 空の DataFrame → ValueError(... "empty" ...)
        - 必須カラム欠落 → ValueError(... "missing required columns" ...)
        """
        if df is None:
            raise ValueError("DataFrame is empty.")

        if df.empty:
            # ⇨ "empty" を含むメッセージにしておく
            raise ValueError("DataFrame is empty.")

        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            # ⇨ "missing required columns" というフレーズが必要
            missing_list = sorted(missing)
            raise ValueError(
                f"DataFrame is missing required columns: {missing_list}"
            )

        # index は DatetimeIndex 想定
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a pandas.DatetimeIndex.")

        # 日付で昇順ソート（in-place）
        if not df.index.is_monotonic_increasing:
            df.sort_index(inplace=True)

    def __str__(self) -> str:  # ログ・UI で名前＋パラメータを表示する用
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})"
