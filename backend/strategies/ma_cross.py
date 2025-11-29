# backend/strategies/ma_cross.py
from __future__ import annotations

from typing import Tuple, Optional

import pandas as pd

from .base import BaseStrategy


class MACrossStrategy(BaseStrategy):
    """
    シンプル移動平均クロス戦略。

    tests/test_strategies.py が期待している仕様：
    - デフォルト: short_window=9, long_window=21
    - strategy.params['short_window'], ['long_window'] が存在
    - short_window >= long_window なら ValueError
    - validate_dataframe で len(df) < long_window のとき
      ValueError("Insufficient data ...") を投げる
    - get_ma_values(df) -> (short_ma, long_ma)
    - generate_signals(df) は Series を返す（値は 0 or 1）
      * 1: ロング
      * 0: ノーポジ
    """

    def __init__(
        self,
        short_window: int = 9,
        long_window: int = 21,
        name: Optional[str] = None,
    ) -> None:
        if short_window <= 0 or long_window <= 0:
            raise ValueError("Window sizes must be positive integers.")

        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window.")

        self.short_window = int(short_window)
        self.long_window = int(long_window)

        params = {
            "short_window": self.short_window,
            "long_window": self.long_window,
        }
        super().__init__(name=name, **params)

    # ---- validation ----
    def validate_dataframe(self, df: pd.DataFrame) -> None:
        # Base の共通チェック（empty, columns, index 等）
        super().validate_dataframe(df)

        # ⇨ テスト: "Insufficient data" という文字列でマッチ
        if len(df) < self.long_window:
            raise ValueError(
                f"Insufficient data for long_window={self.long_window}. "
                f"Got {len(df)} rows."
            )

    # ---- indicator helpers ----
    def get_ma_values(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        短期・長期の移動平均 Series を返す（df.index と同じ長さ）。
        """
        self.validate_dataframe(df)

        close = df["close"].astype(float)

        short_ma = close.rolling(
            window=self.short_window,
            min_periods=self.short_window,
        ).mean()

        long_ma = close.rolling(
            window=self.long_window,
            min_periods=self.long_window,
        ).mean()

        return short_ma, long_ma

    # ---- main signal generator ----
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        - short_ma > long_ma → 1（ロング）
        - それ以外（short_ma <= long_ma, NaN含む）→ 0（ノーポジ）

        ※テストで「0 or 1 だけ」を要求されているので -1 は使わない。
        """
        self.validate_dataframe(df)

        short_ma, long_ma = self.get_ma_values(df)

        # デフォルトは 0（ノーポジ）
        signals = pd.Series(0, index=df.index, dtype="int64")

        valid = short_ma.notna() & long_ma.notna()
        signals[valid & (short_ma > long_ma)] = 1

        return signals
