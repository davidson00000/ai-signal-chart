# backend/strategies/ma_cross.py
"""
Moving Average Cross Strategy implementation.
"""

from __future__ import annotations

from typing import Tuple, Optional

import pandas as pd

from .base import BaseStrategy


class MACrossStrategy(BaseStrategy):
    """
    シンプル移動平均クロス戦略。

    仕様：
    - デフォルト: short_window=9, long_window=21
    - close 価格の単純移動平均(SMA)を使う
    - シグナルは 0 or 1（ノーポジ or ロング）
    """

    def __init__(
        self,
        short_window: int = 9,
        long_window: int = 21,
    ) -> None:
        """
        Initialize MA Cross Strategy.

        Args:
            short_window: Short-term moving average window
            long_window: Long-term moving average window
        """
        # BaseStrategy の __init__ は空なので呼ばなくても良いが、一応呼んでおく
        super().__init__()

        if short_window < 1:
            raise ValueError("short_window must be >= 1.")
        if long_window <= short_window:
            raise ValueError(
                f"long_window must be > short_window (got {long_window} <= {short_window})."
            )

        self.short_window = int(short_window)
        self.long_window = int(long_window)

    def __str__(self) -> str:
        """Strategy name for display."""
        return f"MA Cross ({self.short_window}/{self.long_window})"

    # ---- validation ----
    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate DataFrame has required columns and sufficient data.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If validation fails
        """
        super().validate_dataframe(df)

        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column.")

        if len(df) < self.long_window:
            raise ValueError(
                f"Insufficient data for long_window={self.long_window}. "
                f"Got {len(df)} rows."
            )

    # ---- indicator helpers ----
    def get_ma_values(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate short and long moving averages.

        Args:
            df: DataFrame with 'close' column

        Returns:
            Tuple of (short_ma, long_ma) Series
        """
        self.validate_dataframe(df)

        close = df["close"].astype(float)

        short_ma = close.rolling(window=self.short_window, min_periods=self.short_window).mean()
        long_ma = close.rolling(window=self.long_window, min_periods=self.long_window).mean()

        return short_ma, long_ma

    # ---- signal generation ----
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MA crossover.

        Rules:
        - short_ma > long_ma → 1 (Long)
        - short_ma <= long_ma or NaN → 0 (No position)

        Args:
            df: DataFrame with OHLCV data

        Returns:
            pd.Series: Signals (0 or 1)
        """
        self.validate_dataframe(df)

        short_ma, long_ma = self.get_ma_values(df)

        # デフォルトは 0（ノーポジ）
        signals = pd.Series(0, index=df.index, dtype="int64")

        # MA が計算可能な行のみでシグナル判定
        valid = short_ma.notna() & long_ma.notna()
        signals[valid & (short_ma > long_ma)] = 1

        return signals
