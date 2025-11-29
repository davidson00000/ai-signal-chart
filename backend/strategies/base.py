"""
Base strategy class for backtesting.
All concrete strategies must implement `generate_signals(df)` which returns
a pandas Series of signals (+1, -1, 0).
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseStrategy(ABC):
    """
    抽象戦略基底クラス。

    すべての戦略は generate_signals(df: pd.DataFrame) -> pd.Series を実装する必要がある。
    """

    def __init__(self):
        """
        基本的な初期化。サブクラスで super().__init__() を呼び出す必要はない。
        """
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        売買シグナルを生成する（抽象メソッド）。

        Args:
            df: OHLCV データを含む DataFrame（少なくとも 'close' カラムが必要）

        Returns:
            pd.Series: シグナル (1 = ロング, -1 = ショート/売り, 0 = ノーポジション)
        """
        raise NotImplementedError("Strategy must implement generate_signals()")

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        BacktestEngine との互換性のため、generate() も提供。
        デフォルトでは generate_signals() を呼び出す。
        """
        return self.generate_signals(df)

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        DataFrame の基本的な検証。

        Args:
            df: 検証する DataFrame

        Raises:
            ValueError: DataFrame が空、または基本要件を満たさない場合
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")
