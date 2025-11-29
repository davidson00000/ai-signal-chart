"""
RSI Strategy Implementation
"""
from backend.strategies.base import BaseStrategy
import pandas as pd
from typing import Optional


class RSIStrategy(BaseStrategy):
    """
    RSI (Relative Strength Index) 戦略
    
    - RSI < oversold → BUY signal (1)
    - RSI > overbought → SELL signal (-1)
    - Otherwise → HOLD (0)
    
    Args:
        period: RSI calculation period (default: 14)
        oversold: Oversold threshold (default: 30)
        overbought: Overbought threshold (default: 70)
    """
    
    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0
    ):
        super().__init__()
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def __str__(self) -> str:
        return f"RSI({self.period}, {self.oversold}/{self.overbought})"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on RSI
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series of signals:  1 (BUY), -1 (SELL), 0 (HOLD)
        """
        self.validate_dataframe(df)
        
        from backend.utils.indicators import rsi as calc_rsi
        
        closes = df["close"].values.tolist()
        rsi_values = calc_rsi(closes, self.period)
        
        signals = pd.Series(0, index=df.index)
        
        for i in range(len(rsi_values)):
            if rsi_values[i] is None:
                continue
            
            if rsi_values[i] < self.oversold:
                signals.iloc[i] = 1  # BUY
            elif rsi_values[i] > self.overbought:
                signals.iloc[i] = -1  # SELL
        
        return signals
