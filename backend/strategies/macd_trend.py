from typing import Dict, Any
import pandas as pd
import numpy as np
from backend.strategies.base import StrategyBase

class MACDTrendStrategy(StrategyBase):
    """
    MACD Trend Strategy.
    Buy when MACD Line crosses above Signal Line.
    Exit when MACD Line crosses below Signal Line.
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, **kwargs):
        super().__init__(**kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Calculate MACD
        ema_fast = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        # Generate signals
        signals = np.where(macd_line > signal_line, 1, 0)
        
        return pd.Series(signals, index=df.index)

    def explain(self, df: pd.DataFrame, idx: int) -> Dict[str, Any]:
        """Explain the MACD trend signal at a given index."""
        # Calculate MACD
        ema_fast = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        
        macd_val = float(macd_line.iloc[idx]) if not pd.isna(macd_line.iloc[idx]) else 0.0
        signal_val = float(signal_line.iloc[idx]) if not pd.isna(signal_line.iloc[idx]) else 0.0
        hist_val = float(histogram.iloc[idx]) if not pd.isna(histogram.iloc[idx]) else 0.0
        close_val = float(df['close'].iloc[idx])
        
        # Check conditions
        conditions = []
        if macd_val > signal_val:
            conditions.append("MACD Line > Signal Line - Bullish")
        else:
            conditions.append("MACD Line < Signal Line - Bearish")
        
        if hist_val > 0:
            conditions.append("MACD Histogram positive")
        else:
            conditions.append("MACD Histogram negative")
        
        # Check for crossover
        if idx > 0:
            prev_macd = macd_line.iloc[idx - 1]
            prev_signal = signal_line.iloc[idx - 1]
            if not pd.isna(prev_macd) and not pd.isna(prev_signal):
                if prev_macd <= prev_signal and macd_val > signal_val:
                    conditions.append("MACD bullish crossover")
                elif prev_macd >= prev_signal and macd_val < signal_val:
                    conditions.append("MACD bearish crossover")
        
        # Calculate confidence based on histogram strength
        close_range = df['close'].rolling(20).std().iloc[idx] if idx >= 20 else df['close'].std()
        if not pd.isna(close_range) and close_range > 0:
            confidence = min(0.95, 0.5 + abs(hist_val) / close_range * 2)
        else:
            confidence = 0.5
        
        return {
            "indicators": {
                "macd_line": round(macd_val, 4),
                "signal_line": round(signal_val, 4),
                "histogram": round(hist_val, 4),
                "close": round(close_val, 2)
            },
            "conditions_triggered": conditions,
            "confidence": round(confidence, 2)
        }

    @classmethod
    def get_params_schema(cls) -> Dict[str, Any]:
        return {
            "fast_period": {
                "type": "int",
                "min": 2,
                "max": 100,
                "default": 12,
                "step": 1,
                "label": "Fast Period"
            },
            "slow_period": {
                "type": "int",
                "min": 2,
                "max": 200,
                "default": 26,
                "step": 1,
                "label": "Slow Period"
            },
            "signal_period": {
                "type": "int",
                "min": 2,
                "max": 50,
                "default": 9,
                "step": 1,
                "label": "Signal Period"
            }
        }

    @classmethod
    def get_optimization_config(cls) -> Dict[str, Any]:
        return {
            "x_param": "fast_period",
            "y_param": "slow_period",
            "x_range": {"min": 5, "max": 30, "step": 1},
            "y_range": {"min": 20, "max": 60, "step": 2},
            "fixed_params": {"signal_period": 9}
        }
