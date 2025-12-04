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
