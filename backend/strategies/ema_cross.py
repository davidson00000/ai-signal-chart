from typing import Dict, Any
import pandas as pd
import numpy as np
from backend.strategies.base import StrategyBase

class EMACrossStrategy(StrategyBase):
    """
    Exponential Moving Average Crossover Strategy.
    Buy when Short EMA crosses above Long EMA.
    Exit when Short EMA crosses below Long EMA.
    """

    def __init__(self, short_window: int = 9, long_window: int = 21, **kwargs):
        super().__init__(**kwargs)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Calculate EMAs
        short_ema = df['close'].ewm(span=self.short_window, adjust=False).mean()
        long_ema = df['close'].ewm(span=self.long_window, adjust=False).mean()

        # Generate signals
        signals = np.where(short_ema > long_ema, 1, 0)
        
        return pd.Series(signals, index=df.index)

    @classmethod
    def get_params_schema(cls) -> Dict[str, Any]:
        return {
            "short_window": {
                "type": "int",
                "min": 1,
                "max": 200,
                "default": 9,
                "step": 1,
                "label": "Short Window (EMA)"
            },
            "long_window": {
                "type": "int",
                "min": 1,
                "max": 400,
                "default": 21,
                "step": 1,
                "label": "Long Window (EMA)"
            }
        }

    @classmethod
    def get_optimization_config(cls) -> Dict[str, Any]:
        return {
            "x_param": "short_window",
            "y_param": "long_window",
            "x_range": {"min": 5, "max": 50, "step": 5},
            "y_range": {"min": 20, "max": 200, "step": 10}
        }
