from typing import Dict, Any
import pandas as pd
import numpy as np
from backend.strategies.base import StrategyBase

class MACrossStrategy(StrategyBase):
    """
    Moving Average Crossover Strategy.
    Buy when Short MA crosses above Long MA.
    Exit when Short MA crosses below Long MA.
    """

    def __init__(self, short_window: int = 10, long_window: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Calculate MAs
        short_ma = df['close'].rolling(window=self.short_window).mean()
        long_ma = df['close'].rolling(window=self.long_window).mean()

        # Generate signals
        # 1 where short_ma > long_ma, -1 otherwise (Sell/Exit)
        signals = np.where(short_ma > long_ma, 1, -1)
        
        return pd.Series(signals, index=df.index)

    @classmethod
    def get_params_schema(cls) -> Dict[str, Any]:
        return {
            "short_window": {
                "type": "int",
                "min": 1,
                "max": 200,
                "default": 10,
                "step": 1,
                "label": "Short Window"
            },
            "long_window": {
                "type": "int",
                "min": 1,
                "max": 400,
                "default": 30,
                "step": 1,
                "label": "Long Window"
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
