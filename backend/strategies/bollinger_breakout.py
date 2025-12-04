from typing import Dict, Any
import pandas as pd
import numpy as np
from backend.strategies.base import StrategyBase

class BollingerBreakoutStrategy(StrategyBase):
    """
    Bollinger Breakout Strategy.
    Buy when Close > Upper Band.
    Exit when Close < Middle Band.
    """

    def __init__(self, window: int = 20, num_std: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Calculate Bollinger Bands
        middle_band = df['close'].rolling(window=self.window).mean()
        std_dev = df['close'].rolling(window=self.window).std()
        upper_band = middle_band + (std_dev * self.num_std)
        
        # Generate signals with latch logic
        signals = pd.Series(np.nan, index=df.index)
        
        # Buy condition: Close > Upper Band
        signals[df['close'] > upper_band] = 1
        
        # Exit condition: Close < Middle Band
        signals[df['close'] < middle_band] = 0
        
        # Forward fill
        signals = signals.ffill().fillna(0)
        
        return signals

    @classmethod
    def get_params_schema(cls) -> Dict[str, Any]:
        return {
            "window": {
                "type": "int",
                "min": 5,
                "max": 100,
                "default": 20,
                "step": 1,
                "label": "Window"
            },
            "num_std": {
                "type": "float",
                "min": 0.1,
                "max": 5.0,
                "default": 2.0,
                "step": 0.1,
                "label": "Std Dev Multiplier"
            }
        }

    @classmethod
    def get_optimization_config(cls) -> Dict[str, Any]:
        return {
            "x_param": "window",
            "y_param": "num_std",
            "x_range": {"min": 10, "max": 50, "step": 5},
            "y_range": {"min": 1.0, "max": 3.0, "step": 0.2}
        }
