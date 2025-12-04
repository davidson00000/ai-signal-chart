from typing import Dict, Any
import pandas as pd
import numpy as np
from backend.strategies.base import StrategyBase

class DonchianBreakoutStrategy(StrategyBase):
    """
    Donchian Breakout Strategy.
    Buy when Close > Highest High of last N periods.
    Exit when Close < Lowest Low of last N periods.
    """

    def __init__(self, window: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.window = window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Calculate Donchian Channels (shifted by 1 to avoid lookahead/self-inclusion)
        upper_channel = df['high'].rolling(window=self.window).max().shift(1)
        lower_channel = df['low'].rolling(window=self.window).min().shift(1)
        
        # Generate signals with latch logic
        signals = pd.Series(np.nan, index=df.index)
        
        # Buy condition
        signals[df['close'] > upper_channel] = 1
        
        # Exit condition
        signals[df['close'] < lower_channel] = 0
        
        # Forward fill
        signals = signals.ffill().fillna(0)
        
        return signals

    @classmethod
    def get_params_schema(cls) -> Dict[str, Any]:
        return {
            "window": {
                "type": "int",
                "min": 5,
                "max": 200,
                "default": 20,
                "step": 1,
                "label": "Lookback Window"
            }
        }
