from typing import Dict, Any
import pandas as pd
import numpy as np
from backend.strategies.base import StrategyBase

class StochasticOscillatorStrategy(StrategyBase):
    """
    Stochastic Oscillator Strategy.
    Buy when %K crosses above %D.
    Exit when %K crosses below %D.
    """

    def __init__(self, k_period: int = 14, d_period: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.k_period = k_period
        self.d_period = d_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Calculate Stochastic Oscillator
        low_min = df['low'].rolling(window=self.k_period).min()
        high_max = df['high'].rolling(window=self.k_period).max()
        
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=self.d_period).mean()

        # Generate signals
        signals = np.where(k_percent > d_percent, 1, 0)
        
        return pd.Series(signals, index=df.index)

    @classmethod
    def get_params_schema(cls) -> Dict[str, Any]:
        return {
            "k_period": {
                "type": "int",
                "min": 2,
                "max": 100,
                "default": 14,
                "step": 1,
                "label": "%K Period"
            },
            "d_period": {
                "type": "int",
                "min": 2,
                "max": 50,
                "default": 3,
                "step": 1,
                "label": "%D Period"
            }
        }
