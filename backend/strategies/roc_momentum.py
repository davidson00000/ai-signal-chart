from typing import Dict, Any
import pandas as pd
import numpy as np
from backend.strategies.base import StrategyBase

class ROCMomentumStrategy(StrategyBase):
    """
    ROC Momentum Strategy.
    Buy when ROC > Threshold.
    Exit when ROC <= Threshold.
    """

    def __init__(self, period: int = 12, threshold: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Calculate ROC
        roc = df['close'].pct_change(periods=self.period) * 100
        
        # Generate signals
        signals = np.where(roc > self.threshold, 1, 0)
        
        return pd.Series(signals, index=df.index)

    @classmethod
    def get_params_schema(cls) -> Dict[str, Any]:
        return {
            "period": {
                "type": "int",
                "min": 1,
                "max": 100,
                "default": 12,
                "step": 1,
                "label": "ROC Period"
            },
            "threshold": {
                "type": "float",
                "min": -10.0,
                "max": 10.0,
                "default": 0.0,
                "step": 0.1,
                "label": "Threshold (%)"
            }
        }
