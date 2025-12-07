from typing import Dict, Any
import pandas as pd
import numpy as np
from backend.strategies.base import StrategyBase

class RSIMeanReversionStrategy(StrategyBase):
    """
    RSI Mean Reversion Strategy.
    Buy when RSI < Oversold.
    Hold until RSI > Overbought (Exit).
    """

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals with latch logic
        signals = pd.Series(np.nan, index=df.index)
        
        # Buy condition
        signals[rsi < self.oversold] = 1
        
        # Exit condition
        signals[rsi > self.overbought] = 0
        
        # Forward fill to maintain position between signals
        signals = signals.ffill().fillna(0)
        
        return signals

    def explain(self, df: pd.DataFrame, idx: int) -> Dict[str, Any]:
        """Explain the RSI mean reversion signal at a given index."""
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        rsi_val = float(rsi.iloc[idx]) if not pd.isna(rsi.iloc[idx]) else 50.0
        close_val = float(df['close'].iloc[idx])
        
        # Check conditions
        conditions = []
        if rsi_val < self.oversold:
            conditions.append(f"RSI ({rsi_val:.1f}) < Oversold ({self.oversold}) - BUY signal")
        elif rsi_val > self.overbought:
            conditions.append(f"RSI ({rsi_val:.1f}) > Overbought ({self.overbought}) - EXIT signal")
        else:
            conditions.append(f"RSI ({rsi_val:.1f}) is neutral (between {self.oversold} and {self.overbought})")
        
        # Calculate confidence based on distance from thresholds
        if rsi_val < self.oversold:
            confidence = 0.5 + (self.oversold - rsi_val) / self.oversold * 0.4
        elif rsi_val > self.overbought:
            confidence = 0.5 + (rsi_val - self.overbought) / (100 - self.overbought) * 0.4
        else:
            confidence = 0.5
        
        return {
            "indicators": {
                "rsi": round(rsi_val, 2),
                "close": round(close_val, 2),
                "rsi_period": self.period,
                "oversold_level": self.oversold,
                "overbought_level": self.overbought
            },
            "conditions_triggered": conditions,
            "confidence": round(min(0.95, confidence), 2)
        }

    @classmethod
    def get_params_schema(cls) -> Dict[str, Any]:
        return {
            "period": {
                "type": "int",
                "min": 2,
                "max": 100,
                "default": 14,
                "step": 1,
                "label": "RSI Period"
            },
            "oversold": {
                "type": "int",
                "min": 1,
                "max": 49,
                "default": 30,
                "step": 1,
                "label": "Oversold Level"
            },
            "overbought": {
                "type": "int",
                "min": 51,
                "max": 99,
                "default": 70,
                "step": 1,
                "label": "Overbought Level"
            }
        }

    @classmethod
    def get_optimization_config(cls) -> Dict[str, Any]:
        return {
            "x_param": "oversold",
            "y_param": "overbought",
            "x_range": {"min": 20, "max": 45, "step": 5},
            "y_range": {"min": 55, "max": 80, "step": 5},
            "fixed_params": {"period": 14}
        }


# Strategy metadata for registry
STRATEGY_METADATA = {
    "id": "rsi_mean_reversion",
    "name": "RSI Mean Reversion",
    "description": "Buy when RSI < Oversold, sell when RSI > Overbought. Long only.",
    "docs_path": "docs/strategies/rsi_mean_reversion.md"
}
