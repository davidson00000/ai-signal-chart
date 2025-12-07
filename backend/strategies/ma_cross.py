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

    def explain(self, df: pd.DataFrame, idx: int) -> Dict[str, Any]:
        """Explain the MA crossover signal at a given index."""
        short_ma = df['close'].rolling(window=self.short_window).mean()
        long_ma = df['close'].rolling(window=self.long_window).mean()
        
        short_val = float(short_ma.iloc[idx]) if not pd.isna(short_ma.iloc[idx]) else 0.0
        long_val = float(long_ma.iloc[idx]) if not pd.isna(long_ma.iloc[idx]) else 0.0
        close_val = float(df['close'].iloc[idx])
        
        # Check conditions
        conditions = []
        if short_val > long_val:
            conditions.append(f"Short MA ({self.short_window}) > Long MA ({self.long_window})")
        else:
            conditions.append(f"Short MA ({self.short_window}) < Long MA ({self.long_window})")
        
        # Check for crossover (if previous bar had opposite condition)
        if idx > 0:
            prev_short = short_ma.iloc[idx - 1]
            prev_long = long_ma.iloc[idx - 1]
            if not pd.isna(prev_short) and not pd.isna(prev_long):
                if prev_short <= prev_long and short_val > long_val:
                    conditions.append("Golden Cross (bullish crossover)")
                elif prev_short >= prev_long and short_val < long_val:
                    conditions.append("Death Cross (bearish crossover)")
        
        # Calculate confidence based on MA spread
        ma_spread = abs(short_val - long_val) / close_val if close_val > 0 else 0
        confidence = min(0.95, 0.5 + ma_spread * 10)  # Scale spread to confidence
        
        return {
            "indicators": {
                "short_ma": round(short_val, 2),
                "long_ma": round(long_val, 2),
                "close": round(close_val, 2),
                "ma_spread_pct": round(ma_spread * 100, 2)
            },
            "conditions_triggered": conditions,
            "confidence": round(confidence, 2)
        }

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
