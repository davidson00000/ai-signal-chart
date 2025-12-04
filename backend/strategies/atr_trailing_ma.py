from typing import Dict, Any
import pandas as pd
import numpy as np
from backend.strategies.base import StrategyBase

class ATRTrailingMAStrategy(StrategyBase):
    """
    ATR Trailing Stop Strategy.
    Entry: MA Cross (Short > Long).
    Exit: Trailing Stop based on ATR (High - ATR * Multiplier).
    """

    def __init__(self, short_window: int = 10, long_window: int = 30, 
                 atr_period: int = 14, atr_multiplier: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        self.short_window = short_window
        self.long_window = long_window
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Calculate MAs
        short_ma = df['close'].rolling(window=self.short_window).mean()
        long_ma = df['close'].rolling(window=self.long_window).mean()
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        
        # Generate signals using loop for stateful logic
        signals = pd.Series(0, index=df.index)
        in_position = False
        highest_close = 0.0
        
        # Pre-calculate entry condition to speed up
        ma_cross_entry = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        
        # Iterate
        # Note: This is slower than vectorization but necessary for trailing stop
        # We can optimize by iterating over numpy arrays
        close_arr = df['close'].values
        atr_arr = atr.values
        entry_arr = ma_cross_entry.values
        signal_arr = np.zeros(len(df))
        
        for i in range(len(df)):
            if i < max(self.long_window, self.atr_period):
                continue
                
            if not in_position:
                if entry_arr[i]:
                    in_position = True
                    highest_close = close_arr[i]
                    signal_arr[i] = 1
            else:
                # Update highest close
                if close_arr[i] > highest_close:
                    highest_close = close_arr[i]
                
                # Check exit
                stop_price = highest_close - (atr_arr[i] * self.atr_multiplier)
                if close_arr[i] < stop_price:
                    in_position = False
                    signal_arr[i] = 0
                else:
                    signal_arr[i] = 1
                    
        return pd.Series(signal_arr, index=df.index)

    @classmethod
    def get_params_schema(cls) -> Dict[str, Any]:
        return {
            "short_window": {
                "type": "int",
                "min": 1,
                "max": 200,
                "default": 10,
                "step": 1,
                "label": "Short Window (MA)"
            },
            "long_window": {
                "type": "int",
                "min": 1,
                "max": 400,
                "default": 30,
                "step": 1,
                "label": "Long Window (MA)"
            },
            "atr_period": {
                "type": "int",
                "min": 2,
                "max": 50,
                "default": 14,
                "step": 1,
                "label": "ATR Period"
            },
            "atr_multiplier": {
                "type": "float",
                "min": 0.5,
                "max": 10.0,
                "default": 3.0,
                "step": 0.1,
                "label": "ATR Multiplier"
            }
        }
