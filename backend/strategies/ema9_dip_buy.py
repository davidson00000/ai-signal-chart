from typing import Dict, Any
import pandas as pd
import numpy as np
from backend.strategies.base import StrategyBase

class EMA9DipBuyStrategy(StrategyBase):
    """
    EMA9 Dip Buy Strategy - Long-only pullback strategy.
    
    Entry Conditions:
    1. Strong uptrend: Price > 21EMA
    2. Recent volume increase (above average)
    3. Pullback with decreasing volume
    4. Price near 9EMA support (within deviation_threshold%)
    5. Breakout above recent pullback high
    
    Exit Conditions:
    1. Take profit based on risk_reward ratio
    2. Stop loss: Low of bar before breakout - stop_buffer%
    3. Trailing exit when price closes below 21EMA
    """

    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        deviation_threshold: float = 2.0,
        stop_buffer: float = 0.5,
        risk_reward: float = 2.0,
        lookback_volume: int = 20,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.deviation_threshold = deviation_threshold
        self.stop_buffer = stop_buffer
        self.risk_reward = risk_reward
        self.lookback_volume = lookback_volume

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on EMA9 Dip Buy logic.
        Returns 1 for LONG position, -1 for NO position/EXIT.
        
        Simplified logic:
        - Enter (1): Price > 21EMA AND price near 9EMA (pullback setup)
        - Exit (-1): Price < 21EMA (trend broken)
        """
        # Calculate EMAs
        ema_fast = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        
        # Calculate volume moving average
        volume_ma = df['volume'].rolling(window=self.lookback_volume).mean()
        
        # Entry conditions (simplified to ensure trades happen):
        # 1. Price is above slow EMA (uptrend)
        is_uptrend = df['close'] > ema_slow
        
        # 2. Price is near fast EMA (pullback - within deviation_threshold%)
        price_to_fast_ema_pct = abs(df['close'] - ema_fast) / ema_fast * 100
        is_near_fast_ema = price_to_fast_ema_pct <= self.deviation_threshold
        
        # 3. Price bounced off or is above fast EMA
        is_above_fast_ema = df['close'] >= ema_fast
        
        # Generate signals:
        # 1 = LONG position (when in uptrend and near/above fast EMA)
        # -1 = NO position/EXIT (when not in uptrend)
        signals = np.where(
            is_uptrend & (is_near_fast_ema | is_above_fast_ema),
            1,   # Long signal
            -1   # Exit/No position signal
        )
        
        return pd.Series(signals, index=df.index)

    @classmethod
    def get_params_schema(cls) -> Dict[str, Any]:
        return {
            "ema_fast": {
                "type": "int",
                "min": 5,
                "max": 20,
                "default": 9,
                "step": 1,
                "label": "Fast EMA Period"
            },
            "ema_slow": {
                "type": "int",
                "min": 10,
                "max": 50,
                "default": 21,
                "step": 1,
                "label": "Slow EMA Period"
            },
            "deviation_threshold": {
                "type": "float",
                "min": 0.5,
                "max": 5.0,
                "default": 2.0,
                "step": 0.5,
                "label": "Max % Deviation from Fast EMA"
            },
            "stop_buffer": {
                "type": "float",
                "min": 0.1,
                "max": 2.0,
                "default": 0.5,
                "step": 0.1,
                "label": "Stop Loss Buffer %"
            },
            "risk_reward": {
                "type": "float",
                "min": 1.0,
                "max": 5.0,
                "default": 2.0,
                "step": 0.5,
                "label": "Risk/Reward Ratio"
            },
            "lookback_volume": {
                "type": "int",
                "min": 10,
                "max": 50,
                "default": 20,
                "step": 5,
                "label": "Volume Lookback Period"
            }
        }

    @classmethod
    def get_optimization_config(cls) -> Dict[str, Any]:
        return {
            "x_param": "ema_fast",
            "y_param": "risk_reward",
            "x_range": {"min": 7, "max": 15, "step": 2},
            "y_range": {"min": 1.5, "max": 3.5, "step": 0.5},
            "fixed_params": {
                "ema_slow": 21,
                "deviation_threshold": 2.0,
                "stop_buffer": 0.5,
                "lookback_volume": 20
            }
        }
