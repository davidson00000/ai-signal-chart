"""
Buy & Hold Strategy

Simple strategy that buys at the start and holds until the end.
Used as a baseline comparison for other strategies.
"""

import pandas as pd
from typing import Dict, Any

from backend.strategies.base import StrategyBase


class BuyAndHoldStrategy(StrategyBase):
    """
    Buy & Hold Strategy.
    
    Logic:
    - Buy on the first bar (full position)
    - Hold until the end of the test period
    - No exits, no stop losses
    
    This strategy is useful as a benchmark to compare other strategies.
    """
    
    def __init__(self, **params):
        super().__init__(**params)
    
    @classmethod
    def get_params_schema(cls) -> Dict[str, Any]:
        """
        Return the schema for strategy parameters.
        Buy & Hold has no parameters.
        """
        return {}
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.Series: Signal series (1=buy on first bar, 0=hold after)
        """
        # Initialize all signals to 0 (hold)
        signals = pd.Series(0, index=df.index)
        
        # Buy on the first bar
        if len(signals) > 0:
            signals.iloc[0] = 1  # Buy signal
        
        return signals


# Strategy metadata for registry
STRATEGY_METADATA = {
    "id": "buy_and_hold",
    "name": "Buy & Hold",
    "description": "Buy at start, hold until end of test period. Used as a baseline comparison.",
    "docs_path": "docs/strategies/buy_and_hold.md"
}
