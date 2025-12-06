from typing import Tuple, Dict, Any
import pandas as pd
from backend.strategies.base_strategy import BaseStrategy

class BollingerStrategy(BaseStrategy):
    """
    Bollinger Mean Reversion Strategy.
    
    Entry: Close < Lower Band -> BUY
    Exit: Close >= Middle Band (SMA) -> SELL
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.period = config.get("bb_period", 20)
        self.std_dev = config.get("bb_std", 2.0)

    def generate_action(self, df: pd.DataFrame, idx: int) -> Tuple[str, Dict[str, Any]]:
        if idx < self.period:
            return "hold", {"reason": "Insufficient data"}
            
        subset = df.iloc[:idx+1]
        
        # Calculate Bollinger Bands
        close = subset['close']
        mid = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        
        upper = mid + (std * self.std_dev)
        lower = mid - (std * self.std_dev)
        
        curr_close = close.iloc[-1]
        curr_lower = lower.iloc[-1]
        curr_mid = mid.iloc[-1]
        curr_upper = upper.iloc[-1]
        
        action = "hold"
        reason = f"Close={curr_close:.2f}, Lower={curr_lower:.2f}, Mid={curr_mid:.2f}"
        
        if curr_close < curr_lower:
            action = "buy"
            reason = f"Oversold: Close {curr_close:.2f} < Lower Band {curr_lower:.2f}"
        elif curr_close >= curr_mid:
            action = "sell"
            reason = f"Mean Reversion: Close {curr_close:.2f} >= Mid Band {curr_mid:.2f}"
            
        return action, {
            "strategy": "bollinger",
            "upper": curr_upper,
            "mid": curr_mid,
            "lower": curr_lower,
            "reason": reason
        }
