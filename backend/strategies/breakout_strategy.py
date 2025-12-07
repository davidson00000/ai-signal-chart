from typing import Tuple, Dict, Any
import pandas as pd
from backend.strategies.base_strategy import BaseStrategy

class BreakoutStrategy(BaseStrategy):
    """
    Breakout Strategy (20-day High).
    
    Entry: Close > Highest(High, 20) (Donchian Channel Breakout)
    Exit: Close < Lowest(Low, 10)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.breakout_window = config.get("breakout_window", 20)
        self.exit_window = config.get("exit_window", 10)

    def generate_action(self, df: pd.DataFrame, idx: int) -> Tuple[str, Dict[str, Any]]:
        lookback = max(self.breakout_window, self.exit_window)
        if idx < lookback:
            return "hold", {"reason": "Insufficient data"}
            
        # We need previous bars to calculate the channel, excluding current bar to avoid lookahead?
        # Standard Donchian: High of last N bars.
        # If we use current bar's close vs High of LAST N bars (excluding current), it's a breakout.
        
        subset = df.iloc[:idx] # Exclude current bar for channel calculation
        if len(subset) < lookback:
             return "hold", {"reason": "Insufficient data"}
             
        current_bar = df.iloc[idx]
        current_close = current_bar['close']
        
        # Calculate channels based on PREVIOUS bars
        recent_highs = subset['high'].tail(self.breakout_window)
        highest_high = recent_highs.max()
        
        recent_lows = subset['low'].tail(self.exit_window)
        lowest_low = recent_lows.min()
        
        action = "hold"
        reason = f"Close={current_close:.2f}, High({self.breakout_window})={highest_high:.2f}, Low({self.exit_window})={lowest_low:.2f}"
        
        if current_close > highest_high:
            action = "buy"
            reason = f"Breakout: Close {current_close:.2f} > {self.breakout_window}-day High {highest_high:.2f}"
        elif current_close < lowest_low:
            action = "sell"
            reason = f"Exit: Close {current_close:.2f} < {self.exit_window}-day Low {lowest_low:.2f}"
            
        return action, {
            "strategy": "breakout",
            "highest_high": highest_high,
            "lowest_low": lowest_low,
            "reason": reason
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for the entire DataFrame at once.
        """
        df = df.copy()
        
        # Calculate Channels (shifted by 1 to exclude current bar)
        df['highest_high'] = df['high'].rolling(window=self.breakout_window).max().shift(1)
        df['lowest_low'] = df['low'].rolling(window=self.exit_window).min().shift(1)
        
        # Generate Signals
        df['signal'] = 0
        
        # Buy when Close > Highest High
        buy_cond = df['close'] > df['highest_high']
        
        # Sell when Close < Lowest Low
        sell_cond = df['close'] < df['lowest_low']
        
        df.loc[buy_cond, 'signal'] = 1
        df.loc[sell_cond, 'signal'] = -1
        
        return df
