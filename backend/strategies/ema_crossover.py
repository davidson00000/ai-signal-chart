from typing import Tuple, Dict, Any
import pandas as pd
import pandas_ta as ta
from backend.strategies.base_strategy import BaseStrategy

class EmaCrossoverStrategy(BaseStrategy):
    """
    EMA Crossover Strategy (12EMA / 26EMA).
    
    Entry: 12EMA crosses above 26EMA -> BUY
    Exit: 12EMA crosses below 26EMA -> SELL
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.short_window = config.get("ema_short", 12)
        self.long_window = config.get("ema_long", 26)

    def generate_action(self, df: pd.DataFrame, idx: int) -> Tuple[str, Dict[str, Any]]:
        # Need enough data
        if idx < self.long_window:
            return "hold", {"reason": "Insufficient data"}
            
        # Calculate EMAs
        # We need data up to current index
        # To optimize, we could calculate for the whole DF once, but for simulation structure
        # we often process bar by bar. However, pandas_ta works best on series.
        # For efficiency in backtest loop, we assume df contains all data up to now.
        
        # Calculate indicators for the whole dataframe (or slice)
        # In a real optimized engine, we would pre-calculate. 
        # Here we calculate on the fly for simplicity and safety against lookahead bias if df grows.
        # Assuming df is the full historical dataframe passed from the engine.
        
        # Optimization: If df is large, this is slow. 
        # But auto_sim_lab passes the full df and iterates index.
        # We should pre-calculate indicators outside the loop if possible.
        # However, the interface is generate_action(df, idx).
        
        # Let's calculate on a slice to be safe, or assume pre-calculated columns exist?
        # The current auto_sim_lab doesn't pre-calculate.
        # We will calculate on a slice to ensure correctness.
        
        subset = df.iloc[:idx+1]
        if len(subset) < self.long_window + 1:
             return "hold", {"reason": "Insufficient data"}

        # Use pandas_ta or simple pandas ewm
        ema_short = subset['close'].ewm(span=self.short_window, adjust=False).mean()
        ema_long = subset['close'].ewm(span=self.long_window, adjust=False).mean()
        
        curr_short = ema_short.iloc[-1]
        curr_long = ema_long.iloc[-1]
        prev_short = ema_short.iloc[-2]
        prev_long = ema_long.iloc[-2]
        
        action = "hold"
        reason = f"EMA{self.short_window}={curr_short:.2f}, EMA{self.long_window}={curr_long:.2f}"
        
        # Crossover Logic
        if prev_short <= prev_long and curr_short > curr_long:
            action = "buy"
            reason = f"Golden Cross: EMA{self.short_window} ({curr_short:.2f}) > EMA{self.long_window} ({curr_long:.2f})"
        elif prev_short >= prev_long and curr_short < curr_long:
            action = "sell"
            reason = f"Death Cross: EMA{self.short_window} ({curr_short:.2f}) < EMA{self.long_window} ({curr_long:.2f})"
            
        return action, {
            "strategy": "ema_crossover",
            "ema_short": curr_short,
            "ema_long": curr_long,
            "reason": reason
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for the entire DataFrame at once.
        
        Returns:
            DataFrame with 'signal' column (1: Buy, -1: Sell, 0: Hold)
            and indicator columns.
        """
        df = df.copy()
        
        # Calculate EMAs
        df['ema_short'] = df['close'].ewm(span=self.short_window, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=self.long_window, adjust=False).mean()
        
        # Generate Signals
        # 1 where short > long, -1 where short < long
        # We use crossover logic: 
        # Buy when short crosses above long (prev short <= prev long AND curr short > curr long)
        # Sell when short crosses below long (prev short >= prev long AND curr short < curr long)
        
        df['signal'] = 0
        
        # Vectorized crossover detection
        # Condition: Short > Long
        bullish = df['ema_short'] > df['ema_long']
        bearish = df['ema_short'] < df['ema_long']
        
        # Crossover
        # Bullish Crossover: Bullish now AND Bearish/Neutral before
        crossover_bull = bullish & (~bullish.shift(1).fillna(False))
        
        # Bearish Crossover: Bearish now AND Bullish/Neutral before
        crossover_bear = bearish & (~bearish.shift(1).fillna(False))
        
        df.loc[crossover_bull, 'signal'] = 1
        df.loc[crossover_bear, 'signal'] = -1
        
        return df
