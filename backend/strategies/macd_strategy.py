from typing import Tuple, Dict, Any
import pandas as pd
from backend.strategies.base_strategy import BaseStrategy

class MacdStrategy(BaseStrategy):
    """
    MACD Signal Line Strategy (12-26-9).
    
    Entry: MACD line crosses above Signal line -> BUY
    Exit: MACD line crosses below Signal line -> SELL
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.fast = config.get("macd_fast", 12)
        self.slow = config.get("macd_slow", 26)
        self.signal = config.get("macd_signal", 9)

    def generate_action(self, df: pd.DataFrame, idx: int) -> Tuple[str, Dict[str, Any]]:
        if idx < self.slow + self.signal:
            return "hold", {"reason": "Insufficient data"}
            
        subset = df.iloc[:idx+1]
        
        # Calculate MACD
        ema_fast = subset['close'].ewm(span=self.fast, adjust=False).mean()
        ema_slow = subset['close'].ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        
        curr_macd = macd_line.iloc[-1]
        curr_signal = signal_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        
        action = "hold"
        reason = f"MACD={curr_macd:.2f}, Signal={curr_signal:.2f}"
        
        if prev_macd <= prev_signal and curr_macd > curr_signal:
            action = "buy"
            reason = f"MACD Cross Up: {curr_macd:.2f} > {curr_signal:.2f}"
        elif prev_macd >= prev_signal and curr_macd < curr_signal:
            action = "sell"
            reason = f"MACD Cross Down: {curr_macd:.2f} < {curr_signal:.2f}"
            
        return action, {
            "strategy": "macd",
            "macd": curr_macd,
            "signal": curr_signal,
            "histogram": curr_macd - curr_signal,
            "reason": reason
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for the entire DataFrame at once.
        """
        df = df.copy()
        
        # Calculate MACD
        ema_fast = df['close'].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.slow, adjust=False).mean()
        df['macd_line'] = ema_fast - ema_slow
        df['signal_line'] = df['macd_line'].ewm(span=self.signal, adjust=False).mean()
        df['macd_hist'] = df['macd_line'] - df['signal_line']
        
        # Generate Signals
        df['signal'] = 0
        
        bullish = df['macd_line'] > df['signal_line']
        bearish = df['macd_line'] < df['signal_line']
        
        crossover_bull = bullish & (~bullish.shift(1).fillna(False))
        crossover_bear = bearish & (~bearish.shift(1).fillna(False))
        
        df.loc[crossover_bull, 'signal'] = 1
        df.loc[crossover_bear, 'signal'] = -1
        
        return df
