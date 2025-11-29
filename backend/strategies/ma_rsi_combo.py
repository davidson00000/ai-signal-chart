"""
MA + RSI Composite Strategy
"""
from backend.strategies.base import BaseStrategy
import pandas as pd


class MARSIComboStrategy(BaseStrategy):
    """
    MA × RSI 複合戦略
    
    Buy Conditions:
    - Golden Cross (short MA crosses above long MA)
    - AND RSI < oversold
    
    Sell Conditions:
    - Dead Cross (short MA crosses below long MA)  
    - AND RSI > overbought
    
    Args:
        short_window: Short MA period (default: 9)
        long_window: Long MA period (default: 21)
        rsi_period: RSI calculation period (default: 14)
        rsi_oversold: RSI oversold threshold (default: 30)
        rsi_overbought: RSI overbought threshold (default: 70)
    """
    
    def __init__(
        self,
        short_window: int = 9,
        long_window: int = 21,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0
    ):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def __str__(self) -> str:
        return f"MA+RSI({self.short_window}/{self.long_window}, RSI:{self.rsi_period})"
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MA cross + RSI
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            Series of signals: 1 (BUY), -1 (SELL), 0 (HOLD)
        """
        self.validate_dataframe(df)
        
        from backend.utils.indicators import simple_moving_average, rsi as calc_rsi
        
        closes = df["close"].values.tolist()
        
        # MA calculation
        short_ma = simple_moving_average(closes, self.short_window)
        long_ma = simple_moving_average(closes, self.long_window)
        
        # RSI calculation
        rsi_values = calc_rsi(closes, self.rsi_period)
        
        signals = pd.Series(0, index=df.index)
        
        for i in range(1, len(closes)):
            if short_ma[i] is None or long_ma[i] is None or rsi_values[i] is None:
                continue
            
            if short_ma[i-1] is None or long_ma[i-1] is None:
                continue
            
            # MA Cross Detection
            prev_diff = short_ma[i-1] - long_ma[i-1]
            curr_diff = short_ma[i] - long_ma[i]
            
            # Golden Cross + RSI oversold → BUY
            if prev_diff <= 0 and curr_diff > 0 and rsi_values[i] < self.rsi_oversold:
                signals.iloc[i] = 1
            
            # Dead Cross + RSI overbought → SELL
            elif prev_diff >= 0 and curr_diff < 0 and rsi_values[i] > self.rsi_overbought:
                signals.iloc[i] = -1
        
        return signals
