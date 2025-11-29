"""
Technical indicators calculation
Migrated from main_legacy.py - DO NOT MODIFY LOGIC (bug-fixed version)
"""
from typing import List, Optional


def simple_moving_average(values: List[float], window: int) -> List[Optional[float]]:
    """
    単純移動平均。
    window に満たない最初の部分は None を返す（NaN は JSON でエラーになるため使用しない）。
    
    Args:
        values: List of prices (close prices typically)
        window: Moving average window size
        
    Returns:
        List of MA values with None for insufficient data points
        
    Note:
        This function is migrated from main_legacy.py without any logic changes.
        It has been tested and verified to work correctly.
        DO NOT MODIFY THE CALCULATION LOGIC.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    ma: List[Optional[float]] = []
    sum_val = 0.0
    for i, v in enumerate(values):
        sum_val += v
        if i >= window:
            sum_val -= values[i - window]
        if i >= window - 1:
            ma.append(sum_val / window)
        else:
            ma.append(None)
    return ma


def rsi(values: List[float], period: int = 14) -> List[Optional[float]]:
    """
    Relative Strength Index (RSI) 計算
    
    Args:
        values: List of prices (close prices typically)
        period: RSI period (default: 14)
        
    Returns:
        List of RSI values (0-100) with None for insufficient data points
    """
    if period <= 0:
        raise ValueError("period must be positive")
    
    if len(values) < period + 1:
        return [None] * len(values)
    
    # Calculate price changes
    deltas = [values[i] - values[i-1] for i in range(1, len(values))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]
    
    # Initial average
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    rsi_values: List[Optional[float]] = [None] * (period + 1)
    
    # First RSI value
    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100.0 - (100.0 / (1.0 + rs)))
    
    # Subsequent RSI values using smoothed moving average
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100.0 - (100.0 / (1.0 + rs)))
    
    return rsi_values
