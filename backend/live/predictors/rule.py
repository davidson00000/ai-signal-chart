import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from backend.utils.indicators import simple_moving_average, rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD, Signal, and Histogram.
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands (Upper, Middle, Lower).
    """
    middle = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower

def predict(price_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Rule-based Predictor v1: Technical Analysis Consensus.
    
    Indicators:
    - SMA (10, 30): Trend direction
    - RSI (14): Overbought/Oversold reversion
    - MACD (12, 26, 9): Momentum
    - Bollinger Bands (20, 2): Mean reversion
    
    Args:
        price_df: DataFrame with OHLCV data
        
    Returns:
        {
            "direction": "up" | "down" | "flat",
            "score": float,  # 0.0 - 1.0
            "details": Dict[str, str] # Breakdown of signals
        }
    """
    # Need enough data for Long MA (30) + buffer
    if price_df.empty or len(price_df) < 35:
        return {"direction": "flat", "score": 0.0, "details": {}}
    
    closes = price_df["close"]
    # Ensure pandas Series
    if not isinstance(closes, pd.Series):
        closes = pd.Series(closes)
        
    # 1. Calculate Indicators
    # SMA
    ma_short_window = 10
    ma_long_window = 30
    ma_short = closes.rolling(window=ma_short_window).mean()
    ma_long = closes.rolling(window=ma_long_window).mean()
    
    # RSI
    # Using backend.utils.indicators.rsi which returns list, convert to Series
    rsi_list = rsi(closes.tolist(), 14)
    # Fix: indicators.rsi might return length + 1 due to off-by-one in init
    if len(rsi_list) > len(closes):
        rsi_list = rsi_list[-len(closes):]
    rsi_series = pd.Series(rsi_list, index=closes.index)
    
    # MACD
    macd_line, macd_signal, macd_hist = calculate_macd(closes)
    
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(closes)
    
    # Get latest values
    current_close = closes.iloc[-1]
    curr_ma_short = ma_short.iloc[-1]
    curr_ma_long = ma_long.iloc[-1]
    curr_rsi = rsi_series.iloc[-1]
    curr_macd_hist = macd_hist.iloc[-1]
    curr_bb_upper = bb_upper.iloc[-1]
    curr_bb_lower = bb_lower.iloc[-1]
    
    # Check for NaN (insufficient data for some indicators)
    if pd.isna(curr_ma_long) or pd.isna(curr_rsi) or pd.isna(curr_macd_hist) or pd.isna(curr_bb_upper):
        return {"direction": "flat", "score": 0.0, "details": {"error": "Insufficient data for indicators"}}

    # 2. Scoring Logic
    up_score = 0
    down_score = 0
    details = {}
    
    # Rule 1: MA Trend (Short vs Long)
    if curr_ma_short > curr_ma_long:
        up_score += 1
        details["ma_trend"] = "bullish"
    else:
        down_score += 1
        details["ma_trend"] = "bearish"
        
    # Rule 2: Price Trend (Price vs Long MA)
    if current_close > curr_ma_long:
        up_score += 1
        details["price_trend"] = "bullish"
    else:
        down_score += 1
        details["price_trend"] = "bearish"
        
    # Rule 3: RSI (Reversion)
    if curr_rsi < 30:
        up_score += 1
        details["rsi"] = "oversold_bullish"
    elif curr_rsi > 70:
        down_score += 1
        details["rsi"] = "overbought_bearish"
    else:
        details["rsi"] = "neutral"
        
    # Rule 4: MACD (Momentum)
    if curr_macd_hist > 0:
        up_score += 1
        details["macd"] = "bullish"
    else:
        down_score += 1
        details["macd"] = "bearish"
        
    # Rule 5: Bollinger Bands (Mean Reversion)
    if current_close < curr_bb_lower:
        up_score += 1
        details["bb"] = "oversold_bullish"
    elif current_close > curr_bb_upper:
        down_score += 1
        details["bb"] = "overbought_bearish"
    else:
        details["bb"] = "neutral"
        
    # 3. Aggregation
    net_score = up_score - down_score
    max_score = 5.0 # Total rules
    
    direction = "flat"
    if net_score >= 2:
        direction = "up"
    elif net_score <= -2:
        direction = "down"
        
    # Normalize score (0.0 - 1.0)
    # Use absolute net score ratio
    final_score = min(abs(net_score) / max_score, 1.0)
    
    # Boost score slightly if direction is strong to avoid 0.4 being "weak"
    # But keep it proportional. 
    # 2/5 = 0.4 (Weak signal)
    # 3/5 = 0.6 (Moderate)
    # 4/5 = 0.8 (Strong)
    # 5/5 = 1.0 (Very Strong)
            
    return {
        "direction": direction,
        "score": final_score,
        "details": details
    }
