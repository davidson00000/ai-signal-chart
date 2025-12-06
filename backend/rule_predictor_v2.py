import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from backend.utils.indicators import simple_moving_average, rsi

# --- Indicator Calculations ---

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

def calculate_stochastic(close: pd.Series, high: pd.Series, low: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic Oscillator (K and D).
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent, d_percent

# --- Helpers ---

def clip_signal(x: float, threshold: float = 1.0) -> float:
    if x is None or np.isnan(x):
        return 0.0
    return float(np.clip(x, -threshold, threshold))

def to_display_signal(x: float, dead_zone: float = 0.1) -> int:
    if abs(x) < dead_zone:
        return 0
    return 1 if x > 0 else -1

# --- Signal Logic ---

def compute_indicator_signals(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute continuous signals (-1.0 to +1.0) for each indicator.
    """
    signals = {
        "sma": 0.0,
        "rsi": 0.0,
        "macd": 0.0,
        "stoch": 0.0,
        "bbands": 0.0
    }
    
    # Ensure sufficient data
    if len(df) < 35:
        return signals
        
    closes = df["close"]
    highs = df["high"]
    lows = df["low"]
    
    # 1. SMA (Short=10, Long=30)
    # Keeping discrete for now as per spec, but float
    sma_short = closes.rolling(window=10).mean().iloc[-1]
    sma_long = closes.rolling(window=30).mean().iloc[-1]
    
    if pd.notna(sma_short) and pd.notna(sma_long):
        if sma_short > sma_long:
            signals["sma"] = 1.0
        elif sma_short < sma_long:
            signals["sma"] = -1.0

    # 2. RSI (14) -> Continuous
    rsi_vals = rsi(closes.tolist(), 14)
    curr_rsi = rsi_vals[-1]
    
    if curr_rsi is not None:
        # rsi around 50 => 0 (neutral)
        # rsi around 25 => +1 (bullish / oversold)
        # rsi around 75 => -1 (bearish / overbought)
        raw_rsi_signal = (50.0 - curr_rsi) / 25.0
        rsi_signal = clip_signal(raw_rsi_signal, threshold=1.0)
        
        # Dead zone
        if abs(rsi_signal) < 0.1:
            rsi_signal = 0.0
        signals["rsi"] = rsi_signal

    # 3. MACD (12, 26, 9) -> Discrete (as per spec)
    _, _, macd_hist = calculate_macd(closes)
    curr_hist = macd_hist.iloc[-1]
    
    if pd.notna(curr_hist):
        if curr_hist > 0:
            signals["macd"] = 1.0
        elif curr_hist < 0:
            signals["macd"] = -1.0

    # 4. Stochastic (14, 3, 3) -> Continuous
    k, d = calculate_stochastic(closes, highs, lows)
    curr_k = k.iloc[-1]
    
    if pd.notna(curr_k):
        # k around 50 => 0 (neutral)
        # k around 20 => +1 (bullish / oversold)
        # k around 80 => -1 (bearish / overbought)
        raw_stoch_signal = (50.0 - curr_k) / 25.0
        stoch_signal = clip_signal(raw_stoch_signal, threshold=1.0)
        
        # Dead zone
        if abs(stoch_signal) < 0.1:
            stoch_signal = 0.0
        signals["stoch"] = stoch_signal

    # 5. Bollinger Bands (20, 2) -> Continuous
    upper, middle, lower = calculate_bollinger_bands(closes)
    curr_upper = upper.iloc[-1]
    curr_middle = middle.iloc[-1]
    curr_close = closes.iloc[-1]
    
    if pd.notna(curr_upper) and pd.notna(curr_middle):
        # assume upper = middle + 2*std, lower = middle - 2*std
        std = (curr_upper - curr_middle) / 2.0
        
        z = 0.0
        if std > 0:
            # If close == middle => 0
            # close near upper band => negative (downward bias)
            # close near lower band => positive (upward bias)
            z = (curr_middle - curr_close) / (2.0 * std)
            
        bbands_signal = clip_signal(z, threshold=1.0)
        
        # Dead zone
        if abs(bbands_signal) < 0.1:
            bbands_signal = 0.0
        signals["bbands"] = bbands_signal
            
    return signals

def weighted_vote(signals: Dict[str, float]) -> Tuple[float, float, float]:
    """
    Compute weighted score and probabilities.
    Returns: (score, prob_up, prob_down)
    """
    weights = {
        "sma": 0.30,
        "rsi": 0.20,
        "macd": 0.20,
        "stoch": 0.15,
        "bbands": 0.15
    }
    
    score = 0.0
    total_weight = 0.0
    
    for indicator, weight in weights.items():
        sig = signals.get(indicator, 0.0)
        score += sig * weight
        total_weight += weight
        
    # Normalize if weights don't sum to 1 (they do here, but good practice)
    if total_weight > 0:
        score /= total_weight
        
    score = clip_signal(score, 1.0)
    
    prob_up = (score + 1.0) / 2.0
    prob_down = 1.0 - prob_up
    
    return score, prob_up, prob_down

def predict(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Main prediction function for Rule Predictor v2.
    """
    # Ensure numeric
    cols = ["open", "high", "low", "close"]
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Compute continuous signals
    continuous_signals = compute_indicator_signals(df)
    
    # Compute score and probabilities
    score, prob_up, prob_down = weighted_vote(continuous_signals)
    
    # Map to discrete signals for UI display
    display_signals = {
        name: to_display_signal(value, dead_zone=0.1)
        for name, value in continuous_signals.items()
    }
    
    return {
        "prob_up": round(prob_up, 2),
        "prob_down": round(prob_down, 2),
        "score": round(score, 2),
        "signals": display_signals,      # Discrete for UI icons
        "raw_signals": continuous_signals # Continuous for debug/future use
    }
