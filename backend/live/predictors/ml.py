import pandas as pd
import numpy as np
from typing import Dict, Any
from backend.utils.indicators import rsi

def predict(price_df: pd.DataFrame) -> Dict[str, Any]:
    """
    ML Predictor v1: Lightweight model (Logistic Regression / RandomForest).
    For v1, we use a heuristic linear model to simulate ML output.
    
    Args:
        price_df: DataFrame with OHLCV data
        
    Returns:
        {
            "direction": "up" | "down" | "flat",
            "score": float  # 0.0 - 1.0
        }
    """
    if price_df.empty or len(price_df) < 20:
        return {"direction": "flat", "score": 0.0}
        
    # Feature Engineering (Last candle)
    last_row = price_df.iloc[-1]
    prev_row = price_df.iloc[-2]
    
    # 1. Return (Close vs Open)
    feat_return = (last_row["close"] - last_row["open"]) / last_row["open"]
    
    # 2. Momentum (Close vs Prev Close)
    feat_mom = (last_row["close"] - prev_row["close"]) / prev_row["close"]
    
    # 3. Volatility (High - Low)
    feat_vol = (last_row["high"] - last_row["low"]) / last_row["close"]
    
    # 4. RSI (Normalized 0-1)
    closes = price_df["close"].tolist()
    rsi_val = rsi(closes, 14)[-1]
    feat_rsi = (rsi_val / 100.0) if rsi_val is not None else 0.5
    
    # Simple Linear Model (Weights simulated)
    # w_return = 10.0, w_mom = 5.0, w_vol = -2.0, w_rsi = 2.0 (mean reversion bias)
    # bias = 0.0
    
    logit = (feat_return * 10.0) + (feat_mom * 5.0) + (feat_vol * -2.0) + ((feat_rsi - 0.5) * 2.0)
    
    # Sigmoid
    prob_up = 1.0 / (1.0 + np.exp(-logit))
    
    # Thresholds
    direction = "flat"
    score = 0.0
    
    if prob_up > 0.55:
        direction = "up"
        score = (prob_up - 0.5) * 2.0 # Scale 0.55-1.0 to 0.1-1.0
    elif prob_up < 0.45:
        direction = "down"
        score = (0.5 - prob_up) * 2.0
    else:
        direction = "flat"
        score = 1.0 - (abs(prob_up - 0.5) * 10.0) # Higher score if closer to 0.5
        
    return {
        "direction": direction,
        "score": float(np.clip(score, 0.0, 1.0))
    }
