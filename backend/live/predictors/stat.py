import pandas as pd
import numpy as np
from typing import Dict, Any

def predict(price_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Stat Predictor: Simple statistical analysis (slope, MA diff).
    
    Args:
        price_df: DataFrame with OHLCV data (must have 'close')
        
    Returns:
        {
            "direction": "up" | "down" | "flat",
            "score": float  # 0.0 - 1.0
        }
    """
    if price_df.empty or len(price_df) < 5:
        return {"direction": "flat", "score": 0.0}
    
    # 1. Calculate Slope (Linear Regression on last 5 points)
    closes = price_df["close"].values
    y = closes[-5:]
    x = np.arange(len(y))
    
    # Simple linear regression: y = mx + c
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    # Normalize slope (percentage change per bar)
    slope_pct = (m / y[0]) * 100
    
    # 2. Determine Direction & Score
    direction = "flat"
    score = 0.0
    
    if slope_pct > 0.1:  # Strong Up
        direction = "up"
        score = min(abs(slope_pct) * 2, 1.0) # Scale score
    elif slope_pct < -0.1: # Strong Down
        direction = "down"
        score = min(abs(slope_pct) * 2, 1.0)
    else:
        direction = "flat"
        score = 1.0 - (abs(slope_pct) / 0.1) # Higher score if closer to 0
        
    return {
        "direction": direction,
        "score": float(np.clip(score, 0.0, 1.0))
    }
