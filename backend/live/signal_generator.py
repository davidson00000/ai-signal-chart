import pandas as pd
from typing import Dict, Any
from datetime import datetime
from backend import data_feed
from backend.live.predictors import stat, rule, ml

def generate_live_signal(symbol: str, timeframe: str, lookback: int) -> Dict[str, Any]:
    """
    Generate live trading signal for EXITON v1.
    
    1. Fetch price data from DataFeed
    2. Call 3 predictors (Stat, Rule, ML)
    3. Combine results into final signal
    4. Return EXITON v1 spec JSON
    
    Args:
        symbol: Ticker symbol
        timeframe: Candle timeframe
        lookback: Number of candles to fetch
        
    Returns:
        Dict matching EXITON v1 /live/signal response spec
    """
    # 1. Fetch Data
    candles = data_feed.get_chart_data(
        symbol=symbol,
        timeframe=timeframe,
        limit=lookback + 50 # Buffer for indicators
    )
    
    if not candles:
        # Return empty/error response if no data
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "latest_price": 0.0,
            "price_time": datetime.utcnow().isoformat() + "Z",
            "predictions": {},
            "final_signal": {"action": "hold", "confidence": 0.0, "reason": "No data"},
            "history": []
        }
        
    df = pd.DataFrame(candles)
    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    latest_price = float(df.iloc[-1]["close"])
    # Handle timestamp format (assuming 'time' or 'timestamp' field)
    price_time_val = df.iloc[-1].get("time") or df.iloc[-1].get("timestamp")
    # If unix timestamp
    if isinstance(price_time_val, (int, float)):
        price_time = datetime.utcfromtimestamp(price_time_val).isoformat() + "Z"
    else:
        price_time = str(price_time_val)

    # 2. Call Predictors
    stat_res = stat.predict(df)
    
    # Rule Predictor v2 Integration
    from backend import rule_predictor_v2
    v2_res = rule_predictor_v2.predict(df)
    
    # Map v2 result to common schema
    prob_up = v2_res["prob_up"]
    prob_down = v2_res["prob_down"]
    confidence = max(prob_up, prob_down)
    
    rule_direction = "flat"
    if prob_up > 0.55:
        rule_direction = "up"
    elif prob_down > 0.55:
        rule_direction = "down"
        
    rule_res = {
        "direction": rule_direction,
        "score": v2_res["score"], # Raw score (-1 to 1)
        "confidence": confidence, # Confidence (0-1)
        "prob_up": prob_up,
        "prob_down": prob_down,
        "signals": v2_res["signals"],
        "raw_signals": v2_res.get("raw_signals"),
        "name": "Rule Predictor v2"
    }

    ml_res = ml.predict(df)
    
    predictions = {
        "stat": stat_res,
        "rule": rule_res,
        "ml": ml_res
    }
    
    # 3. Combine Logic (Confidence Model v1)
    
    # Calculate base confidence (Average of scores)
    scores = [p["score"] for p in predictions.values()]
    base_confidence = sum(scores) / len(scores)
    
    # Check directions
    directions = [p["direction"] for p in predictions.values()]
    up_count = directions.count("up")
    down_count = directions.count("down")
    flat_count = directions.count("flat")
    
    # Determine Final Action and Apply Penalties
    final_action = "hold"
    confidence = base_confidence
    reason = ""
    
    if up_count == 3:
        final_action = "buy"
        reason = "Strong bullish consensus across all predictors."
        # No penalty
    elif down_count == 3:
        final_action = "sell"
        reason = "Strong bearish consensus across all predictors."
        # No penalty
    elif up_count >= 2:
        final_action = "buy"
        confidence *= 0.85 # Penalty for disagreement
        reason = "Bullish bias. Majority of predictors signal UP."
    elif down_count >= 2:
        final_action = "sell"
        confidence *= 0.85 # Penalty for disagreement
        reason = "Bearish bias. Majority of predictors signal DOWN."
    elif up_count == 1 and down_count == 1:
        final_action = "hold"
        confidence *= 0.70 # Max penalty for confusion
        reason = "Predictors disagree (Mixed signals). Market is noisy."
    else:
        # Mostly flat or weak signals
        final_action = "hold"
        reason = "Market is neutral or predictors are undecided."

    # Dynamic Reason Refinement (Specific cases)
    if final_action == "buy" and predictions["rule"]["direction"] == "down":
        reason = "Stat+ML are bullish, but Rule is bearish (possible trend reversal)."
    elif final_action == "sell" and predictions["rule"]["direction"] == "up":
        reason = "Stat+ML are bearish, but Rule is bullish (possible trend reversal)."

    # 4. Return Response
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "latest_price": latest_price,
        "price_time": price_time,
        "predictions": predictions,
        "final_signal": {
            "action": final_action,
            "confidence": round(confidence, 2),
            "reason": reason
        },
        "history": [] # TODO: Implement history if needed
    }
