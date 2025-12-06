"""
EXITON Market Scanner

Scans a universe of symbols using Rule Predictor v2 and Stat Predictor
to identify bullish/bearish opportunities.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import concurrent.futures

from backend import data_feed
from backend import rule_predictor_v2
from backend.live.predictors import stat as stat_predictor

# Path to universe files
TOOLS_DIR = Path(__file__).parent.parent / "tools"

def load_symbol_universe(universe_name: str) -> List[str]:
    """
    Load symbol list from CSV.
    universe_name: "sp500", "mvp", "default" (maps to symbols_universe.csv)
    """
    filename = "symbols_universe.csv"
    if universe_name == "sp500":
        filename = "symbols_universe_sp500.csv"
    elif universe_name == "mvp":
        filename = "symbols_universe_mvp.csv"
    
    path = TOOLS_DIR / filename
    if not path.exists():
        # Fallback to default list if file not found
        return ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD", "INTC", "SMCI", "COIN", "MSTR"]
    
    try:
        df = pd.read_csv(path)
        # Assume first column is symbol
        return df.iloc[:, 0].tolist()
    except Exception:
        return ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"]


def analyze_symbol(symbol: str, timeframe: str, lookback: int) -> Optional[Dict[str, Any]]:
    """
    Run analysis for a single symbol.
    """
    try:
        # 1. Fetch data
        candles = data_feed.get_chart_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=lookback + 50, # Add buffer for indicators
        )
        
        if not candles or len(candles) < 50:
            return None
            
        df = pd.DataFrame(candles)
        
        # Normalize columns
        if "time" in df.columns:
            df["date"] = pd.to_datetime(df["time"], unit="s")
        elif "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"])
            
        df = df.sort_values("date").reset_index(drop=True)
        latest_price = float(df.iloc[-1]["close"])
        
        # 2. Run Predictors
        # Rule Predictor v2
        rule_res = rule_predictor_v2.predict(df)
        rule_score = rule_res.get("score", 0.0)
        rule_prob_up = rule_res.get("prob_up", 0.0)
        rule_prob_down = rule_res.get("prob_down", 0.0)
        
        rule_direction = "FLAT"
        if rule_prob_up > 0.55: rule_direction = "UP"
        elif rule_prob_down > 0.55: rule_direction = "DOWN"
        
        # Stat Predictor
        stat_res = stat_predictor.predict(df)
        stat_direction = stat_res.get("direction", "flat").upper()
        stat_score = stat_res.get("score", 0.0) # 0-1 confidence
        
        # 3. Calculate Combined Score
        # Rule component: -1 to 1
        rule_component = rule_score
        
        # Stat component: Map 0-1 to -1 to 1 based on direction
        stat_component = 0.0
        if stat_direction == "UP":
            stat_component = stat_score
        elif stat_direction == "DOWN":
            stat_component = -stat_score
            
        # Agreement bonus
        agreement = 0.0
        if rule_direction == stat_direction == "UP":
            agreement = 0.2
        elif rule_direction == stat_direction == "DOWN":
            agreement = -0.2
            
        # Weighted sum
        raw_combined = (rule_component * 0.7) + (stat_component * 0.3) + agreement
        combined_score = float(np.clip(raw_combined, -1.0, 1.0))
        
        # 4. Determine Final Signal
        final_signal = "FLAT"
        if combined_score > 0.3:
            final_signal = "STRONG_UP"
        elif combined_score > 0.05:
            final_signal = "UP"
        elif combined_score < -0.3:
            final_signal = "STRONG_DOWN"
        elif combined_score < -0.05:
            final_signal = "DOWN"
            
        return {
            "symbol": symbol,
            "latest_price": latest_price,
            "rule_direction": rule_direction,
            "rule_score": round(rule_score, 2),
            "rule_prob_up": round(rule_prob_up, 2),
            "stat_direction": stat_direction,
            "stat_conf": round(stat_score, 2),
            "combined_score": round(combined_score, 2),
            "final_signal": final_signal
        }
        
    except Exception as e:
        # print(f"Error analyzing {symbol}: {e}")
        return None


def scan_market(
    universe_name: str = "default",
    timeframe: str = "1d",
    lookback: int = 200,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Scan the market for opportunities.
    """
    symbols = load_symbol_universe(universe_name)
    results = []
    
    # Run sequentially to avoid yfinance threading issues (identical results bug)
    # TODO: Implement batch fetching with yf.download(tickers=...) for better performance
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future_to_symbol = {
            executor.submit(analyze_symbol, sym, timeframe, lookback): sym 
            for sym in symbols
        }
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            res = future.result()
            if res:
                results.append(res)
                
    # Sort by combined_score descending (Bullish first)
    results.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # Limit results
    top_results = results[:limit]
    
    return {
        "universe": universe_name,
        "timeframe": timeframe,
        "lookback": lookback,
        "total_scanned": len(symbols),
        "results": top_results
    }


if __name__ == "__main__":
    import time
    start = time.time()
    print("Running market scan (standalone)...")
    # Test with a small universe or default
    res = scan_market(universe_name="default", limit=10)
    print(f"Scan complete in {time.time() - start:.2f}s")
    from pprint import pprint
    pprint(res["results"][:3])
