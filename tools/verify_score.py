import sys
import os
import pandas as pd
from typing import Dict, Any

# Add project root to path
sys.path.append(os.getcwd())

from backend.optimizer import GridSearchOptimizer

def calculate_expected_score(return_pct: float, sharpe: float, max_dd: float, trades: int) -> float:
    """
    Independently calculate the score based on the formula:
    Score = Sharpe * 2.0 + Return * 0.5
    Penalties:
    - Trade Count < 5 => -1e9
    - Max Drawdown > 70.0 => -1e9
    """
    if trades < 5:
        return -1e9
    if max_dd > 70.0:
        return -1e9
    
    return (sharpe * 2.0) + (return_pct * 0.5)

def verify_symbol(symbol: str, start_date: str, end_date: str) -> bool:
    print(f"\n--- Verifying Score for {symbol} ({start_date} to {end_date}) ---")
    
    optimizer = GridSearchOptimizer()
    
    # Define a small grid for verification purposes
    # We want enough combinations to test various scenarios (low trades, high DD, good performance)
    param_grid = {
        "short_window": [5, 10, 20],
        "long_window": [20, 50, 100]
    }
    
    try:
        results = optimizer.optimize(
            symbol=symbol,
            timeframe="1d",
            start_date=start_date,
            end_date=end_date,
            param_grid=param_grid,
            strategy_type="ma_cross"
        )
        
        all_matched = True
        
        print(f"{'Params':<15} | {'Return':<10} | {'Sharpe':<8} | {'MaxDD':<8} | {'Trades':<6} | {'GUI Score':<12} | {'Calc Score':<12} | {'Status'}")
        print("-" * 100)
        
        for res in results:
            params_str = f"{res.params['short_window']}/{res.params['long_window']}"
            
            # Extract metrics
            ret = res.return_pct
            sharpe = res.sharpe_ratio
            dd = res.max_drawdown * 100 # Optimizer returns decimal, score calc uses percent for threshold check? 
            # Wait, let's check backend/optimizer.py logic again.
            # In optimizer.py:
            # max_dd = stats["max_drawdown"] * 100 
            # if max_dd > 70.0: score = -1e9
            # So the threshold check uses percentage (0-100).
            # The OptimizationResult object stores max_drawdown as decimal (0.0 - 1.0).
            # So we need to convert it to percent for the check.
            
            trades = res.trade_count
            gui_score = res.score
            
            # Calculate expected score
            # Note: OptimizationResult stores max_drawdown as decimal (e.g. 0.15 for 15%)
            # The score calculation logic in optimizer.py converts it to percent for the check:
            # max_dd = stats["max_drawdown"] * 100
            # if max_dd > 70.0 ...
            # But the score formula itself doesn't use max_dd, only the penalty check does.
            
            calc_score = calculate_expected_score(ret, sharpe, dd, trades)
            
            # Compare
            diff = abs(gui_score - calc_score)
            is_match = diff < 0.001
            
            status = "✅ Matched" if is_match else "❌ Mismatch"
            if not is_match:
                all_matched = False
            
            print(f"{params_str:<15} | {ret:<10.2f} | {sharpe:<8.2f} | {dd:<8.2f} | {trades:<6} | {gui_score:<12.4f} | {calc_score:<12.4f} | {status}")
            
        return all_matched

    except Exception as e:
        print(f"⚠️ Error verifying {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    symbols = ["SMCI", "HOOD", "PLTR", "AAPL"]
    start_date = "2025-01-01"
    end_date = "2025-12-31"
    
    all_passed = True
    
    for sym in symbols:
        if not verify_symbol(sym, start_date, end_date):
            all_passed = False
            
    print("\n==================================================")
    if all_passed:
        print("🎉 All Scores Verified! Logic is consistent.")
    else:
        print("⚠️ Some Scores Mismatched! Check logs above.")
        sys.exit(1)
    print("==================================================")
