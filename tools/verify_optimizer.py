import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from backend.strategies.runner import run_ma_cross_backtest
from backend.optimizer import GridSearchOptimizer
from backend import data_feed

def verify_case(symbol: str, start_date: str, end_date: str) -> bool:
    """
    Verify optimizer consistency for a specific symbol and date range.
    Returns True if all checks pass, False otherwise.
    """
    print(f"--- Verifying Optimizer Consistency for {symbol} ({start_date} to {end_date}) ---")
    
    timeframe = "1d"
    
    # 1. Define Test Params
    test_params = [
        (5, 25),
        (9, 21),
        (10, 50),
        (20, 100)
    ]
    
    # 2. Run Direct Backtests (Ground Truth)
    ground_truth = {}
    
    # Fetch data for runner
    # Logic from optimizer to fetch data
    try:
        fetch_start = pd.Timestamp(start_date) - pd.Timedelta(days=365)
        candles = data_feed.get_chart_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=3000,
            start=fetch_start.strftime("%Y-%m-%d"),
            end=end_date
        )
        if not candles:
            print(f"⚠️ No data found for {symbol}. Skipping case.")
            return False
            
        df = pd.DataFrame(candles)
        df = df.set_index(pd.to_datetime(df["time"], unit="s"))
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        
        start_ts = pd.Timestamp(start_date).tz_localize("UTC")
        
        print("\n[Direct Run Results]")
        for short_w, long_w in test_params:
            res = run_ma_cross_backtest(
                df=df,
                short_window=short_w,
                long_window=long_w,
                start_date=start_ts
            )
            metrics = res["stats"]
            ground_truth[(short_w, long_w)] = metrics
            print(f"Params ({short_w}, {long_w}): Return={metrics['return_pct']:.2f}%, Trades={metrics['trade_count']}, MaxDD={metrics['max_drawdown']*100:.2f}%")

        # 3. Run Optimizer
        print("\n[Optimizer Results]")
        optimizer = GridSearchOptimizer()
        
        shorts = sorted(list(set(p[0] for p in test_params)))
        longs = sorted(list(set(p[1] for p in test_params)))
        
        param_grid = {
            "short_window": shorts,
            "long_window": longs
        }
        
        opt_results = optimizer.optimize(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            param_grid=param_grid,
            strategy_type="ma_cross"
        )
        
        # 4. Compare
        print("\n[Comparison]")
        case_passed = True
        
        for res in opt_results:
            p = (res.params["short_window"], res.params["long_window"])
            if p in ground_truth:
                gt = ground_truth[p]
                
                # Compare Return
                diff_ret = abs(res.return_pct - gt["return_pct"])
                diff_trades = abs(res.trade_count - gt["trade_count"])
                # Optimizer stores raw float for max_drawdown
                diff_dd = abs(res.max_drawdown - gt["max_drawdown"]) 
                
                match = True
                if diff_ret > 0.0001: match = False
                if diff_trades != 0: match = False
                if diff_dd > 0.0001: match = False
                
                status = "✅ PASS" if match else "❌ FAIL"
                if not match: case_passed = False
                
                print(f"{status} Params {p}:")
                print(f"  Optimizer: Ret={res.return_pct:.2f}%, Trades={res.trade_count}, DD={res.max_drawdown:.4f}")
                print(f"  Ground Truth: Ret={gt['return_pct']:.2f}%, Trades={gt['trade_count']}, DD={gt['max_drawdown']:.4f}")
        
        if case_passed:
            print(f"🎉 Verification Passed for {symbol} ({start_date} to {end_date})")
        else:
            print(f"⚠️ Verification FAILED for {symbol} ({start_date} to {end_date})")
            
        return case_passed

    except Exception as e:
        print(f"⚠️ Error verifying case {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return False

# Define Test Cases
TEST_CASES = [
    ("SMCI", "2025-01-01", "2025-12-31"),
    ("SMCI", "2020-01-01", "2023-12-31"),
    ("HOOD", "2025-01-01", "2025-12-31"),
    ("PLTR", "2025-01-01", "2025-12-31"),
    ("AAPL", "2020-01-01", "2023-12-31"),
]

if __name__ == "__main__":
    all_passed = True
    
    print("==================================================")
    print("🚀 Starting Optimizer Consistency Verification")
    print("==================================================\n")

    for symbol, start_date, end_date in TEST_CASES:
        ok = verify_case(symbol, start_date, end_date)
        if not ok:
            all_passed = False
        print("\n--------------------------------------------------\n")

    print("==================================================")
    if all_passed:
        print("🎉 All test cases PASSED! Optimizer is consistent across symbols / ranges.")
    else:
        print("⚠️ Some test cases FAILED. Please check the logs above.")
        sys.exit(1)
    print("==================================================\n")
