import pandas as pd
import yfinance as yf
from backend.rule_predictor_v2 import compute_indicator_signals, calculate_stochastic, calculate_bollinger_bands
from backend.utils.indicators import rsi

def debug_symbol(symbol):
    print(f"\n--- Debugging {symbol} ---")
    df = yf.download(symbol, period="1y", interval="1d", progress=False)
    if df.empty:
        print("No data found.")
        return

    # Standardize columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    if 'adj close' in df.columns:
        df['close'] = df['adj close']
    
    # Run logic (full predict to see mapping)
    from backend.rule_predictor_v2 import predict
    result = predict(df)
    
    # Inspect raw values
    closes = df["close"]
    highs = df["high"]
    lows = df["low"]
    
    # RSI
    rsi_vals = rsi(closes.tolist(), 14)
    curr_rsi = rsi_vals[-1]
    print(f"RSI (14): {curr_rsi:.2f}")
    
    # Stoch
    k, d = calculate_stochastic(closes, highs, lows)
    curr_k = k.iloc[-1]
    print(f"Stoch K: {curr_k:.2f}")
    
    # BBands
    upper, mid, lower = calculate_bollinger_bands(closes)
    curr_upper = upper.iloc[-1]
    curr_lower = lower.iloc[-1]
    curr_close = closes.iloc[-1]
    print(f"BBands Upper: {curr_upper:.2f}, Lower: {curr_lower:.2f}, Close: {curr_close:.2f}")
    
    print(f"Discrete Signals: {result['signals']}")
    print(f"Continuous Signals: {result['raw_signals']}")
    print(f"Score: {result['score']}, Prob Up: {result['prob_up']}")

if __name__ == "__main__":
    for sym in ["MSTR", "COIN", "MARA", "CLSK"]:
        debug_symbol(sym)
