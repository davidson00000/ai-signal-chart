import argparse
import pandas as pd
import os
import sys
from .data_loader import download_data
from .stats_calc import calculate_stats
from .utils import setup_logger

logger = setup_logger(__name__)

def load_symbols_from_file(filepath):
    try:
        df = pd.read_csv(filepath)
        # Handle potential whitespace in column names
        df.columns = [c.strip() for c in df.columns]
        if 'symbol' in df.columns:
            return df['symbol'].tolist()
        else:
            logger.error(f"Column 'symbol' not found in {filepath}")
            return []
    except Exception as e:
        logger.error(f"Error reading symbols file {filepath}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Calculate up-move statistics for stocks.")
    parser.add_argument('--symbols_file', type=str, default='tools/symbols_universe.csv', help='Path to CSV file with symbols')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols (overrides file)')
    parser.add_argument('--lookback_days', type=int, default=365, help='Number of days to look back')
    parser.add_argument('--output', type=str, default='up_move_stats_result.csv', help='Output CSV file path')
    parser.add_argument('--fx_rate', type=float, default=150.0, help='USD/JPY rate. Set to <= 0 to disable JPY conversion.')
    
    args = parser.parse_args()
    
    symbols = []
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        if os.path.exists(args.symbols_file):
            symbols = load_symbols_from_file(args.symbols_file)
        else:
            logger.warning(f"Symbols file '{args.symbols_file}' not found.")
            
    if not symbols:
        logger.error("No symbols found to process. Please provide --symbols or ensure symbols file exists.")
        return

    results = []
    logger.info(f"Processing {len(symbols)} symbols with lookback {args.lookback_days} days...")
    
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] Processing {symbol}...")
        try:
            df = download_data(symbol, args.lookback_days)
            if df is not None:
                stats = calculate_stats(df, args.fx_rate)
                if stats:
                    stats['symbol'] = symbol
                    results.append(stats)
            else:
                logger.warning(f"Skipping {symbol} due to download failure.")
        except Exception as e:
            logger.error(f"Unexpected error processing {symbol}: {e}")
            continue

    if not results:
        logger.error("No results generated.")
        return

    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['symbol', 'days_total', 'up_1pct_days', 'up_5pct_days', 'up_10pct_days', 'start_date', 'end_date', 'last_price_usd']
    if args.fx_rate > 0:
        cols.append('min_invest_jpy')
        
    # Ensure all columns exist (in case of empty results but that's handled above)
    # Filter columns that actually exist in the dataframe (min_invest_jpy might not be there if all failed or price 0)
    existing_cols = [c for c in cols if c in results_df.columns]
    results_df = results_df[existing_cols]
    
    # Console output
    print("\n" + "="*80)
    print(f"UP MOVE STATS (Last {args.lookback_days} days)")
    print("="*80)
    # Use to_string for nice formatting
    print(results_df.to_string(index=False))
    print("="*80 + "\n")
    
    # CSV output
    try:
        results_df.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")
    except Exception as e:
        logger.error(f"Error saving CSV to {args.output}: {e}")

if __name__ == "__main__":
    main()
