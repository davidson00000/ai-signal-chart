# Custom Up‑Move Stats with Date Range & Thresholds

"""Utility to calculate up‑move statistics for a list of symbols with
user‑specified start/end dates and custom percentage thresholds.

Usage example:

```bash
python -m tools.up_move_stats.custom_up_move_stats \
    --symbols_file ./tools/symbols_universe_sp500.csv \
    --start 2025-01-01 \
    --end   2025-12-31 \
    --thresholds 2,4,8 \
    --output sp500_custom_2025.csv
```

The script re‑uses the existing `download_data` and `download_data_range`
helpers from `tools.up_move_stats.data_loader`. If `--start`/`--end` are not
provided, it falls back to the original look‑back behaviour.
"""

import argparse
import pandas as pd
from datetime import datetime
from .data_loader import download_data, download_data_range
from .stats_calc import calculate_stats
from .utils import setup_logger

logger = setup_logger(__name__)


def parse_thresholds(threshold_str: str):
    """Convert a comma‑separated string like ``"2,4,8"`` to a list of
    decimal fractions ``[0.02, 0.04, 0.08]``.
    """
    try:
        parts = [float(p.strip()) for p in threshold_str.split(',') if p.strip()]
        return [p / 100.0 for p in parts]
    except Exception as e:
        logger.error(f"Failed to parse thresholds '{threshold_str}': {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Custom up‑move stats with date range and thresholds.")
    parser.add_argument('--symbols_file', type=str, default='tools/symbols_universe.csv', help='Path to CSV file with a column named "symbol"')
    parser.add_argument('--symbols', type=str, help='Comma‑separated symbols (overrides file)')
    parser.add_argument('--start', type=str, help='Start date (YYYY‑MM‑DD) for custom range')
    parser.add_argument('--end', type=str, help='End date (YYYY‑MM‑DD) for custom range')
    parser.add_argument('--thresholds', type=str, default='1,5,10', help='Comma‑separated percentage thresholds, e.g. "2,4,8"')
    parser.add_argument('--output', type=str, default='custom_up_move_stats.csv', help='Output CSV file')
    parser.add_argument('--fx_rate', type=float, default=150.0, help='USD/JPY conversion rate (<=0 disables conversion)')
    args = parser.parse_args()

    # Resolve symbols list
    symbols = []
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        import pandas as pd
        df_symbols = pd.read_csv(args.symbols_file)
        if 'symbol' not in df_symbols.columns:
            logger.error(f"Column 'symbol' not found in {args.symbols_file}")
            return
        symbols = df_symbols['symbol'].tolist()

    thresholds = parse_thresholds(args.thresholds)

    results = []
    logger.info(f"Processing {len(symbols)} symbols with thresholds {thresholds}")

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] Downloading {symbol}")
        try:
            if args.start and args.end:
                df = download_data_range(symbol, args.start, args.end)
            else:
                # fallback to original look‑back (365 days) behaviour
                df = download_data(symbol)
            if df is None:
                logger.warning(f"Skipping {symbol} – no data")
                continue
            stats = calculate_stats(df, fx_rate=args.fx_rate)
            if not stats:
                continue
            # Add custom threshold counts
            # calculate_stats returns up_1pct_days, up_5pct_days, up_10pct_days – we replace them
            # with the user‑provided thresholds.
            # Re‑compute using the same logic as calculate_stats but with custom thresholds.
            # We'll duplicate the small portion here to avoid altering the original function.
            df = df.copy()
            df['Return'] = df['Adj Close'].pct_change()
            df = df.dropna(subset=['Return'])
            counts = []
            for thr in thresholds:
                counts.append((df['Return'] >= thr).sum())
            # Map counts to generic column names up_Xpct_days where X is the threshold integer.
            for idx, thr in enumerate(thresholds):
                col_name = f"up_{int(thr*100)}pct_days"
                stats[col_name] = counts[idx]
            # Remove the default columns if they exist to avoid confusion.
            for default_col in ['up_1pct_days', 'up_5pct_days', 'up_10pct_days']:
                stats.pop(default_col, None)

            stats['symbol'] = symbol
            results.append(stats)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue

    if not results:
        logger.error("No results generated.")
        return

    results_df = pd.DataFrame(results)
    # Order columns: symbol first, then dates, then custom up‑move columns, then others.
    cols_order = ['symbol']
    if 'start_date' in results_df.columns:
        cols_order.append('start_date')
    if 'end_date' in results_df.columns:
        cols_order.append('end_date')
    # add custom up‑move columns in the order they were supplied
    for thr in thresholds:
        cols_order.append(f"up_{int(thr*100)}pct_days")
    # add any remaining columns
    for c in results_df.columns:
        if c not in cols_order:
            cols_order.append(c)
    results_df = results_df[cols_order]

    results_df.to_csv(args.output, index=False)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
