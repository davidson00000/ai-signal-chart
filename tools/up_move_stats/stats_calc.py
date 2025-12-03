import pandas as pd

def calculate_stats(df, fx_rate=None):
    """
    Calculates return statistics from a DataFrame containing 'Adj Close'.
    Returns a dictionary with stats.
    """
    if df is None or df.empty:
        return None
    
    # Calculate daily returns
    # pct_change() calculates (current - prior) / prior, which is equivalent to (current / prior) - 1
    df = df.copy()
    df['Return'] = df['Adj Close'].pct_change()
    
    # Drop NaN created by pct_change (first row)
    df = df.dropna(subset=['Return'])
    
    total_days = len(df)
    up_1pct = (df['Return'] >= 0.01).sum()
    up_5pct = (df['Return'] >= 0.05).sum()
    up_10pct = (df['Return'] >= 0.10).sum()
    
    if total_days > 0:
        start_date = df.index[0].strftime('%Y-%m-%d')
        end_date = df.index[-1].strftime('%Y-%m-%d')
        last_price_usd = float(df['Adj Close'].iloc[-1])
    else:
        start_date = None
        end_date = None
        last_price_usd = 0.0
    
    stats = {
        'days_total': total_days,
        'up_1pct_days': up_1pct,
        'up_5pct_days': up_5pct,
        'up_10pct_days': up_10pct,
        'start_date': start_date,
        'end_date': end_date,
        'last_price_usd': round(last_price_usd, 2)
    }

    if fx_rate is not None and fx_rate > 0 and last_price_usd > 0:
        min_invest_jpy = last_price_usd * fx_rate
        stats['min_invest_jpy'] = int(round(min_invest_jpy))

    return stats
