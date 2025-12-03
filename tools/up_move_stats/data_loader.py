import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from .utils import setup_logger

logger = setup_logger(__name__)

def download_data(symbol, lookback_days=365):
    """
    Downloads historical data for a given symbol using yfinance.
    Returns a DataFrame with 'Adj Close'.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    # logger.info(f"Downloading data for {symbol}...")
    
    try:
        # yfinance download
        # We use auto_adjust=False to ensure we get 'Adj Close' column explicitly.
        df = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=False, multi_level_index=False)
        
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return None
            
        # Check for Adj Close
        if 'Adj Close' not in df.columns:
            # Sometimes yfinance might return just 'Close' if it's already adjusted or data source differs
            if 'Close' in df.columns:
                 # logger.warning(f"'Adj Close' not found for {symbol}, using 'Close'")
                 df['Adj Close'] = df['Close']
            else:
                 logger.error(f"Neither 'Adj Close' nor 'Close' found for {symbol}")
                 return None

        return df[['Adj Close']]
        
    except Exception as e:
        logger.error(f"Error downloading {symbol}: {e}")
        return None
def download_data_range(symbol, start_date, end_date):
    """Download data for an explicit date range.
    * ``start_date`` and ``end_date`` can be ``datetime`` objects or ISO‑8601 strings.
    Returns a ``DataFrame`` containing only the ``Adj Close`` column.
    """
    # Normalise inputs to datetime objects
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date)

    # logger.info(f"Downloading {symbol} from {start_date.date()} to {end_date.date()}")
    try:
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
        )
        if df.empty:
            logger.warning(f"No data found for {symbol} in the given range")
            return None
        # Ensure Adj Close column exists
        if "Adj Close" not in df.columns:
            if "Close" in df.columns:
                df["Adj Close"] = df["Close"]
            else:
                logger.error(f"Neither 'Adj Close' nor 'Close' found for {symbol}")
                return None
        return df[["Adj Close"]]
    except Exception as e:
        logger.error(f"Error downloading {symbol}: {e}")
        return None
