"""
Data Feed Layer - Market data acquisition
Migrated from main_legacy.py
Enhanced to support up to 3000 historical candles
"""
import ccxt
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from fastapi import HTTPException


# --- Configuration ---
# Binance returns 451 from Vercel, so use Bybit for cloud deployment
CRYPTO_EXCHANGE_ID = "bybit"  # Can use "binance" for local only

# Maximum data points to return
MAX_DATA_POINTS = 3000


def get_exchange():
    """
    Get ccxt exchange instance
    
    Returns:
        ccxt Exchange instance configured with rate limiting
    """
    exchange_class = getattr(ccxt, CRYPTO_EXCHANGE_ID)
    return exchange_class({"enableRateLimit": True})


def ohlcv_to_candles(ohlcv: List[List[float]]) -> List[Dict[str, Any]]:
    """
    Convert ccxt OHLCV format to frontend candle format
    
    ccxt format: [timestamp_ms, open, high, low, close, volume]
    Frontend format: {time, open, high, low, close, volume}
    
    Args:
        ohlcv: List of OHLCV arrays from ccxt
        
    Returns:
        List of candle dictionaries for Lightweight Charts
    """
    candles = []
    for t, o, h, l, c, v in ohlcv:
        candles.append(
            {
                "time": int(t / 1000),  # Convert ms to seconds for Lightweight Charts
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v),
            }
        )
    return candles


def df_to_candles(df, limit: int) -> List[Dict[str, Any]]:
    """
    Convert yfinance DataFrame to frontend candle format
    
    Args:
        df: yfinance DataFrame with DatetimeIndex and OHLCV columns
        limit: Maximum number of candles to return
        
    Returns:
        List of candle dictionaries for Lightweight Charts
    """
    candles: List[Dict[str, Any]] = []
    
    # Sort by time (newest first in DataFrame, then take tail)
    df = df.sort_index(ascending=True)
    
    # Trim to limit
    df = df.dropna().tail(limit)

    for ts, row in df.iterrows():
        candles.append(
            {
                "time": int(ts.timestamp()),  # Unix timestamp in seconds
                "open": row["Open"].item() if hasattr(row["Open"], 'item') else float(row["Open"]),
                "high": row["High"].item() if hasattr(row["High"], 'item') else float(row["High"]),
                "low": row["Low"].item() if hasattr(row["Low"], 'item') else float(row["Low"]),
                "close": row["Close"].item() if hasattr(row["Close"], 'item') else float(row["Close"]),
                "volume": row.get("Volume", 0.0).item() if hasattr(row.get("Volume", 0.0), 'item') else float(row.get("Volume", 0.0)),
            }
        )
    return candles


def fetch_crypto_candles(
    symbol: str,
    timeframe: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Fetch cryptocurrency candles using ccxt
    
    Args:
        symbol: Crypto pair (e.g., "BTC/USDT")
        timeframe: Timeframe string (e.g., "1m", "5m", "1h", "1d")
        limit: Number of candles to fetch
        
    Returns:
        List of candle dictionaries
        
    Raises:
        HTTPException: If data fetch fails
    """
    exchange = get_exchange()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch crypto data: {e}")

    if not ohlcv:
        raise HTTPException(status_code=400, detail="No OHLCV data for crypto symbol")

    return ohlcv_to_candles(ohlcv)


def fetch_stock_candles(
    symbol: str,
    timeframe: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Fetch stock price candles using yfinance with extended data support (up to 3000 points)
    
    Strategy:
    1. Try period="max" first for maximum historical data
    2. If that fails or returns insufficient data, use start/end dates
    3. Sort and trim to requested limit (max 3000)
    
    Args:
        symbol: Stock symbol (e.g., "AAPL", "7203.T")
        timeframe: Timeframe string (e.g., "1m", "5m", "1h", "1d")
        limit: Number of candles to return (max 3000)
        
    Returns:
        List of candle dictionaries
        
    Raises:
        HTTPException: If data fetch fails or timeframe unsupported
    """
    # Ensure limit doesn't exceed maximum
    limit = min(limit, MAX_DATA_POINTS)
    
    # Map frontend timeframe to yfinance interval
    tf_to_interval = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "60m",
        "4h": "60m",   # 4h approximated with 60m
        "1d": "1d",
    }

    if timeframe not in tf_to_interval:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported timeframe for stocks: {timeframe}"
        )

    interval = tf_to_interval[timeframe]
    
    # Determine fetch strategy based on timeframe
    df = None
    
    # Strategy 1: Try period="max" for daily data
    if timeframe == "1d":
        try:
            df = yf.download(
                symbol,
                period="max",
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            
            # If we got data and it's sufficient, use it
            if df is not None and not df.empty and len(df) >= limit:
                return df_to_candles(df, limit)
        except Exception as e:
            # max period failed, will try date range fallback
            pass
    
    # Strategy 2: Use calculated date ranges
    end_date = datetime.now()
    
    # Calculate start date based on timeframe and desired data points
    if timeframe in ("1m", "5m"):
        # Intraday minute data: limited by Yahoo (max ~7 days)
        start_date = end_date - timedelta(days=7)
    elif timeframe in ("15m", "30m", "1h"):
        # Hourly data: try to get ~60 days
        start_date = end_date - timedelta(days=60)
    elif timeframe == "4h":
        # 4-hour approximated with 60m: ~120 days
        start_date = end_date - timedelta(days=120)
    else:  # "1d"
        # Daily data: calculate days needed for limit candles
        # Add buffer for weekends/holidays
        days_needed = int(limit * 1.5)  # 1.5x buffer for non-trading days
        start_date = end_date - timedelta(days=min(days_needed, 3650))  # Max 10 years

    try:
        df = yf.download(
            symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval=interval,
            progress=False,
            auto_adjust=True
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to fetch stock data: {e}"
        )

    if df is None or df.empty:
        raise HTTPException(
            status_code=400, 
            detail=f"No stock data for symbol: {symbol}"
        )
    
    # Ensure DataFrame is sorted by time (ascending)
    df = df.sort_index(ascending=True)
    
    # Trim to requested limit (from the end, most recent data)
    return df_to_candles(df, limit)


def get_chart_data(
    symbol: str,
    timeframe: str = "1m",
    limit: int = 200,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Get chart data for a symbol (auto-detect crypto vs stock)
    
    Detection logic:
    - Contains "/" → Cryptocurrency (use ccxt)
    - No "/" → Stock (use yfinance)
    
    Args:
        symbol: Symbol string
        timeframe: Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)
        limit: Number of candles (max 3000)
        start: Start date (optional, for future use)
        end: End date (optional, for future use)
        
    Returns:
        List of candle dictionaries (sorted by time ascending)
    """
    # Ensure limit doesn't exceed maximum
    limit = min(limit, MAX_DATA_POINTS)
    
    is_crypto = "/" in symbol

    if is_crypto:
        return fetch_crypto_candles(symbol, timeframe, limit)
    else:
        return fetch_stock_candles(symbol, timeframe, limit)


def get_latest_price(symbol: str, timeframe: str = "1m") -> float:
    """
    Get the latest market price for a symbol
    
    This function fetches the most recent candle and returns its close price.
    Used by PaperTrader to execute orders at current market prices.
    
    Args:
        symbol: Symbol string (crypto or stock)
        timeframe: Timeframe to use (default: "1m" for most recent data)
        
    Returns:
        Latest close price as float
        
    Raises:
        HTTPException: If unable to fetch price data
    """
    try:
        # Fetch just a few recent candles (enough to get latest)
        candles = get_chart_data(symbol, timeframe=timeframe, limit=5)
        
        if not candles:
            raise HTTPException(
                status_code=400,
                detail=f"No price data available for symbol: {symbol}"
            )
        
        # Return the close price of the most recent candle
        latest_close = candles[-1]["close"]
        return float(latest_close)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch latest price for {symbol}: {e}"
        )

def get_historical_candles(
    symbol: str,
    timeframe: str,
    limit: int = 500,
):
    """
    Backtester 用の公式データ取得関数。
    main.py がこの名前の関数を import するので必須。

    実体は get_chart_data のラッパー。
    将来の仕様変更に強く、互換性を保つためのアダプター関数。
    """
    # get_chart_data がすでに crypto / stock を自動判定して返す
    candles = get_chart_data(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
    )

    # timestamp を ISO8601 文字列に変換（BacktestEngine 要件）
    for c in candles:
        c["timestamp"] = datetime.utcfromtimestamp(c["time"]).isoformat() + "Z"

    return candles
