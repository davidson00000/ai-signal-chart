"""
Data Feed Layer - Market data acquisition
Migrated from main_legacy.py
"""
import ccxt
import yfinance as yf
from typing import List, Dict, Any, Optional
from fastapi import HTTPException


# --- Configuration ---
# Binance returns 451 from Vercel, so use Bybit for cloud deployment
CRYPTO_EXCHANGE_ID = "bybit"  # Can use "binance" for local only


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
    df = df.dropna().tail(limit)

    for ts, row in df.iterrows():
        candles.append(
            {
                "time": int(ts.timestamp()),  # Unix timestamp in seconds
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0.0)),
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
    Fetch stock price candles using yfinance
    
    Args:
        symbol: Stock symbol (e.g., "AAPL", "7203.T")
        timeframe: Timeframe string (e.g., "1m", "5m", "1h", "1d")
        limit: Number of candles to return
        
    Returns:
        List of candle dictionaries
        
    Raises:
        HTTPException: If data fetch fails or timeframe unsupported
    """
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

    # Determine period based on timeframe
    if timeframe in ("1m", "5m", "15m", "30m", "1h", "4h"):
        period = "7d"   # Intraday: recent 7 days
    else:
        period = "2y"   # Daily: 2 years

    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
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
        limit: Number of candles
        start: Start date (optional, for future use)
        end: End date (optional, for future use)
        
    Returns:
        List of candle dictionaries
    """
    is_crypto = "/" in symbol

    if is_crypto:
        return fetch_crypto_candles(symbol, timeframe, limit)
    else:
        return fetch_stock_candles(symbol, timeframe, limit)
