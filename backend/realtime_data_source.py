"""
Realtime Data Source for Auto Sim Lab

This module provides an abstract interface and implementations for
fetching real-time price data for the Realtime Auto Sim Lab.

Current implementation uses yfinance for price polling.
Future implementations could use WebSocket feeds or other APIs.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any
import yfinance as yf
from pydantic import BaseModel


class RealtimeCandle(BaseModel):
    """
    Represents a single OHLCV candle from realtime data source.
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class RealtimeDataSource(ABC):
    """
    Abstract base class for realtime data sources.
    Implementations must provide methods to fetch latest price and candle data.
    """
    
    @abstractmethod
    async def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Stock/crypto symbol (e.g., 'AAPL', 'BTC-USD')
            
        Returns:
            Current price as float
        """
        pass
    
    @abstractmethod
    async def get_or_update_candle(self, symbol: str, timeframe: str) -> Optional[RealtimeCandle]:
        """
        Get or update the current candle for the given timeframe.
        
        Args:
            symbol: Stock/crypto symbol
            timeframe: Candle timeframe (e.g., '1m', '5m')
            
        Returns:
            Current/latest candle or None if unavailable
        """
        pass


class SimpleYahooDataSource(RealtimeDataSource):
    """
    Simple polling-based data source using yfinance.
    
    This is a basic implementation that polls Yahoo Finance for price data.
    Not suitable for high-frequency trading but works for demonstration.
    """
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._current_candles: Dict[str, RealtimeCandle] = {}
    
    async def get_latest_price(self, symbol: str) -> float:
        """
        Fetch latest price from Yahoo Finance.
        
        Uses yfinance to get the current market price.
        Falls back to previous close if market is closed.
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Try to get realtime price
            info = ticker.info
            
            # Try different price fields
            price = info.get('regularMarketPrice') or info.get('currentPrice')
            
            if price is None:
                # Fallback to last close from history
                hist = ticker.history(period="1d")
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
                else:
                    raise ValueError(f"No price data available for {symbol}")
            
            return float(price)
            
        except Exception as e:
            # If all else fails, try to get from recent history
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                if not hist.empty:
                    return float(hist['Close'].iloc[-1])
            except:
                pass
            raise ValueError(f"Failed to get price for {symbol}: {e}")
    
    async def get_or_update_candle(self, symbol: str, timeframe: str) -> Optional[RealtimeCandle]:
        """
        Get or update current candle for the symbol.
        
        For simplicity, this creates a new candle each time based on current price.
        In a production system, this would aggregate ticks into proper OHLC candles.
        """
        try:
            price = await self.get_latest_price(symbol)
            now = datetime.utcnow()
            
            key = f"{symbol}_{timeframe}"
            
            # Check if we have an existing candle for this period
            if key in self._current_candles:
                existing = self._current_candles[key]
                # Update existing candle
                updated = RealtimeCandle(
                    timestamp=existing.timestamp,
                    open=existing.open,
                    high=max(existing.high, price),
                    low=min(existing.low, price),
                    close=price,
                    volume=existing.volume  # Volume tracking not implemented
                )
                self._current_candles[key] = updated
                return updated
            else:
                # Create new candle
                candle = RealtimeCandle(
                    timestamp=now,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=0.0
                )
                self._current_candles[key] = candle
                return candle
                
        except Exception as e:
            print(f"Error fetching candle for {symbol}: {e}")
            return None
    
    def reset_candle(self, symbol: str, timeframe: str) -> None:
        """
        Reset the candle for a new period.
        Called when moving to a new timeframe period.
        """
        key = f"{symbol}_{timeframe}"
        if key in self._current_candles:
            del self._current_candles[key]


# Singleton instance for shared use
_yahoo_data_source: Optional[SimpleYahooDataSource] = None


def get_yahoo_data_source() -> SimpleYahooDataSource:
    """Get or create singleton Yahoo data source instance."""
    global _yahoo_data_source
    if _yahoo_data_source is None:
        _yahoo_data_source = SimpleYahooDataSource()
    return _yahoo_data_source
