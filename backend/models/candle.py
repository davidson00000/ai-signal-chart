"""
Candle (OHLCV) data model
"""
from pydantic import BaseModel, Field
from typing import Optional


class Candle(BaseModel):
    """
    Single candle (OHLCV) data point
    Compatible with Lightweight Charts format
    """
    time: int = Field(..., description="Unix timestamp in seconds")
    open: float
    high: float
    low: float
    close: float
    volume: float = Field(default=0.0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "time": 1701158400,
                "open": 191.50,
                "high": 192.80,
                "low": 191.20,
                "close": 192.50,
                "volume": 1234567.0
            }
        }
