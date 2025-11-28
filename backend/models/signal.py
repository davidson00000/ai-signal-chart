"""
Signal data model
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any


class Signal(BaseModel):
    """
    Trading signal from strategy
    """
    id: Optional[int] = None
    symbol: str
    time: int = Field(..., description="Unix timestamp in seconds")
    side: Literal["BUY", "SELL", "HOLD"]
    price: float
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    reason: Optional[str] = None
    strategy: Optional[str] = "ma_cross"
    
    # MA cross specific fields (for backward compatibility)
    tp: Optional[float] = Field(None, description="Take Profit price")
    sl: Optional[float] = Field(None, description="Stop Loss price")
    index: Optional[int] = Field(None, description="Candle index")
    
    # Additional metadata
    meta: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "time": 1701158400,
                "side": "BUY",
                "price": 192.50,
                "confidence": 0.73,
                "reason": "short_ma crossed above long_ma",
                "strategy": "ma_cross",
                "tp": 194.43,
                "sl": 191.54
            }
        }
