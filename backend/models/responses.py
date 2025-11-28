"""
API Response models
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = "ok"
    version: Optional[str] = "0.1.0"
    timestamp: Optional[str] = None


class ChartDataResponse(BaseModel):
    """
    Chart data response - maintains backward compatibility with frontend
    """
    symbol: str
    timeframe: str
    candles: List[Dict[str, Any]]  # Keep as Dict for flexibility
    shortMA: List[Optional[float]] = Field(default_factory=list)
    longMA: List[Optional[float]] = Field(default_factory=list)
    signals: List[Dict[str, Any]] = Field(default_factory=list)
    trades: List[Dict[str, Any]] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
    meta: Optional[Dict[str, Any]] = None


class SignalResponse(BaseModel):
    """Signal response per API_SPEC.md"""
    symbol: str
    date: Optional[str] = None
    timeframe: Optional[str] = "1d"
    strategy: str = "ma_cross"
    signal: str  # BUY, SELL, HOLD
    confidence: Optional[float] = None
    reason: Optional[str] = None
    price: float
    meta: Optional[Dict[str, Any]] = None


class OrderResponse(BaseModel):
    """Paper order execution response"""
    order_id: str
    status: str
    symbol: str
    side: str
    quantity: int
    executed_price: float
    executed_at: str
    pnl: float = 0.0


class PositionsResponse(BaseModel):
    """Positions list response"""
    positions: List[Dict[str, Any]]
    total_unrealized_pnl: float = 0.0


class TradesResponse(BaseModel):
    """Trades list response"""
    trades: List[Dict[str, Any]]


class PnLResponse(BaseModel):
    """P&L summary response"""
    mode: str = "daily"
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    pnl: List[Dict[str, Any]] = Field(default_factory=list)
