"""
Backtest-related Pydantic models
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class BacktestRequest(BaseModel):
    """Request model for backtest simulation"""
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL, 7203.T)")
    timeframe: str = Field("1d", description="Timeframe (1m,5m,15m,30m,1h,4h,1d)")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    strategy: str = Field("ma_cross", description="Strategy name")
    initial_capital: float = Field(1000000.0, description="Initial capital")
    commission: float = Field(0.0005, description="Commission rate (0.0005 = 0.05%)")
    position_size: float = Field(1.0, description="Position size fraction (1.0 = 100%)")
    short_window: Optional[int] = Field(9, description="Short MA window")
    long_window: Optional[int] = Field(21, description="Long MA window")


class BacktestTrade(BaseModel):
    """Model for a single trade"""
    date: datetime
    side: str  # BUY or SELL
    price: float
    quantity: int
    commission: float
    pnl: Optional[float] = None
    cash_after: float


class EquityPoint(BaseModel):
    """Model for equity curve point"""
    date: datetime
    equity: float
    cash: float
    position_value: float


class BacktestMetrics(BaseModel):
    """Model for backtest performance metrics"""
    initial_capital: float
    final_equity: float
    total_pnl: float
    return_pct: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    winning_trades: int
    losing_trades: int


class BacktestResponse(BaseModel):
    """Response model for backtest simulation"""
    symbol: str
    timeframe: str
    strategy: str
    metrics: BacktestMetrics
    trades: List[BacktestTrade]
    equity_curve: List[EquityPoint]
    data_points: int = Field(description="Number of data points used")
