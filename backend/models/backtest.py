"""
Backtest-related Pydantic models (Pydantic v2 compatible)
Consolidated from multiple model files for consistency
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    """
    バックテスト実行用のリクエストモデル。

    フロントエンドからの POST /simulate リクエストで使用される。
    """

    symbol: str = Field(..., description="Symbol code, e.g. AAPL, BTC/USDT")
    timeframe: str = Field(default="1d", description="Timeframe, e.g. 1d, 1h, 5m")

    start_date: Optional[str] = Field(
        default=None,
        description="Start datetime for backtest (ISO8601 string)",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End datetime for backtest (ISO8601 string)",
    )

    strategy: str = Field(
        default="ma_cross",
        description="Strategy name (currently only 'ma_cross' supported)",
    )

    initial_capital: float = Field(
        default=1_000_000,
        description="Initial capital for the backtest",
    )
    commission_rate: float = Field(
        default=0.0,
        description="Commission rate per trade (e.g., 0.001 = 0.1%)",
    )
    position_size: float = Field(
        default=1.0,
        description="Position size multiplier (0.0-1.0)",
    )

    short_window: int = Field(
        default=9,
        description="Short moving average window for MA cross strategy",
    )
    long_window: int = Field(
        default=21,
        description="Long moving average window for MA cross strategy",
    )


class EquityCurvePoint(BaseModel):
    """
    エクイティカーブ（残高推移）の1点。
    """

    date: str = Field(..., description="Time in ISO8601 format")
    equity: float = Field(..., description="Account equity at this timestamp")
    cash: Optional[float] = Field(default=None, description="Cash balance (optional)")


class TradeSummary(BaseModel):
    """
    バックテスト結果としての 1 トレード情報。
    """

    date: str = Field(..., description="Trade execution time (ISO8601)")
    side: str = Field(..., description="BUY or SELL")
    price: float = Field(..., description="Execution price")
    quantity: float = Field(..., description="Quantity traded")
    commission: float = Field(..., description="Commission paid")
    pnl: Optional[float] = Field(default=None, description="Profit/Loss (null for BUY)")
    cash_after: float = Field(..., description="Cash balance after trade")


class BacktestStats(BaseModel):
    """
    バックテスト全体の集計メトリクス。
    """

    initial_capital: float = Field(..., description="Initial capital")
    final_equity: float = Field(..., description="Final equity")
    total_pnl: float = Field(..., description="Total profit/loss")
    return_pct: float = Field(..., description="Return percentage")

    trade_count: int = Field(default=0, description="Total number of trades")
    winning_trades: int = Field(default=0, description="Number of winning trades")
    losing_trades: int = Field(default=0, description="Number of losing trades")
    win_rate: float = Field(default=0.0, description="Win rate (0.0-1.0)")

    max_drawdown: float = Field(default=0.0, description="Maximum drawdown")
    sharpe_ratio: Optional[float] = Field(default=None, description="Sharpe ratio")


class BacktestResponse(BaseModel):
    """
    /simulate バックテスト API のレスポンスモデル。
    """

    symbol: str = Field(..., description="Symbol backtested")
    timeframe: str = Field(..., description="Timeframe used")
    strategy: str = Field(..., description="Strategy name")

    equity_curve: List[EquityCurvePoint] = Field(
        default_factory=list,
        description="Equity curve data",
    )
    trades: List[TradeSummary] = Field(
        default_factory=list,
        description="List of all trades",
    )
    metrics: BacktestStats = Field(..., description="Summary statistics")

    data_points: int = Field(
        default=0,
        description="Number of candles used in backtest",
    )


__all__ = [
    "BacktestRequest",
    "BacktestResponse",
    "BacktestStats",
    "EquityCurvePoint",
    "TradeSummary",
]
