from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class EquityCurvePoint(BaseModel):
    timestamp: datetime = Field(..., description="時刻")
    equity: float = Field(..., description="その時点の口座残高")


class TradeSummary(BaseModel):
    entry_time: datetime = Field(..., description="エントリー時刻")
    exit_time: datetime = Field(..., description="手仕舞い時刻")
    entry_price: float = Field(..., description="エントリー価格")
    exit_price: float = Field(..., description="手仕舞い価格")
    qty: float = Field(..., description="数量（株数・枚数など）")
    pnl: float = Field(..., description="損益")


class BacktestResponse(BaseModel):
    """
    /simulate エンドポイントのレスポンス用スキーマ
    """

    symbol: str = Field(..., description="バックテスト対象銘柄")
    timeframe: str = Field(..., description="足種別")
    strategy: str = Field(..., description="使用した戦略名")

    initial_capital: float = Field(..., description="初期資金")
    final_equity: float = Field(..., description="最終残高")
    total_pnl: float = Field(..., description="トータル損益")
    return_pct: float = Field(..., description="リターン（%）")

    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="その他のメトリクス（勝率、最大ドローダウンなど）",
    )
    equity_curve: List[EquityCurvePoint] = Field(
        default_factory=list,
        description="残高推移",
    )
    trades: List[TradeSummary] = Field(
        default_factory=list,
        description="全トレード一覧",
    )
