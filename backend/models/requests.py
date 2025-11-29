from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    """
    リクエストで受け取るバックテスト条件。

    - symbol: 銘柄コード（例: "AAPL", "7203.T", "BTC/USDT"）
    - timeframe: 足種別（例: "1d", "1h"）
    - exchange: データ取得元（"yfinance" / "bybit" など）
    - start: 期間指定の開始日時（ISO8601文字列, 例: "2020-01-01T00:00:00Z"）
    - end: 期間指定の終了日時（指定しない場合は最新まで）
    - limit: ローソク足の最大本数（start/end が無い場合に使用）
    - initial_capital: 初期資金
    - position_size: 常に使う資金の割合（0.0〜1.0）
    - commission_rate: 売買手数料率（例: 0.001 = 0.1%）
    - slippage: スリッページ（価格に対する割合）
    - strategy: 戦略名（"ma_cross" など）
    - fast_ma, slow_ma: MA クロス戦略の期間
    """

    symbol: str = Field(..., description="銘柄コード（例: AAPL, 7203.T, BTC/USDT）")
    timeframe: str = Field("1d", description="足種別（1m, 5m, 1h, 1d など）")
    exchange: str = Field("yfinance", description="データ取得元（yfinance / bybit など）")

    # 期間指定（どちらか、または limit で指定）
    start: Optional[datetime] = Field(
        None,
        description='開始日時（ISO8601, 例: "2020-01-01T00:00:00Z"）',
    )
    end: Optional[datetime] = Field(
        None,
        description='終了日時（ISO8601, 例: "2024-01-01T00:00:00Z"）',
    )
    limit: int = Field(
        500,
        ge=10,
        le=3000,
        description="取得する最大ローソク足本数（start/end なしの場合に使用）",
    )

    initial_capital: float = Field(1_000_000, ge=0, description="初期資金")
    position_size: float = Field(1.0, ge=0.0, le=1.0, description="常に使う資金の割合")
    commission_rate: float = Field(0.0005, ge=0.0, description="売買手数料率")
    slippage: float = Field(0.0, ge=0.0, description="スリッページ（割合）")

    strategy: str = Field("ma_cross", description='戦略名（例: "ma_cross"）')
    fast_ma: int = Field(5, ge=1, description="短期移動平均の期間")
    slow_ma: int = Field(20, ge=2, description="長期移動平均の期間")

    model_config = {"populate_by_name": True}
