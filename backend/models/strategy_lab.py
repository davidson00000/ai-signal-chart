from pydantic import BaseModel, Field
from typing import List, Literal, Optional

class StrategyLabBatchRequest(BaseModel):
    study_name: str = Field(..., description="Name of the study")
    symbols: List[str] = Field(..., description="List of symbols to optimize")
    timeframe: str = Field(default="1d", description="Timeframe")
    strategy_type: Literal["ma_cross"] = Field(default="ma_cross", description="Strategy type")
    initial_capital: float = Field(default=1000000, description="Initial capital")
    commission_rate: float = Field(default=0.001, description="Commission rate")
    position_size: float = Field(default=1.0, description="Position size (0-1)")

    short_ma_min: int = Field(default=5, description="Min short window")
    short_ma_max: int = Field(default=20, description="Max short window")
    short_ma_step: int = Field(default=1, description="Short window step")
    long_ma_min: int = Field(default=20, description="Min long window")
    long_ma_max: int = Field(default=60, description="Max long window")
    long_ma_step: int = Field(default=5, description="Long window step")

    metric: Literal["total_return", "sharpe"] = Field(default="total_return", description="Optimization metric")

class StrategyLabSymbolResult(BaseModel):
    symbol: str
    short_window: int
    long_window: int
    total_return: float
    sharpe: Optional[float]
    max_drawdown: float
    win_rate: float
    trades: int
    metric_score: float
    rank: int
    error: Optional[str] = None

class StrategyLabBatchResponse(BaseModel):
    study_name: str
    metric: str
    results: List[StrategyLabSymbolResult]
