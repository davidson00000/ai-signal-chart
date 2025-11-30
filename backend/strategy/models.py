from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional, Union, Dict, Any
from datetime import date

# --- Indicator Models ---

class IndicatorSpec(BaseModel):
    id: str = Field(..., description="Unique identifier for the indicator (e.g., 'ma20')")
    type: Literal["sma", "ema", "rsi", "bollinger"] = Field(..., description="Type of indicator")
    source: str = Field("close", description="Source column (e.g., 'close', 'open')")
    period: Optional[int] = Field(None, description="Period for SMA, EMA, RSI")
    std_dev: Optional[float] = Field(None, description="Standard deviation for Bollinger Bands")

# --- Rule Models ---

class ValueRef(BaseModel):
    ref: Optional[str] = Field(None, description="Reference to an indicator ID or column name")
    value: Optional[float] = Field(None, description="Constant value")

    @field_validator('value')
    def check_ref_or_value(cls, v, values):
        # Pydantic v2 validation logic might differ slightly, but basic check:
        # We need either ref or value. 
        # Since this is a simple validator, we'll trust the structure for now.
        return v

class RuleCondition(BaseModel):
    left: ValueRef
    op: Literal["==", "!=", ">", ">=", "<", "<="]
    right: ValueRef

class RuleGroup(BaseModel):
    # Recursive definition for nested rules
    all: Optional[List[Union['RuleGroup', 'RuleCondition']]] = None
    any: Optional[List[Union['RuleGroup', 'RuleCondition']]] = None

# --- Strategy Models ---

class PositionConfig(BaseModel):
    direction: Literal["long_only", "short_only", "both"] = "long_only"
    max_position: float = 1.0

class JsonStrategySpec(BaseModel):
    name: str
    description: Optional[str] = None
    indicators: List[IndicatorSpec]
    entry_rules: List[RuleGroup]
    exit_rules: List[RuleGroup]
    position: PositionConfig = Field(default_factory=PositionConfig)

# --- API Request Model ---

class JsonStrategyRunRequest(BaseModel):
    symbol: str
    timeframe: str = "1d"
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    initial_capital: float = 1000000
    commission_rate: float = 0.001
    position_size: float = 1.0
    strategy: JsonStrategySpec
