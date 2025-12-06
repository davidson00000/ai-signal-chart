from datetime import datetime
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
import json
from pathlib import Path

# Path to store the live strategy JSON
# Assuming this file is in backend/models/, we go up one level to backend/ then to strategies/
LIVE_STRATEGY_PATH = Path(__file__).parent.parent / "strategies" / "live_strategy.json"


class RiskConfig(BaseModel):
    position_mode: Literal["fixed_shares", "fixed_amount_jpy"] = "fixed_shares"
    position_value: float = Field(..., description="Number of shares if mode=fixed_shares, or JPY amount if mode=fixed_amount_jpy")


class LiveStrategyMeta(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    note: Optional[str] = ""


class LiveStrategyConfig(BaseModel):
    symbol: str
    timeframe: str = "1d"
    strategy_name: str
    strategy_type: str
    params: Dict[str, Any]
    risk: RiskConfig
    meta: LiveStrategyMeta = Field(default_factory=LiveStrategyMeta)


def save_live_strategy(config: LiveStrategyConfig) -> None:
    """Save the live strategy configuration to a JSON file."""
    LIVE_STRATEGY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LIVE_STRATEGY_PATH.open("w", encoding="utf-8") as f:
        # model_dump() is for Pydantic v2, dict() for v1. Using model_dump assuming v2 or compatible.
        # If using older pydantic, might need .dict()
        # The user prompt used .model_dump(), so I will stick to that.
        json.dump(config.model_dump(mode='json'), f, ensure_ascii=False, indent=2)


def load_live_strategy() -> LiveStrategyConfig:
    """Load the live strategy configuration from the JSON file."""
    if not LIVE_STRATEGY_PATH.exists():
        raise FileNotFoundError("Live strategy not set yet.")
    with LIVE_STRATEGY_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return LiveStrategyConfig.model_validate(data)
