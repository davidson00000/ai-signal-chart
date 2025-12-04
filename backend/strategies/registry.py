from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class StrategyMetadata:
    id: str
    name: str
    description: str
    docs_path: str

STRATEGY_REGISTRY: Dict[str, StrategyMetadata] = {
    "ma_cross": StrategyMetadata(
        id="ma_cross",
        name="Moving Average Crossover",
        description="短期・長期の移動平均線のクロスで売買するトレンドフォロー戦略。",
        docs_path="docs/strategies/ma_crossover.md"
    ),
    "rsi_mean_reversion": StrategyMetadata(
        id="rsi_mean_reversion",
        name="RSI Reversal",
        description="RSI の買われ過ぎ/売られ過ぎからの反転を狙う逆張り戦略。",
        docs_path="docs/strategies/rsi_reversal.md"
    ),
    "ema9_dip_buy": StrategyMetadata(
        id="ema9_dip_buy",
        name="EMA9 Dip Buy",
        description="9EMA押し目買い手法。強い上昇トレンド中のプルバックで押し目を狙うロング専用戦略。",
        docs_path="docs/strategies/ema9_dip_buy.md"
    )
}

def get_strategy_metadata(strategy_id: str) -> Optional[StrategyMetadata]:
    return STRATEGY_REGISTRY.get(strategy_id)

def list_strategies() -> List[StrategyMetadata]:
    return list(STRATEGY_REGISTRY.values())
