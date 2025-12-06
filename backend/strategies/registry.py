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
    ),
    "ema_crossover": StrategyMetadata(
        id="ema_crossover",
        name="EMA Crossover",
        description="短期・長期の指数平滑移動平均線(EMA)のクロスで売買するトレンドフォロー戦略。",
        docs_path="docs/strategies/ema_crossover.md"
    ),
    "macd": StrategyMetadata(
        id="macd",
        name="MACD Signal Line",
        description="MACD線とシグナル線のクロスで売買するモメンタム戦略。",
        docs_path="docs/strategies/macd.md"
    ),
    "breakout": StrategyMetadata(
        id="breakout",
        name="Breakout Strategy",
        description="過去N日間の高値更新で買い、安値更新で売るブレイクアウト戦略。",
        docs_path="docs/strategies/breakout.md"
    ),
    "bollinger": StrategyMetadata(
        id="bollinger",
        name="Bollinger Mean Reversion",
        description="ボリンジャーバンドの±2σバンドブレイクからの平均回帰を狙う逆張り戦略。",
        docs_path="docs/strategies/bollinger.md"
    )
}

def get_strategy_metadata(strategy_id: str) -> Optional[StrategyMetadata]:
    return STRATEGY_REGISTRY.get(strategy_id)

def list_strategies() -> List[StrategyMetadata]:
    return list(STRATEGY_REGISTRY.values())
