"""strategy_lab/auto_generator.py

Strategy DSL (.dsl JSON) を読み込んで、
対応する Python Strategy クラスのインスタンスを生成するモジュール。

v0.1 では:
- type="ma_cross" のみサポート
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from backend.strategies.ma_cross import MACrossStrategy
from backend.strategies.base import BaseStrategy


@dataclass
class StrategySpec:
    """DSL をパースした結果を保持するクラス"""

    name: str
    type: str
    params: Dict[str, Any]
    description: str | None = None
    meta: Dict[str, Any] | None = None


def load_dsl(path: str | Path) -> StrategySpec:
    """.dsl (JSON) ファイルから StrategySpec を生成する。"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return StrategySpec(
        name=data.get("name", "Unnamed Strategy"),
        type=data["type"],
        params=data.get("params", {}),
        description=data.get("description"),
        meta=data.get("meta"),
    )


def create_strategy_from_spec(spec: StrategySpec) -> BaseStrategy:
    """StrategySpec から実際の Strategy インスタンスを作る。"""
    stype = spec.type.lower()

    if stype == "ma_cross":
        return MACrossStrategy(**spec.params)

    raise ValueError(f"Unsupported strategy type: {spec.type}")


def load_strategy_from_dsl(path: str | Path) -> BaseStrategy:
    """DSL を読み込み Strategy インスタンスを返すワンショット関数。"""
    spec = load_dsl(path)
    return create_strategy_from_spec(spec)


def suggest_ma_params_with_llm(prompt: str) -> Dict[str, int]:
    """将来 LLM 接続するためのフック（今はダミー実装）。"""
    # TODO: OpenAI / Gemini 等と接続
    return {"short_window": 9, "long_window": 21}
