# backend/utils/experiments.py
from __future__ import annotations

from pathlib import Path
from typing import List

from backend.models.backtest import (
    BacktestExperiment,
    BacktestExperimentSummary,
    BacktestExperimentCreate,
    BacktestRequest,
)
from datetime import datetime
from uuid import uuid4
import json

EXPERIMENTS_DIR = Path("experiments")


def _ensure_dir() -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def _experiment_path(exp_id: str) -> Path:
    return EXPERIMENTS_DIR / f"{exp_id}.json"


def create_experiment(payload: BacktestExperimentCreate) -> BacktestExperiment:
    """新しい実験を作成して保存"""
    _ensure_dir()
    now = datetime.utcnow()
    exp = BacktestExperiment(
        id=uuid4().hex,
        name=payload.name,
        description=payload.description,
        created_at=now,
        updated_at=now,
        config=payload.config,
    )
    path = _experiment_path(exp.id)
    with path.open("w", encoding="utf-8") as f:
        # Pydantic v2 を想定
        json.dump(exp.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
    return exp


def load_experiment(exp_id: str) -> BacktestExperiment:
    """ID から実験を読み込む"""
    path = _experiment_path(exp_id)
    if not path.exists():
        raise FileNotFoundError(f"Experiment {exp_id} not found")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return BacktestExperiment.model_validate(data)


def list_experiments() -> List[BacktestExperimentSummary]:
    """保存済み実験の一覧を取得"""
    _ensure_dir()
    summaries: List[BacktestExperimentSummary] = []
    for path in sorted(EXPERIMENTS_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        exp = BacktestExperiment.model_validate(data)
        summaries.append(
            BacktestExperimentSummary(
                id=exp.id,
                name=exp.name,
                description=exp.description,
                created_at=exp.created_at,
                updated_at=exp.updated_at,
            )
        )
    return summaries
