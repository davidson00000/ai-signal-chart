"""
Experiments Management Module
Handles saving, loading, and managing backtest experiments
"""
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import uuid

from pydantic import BaseModel, Field


# Experiments storage path
EXPERIMENTS_FILE = Path(__file__).parent.parent.parent / "experiments" / "experiments.json"


class ExperimentCreate(BaseModel):
    """Create new experiment request"""
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(default=None, description="Experiment description")
    strategy_type: str = Field(..., description="Strategy type: ma_cross, rsi, ma_rsi_combo")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    start_date: Optional[str] = Field(default=None, description="Start date")
    end_date: Optional[str] = Field(default=None, description="End date")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    results: Optional[Dict[str, Any]] = Field(default=None, description="Backtest results")


class Experiment(BaseModel):
    """Experiment model with ID and timestamps"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique experiment ID")
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(default=None, description="Experiment description")
    strategy_type: str = Field(..., description="Strategy type")
    symbol: str = Field(..., description="Trading symbol")
    timeframe: str = Field(..., description="Timeframe")
    start_date: Optional[str] = Field(default=None, description="Start date")
    end_date: Optional[str] = Field(default=None, description="End date")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    results: Optional[Dict[str, Any]] = Field(default=None, description="Backtest results")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Creation timestamp")
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Update timestamp")


class ExperimentsManager:
    """Manages experiments storage and retrieval"""
    
    def __init__(self, storage_path: Path = EXPERIMENTS_FILE):
        self.storage_path = storage_path
        self._ensure_storage()
    
    def _ensure_storage(self):
        """Ensure experiments directory and file exist"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self._save_experiments([])
    
    def _load_experiments(self) -> List[Experiment]:
        """Load all experiments from storage"""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [Experiment(**exp) for exp in data]
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _save_experiments(self, experiments: List[Experiment]):
        """Save all experiments to storage"""
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump([exp.model_dump() for exp in experiments], f, indent=2, ensure_ascii=False)
    
    def create_experiment(self, exp_create: ExperimentCreate) -> Experiment:
        """Create and save new experiment"""
        experiment = Experiment(**exp_create.model_dump())
        experiments = self._load_experiments()
        experiments.append(experiment)
        self._save_experiments(experiments)
        return experiment
    
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID"""
        experiments = self._load_experiments()
        for exp in experiments:
            if exp.id == experiment_id:
                return exp
        return None
    
    def list_experiments(self, limit: int = 100, offset: int = 0) -> List[Experiment]:
        """List all experiments with pagination"""
        experiments = self._load_experiments()
        # Sort by created_at descending (newest first)
        experiments.sort(key=lambda x: x.created_at, reverse=True)
        return experiments[offset:offset + limit]
    
    def update_experiment(self, experiment_id: str, updates: Dict[str, Any]) -> Optional[Experiment]:
        """Update existing experiment"""
        experiments = self._load_experiments()
        for i, exp in enumerate(experiments):
            if exp.id == experiment_id:
                # Update fields
                exp_dict = exp.model_dump()
                exp_dict.update(updates)
                exp_dict['updated_at'] = datetime.utcnow().isoformat()
                experiments[i] = Experiment(**exp_dict)
                self._save_experiments(experiments)
                return experiments[i]
        return None
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment by ID"""
        experiments = self._load_experiments()
        original_count = len(experiments)
        experiments = [exp for exp in experiments if exp.id != experiment_id]
        if len(experiments) < original_count:
            self._save_experiments(experiments)
            return True
        return False
    
    def get_experiments_summary(self) -> Dict[str, Any]:
        """Get summary statistics of experiments"""
        experiments = self._load_experiments()
        
        strategy_counts = {}
        symbol_counts = {}
        
        for exp in experiments:
            # Count by strategy
            strategy_counts[exp.strategy_type] = strategy_counts.get(exp.strategy_type, 0) + 1
            # Count by symbol
            symbol_counts[exp.symbol] = symbol_counts.get(exp.symbol, 0) + 1
        
        return {
            "total_experiments": len(experiments),
            "by_strategy": strategy_counts,
            "by_symbol": symbol_counts,
        }
