"""
Decision Log Models for Auto Sim Lab

This module defines the data structures for tracking trading decisions
during automated paper trading simulations.
"""

from datetime import datetime
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel, Field


class DecisionEvent(BaseModel):
    """
    Represents a single decision event during Auto Sim Lab simulation.
    
    Events can be:
    - signal_decision: The signal generator produced a signal
    - entry: A new position was opened
    - exit: A position was closed
    """
    timestamp: datetime
    symbol: str
    timeframe: str  # e.g., '1d', '1h'
    
    event_type: Literal["signal_decision", "entry", "exit"]
    
    # Signal-related fields
    final_signal: Optional[str] = None  # "buy", "sell", "hold"
    raw_signals: Optional[Dict[str, Any]] = None  # Predictor details
    
    # Position / Trade fields
    position_side: Optional[Literal["long", "flat"]] = None
    position_size: Optional[float] = None
    price: Optional[float] = None
    
    # Capital management fields
    equity_before: Optional[float] = None
    equity_after: Optional[float] = None
    risk_per_trade: Optional[float] = None  # 0.01 = 1%
    
    # Human-readable reason
    reason: str = Field(..., description="Human-readable explanation of the decision")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DecisionLog:
    """
    Container for collecting DecisionEvent objects during simulation.
    """
    def __init__(self) -> None:
        self.events: List[DecisionEvent] = []
    
    def add(self, event: DecisionEvent) -> None:
        """Add a new decision event to the log."""
        self.events.append(event)
    
    def to_list(self) -> List[Dict]:
        """Convert all events to a list of dictionaries."""
        return [e.model_dump() for e in self.events]
    
    def __len__(self) -> int:
        return len(self.events)
