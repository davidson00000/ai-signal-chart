"""
Decision Log Models for Auto Sim Lab

This module defines the data structures for tracking trading decisions
during automated paper trading simulations.

Enhanced with R-management, execution mode, and loss control fields.
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
    - halt: Simulation halted due to loss control
    """
    timestamp: datetime
    symbol: str
    timeframe: str  # e.g., '1d', '1h'
    
    event_type: Literal["signal_decision", "entry", "exit", "halt"]
    
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
    
    # R-Management fields
    atr_value: Optional[float] = None
    stop_price: Optional[float] = None
    risk_amount: Optional[float] = None  # Dollar risk (R)
    r_value: Optional[float] = None  # PnL in R multiples
    
    # Execution fields
    execution_price: Optional[float] = None
    execution_mode: Optional[str] = None  # "same_bar_close" or "next_bar_open"
    commission: Optional[float] = None
    slippage: Optional[float] = None
    
    # Loss control fields
    halt_reason: Optional[str] = None
    current_drawdown: Optional[float] = None
    daily_r_loss: Optional[float] = None
    
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
