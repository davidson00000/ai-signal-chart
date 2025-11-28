"""
Trade and Position data models
"""
from pydantic import BaseModel, Field
from typing import Literal, Optional


class Trade(BaseModel):
    """
    Executed trade record
    """
    id: Optional[int] = None
    order_id: Optional[str] = None
    symbol: str
    side: Literal["LONG", "SHORT", "BUY", "SELL"]
    
    # Entry
    entry_time: int = Field(..., description="Unix timestamp in seconds")
    entry_price: float
    
    # Exit
    exit_time: Optional[int] = Field(None, description="Unix timestamp in seconds")
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = Field(None, description="TP, SL, or Reverse")
    
    # Trade details
    quantity: Optional[int] = Field(None, ge=1)
    pnl: Optional[float] = Field(None, description="P&L as ratio (0.021 = +2.1%)")
    
    # Strategy info
    strategy: Optional[str] = None
    signal_id: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "order_id": "paper-20251102-0001",
                "symbol": "AAPL",
                "side": "LONG",
                "entry_time": 1701158400,
                "entry_price": 192.50,
                "exit_time": 1701244800,
                "exit_price": 195.10,
                "exit_reason": "TP",
                "quantity": 10,
                "pnl": 0.0135,
                "strategy": "ma_cross"
            }
        }


class Position(BaseModel):
    """
    Current position holding
    """
    symbol: str
    quantity: int = Field(..., ge=0)
    avg_price: float = Field(..., gt=0)
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = Field(None, description="Unrealized P&L in currency")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "quantity": 30,
                "avg_price": 190.20,
                "current_price": 192.50,
                "unrealized_pnl": 69.0
            }
        }
