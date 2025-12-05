from __future__ import annotations
from datetime import datetime
from typing import Literal, List, Optional, Dict
from pydantic import BaseModel, Field

# ============================================================================
# 1. PaperAccount
# ============================================================================

class PaperAccount(BaseModel):
    """
    Represents the overall state of a paper trading account.
    """
    account_id: str = Field(..., description="Unique ID for this account (e.g., 'default')")
    base_currency: str = "USD"
    initial_equity: float
    equity: float = Field(..., description="Current total equity (cash + open PnL)")
    cash: float = Field(..., description="Available cash")
    open_risk: float = Field(0.0, description="Sum of risk of all open positions (in currency)")
    created_at: datetime
    updated_at: datetime

    def risk_pct(self) -> float:
        """Return current open risk as percentage of equity (0.0 ~ 1.0)."""
        if self.equity == 0:
            return 0.0
        return self.open_risk / self.equity


# ============================================================================
# 2. PaperPosition
# ============================================================================

class PaperPosition(BaseModel):
    """
    Represents a single open or closed position.
    """
    position_id: str
    account_id: str
    symbol: str
    direction: Literal["LONG", "SHORT"]
    size: float = Field(..., description="Number of shares/contracts (can be non-integer for crypto)")
    entry_price: float
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    opened_at: datetime
    closed_at: Optional[datetime] = None
    status: Literal["OPEN", "CLOSED"] = "OPEN"
    tags: List[str] = Field(default_factory=list)

    def is_open(self) -> bool:
        return self.status == "OPEN"

    def r_risk(self) -> float:
        """
        Return the monetary risk amount (1R) for this position.
        If stop_price is not set or risk is invalid/negative, return 0.0.
        """
        if self.stop_price is None or self.size == 0:
            return 0.0

        if self.direction == "LONG":
            risk_per_share = self.entry_price - self.stop_price
        else:  # SHORT
            risk_per_share = self.stop_price - self.entry_price

        risk = risk_per_share * self.size
        if risk <= 0:
            return 0.0
        return risk


# ============================================================================
# 3. PaperTradeLog
# ============================================================================

class PaperTradeLog(BaseModel):
    """
    Represents a completed trade (one round trip).
    """
    trade_id: str
    account_id: str
    symbol: str
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    exit_price: float
    size: float
    opened_at: datetime
    closed_at: datetime
    gross_pnl: float
    net_pnl: float
    r_multiple: float = Field(..., description="PnL in units of initial risk R (e.g., +2.0R, -0.5R)")
    notes: Optional[str] = None


# ============================================================================
# 4. InMemoryPaperStore
# ============================================================================

class InMemoryPaperStore:
    """
    A simple in-memory 'repository' for paper trading data.
    No persistence; data is lost when the process restarts.
    """
    def __init__(self):
        self.accounts: Dict[str, PaperAccount] = {}
        self.positions: Dict[str, PaperPosition] = {}
        self.trades: Dict[str, PaperTradeLog] = {}

    def create_account(self, account: PaperAccount) -> None:
        if account.account_id in self.accounts:
            raise ValueError(f"Account {account.account_id} already exists")
        self.accounts[account.account_id] = account

    def get_account(self, account_id: str) -> Optional[PaperAccount]:
        return self.accounts.get(account_id)

    def upsert_account(self, account: PaperAccount) -> None:
        self.accounts[account.account_id] = account

    def add_position(self, position: PaperPosition) -> None:
        if position.position_id in self.positions:
            raise ValueError(f"Position {position.position_id} already exists")
        self.positions[position.position_id] = position

    def get_open_positions(self, account_id: str) -> List[PaperPosition]:
        return [
            p for p in self.positions.values()
            if p.account_id == account_id and p.status == "OPEN"
        ]
        
    def get_trades_by_account(self, account_id: str) -> List[PaperTradeLog]:
        return [t for t in self.trades.values() if t.account_id == account_id]

    def delete_position(self, position_id: str) -> bool:
        """
        Hard delete a position (e.g. for cancellation).
        Returns True if deleted, False if not found.
        """
        if position_id in self.positions:
            del self.positions[position_id]
            return True
        return False

    def close_position(self, position_id: str, exit_price: float, closed_at: datetime) -> PaperTradeLog:
        """
        Close an existing position and generate a trade log.
        """
        position = self.positions.get(position_id)
        if not position:
            raise ValueError(f"Position {position_id} not found")
        
        if position.status == "CLOSED":
            raise ValueError(f"Position {position_id} is already closed")
            
        # Update position status
        position.status = "CLOSED"
        position.closed_at = closed_at
        
        # Calculate PnL
        if position.direction == "LONG":
            pnl = (exit_price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - exit_price) * position.size
            
        # Calculate R-multiple
        risk_amount = position.r_risk()
        if risk_amount > 0:
            r_multiple = pnl / risk_amount
        else:
            r_multiple = 0.0 # Undefined or infinite if no risk defined
            
        # Create Trade Log
        import uuid
        trade_log = PaperTradeLog(
            trade_id=str(uuid.uuid4()),
            account_id=position.account_id,
            symbol=position.symbol,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            opened_at=position.opened_at,
            closed_at=closed_at,
            gross_pnl=pnl,
            net_pnl=pnl, # Assuming no fees for now
            r_multiple=r_multiple,
            notes=f"Closed via InMemoryPaperStore at {exit_price}"
        )
        
        self.trades[trade_log.trade_id] = trade_log
        
        # Also update account equity/cash? 
        # The prompt didn't explicitly ask for account update logic here, 
        # but "Mark the position as CLOSED" and "Create a PaperTradeLog" were required.
        # I will stick to the requirements. Account updates might be handled by a service layer.
        
        return trade_log
