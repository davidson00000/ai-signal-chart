"""
Execution Layer - Paper Trading
"""
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime


class PaperTrader:
    """
    Paper trading system for simulated order execution
    
    This is a new implementation for Phase 2 of the ROADMAP.
    Manages virtual positions and tracks P&L without real money.
    """
    
    def __init__(self, initial_cash: float = 100000.0):
        """
        Initialize paper trader
        
        Args:
            initial_cash: Starting cash balance (default: 100,000)
        """
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position info
        self.trades: List[Dict[str, Any]] = []
        self.order_counter = 0
        
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self.order_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"paper-{timestamp}-{self.order_counter:04d}"
    
    def execute_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: int,
        price: Optional[float] = None,
        signal_id: Optional[str] = None,
        order_time: Optional[str] = None,
        mode: str = "market"
    ) -> Dict[str, Any]:
        """
        Execute a paper trading order
        
        Args:
            symbol: Symbol to trade
            side: BUY or SELL
            quantity: Number of shares/units
            price: Limit price (optional, uses market if None)
            signal_id: Associated signal ID (optional)
            order_time: Order timestamp (optional, uses now if None)
            mode: "market" or "limit"
            
        Returns:
            Order response per API_SPEC.md
        """
        order_id = self._generate_order_id()
        executed_price = price if price else 0.0  # In real impl, would fetch market price
        executed_at = order_time if order_time else datetime.now().isoformat()
        
        # Calculate cost
        cost = executed_price * quantity
        
        if side == "BUY":
            # Check if enough cash
            if cost > self.cash:
                return {
                    "order_id": order_id,
                    "status": "rejected",
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "executed_price": 0.0,
                    "executed_at": executed_at,
                    "pnl": 0.0,
                    "message": "Insufficient cash"
                }
            
            # Execute buy
            self.cash -= cost
            
            # Update position
            if symbol not in self.positions:
                self.positions[symbol] = {
                    "quantity": quantity,
                    "avg_price": executed_price,
                    "total_cost": cost
                }
            else:
                pos = self.positions[symbol]
                new_qty = pos["quantity"] + quantity
                new_cost = pos["total_cost"] + cost
                pos["quantity"] = new_qty
                pos["total_cost"] = new_cost
                pos["avg_price"] = new_cost / new_qty if new_qty > 0 else 0.0
        
        else:  # SELL
            # Check if position exists
            if symbol not in self.positions or self.positions[symbol]["quantity"] < quantity:
                return {
                    "order_id": order_id,
                    "status": "rejected",
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "executed_price": 0.0,
                    "executed_at": executed_at,
                    "pnl": 0.0,
                    "message": "Insufficient position"
                }
            
            # Execute sell
            proceeds = executed_price * quantity
            self.cash += proceeds
            
            # Calculate P&L for this specific sale
            pos = self.positions[symbol]
            cost_basis = pos["avg_price"] * quantity
            pnl = proceeds - cost_basis
            
            # Update position
            pos["quantity"] -= quantity
            pos["total_cost"] -= cost_basis
            
            # Remove position if fully closed
            if pos["quantity"] == 0:
                del self.positions[symbol]
        
        # Record trade
        trade_record = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": executed_price,
            "executed_at": executed_at,
            "signal_id": signal_id,
            "pnl": pnl if side == "SELL" else 0.0
        }
        self.trades.append(trade_record)
        
        return {
            "order_id": order_id,
            "status": "accepted",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "executed_price": executed_price,
            "executed_at": executed_at,
            "pnl": pnl if side == "SELL" else 0.0
        }
    
    
    def get_positions(self, price_lookup_fn=None) -> List[Dict[str, Any]]:
        """
        Get current positions with optional real-time pricing
        
        Args:
            price_lookup_fn: Optional callback function(symbol) -> float
                            to fetch current market prices
        
        Returns:
            List of position dictionaries per API_SPEC.md
        """
        positions = []
        for symbol, pos in self.positions.items():
            current_price = None
            unrealized_pnl = None
            
            # If price lookup function provided, fetch current price
            if price_lookup_fn:
                try:
                    current_price = price_lookup_fn(symbol)
                    # Calculate unrealized P&L
                    # P&L = (current_price - avg_price) * quantity
                    unrealized_pnl = (current_price - pos["avg_price"]) * pos["quantity"]
                except Exception:
                    # If price fetch fails, keep as None
                    pass
            
            positions.append({
                "symbol": symbol,
                "quantity": pos["quantity"],
                "avg_price": pos["avg_price"],
                "current_price": current_price,
                "unrealized_pnl": unrealized_pnl
            })
        return positions

    
    def get_trades(
        self,
        symbol: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get trade history
        
        Args:
            symbol: Filter by symbol (optional)
            from_date: Start date filter (optional)
            to_date: End date filter (optional)
            limit: Maximum number of trades to return
            
        Returns:
            List of trade dictionaries per API_SPEC.md
        """
        filtered_trades = self.trades
        
        # Apply filters (basic implementation)
        if symbol:
            filtered_trades = [t for t in filtered_trades if t["symbol"] == symbol]
        
        # Limit results
        return filtered_trades[-limit:]
    
    def total_pnl(self) -> float:
        """
        Calculate total realized P&L
        
        Returns:
            Total P&L from all closed trades
        """
        return sum(t.get("pnl", 0.0) for t in self.trades)
    
    def get_equity(self) -> float:
        """
        Get total equity (cash + position value)
        
        Note: In real implementation, would fetch current prices
        
        Returns:
            Total account equity
        """
        # For now, just return cash (positions would need current market prices)
        return self.cash
