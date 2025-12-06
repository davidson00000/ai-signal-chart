"""
R Analytics Module for Auto Sim Lab

Computes R-based performance metrics from trade data.
R values represent risk multiples: R = PnL / Risk Amount
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any


@dataclass
class RAnalytics:
    """R-based performance analytics for discretionary traders."""
    
    # Core R metrics
    total_r: float
    avg_r_per_trade: float
    max_r: float
    min_r: float
    
    # Win/loss metrics
    win_rate: float  # Percentage as decimal (0.55 = 55%)
    trades_count: int
    winning_trades: int
    losing_trades: int
    
    # Streak metrics
    max_consecutive_wins: int
    max_consecutive_losses: int
    current_streak: int  # Positive for wins, negative for losses
    
    # Time series data
    r_equity_curve: List[float]  # Cumulative R over trades
    r_values: List[float]        # Individual R values per trade
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)


def compute_r_analytics(trades: List[Dict[str, Any]]) -> Optional[RAnalytics]:
    """
    Compute R-based analytics from trade data.
    
    Args:
        trades: List of trade dictionaries containing 'r_value' field.
                Each trade should have at least:
                - r_value: float (PnL in R multiples)
                - exit_time: str (for sorting)
                
    Returns:
        RAnalytics object if valid R data exists, None otherwise.
    """
    # Filter trades with valid r_value
    valid_trades = [
        t for t in trades 
        if t.get('r_value') is not None
    ]
    
    if not valid_trades:
        return None
    
    # Sort by exit_time
    try:
        sorted_trades = sorted(
            valid_trades,
            key=lambda t: t.get('exit_time', '') or ''
        )
    except (TypeError, KeyError):
        sorted_trades = valid_trades
    
    # Extract R values
    r_values = [float(t['r_value']) for t in sorted_trades]
    
    if not r_values:
        return None
    
    # Compute cumulative R equity curve
    r_equity_curve = []
    cumulative = 0.0
    for r in r_values:
        cumulative += r
        r_equity_curve.append(round(cumulative, 2))
    
    # Core metrics
    total_r = sum(r_values)
    avg_r_per_trade = total_r / len(r_values)
    max_r = max(r_values)
    min_r = min(r_values)
    
    # Win/loss count
    winning_trades = sum(1 for r in r_values if r > 0)
    losing_trades = sum(1 for r in r_values if r <= 0)
    win_rate = winning_trades / len(r_values)
    
    # Compute streak metrics
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_win_streak = 0
    current_loss_streak = 0
    
    for r in r_values:
        if r > 0:
            # Winning trade
            current_win_streak += 1
            current_loss_streak = 0
            max_consecutive_wins = max(max_consecutive_wins, current_win_streak)
        else:
            # Losing trade (including breakeven)
            current_loss_streak += 1
            current_win_streak = 0
            max_consecutive_losses = max(max_consecutive_losses, current_loss_streak)
    
    # Current streak (positive = wins, negative = losses)
    if current_win_streak > 0:
        current_streak = current_win_streak
    else:
        current_streak = -current_loss_streak
    
    return RAnalytics(
        total_r=round(total_r, 2),
        avg_r_per_trade=round(avg_r_per_trade, 2),
        max_r=round(max_r, 2),
        min_r=round(min_r, 2),
        win_rate=round(win_rate, 4),
        trades_count=len(r_values),
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        max_consecutive_wins=max_consecutive_wins,
        max_consecutive_losses=max_consecutive_losses,
        current_streak=current_streak,
        r_equity_curve=r_equity_curve,
        r_values=[round(r, 2) for r in r_values]
    )
