"""
Multi-Symbol Auto Sim Models

Pydantic models for multi-symbol simulation requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class MultiSimConfig(BaseModel):
    """Configuration for multi-symbol simulation."""
    symbols: List[str] = Field(..., description="List of symbols to simulate")
    timeframe: str = Field(default="1d", description="Timeframe")
    start_date: Optional[str] = Field(default=None, description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(default=None, description="End date YYYY-MM-DD")
    initial_capital: float = Field(default=100000.0, description="Initial capital per symbol")
    
    # Strategy settings
    strategy_mode: str = Field(default="ma_crossover", description="Strategy mode")
    ma_short_window: int = Field(default=50, description="Short MA window")
    ma_long_window: int = Field(default=60, description="Long MA window")
    
    # Execution settings
    execution_mode: str = Field(default="same_bar_close", description="Execution mode")
    position_sizing_mode: str = Field(default="full_equity", description="Position sizing")
    
    # R-management settings
    use_r_management: bool = Field(default=True, description="Enable R management")
    virtual_stop_method: str = Field(default="percent", description="Stop calculation method")
    virtual_stop_percent: float = Field(default=0.03, description="Virtual stop percentage")
    
    # Limits
    max_bars: int = Field(default=0, description="Maximum bars to analyze (0=all)")


class MultiSimSymbolResult(BaseModel):
    """Result for a single symbol in multi-sim."""
    rank: int = Field(description="Ranking position")
    symbol: str = Field(description="Symbol")
    final_equity: float = Field(description="Final equity value")
    total_return: float = Field(description="Total return percentage")
    total_r: Optional[float] = Field(default=None, description="Total R (if R-management enabled)")
    avg_r: Optional[float] = Field(default=None, description="Average R per trade")
    best_r: Optional[float] = Field(default=None, description="Best R")
    worst_r: Optional[float] = Field(default=None, description="Worst R")
    win_rate: float = Field(description="Win rate percentage")
    max_dd: float = Field(description="Maximum drawdown percentage")
    trades: int = Field(description="Number of trades")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class MultiSimResult(BaseModel):
    """Response for multi-symbol simulation."""
    results: List[MultiSimSymbolResult] = Field(description="List of symbol results")
    parameters: Dict[str, Any] = Field(description="Simulation parameters used")
    total_symbols: int = Field(description="Total symbols simulated")
    successful: int = Field(description="Number of successful simulations")
    failed: int = Field(description="Number of failed simulations")
