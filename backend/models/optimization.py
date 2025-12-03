"""
Pydantic models for optimization requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class OptimizationRequest(BaseModel):
    """Request model for parameter optimization"""
    
    symbol: str = Field(..., description="Symbol to optimize")
    timeframe: str = Field(default="1d", description="Timeframe")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    
    strategy_type: str = Field(default="ma_cross", description="Strategy type: ma_cross, rsi, ma_rsi_combo")
    
    # MA parameters
    short_window_min: int = Field(default=5, description="Min short window")
    short_window_max: int = Field(default=20, description="Max short window")
    short_window_step: int = Field(default=2, description="Short window step")
    
    long_window_min: int = Field(default=20, description="Min long window")
    long_window_max: int = Field(default=60, description="Max long window")
    long_window_step: int = Field(default=5, description="Long window step")
    
    # RSI parameters (optional)
    rsi_period_min: Optional[int] = Field(default=10, description="Min RSI period")
    rsi_period_max: Optional[int] = Field(default=20, description="Max RSI period")
    rsi_period_step: Optional[int] = Field(default=2, description="RSI period step")
    
    rsi_oversold_min: Optional[float] = Field(default=25, description="Min RSI oversold")
    rsi_oversold_max: Optional[float] = Field(default=35, description="Max RSI oversold")
    rsi_oversold_step: Optional[float] = Field(default=5, description="RSI oversold step")
    
    rsi_overbought_min: Optional[float] = Field(default=65, description="Min RSI overbought")
    rsi_overbought_max: Optional[float] = Field(default=75, description="Max RSI overbought")
    rsi_overbought_step: Optional[float] = Field(default=5, description="RSI overbought step")
    
    # Backtest parameters
    initial_capital: float = Field(default=1000000, description="Initial capital")
    commission_rate: float = Field(default=0.001, description="Commission rate")
    position_size: float = Field(default=1.0, description="Position size (0-1)")
    
    # Output parameters
    top_n: int = Field(default=10, description="Number of top results to return")


class OptimizationResponse(BaseModel):
    """Response model for optimization results"""
    
    symbol: str = Field(..., description="Symbol optimized")
    timeframe: str = Field(..., description="Timeframe used")
    strategy_type: str = Field(..., description="Strategy type")
    total_combinations: int = Field(..., description="Total combinations tested")
    top_results: List[Dict[str, Any]] = Field(..., description="Top optimization results")


class MACrossOptimizationRequest(BaseModel):
    """Request model specifically for MA Cross optimization"""
    
    symbol: str = Field(..., description="Symbol to optimize")
    timeframe: str = Field(default="1d", description="Timeframe")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    
    initial_capital: float = Field(default=1000000, description="Initial capital")
    commission_rate: float = Field(default=0.001, description="Commission rate")
    
    # Grid Search Ranges
    short_min: int = Field(default=5, description="Min short window")
    short_max: int = Field(default=20, description="Max short window")
    short_step: int = Field(default=1, description="Short window step")
    
    long_min: int = Field(default=20, description="Min long window")
    long_max: int = Field(default=60, description="Max long window")
    long_step: int = Field(default=5, description="Long window step")


class GenericOptimizationRequest(BaseModel):
    """Generic request model for parameter optimization"""
    
    symbol: str = Field(..., description="Symbol to optimize")
    timeframe: str = Field(default="1d", description="Timeframe")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    
    strategy_type: str = Field(..., description="Strategy type")
    
    initial_capital: float = Field(default=1000000, description="Initial capital")
    commission_rate: float = Field(default=0.001, description="Commission rate")
    
    # Generic Parameter Grid
    # Example: {"short_window": [5, 10, 15], "long_window": [20, 30, 40]}
    param_grid: Dict[str, List[Any]] = Field(..., description="Dictionary of parameter names and list of values to test")
    fixed_params: Dict[str, Any] = Field(default={}, description="Fixed parameters to pass to strategy")
