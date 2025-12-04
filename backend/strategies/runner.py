from typing import Dict, Any, Optional
import pandas as pd
from backend.strategies.ma_cross import MACrossStrategy
from backend.backtester import BacktestEngine

def run_ma_cross_backtest(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    initial_capital: float = 1000000,
    commission_rate: float = 0.001,
    start_date: Optional[pd.Timestamp] = None
) -> Dict[str, Any]:
    """
    Unified runner for MA Cross Strategy backtest.
    Used by both Single Run and Optimizer to ensure consistency.
    
    Args:
        df: DataFrame with OHLCV data
        short_window: Short MA window
        long_window: Long MA window
        initial_capital: Initial capital
        commission_rate: Commission rate
        start_date: Optional start date for backtest (data before is warm-up)
        
    Returns:
        Dict with keys: equity_curve, trades, stats
    """
    # 1. Instantiate Strategy
    strategy = MACrossStrategy(short_window=short_window, long_window=long_window)
    
    # 2. Instantiate Engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        position_size=1.0
    )
    
    # 3. Run Backtest
    # start_date is passed to skip trading during warm-up period
    result = engine.run_backtest(df, strategy, start_date=start_date)
    
    return result
