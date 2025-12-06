"""
Multi-Symbol Auto Sim Engine

Runs Auto Sim Lab across multiple symbols and aggregates results.
"""

from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from backend.auto_sim_lab import run_auto_simulation, AutoSimConfig
from backend.models.multi_sim import (
    MultiSimConfig,
    MultiSimSymbolResult,
    MultiSimResult
)


def run_single_symbol(symbol: str, config: MultiSimConfig) -> MultiSimSymbolResult:
    """
    Run simulation for a single symbol.
    
    Args:
        symbol: Symbol to simulate
        config: Multi-sim configuration
        
    Returns:
        MultiSimSymbolResult with results or error
    """
    try:
        # Create AutoSimConfig for this symbol
        auto_config = AutoSimConfig(
            symbol=symbol,
            timeframe=config.timeframe,
            strategy_mode=config.strategy_mode,
            ma_short_window=config.ma_short_window,
            ma_long_window=config.ma_long_window,
            initial_capital=config.initial_capital,
            position_sizing_mode=config.position_sizing_mode,
            execution_mode=config.execution_mode,
            start_date=config.start_date,
            end_date=config.end_date,
            use_r_management=config.use_r_management,
            virtual_stop_method=config.virtual_stop_method,
            virtual_stop_percent=config.virtual_stop_percent,
            max_bars=config.max_bars
        )
        
        # Run simulation
        result = run_auto_simulation(auto_config)
        
        # Extract metrics
        summary = result.summary
        
        # Calculate max drawdown from equity curve
        max_dd = 0.0
        if result.equity_curve:
            peak = config.initial_capital
            for point in result.equity_curve:
                equity = point.get('equity', 0)
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak * 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
        
        return MultiSimSymbolResult(
            rank=0,  # Will be set after sorting
            symbol=symbol,
            final_equity=round(result.final_equity, 2),
            total_return=round((result.final_equity / config.initial_capital - 1) * 100, 2),
            total_r=summary.get('total_r'),
            avg_r=summary.get('avg_r'),
            best_r=summary.get('best_r'),
            worst_r=summary.get('worst_r'),
            win_rate=summary.get('win_rate', 0.0),
            max_dd=round(-max_dd, 2),  # Negative value to indicate loss
            trades=summary.get('total_trades', 0),
            error=None
        )
        
    except Exception as e:
        traceback.print_exc()
        return MultiSimSymbolResult(
            rank=0,
            symbol=symbol,
            final_equity=config.initial_capital,
            total_return=0.0,
            total_r=None,
            avg_r=None,
            best_r=None,
            worst_r=None,
            win_rate=0.0,
            max_dd=0.0,
            trades=0,
            error=str(e)
        )


def run_multi_simulation(config: MultiSimConfig) -> MultiSimResult:
    """
    Run multi-symbol simulation.
    
    Args:
        config: Multi-sim configuration
        
    Returns:
        MultiSimResult with ranked results
    """
    results: List[MultiSimSymbolResult] = []
    
    # Run simulations (can be parallelized with ThreadPoolExecutor)
    # For now, run sequentially to avoid rate limiting issues
    for symbol in config.symbols:
        result = run_single_symbol(symbol, config)
        results.append(result)
    
    # Count success/failure
    successful = sum(1 for r in results if r.error is None)
    failed = len(results) - successful
    
    # Sort by ranking criteria:
    # 1. Total R (highest first) - only for successful results with R enabled
    # 2. Avg R (if tie)
    # 3. Max DD (lowest magnitude = better) (if tie)
    # Failed results go to the bottom
    
    def sort_key(r: MultiSimSymbolResult):
        if r.error is not None:
            return (1, 0, 0, 0)  # Failed results at bottom
        
        # Primary: Total R (highest first, so negate for descending)
        total_r = r.total_r if r.total_r is not None else -999999
        
        # Secondary: Avg R (highest first)
        avg_r = r.avg_r if r.avg_r is not None else -999999
        
        # Tertiary: Max DD (less negative is better, so use as-is)
        # max_dd is already negative, so higher (less negative) is better
        max_dd = r.max_dd
        
        return (0, -total_r, -avg_r, -max_dd)
    
    results.sort(key=sort_key)
    
    # Assign ranks
    for i, result in enumerate(results):
        result.rank = i + 1
    
    # Build parameters dict
    parameters = {
        "timeframe": config.timeframe,
        "strategy_mode": config.strategy_mode,
        "ma_short_window": config.ma_short_window,
        "ma_long_window": config.ma_long_window,
        "initial_capital": config.initial_capital,
        "execution_mode": config.execution_mode,
        "position_sizing_mode": config.position_sizing_mode,
        "use_r_management": config.use_r_management,
        "virtual_stop_method": config.virtual_stop_method if config.use_r_management else None,
        "virtual_stop_percent": config.virtual_stop_percent if config.use_r_management else None,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "max_bars": config.max_bars
    }
    
    return MultiSimResult(
        results=results,
        parameters=parameters,
        total_symbols=len(config.symbols),
        successful=successful,
        failed=failed
    )
