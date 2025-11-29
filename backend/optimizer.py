"""
Parameter Optimization Module
Grid Search and Bayesian Optimization for Trading Strategies
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, as_dict
import itertools
import pandas as pd


@dataclass
class OptimizationResult:
    """Single optimization result"""
    params: Dict[str, Any]
    total_pnl: float
    return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trade_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "params": self.params,
            "metrics": {
                "total_pnl": self.total_pnl,
                "return_pct": self.return_pct,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "win_rate": self.win_rate,
                "trade_count": self.trade_count
            }
        }


class GridSearchOptimizer:
    """
    Grid Search Parameter Optimization
    
    Tests all combinations of parameters to find the best one
    """
    
    def optimize(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str],
        end_date: Optional[str],
        param_grid: Dict[str, List[Any]],
        initial_capital: float = 1000000,
        commission_rate: float = 0.001,
        position_size: float = 1.0,
        strategy_type: str = "ma_cross"
    ) -> List[OptimizationResult]:
        """
        Run grid search optimization
        
        Args:
            symbol: Symbol to backtest
            timeframe: Timeframe (1d, 1h, etc.)
            start_date: Start date (YYYY-MM-DD or None)
            end_date: End date (YYYY-MM-DD or None)
            param_grid: Dictionary of parameter ranges
            initial_capital: Starting capital
            commission_rate: Commission rate
            position_size: Position size (0-1)
            strategy_type: Strategy type (ma_cross, rsi, ma_rsi_combo)
        
        Returns:
            List of optimization results sorted by total_pnl
        """
        from backend.backtester import BacktestEngine
        from backend.strategies.ma_cross import MACrossStrategy
        from backend.strategies.rsi_strategy import RSIStrategy
        from backend.strategies.ma_rsi_combo import MARSIComboStrategy
        from backend import data_feed
        
        # Get data
        candles = data_feed.get_chart_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=3000,
            start=start_date,
            end=end_date
        )
        
        if not candles:
            raise ValueError(f"No data found for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        if "time" not in df.columns:
            raise ValueError("Data must contain 'time' column")
        
        df = df.set_index(pd.to_datetime(df["time"], unit="s"))
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        results = []
        
        for combo in combinations:
            params = dict(zip(param_names, combo))
            
            # Validate MA parameters
            if "short_window" in params and "long_window" in params:
                if params["short_window"] >= params["long_window"]:
                    continue  # Skip invalid combinations
            
            # Create strategy instance
            try:
                if strategy_type == "ma_cross":
                    strategy = MACrossStrategy(
                        short_window=params.get("short_window", 9),
                        long_window=params.get("long_window", 21)
                    )
                elif strategy_type == "rsi":
                    strategy = RSIStrategy(
                        period=params.get("rsi_period", 14),
                        oversold=params.get("rsi_oversold", 30),
                        overbought=params.get("rsi_overbought", 70)
                    )
                elif strategy_type == "ma_rsi_combo":
                    strategy = MARSIComboStrategy(
                        short_window=params.get("short_window", 9),
                        long_window=params.get("long_window", 21),
                        rsi_period=params.get("rsi_period", 14),
                        rsi_oversold=params.get("rsi_oversold", 30),
                        rsi_overbought=params.get("rsi_overbought", 70)
                    )
                else:
                    raise ValueError(f"Unknown strategy type: {strategy_type}")
                
                # Run backtest
                engine = BacktestEngine(
                    initial_capital=initial_capital,
                    commission_rate=commission_rate,
                    position_size=position_size
                )
                
                backtest_result = engine.run_backtest(df, strategy)
                
                # Store result
                results.append(OptimizationResult(
                    params=params,
                    total_pnl=backtest_result["total_pnl"],
                    return_pct=backtest_result["return_pct"],
                    sharpe_ratio=backtest_result.get("sharpe_ratio", 0.0),
                    max_drawdown=backtest_result["max_drawdown"],
                    win_rate=backtest_result["win_rate"],
                    trade_count=backtest_result["trade_count"]
                ))
                
            except Exception as e:
                # Skip failed combinations
                print(f"Failed to backtest params {params}: {e}")
                continue
        
        # Sort by total PnL (descending)
        results.sort(key=lambda x: x.total_pnl, reverse=True)
        
        return results
