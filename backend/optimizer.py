"""
Parameter Optimization Module
Grid Search and Bayesian Optimization for Trading Strategies
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
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
        Run grid search optimization for a single symbol
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
                
                # Extract stats from result
                stats = backtest_result.get("stats", {})
                
                # Store result
                results.append(OptimizationResult(
                    params=params,
                    total_pnl=stats.get("total_pnl", 0.0),
                    return_pct=stats.get("return_pct", 0.0),
                    sharpe_ratio=stats.get("sharpe_ratio", 0.0),
                    max_drawdown=stats.get("max_drawdown", 0.0),
                    win_rate=stats.get("win_rate", 0.0),
                    trade_count=stats.get("trade_count", 0)
                ))
                
            except Exception as e:
                # Skip failed combinations
                # print(f"Failed to backtest params {params}: {e}")
                continue
        
        # Sort by total PnL (descending)
        results.sort(key=lambda x: x.total_pnl, reverse=True)
        
        return results

    def optimize_batch(
        self,
        symbols: List[str],
        timeframe: str,
        param_grid: Dict[str, List[Any]],
        metric: str = "total_return",
        initial_capital: float = 1000000,
        commission_rate: float = 0.001,
        position_size: float = 1.0,
        strategy_type: str = "ma_cross"
    ) -> List[Dict[str, Any]]:
        """
        Run batch optimization for multiple symbols
        """
        batch_results = []
        
        for symbol in symbols:
            try:
                # Run optimization for single symbol
                results = self.optimize(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=None, # Use full available data
                    end_date=None,
                    param_grid=param_grid,
                    initial_capital=initial_capital,
                    commission_rate=commission_rate,
                    position_size=position_size,
                    strategy_type=strategy_type
                )
                
                if not results:
                    continue
                    
                # Select best result based on metric
                if metric == "sharpe":
                    # Filter out None sharpe ratios and sort
                    valid_results = [r for r in results if r.sharpe_ratio is not None]
                    if valid_results:
                        valid_results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
                        best = valid_results[0]
                    else:
                        best = results[0] # Fallback to total_pnl
                else:
                    # Default to total_return (already sorted)
                    best = results[0]
                
                # Create result entry
                entry = {
                    "symbol": symbol,
                    "short_window": best.params.get("short_window"),
                    "long_window": best.params.get("long_window"),
                    "total_return": best.return_pct,
                    "sharpe": best.sharpe_ratio,
                    "max_drawdown": best.max_drawdown,
                    "win_rate": best.win_rate,
                    "trades": best.trade_count,
                    "metric_score": best.sharpe_ratio if metric == "sharpe" else best.return_pct,
                    "error": None
                }
                batch_results.append(entry)
                
            except Exception as e:
                # Record error for this symbol
                batch_results.append({
                    "symbol": symbol,
                    "short_window": 0,
                    "long_window": 0,
                    "total_return": 0.0,
                    "sharpe": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "trades": 0,
                    "metric_score": -999.0,
                    "error": str(e)
                })
        
        # Sort batch results by metric score
        batch_results.sort(key=lambda x: x["metric_score"], reverse=True)
        
        # Add rank
        for i, res in enumerate(batch_results):
            res["rank"] = i + 1
            
        return batch_results
