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
    score: float = 0.0
    
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
                "trade_count": self.trade_count,
                "score": self.score
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
        strategy_type: str = "ma_cross",
        fixed_params: Dict[str, Any] = None
    ) -> List[OptimizationResult]:
        """
        Run grid search optimization for a single symbol
        """
        from backend.backtester import BacktestEngine
        from backend import data_feed
        from backend.strategies import (
            MACrossStrategy, EMACrossStrategy, MACDTrendStrategy,
            RSIMeanReversionStrategy, StochasticOscillatorStrategy,
            BollingerMeanReversionStrategy, BollingerBreakoutStrategy,
            DonchianBreakoutStrategy, ATRTrailingMAStrategy, ROCMomentumStrategy,
            EMA9DipBuyStrategy
        )
        
        # Strategy Factory Map
        STRATEGY_CLASSES = {
            "ma_cross": MACrossStrategy,
            "ema_cross": EMACrossStrategy,
            "macd_trend": MACDTrendStrategy,
            "rsi_mean_reversion": RSIMeanReversionStrategy,
            "stoch_oscillator": StochasticOscillatorStrategy,
            "bollinger_mean_reversion": BollingerMeanReversionStrategy,
            "bollinger_breakout": BollingerBreakoutStrategy,
            "donchian_breakout": DonchianBreakoutStrategy,
            "atr_trailing_ma": ATRTrailingMAStrategy,
            "roc_momentum": ROCMomentumStrategy,
            "ema9_dip_buy": EMA9DipBuyStrategy,
        }
        
        if strategy_type not in STRATEGY_CLASSES:
             raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        strategy_cls = STRATEGY_CLASSES[strategy_type]
        fixed_params = fixed_params or {}

        # Get data
        fetch_start_date = start_date
        backtest_start_dt = None
        
        if start_date:
            # Calculate buffer for warm-up (e.g. 365 days)
            # This matches the logic in /simulate endpoint
            try:
                start_dt = pd.Timestamp(start_date)
                buffer_dt = start_dt - pd.Timedelta(days=365)
                fetch_start_date = buffer_dt.strftime("%Y-%m-%d")
                
                # Set backtest start date to original requested date
                # Ensure it's timezone-aware (UTC) if needed, but BacktestEngine handles naive/aware comparison
                # We'll pass the timestamp directly
                backtest_start_dt = start_dt
            except Exception:
                # Fallback if date parsing fails
                pass

        candles = data_feed.get_chart_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=3000,
            start=fetch_start_date,
            end=end_date
        )
        
        if not candles:
            raise ValueError(f"No data found for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        if "time" not in df.columns:
            raise ValueError("Data must contain 'time' column")
        
        # Set datetime index and drop original time column to avoid conflict
        df.index = pd.to_datetime(df["time"], unit="s")
        df = df.drop(columns=["time"])
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        results = []
        
        for combo in combinations:
            params = dict(zip(param_names, combo))
            
            # Merge with fixed params
            full_params = {**fixed_params, **params}
            
            # Basic validation for window params (specific to MA strategies, but harmless for others if keys don't exist)
            if "short_window" in full_params and "long_window" in full_params:
                if full_params["short_window"] >= full_params["long_window"]:
                    continue
            
            # Create strategy instance and run backtest
            try:
                # Generic Backtest Execution
                strategy = strategy_cls(**full_params)
                
                engine = BacktestEngine(
                    initial_capital=initial_capital,
                    commission_rate=commission_rate,
                    position_size=position_size
                )
                
                # Run Backtest
                result = engine.run_backtest(df, strategy, start_date=backtest_start_dt)
                
                stats = result["stats"]
                
                # Compute Score (Objective Function)
                # Score = Sharpe * 2.0 + Return * 0.5
                # Penalties: Trade Count < 5 => -1e9, Max DD > 70 => -1e9
                
                trade_count = stats["trade_count"]
                max_dd = stats["max_drawdown"] * 100 # Convert to percentage
                sharpe = stats["sharpe_ratio"]
                ret_pct = stats["return_pct"]
                
                if trade_count < 5:
                    score = -1e9
                elif max_dd > 70.0:
                    score = -1e9
                else:
                    score = (sharpe * 2.0) + (ret_pct * 0.5)
                
                opt_res = OptimizationResult(
                    params=params,
                    total_pnl=stats["total_pnl"],
                    return_pct=stats["return_pct"],
                    sharpe_ratio=stats["sharpe_ratio"],
                    max_drawdown=stats["max_drawdown"],
                    win_rate=stats["win_rate"],
                    trade_count=stats["trade_count"],
                    score=score
                )
                
                results.append(opt_res)
                
            except Exception as e:
                # Silently skip failed combinations
                continue
                
        # Sort by Score (Descending)
        results.sort(key=lambda x: x.score, reverse=True)
        
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
