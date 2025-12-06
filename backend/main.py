"""
FastAPI Backend for AI Signal Chart Backtest System
"""

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel
import pandas as pd
from datetime import datetime, timedelta

from backend import data_feed
from backend.backtester import BacktestEngine
from backend.strategies.ma_cross import MACrossStrategy
from backend.strategies.ema_cross import EMACrossStrategy
from backend.strategies.macd_trend import MACDTrendStrategy
from backend.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from backend.strategies.stoch_oscillator import StochasticOscillatorStrategy
from backend.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from backend.strategies.bollinger_breakout import BollingerBreakoutStrategy
from backend.strategies.donchian_breakout import DonchianBreakoutStrategy
from backend.strategies.atr_trailing_ma import ATRTrailingMAStrategy
from backend.strategies.roc_momentum import ROCMomentumStrategy
from backend.strategies.ema9_dip_buy import EMA9DipBuyStrategy

from backend.models.backtest import (
    BacktestRequest,
    BacktestResponse,
    BacktestStats,
    EquityCurvePoint,
    TradeSummary,
)


from backend.models.optimization import (
    OptimizationRequest, OptimizationResponse, MACrossOptimizationRequest, GenericOptimizationRequest
)
from backend.models.strategy_lab import (
    StrategyLabBatchRequest,
    StrategyLabBatchResponse,
    StrategyLabSymbolResult
)
from backend.strategy.models import JsonStrategyRunRequest

from backend.models.paper_trade import (
    PaperAccount,
    PaperPosition,
    PaperTradeLog,
    InMemoryPaperStore,
)

# Global In-Memory Store for Paper Trading
paper_store = InMemoryPaperStore()


app = FastAPI(title="AI Signal Chart Backtest API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ローカル開発中なので一旦ゆるく
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict:
    """
    Health check endpoint
    """
    return {"status": "ok", "version": "0.1.0"}


@app.get("/chart-data")
async def get_chart_data_endpoint(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1d",
    with_ma: bool = True,
    short_window: int = 9,
    long_window: int = 21,
    limit: int = 500
) -> Dict[str, Any]:
    """
    チャート表示用データ + MA計算
    
    Args:
        symbol: 銘柄コード (AAPL, 7203.T, BTC/USDT etc.)
        start: 開始日 (YYYY-MM-DD format, optional)
        end: 終了日 (YYYY-MM-DD format, optional)
        interval: 足種別 (1m, 5m, 1h, 1d etc.)
        with_ma: MAを含めるか
        short_window: 短期MA窓
        long_window: 長期MA窓
        limit: 最大データポイント数
    
    Returns:
        チャートデータ（MAを含む）
    """
    try:
        from backend.utils.indicators import simple_moving_average
        
        # 1. データ取得
        candles = data_feed.get_chart_data(
            symbol=symbol,
            timeframe=interval,
            limit=limit,
            start=start,
            end=end
        )
        
        if not candles:
            raise HTTPException(status_code=404, detail="No data found for symbol")
        
        # 2. MA計算（with_ma=Trueの場合）
        if with_ma and candles:
            closes = [c["close"] for c in candles]
            ma_short = simple_moving_average(closes, short_window)
            ma_long = simple_moving_average(closes, long_window)
            
            # 各キャンドルにMAを追加
            for i, candle in enumerate(candles):
                candle["ma_short"] = ma_short[i]
                candle["ma_long"] = ma_long[i]
        
        return {
            "symbol": symbol,
            "interval": interval,
            "data": candles,
            "count": len(candles)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data: {str(e)}")


@app.get("/indicators")
async def get_indicators(
    symbol: str,
    interval: str = "1d",
    indicators: str = "sma,rsi",
    sma_short: int = 9,
    sma_long: int = 21,
    rsi_period: int = 14,
    limit: int = 500
) -> Dict[str, Any]:
    """
    テクニカル指標計算エンドポイント
    
    Args:
        symbol: 銘柄コード
        interval: 足種別
        indicators: 計算する指標（カンマ区切り: sma,rsi）
        sma_short: 短期SMA窓
        sma_long: 長期SMA窓
        rsi_period: RSI期間
        limit: データポイント数
    
    Returns:
        計算された指標データ
    """
    try:
        from backend.utils.indicators import simple_moving_average, rsi as calc_rsi
        
        # データ取得
        candles = data_feed.get_chart_data(symbol, interval, limit=limit)
        
        if not candles:
            raise HTTPException(status_code=404, detail="No data found for symbol")
        
        closes = [c["close"] for c in candles]
        
        result = {
            "symbol": symbol,
            "interval": interval,
            "indicators": {},
            "data_points": len(closes)
        }
        
        requested = [i.strip().lower() for i in indicators.split(",")]
        
        if "sma" in requested:
            result["indicators"]["sma_short"] = simple_moving_average(closes, sma_short)
            result["indicators"]["sma_long"] = simple_moving_average(closes, sma_long)
        
        if "rsi" in requested:
            result["indicators"]["rsi"] = calc_rsi(closes, rsi_period)
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating indicators: {str(e)}")


@app.post("/simulate", response_model=BacktestResponse)
async def run_simulation(request: BacktestRequest):
    """
    Run backtest simulation.
    """
    try:
        # 1. Get Data
        candles = data_feed.get_historical_candles(
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=5000 # Sufficient limit for backtest
        )
        
        if not candles:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
            
        df = pd.DataFrame(candles)
        if df.empty:
             raise HTTPException(status_code=404, detail="Empty data returned")
             
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        # Filter by Date Range if provided
        start_ts = None
        if request.start_date:
            # Robust approach: Convert index to UTC and compare
            if df.index.tz is None:
                 df.index = df.index.tz_localize('UTC')
            
            # Parse request dates as UTC
            start_ts = pd.Timestamp(request.start_date).tz_convert('UTC') if pd.Timestamp(request.start_date).tzinfo else pd.Timestamp(request.start_date).tz_localize('UTC')
            
            # Calculate buffer date (e.g., 365 days before start_date) for warm-up
            buffer_date = start_ts - pd.Timedelta(days=365)
            df = df[df.index >= buffer_date]

        if request.end_date:
            if df.index.tz is None:
                 df.index = df.index.tz_localize('UTC')
            
            end_ts = pd.Timestamp(request.end_date).tz_convert('UTC') if pd.Timestamp(request.end_date).tzinfo else pd.Timestamp(request.end_date).tz_localize('UTC')
            df = df[df.index <= end_ts]

        if df.empty:
             raise HTTPException(status_code=400, detail="No data found for the specified date range")
        params = request.params or {}
        
        # Backward compatibility for ma_cross specific fields
        if request.strategy == "ma_cross" and not params:
            params = {
                "short_window": request.short_window,
                "long_window": request.long_window
            }

        if request.strategy == "ma_cross":
            strategy = MACrossStrategy(**params)
        elif request.strategy == "ema_cross":
            strategy = EMACrossStrategy(**params)
        elif request.strategy == "macd_trend":
            strategy = MACDTrendStrategy(**params)
        elif request.strategy == "rsi_mean_reversion":
            strategy = RSIMeanReversionStrategy(**params)
        elif request.strategy == "stoch_oscillator":
            strategy = StochasticOscillatorStrategy(**params)
        elif request.strategy == "bollinger_mean_reversion":
            strategy = BollingerMeanReversionStrategy(**params)
        elif request.strategy == "bollinger_breakout":
            strategy = BollingerBreakoutStrategy(**params)
        elif request.strategy == "donchian_breakout":
            strategy = DonchianBreakoutStrategy(**params)
        elif request.strategy == "atr_trailing_ma":
            strategy = ATRTrailingMAStrategy(**params)
        elif request.strategy == "roc_momentum":
            strategy = ROCMomentumStrategy(**params)
        elif request.strategy == "ema9_dip_buy":
            strategy = EMA9DipBuyStrategy(**params)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")

        # 3. Run Backtest
        
        # Use unified runner for MA Cross to ensure consistency with optimizer
        if request.strategy == "ma_cross":
            from backend.strategies.runner import run_ma_cross_backtest
            
            # Use params if available, else request fields
            short_window = params.get("short_window", request.short_window)
            long_window = params.get("long_window", request.long_window)
            
            result = run_ma_cross_backtest(
                df=df,
                short_window=short_window,
                long_window=long_window,
                initial_capital=request.initial_capital,
                commission_rate=request.commission_rate,
                start_date=start_ts
            )
        else:
            # Legacy path for other strategies
            engine = BacktestEngine(
                initial_capital=request.initial_capital,
                commission_rate=request.commission_rate,
                position_size=request.position_size
            )
            
            # Pass start_ts to run_backtest to skip trading during warm-up
            result = engine.run_backtest(df, strategy, start_date=start_ts)
        
        # 4. Format Response
        # Construct price_series for visualization
        price_series = []
        
        # Calculate MAs if strategy is ma_cross for visualization
        # Note: This duplicates calculation but ensures we have data for the chart
        # ideally the strategy should return indicators, but for now we re-calc or just pass close
        
        # Create a copy to avoid modifying original df if needed, though we are just reading
        viz_df = df.copy()
        
        if request.strategy == "ma_cross":
            # Re-calculate MAs for visualization
            # Use params if available, else request fields
            short_window = params.get("short_window", request.short_window)
            long_window = params.get("long_window", request.long_window)
            
            viz_df['ma_short'] = viz_df['close'].rolling(window=short_window).mean()
            viz_df['ma_long'] = viz_df['close'].rolling(window=long_window).mean()
        
        # Convert to list of dicts
        viz_df = viz_df.reset_index() # make timestamp a column
        
        # Filter viz_df to match the backtest period (start_ts to end_ts)
        # start_ts and end_ts are calculated earlier in the function
        if start_ts:
             viz_df = viz_df[viz_df["timestamp"] >= start_ts]
        if 'end_ts' in locals() and end_ts:
             viz_df = viz_df[viz_df["timestamp"] <= end_ts]
        
        for _, row in viz_df.iterrows():
            item = {
                "date": row["timestamp"].isoformat(),
                "close": row["close"]
            }
            if "ma_short" in row and pd.notna(row["ma_short"]):
                item["ma_short"] = row["ma_short"]
            if "ma_long" in row and pd.notna(row["ma_long"]):
                item["ma_long"] = row["ma_long"]
            price_series.append(item)

        return BacktestResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=request.strategy,
            equity_curve=result["equity_curve"],
            trades=result["trades"],
            metrics=result["stats"],
            data_points=len(df),
            price_series=price_series
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))





@app.post("/optimize/ma_cross", response_model=OptimizationResponse)
async def optimize_ma_cross(request: MACrossOptimizationRequest) -> OptimizationResponse:
    """
    MA Cross 戦略専用のグリッドサーチ最適化エンドポイント
    """
    try:
        from backend.optimizer import GridSearchOptimizer
        
        # 組み合わせ数の概算チェック
        short_count = (request.short_max - request.short_min) // request.short_step + 1
        long_count = (request.long_max - request.long_min) // request.long_step + 1
        total_combinations = short_count * long_count
        
        if total_combinations > 400:
             raise HTTPException(
                status_code=422, 
                detail=f"Too many combinations: {total_combinations}. Limit is 400."
            )

        optimizer = GridSearchOptimizer()
        
        # Build parameter grid
        param_grid = {
            "short_window": list(range(request.short_min, request.short_max + 1, request.short_step)),
            "long_window": list(range(request.long_min, request.long_max + 1, request.long_step))
        }
        
        # Run optimization
        results = optimizer.optimize(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            param_grid=param_grid,
            initial_capital=request.initial_capital,
            commission_rate=request.commission_rate,
            position_size=1.0,
            strategy_type="ma_cross"
        )
        
        return OptimizationResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy_type="ma_cross",
            total_combinations=len(results),
            top_results=[
                {
                    "rank": i + 1,
                    **r.to_dict()
                }
                for i, r in enumerate(results)
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")





@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_parameters(request: OptimizationRequest) -> OptimizationResponse:
    """
    パラメータ自動最適化エンドポイント
    
    Args:
        request: Optimization request with parameter ranges
    
    Returns:
        Top N optimization results sorted by total_pnl
    """
    try:
        from backend.optimizer import GridSearchOptimizer
        
        optimizer = GridSearchOptimizer()
        
        # Build parameter grid based on strategy type
        param_grid = {}
        
        if request.strategy_type in ["ma_cross", "ma_rsi_combo"]:
            param_grid["short_window"] = list(range(
                request.short_window_min,
                request.short_window_max + 1,
                request.short_window_step
            ))
            param_grid["long_window"] = list(range(
                request.long_window_min,
                request.long_window_max + 1,
                request.long_window_step
            ))
        
        if request.strategy_type in ["rsi", "ma_rsi_combo"]:
            if request.rsi_period_min is not None and request.rsi_period_max is not None:
                param_grid["rsi_period"] = list(range(
                    request.rsi_period_min,
                    request.rsi_period_max + 1,
                    request.rsi_period_step or 1
                ))
            
            if request.rsi_oversold_min is not None and request.rsi_oversold_max is not None:
                param_grid["rsi_oversold"] = [
                    float(x) for x in range(
                        int(request.rsi_oversold_min),
                        int(request.rsi_oversold_max) + 1,
                        int(request.rsi_oversold_step or 5)
                    )
                ]
            
            if request.rsi_overbought_min is not None and request.rsi_overbought_max is not None:
                param_grid["rsi_overbought"] = [
                    float(x) for x in range(
                        int(request.rsi_overbought_min),
                        int(request.rsi_overbought_max) + 1,
                        int(request.rsi_overbought_step or 5)
                    )
                ]
        
        # Run optimization
        results = optimizer.optimize(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            param_grid=param_grid,
            initial_capital=request.initial_capital,
            commission_rate=request.commission_rate,
            position_size=request.position_size,
            strategy_type=request.strategy_type
        )
        
        # Get top N results
        top_results = results[:request.top_n]
        
        return OptimizationResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy_type=request.strategy_type,
            total_combinations=len(results),
            top_results=[
                {
                    "rank": i + 1,
                    **r.to_dict()
                }
                for i, r in enumerate(top_results)
            ]
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization failed: {str(e)}"
        )


@app.post("/strategy-lab/run-batch", response_model=StrategyLabBatchResponse)
async def run_strategy_lab_batch(request: StrategyLabBatchRequest) -> StrategyLabBatchResponse:
    """
    Strategy Lab: Run batch optimization for multiple symbols
    """
    try:
        from backend.optimizer import GridSearchOptimizer
        
        optimizer = GridSearchOptimizer()
        
        # Build parameter grid
        param_grid = {}
        
        if request.strategy_type == "ma_cross":
            param_grid["short_window"] = list(range(
                request.short_ma_min,
                request.short_ma_max + 1,
                request.short_ma_step
            ))
            param_grid["long_window"] = list(range(
                request.long_ma_min,
                request.long_ma_max + 1,
                request.long_ma_step
            ))
        
        # Run batch optimization
        results = optimizer.optimize_batch(
            symbols=request.symbols,
            timeframe=request.timeframe,
            param_grid=param_grid,
            metric=request.metric,
            initial_capital=request.initial_capital,
            commission_rate=request.commission_rate,
            position_size=request.position_size,
            strategy_type=request.strategy_type
        )
        
        # Convert to response model
        symbol_results = []
        for res in results:
            symbol_results.append(StrategyLabSymbolResult(
                symbol=res["symbol"],
                short_window=res["short_window"],
                long_window=res["long_window"],
                total_return=res["total_return"],
                sharpe=res["sharpe"],
                max_drawdown=res["max_drawdown"],
                win_rate=res["win_rate"],
                trades=res["trades"],
                metric_score=res["metric_score"],
                rank=res["rank"],
                error=res["error"]
            ))
            
        return StrategyLabBatchResponse(
            study_name=request.study_name,
            metric=request.metric,
            results=symbol_results
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Strategy Lab batch run failed: {str(e)}"
        )



@app.post("/strategy/run-json", response_model=BacktestResponse)
async def run_json_strategy(request: JsonStrategyRunRequest) -> BacktestResponse:
    """
    Run a strategy defined in JSON format
    """
    try:
        from backend.strategy.engine import JsonStrategyEngine
        from backend import data_feed
        
        # 1. Get Data
        candles = data_feed.get_chart_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=3000,
            start=request.start_date.isoformat() if request.start_date else None,
            end=request.end_date.isoformat() if request.end_date else None
        )
        
        if not candles:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
            
        df = pd.DataFrame(candles)
        if "time" not in df.columns:
            raise ValueError("Data must contain 'time' column")
        df = df.set_index(pd.to_datetime(df["time"], unit="s"))
        
        # 2. Run Strategy Engine
        engine = JsonStrategyEngine()
        result = engine.run(
            df=df,
            strategy=request.strategy,
            initial_capital=request.initial_capital,
            commission_rate=request.commission_rate,
            position_size=request.position_size
        )
        
        # 3. Format Response (Reuse BacktestResponse)
        metrics = result.get("stats", {})
        equity_curve = result.get("equity_curve", [])
        trades = result.get("trades", [])
        
        return BacktestResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=request.strategy.name,
            metrics=metrics,
            equity_curve=equity_curve,
            trades=trades
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Strategy execution failed: {str(e)}")


@app.post("/experiments", status_code=201)
async def create_experiment_endpoint(experiment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create new experiment
    
    Args:
        experiment: Experiment data including name, strategy, parameters, and results
    
    Returns:
        Created experiment with ID
    """
    try:
        from backend.experiments_manager import ExperimentsManager, ExperimentCreate
        
        manager = ExperimentsManager()
        exp_create = ExperimentCreate(**experiment)
        created_exp = manager.create_experiment(exp_create)
        
        return created_exp.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create experiment: {str(e)}")


@app.get("/experiments")
async def list_experiments_endpoint(limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    """
    List all experiments
    
    Args:
        limit: Maximum number of experiments to return
        offset: Number of experiments to skip
    
    Returns:
        List of experiments with summary
    """
    try:
        from backend.experiments_manager import ExperimentsManager
        
        manager = ExperimentsManager()
        experiments = manager.list_experiments(limit=limit, offset=offset)
        summary = manager.get_experiments_summary()
        
        return {
            "experiments": [exp.model_dump() for exp in experiments],
            "summary": summary,
            "count": len(experiments)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list experiments: {str(e)}")


@app.get("/experiments/{experiment_id}")
async def get_experiment_endpoint(experiment_id: str) -> Dict[str, Any]:
    """
    Get experiment by ID
    
    Args:
        experiment_id: Experiment ID
    
    Returns:
        Experiment data
    """
    try:
        from backend.experiments_manager import ExperimentsManager
        
        manager = ExperimentsManager()
        experiment = manager.get_experiment(experiment_id)
        
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return experiment.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get experiment: {str(e)}")


@app.put("/experiments/{experiment_id}")
async def update_experiment_endpoint(experiment_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update experiment
    
    Args:
        experiment_id: Experiment ID
        updates: Fields to update
    
    Returns:
        Updated experiment
    """
    try:
        from backend.experiments_manager import ExperimentsManager
        
        manager = ExperimentsManager()
        updated_exp = manager.update_experiment(experiment_id, updates)
        
        if not updated_exp:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return updated_exp.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update experiment: {str(e)}")


@app.get("/live/signal")
async def get_live_signal(
    symbol: str,
    timeframe: str = "1d",
    lookback: int = 200,
    limit_signals: int = 10
) -> Dict[str, Any]:
    """
    Get live trading signal for EXITON v1.
    
    Args:
        symbol: Ticker symbol (e.g. "AAPL")
        timeframe: Candle timeframe (e.g. "1d")
        lookback: Number of candles to analyze
        limit_signals: Max number of history signals to return
        
    Returns:
        EXITON v1 Signal Response
    """
    try:
        from backend.live.signal_generator import generate_live_signal
        
        result = generate_live_signal(symbol, timeframe, lookback)
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Live signal generation failed: {str(e)}")



@app.delete("/experiments/{experiment_id}")
async def delete_experiment_endpoint(experiment_id: str) -> Dict[str, str]:
    """
    Delete experiment
    
    Args:
        experiment_id: Experiment ID
    
    Returns:
        Success message
    """
    try:
        from backend.experiments_manager import ExperimentsManager
        
        manager = ExperimentsManager()
        success = manager.delete_experiment(experiment_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return {"message": "Experiment deleted successfully", "id": experiment_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete experiment: {str(e)}")



@app.get("/strategies")
async def list_strategies_endpoint() -> List[Dict[str, str]]:
    """
    List available strategies with metadata
    """
    from backend.strategies.registry import list_strategies
    return [
        {
            "id": s.id,
            "name": s.name,
            "description": s.description
        }
        for s in list_strategies()
    ]


@app.get("/strategies/{strategy_id}/doc")
async def get_strategy_doc_endpoint(strategy_id: str) -> Dict[str, Any]:
    """
    Get strategy documentation (markdown) and presets
    """
    from backend.strategies.registry import get_strategy_metadata
    from backend.strategies.docs import parse_strategy_doc
    import os
    
    metadata = get_strategy_metadata(strategy_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Strategy not found")
        
    try:
        # Resolve path relative to project root
        doc_path = metadata.docs_path
        
        markdown_content, presets = parse_strategy_doc(doc_path)
        
        return {
            "id": strategy_id, 
            "markdown": markdown_content,
            "presets": presets
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load strategy doc: {str(e)}")


# Live Strategy Endpoints

@app.post("/live-strategy")
async def set_live_strategy_endpoint(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Set the live trading strategy configuration.
    Accepts generic dict and validates via Pydantic model.
    """
    try:
        from backend.models.live_strategy import LiveStrategyConfig, save_live_strategy
        
        # Validate and convert
        strategy_config = LiveStrategyConfig(**config)
        save_live_strategy(strategy_config)
        
        return {"status": "ok", "message": "Live strategy saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/live-strategy")
async def get_live_strategy_endpoint() -> Dict[str, Any]:
    """
    Get the current live trading strategy configuration.
    """
    try:
        from backend.models.live_strategy import load_live_strategy
        
        config = load_live_strategy()
        return config.model_dump()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Live strategy not set.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/live-signal")
async def get_live_signal_endpoint() -> Dict[str, Any]:
    """
    Generate a daily trading signal based on the saved live strategy.
    """
    try:
        from backend.models.live_strategy import load_live_strategy
        import pandas as pd
        import numpy as np
        
        # 1. Load Live Strategy
        try:
            config = load_live_strategy()
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Live strategy not set.")

        symbol = config.symbol
        timeframe = config.timeframe
        
        # 2. Fetch Historical Data
        # We need enough data for indicators. 
        # MA Cross: long_window + buffer
        # RSI: rsi_period + buffer
        # Defaulting to 365 days to be safe and consistent with backtest
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        # Fetch data using get_chart_data which supports start date
        candles = data_feed.get_chart_data(symbol, timeframe, limit=3000, start=start_date)
        if not candles:
             raise HTTPException(status_code=400, detail=f"No data found for {symbol}")
        
        df = pd.DataFrame(candles)
        # Convert unix time to datetime
        if "time" in df.columns:
            df["date"] = pd.to_datetime(df["time"], unit="s")
            
        # Ensure sorted
        df = df.sort_values("date").reset_index(drop=True)
        
        # 3. Compute Indicators & Signal
        signal = "HOLD"
        params = config.params
        
        if config.strategy_type == "ma_cross":
            short_window = int(params.get("short_window", 9))
            long_window = int(params.get("long_window", 21))
            
            if len(df) < long_window + 2:
                 raise HTTPException(status_code=400, detail="Not enough data for MA calculation")
            
            df["short_ma"] = df["close"].rolling(window=short_window).mean()
            df["long_ma"] = df["close"].rolling(window=long_window).mean()
            
            # Check crossover
            # prev: -2, curr: -1
            prev_short = df["short_ma"].iloc[-2]
            prev_long = df["long_ma"].iloc[-2]
            curr_short = df["short_ma"].iloc[-1]
            curr_long = df["long_ma"].iloc[-1]
            
            prev_diff = prev_short - prev_long
            curr_diff = curr_short - curr_long
            
            if prev_diff <= 0 and curr_diff > 0:
                signal = "BUY"
            elif prev_diff >= 0 and curr_diff < 0:
                signal = "SELL"
                
        elif config.strategy_type == "rsi_mean_reversion":
            rsi_period = int(params.get("rsi_period", 14))
            oversold = int(params.get("oversold_level", 30))
            overbought = int(params.get("overbought_level", 70))
            
            if len(df) < rsi_period + 2:
                 raise HTTPException(status_code=400, detail="Not enough data for RSI calculation")
            
            # Simple RSI calculation
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))
            
            # Check thresholds
            prev_rsi = df["rsi"].iloc[-2]
            curr_rsi = df["rsi"].iloc[-1]
            
            if prev_rsi <= oversold and curr_rsi > oversold:
                signal = "BUY"
            elif prev_rsi >= overbought and curr_rsi < overbought:
                signal = "SELL"
        
        # 4. Calculate Suggested Shares
        latest_price = float(df["close"].iloc[-1])
        mode = config.risk.position_mode
        value = config.risk.position_value
        suggested_shares = 0
        
        if mode == "fixed_shares":
            suggested_shares = int(value)
        elif mode == "fixed_amount_jpy":
            if latest_price > 0:
                suggested_shares = max(int(value // latest_price), 0)
        
        # 5. Return Response
        return {
            "symbol": config.symbol,
            "timeframe": config.timeframe,
            "strategy_type": config.strategy_type,
            "strategy_name": config.strategy_name,
            "signal": signal,
            "latest_price": latest_price,
            "timestamp": df["date"].iloc[-1].isoformat(),
            "params": config.params,
            "risk": {
                "position_mode": config.risk.position_mode,
                "position_value": config.risk.position_value,
                "suggested_shares": suggested_shares,
            },
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rule_predictor_v2")
async def get_rule_predictor_v2(symbol: str) -> Dict[str, Any]:
    """
    Rule Predictor v2: 5 Indicators + Weighted Vote.
    """
    try:
        from backend.rule_predictor_v2 import predict
        from backend import data_feed
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Fetch Data (need enough for 30MA + buffer)
        start_date = (datetime.now() - timedelta(days=200)).strftime("%Y-%m-%d")
        candles = data_feed.get_chart_data(symbol, "1d", limit=300, start=start_date)
        
        if not candles:
             raise HTTPException(status_code=400, detail=f"No data found for {symbol}")
        
        df = pd.DataFrame(candles)
        if "time" in df.columns:
            df["date"] = pd.to_datetime(df["time"], unit="s")
        
        # Run Prediction
        result = predict(df)
        
        return {
            "symbol": symbol,
            "timestamp": df["date"].iloc[-1].isoformat(),
            **result
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Rule Predictor v2 failed: {str(e)}")


@app.get("/predictor_backtest")
async def get_predictor_backtest(
    symbol: str,
    timeframe: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run backtest for all predictors (Stat, Rule v2, Buy & Hold).
    
    Args:
        symbol: Stock symbol
        timeframe: Timeframe (1d, 4h, etc.)
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        
    Returns:
        Comparison results for all predictors
    """
    try:
        from backend.predictor_eval import run_all_predictors_backtest
        
        result = run_all_predictors_backtest(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start,
            end_date=end
        )
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Predictor backtest failed: {str(e)}")


# =============================================================================
# Symbol Universes API
# =============================================================================

@app.get("/symbol-universes")
async def get_symbol_universes() -> Dict[str, Any]:
    """
    Get all available symbol universes.
    
    Returns:
        Dict of universe_id -> {label, description, symbols}
    """
    from backend.config.symbol_universes import get_all_universes
    return get_all_universes()


@app.get("/symbol-universes/{universe_id}")
async def get_symbol_universe(universe_id: str) -> Dict[str, Any]:
    """
    Get a specific symbol universe by ID.
    
    Args:
        universe_id: Universe identifier (e.g., "sp500_all", "mega_caps")
        
    Returns:
        Universe details including label, description, and symbols
    """
    from backend.config.symbol_universes import get_universe
    
    universe = get_universe(universe_id)
    if universe is None:
        raise HTTPException(status_code=404, detail=f"Universe '{universe_id}' not found")
    
    return universe


@app.get("/symbol_preset")
async def get_symbol_preset(symbol: str) -> Dict[str, Any]:
    """
    Get strategy preset for a symbol.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Preset configuration for the symbol
    """
    try:
        from backend.config.presets import load_symbol_preset
        
        preset = load_symbol_preset(symbol)
        return {
            "symbol": symbol,
            "preset": preset
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load preset: {str(e)}")


@app.post("/symbol_preset")
async def save_symbol_preset_endpoint(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save or update a strategy preset for a symbol.
    
    Body:
        { "symbol": "NVDA", "preset": { ... } }
        
    Returns:
        Success status
    """
    try:
        from backend.config.presets import save_symbol_preset
        
        symbol = data.get("symbol")
        preset = data.get("preset")
        
        if not symbol or not preset:
            raise HTTPException(status_code=400, detail="Missing symbol or preset")
        
        success = save_symbol_preset(symbol, preset)
        
        if success:
            return {"status": "success", "symbol": symbol, "message": f"Preset saved for {symbol}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save preset")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save preset: {str(e)}")


@app.post("/optimize/generic", response_model=OptimizationResponse)
async def optimize_generic_endpoint(request: GenericOptimizationRequest) -> OptimizationResponse:
    """
    Generic parameter optimization endpoint.
    """
    try:
        from backend.optimizer import GridSearchOptimizer
        
        optimizer = GridSearchOptimizer()
        
        results = optimizer.optimize(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            param_grid=request.param_grid,
            initial_capital=request.initial_capital,
            commission_rate=request.commission_rate,
            strategy_type=request.strategy_type,
            fixed_params=request.fixed_params
        )
        
        # Convert results to dict
        normalized_results = [r.to_dict() for r in results]
        
        return OptimizationResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy_type=request.strategy_type,
            total_combinations=len(results),
            results=normalized_results
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.get("/scan_market")
async def scan_market_endpoint(
    universe: str = "default",
    timeframe: str = "1d",
    lookback: int = 200,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Scan the market for trading opportunities.
    
    Args:
        universe: "default", "sp500", "mvp"
        timeframe: "1d"
        lookback: 200
        limit: Max results to return
        
    Returns:
        List of symbols ranked by bullish score
    """
    try:
        from backend.market_scanner import scan_market
        
        result = scan_market(
            universe_name=universe,
            timeframe=timeframe,
            lookback=lookback,
            limit=limit
        )
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Market scan failed: {str(e)}")



# ============================================================================
# Paper Trading API
# ============================================================================

class CreatePaperAccountRequest(BaseModel):
    account_id: str = "default"
    initial_equity: float
    base_currency: str = "USD"


class OpenPaperPositionRequest(BaseModel):
    account_id: str = "default"
    symbol: str
    direction: Literal["LONG", "SHORT"]
    size: float
    entry_price: float
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    opened_at: Optional[datetime] = None
    tags: List[str] = []


class ClosePaperPositionRequest(BaseModel):
    exit_price: float
    closed_at: Optional[datetime] = None


paper_router = APIRouter(prefix="/paper", tags=["paper_trading"])


@paper_router.post("/accounts", response_model=PaperAccount)
def create_paper_account(req: CreatePaperAccountRequest):
    now = datetime.utcnow()

    account = PaperAccount(
        account_id=req.account_id,
        base_currency=req.base_currency,
        initial_equity=req.initial_equity,
        equity=req.initial_equity,
        cash=req.initial_equity,
        open_risk=0.0,
        created_at=now,
        updated_at=now,
    )
    paper_store.upsert_account(account)
    return account


@paper_router.get("/accounts/{account_id}", response_model=PaperAccount)
def get_paper_account(account_id: str):
    account = paper_store.get_account(account_id)
    if account is None:
        raise HTTPException(status_code=404, detail="Paper account not found")
    return account


@paper_router.post("/positions", response_model=PaperPosition)
def open_paper_position(req: OpenPaperPositionRequest):
    account = paper_store.get_account(req.account_id)
    if account is None:
        raise HTTPException(status_code=404, detail="Paper account not found")

    opened_at = req.opened_at or datetime.utcnow()

    # Create position_id (simple scheme; can be improved later)
    position_id = f"{req.account_id}-{req.symbol}-{int(opened_at.timestamp())}"

    # Calculate R-risk amount based on account equity and risk percentage
    r_risk_amount = account.calculate_r_risk()

    position = PaperPosition(
        position_id=position_id,
        account_id=req.account_id,
        symbol=req.symbol,
        direction=req.direction,
        size=req.size,
        entry_price=req.entry_price,
        stop_price=req.stop_price,
        target_price=req.target_price,
        opened_at=opened_at,
        closed_at=None,
        status="OPEN",
        tags=req.tags,
        r_risk_amount=r_risk_amount,  # Store the calculated R-risk
    )

    # Store position
    try:
        paper_store.add_position(position)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Update account open_risk
    account.open_risk += r_risk_amount
    account.updated_at = datetime.utcnow()
    paper_store.upsert_account(account)

    return position


@paper_router.get("/positions/open", response_model=List[PaperPosition])
def get_open_positions(account_id: str = "default"):
    return paper_store.get_open_positions(account_id)


@paper_router.post("/positions/{position_id}/close", response_model=PaperTradeLog)
def close_paper_position(position_id: str, req: ClosePaperPositionRequest):
    position = paper_store.positions.get(position_id)
    if position is None:
        raise HTTPException(status_code=404, detail="Position not found")

    closed_at = req.closed_at or datetime.utcnow()

    try:
        trade_log = paper_store.close_position(
            position_id=position_id,
            exit_price=req.exit_price,
            closed_at=closed_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Update account risk
    account = paper_store.get_account(trade_log.account_id)
    if account:
        # Subtract old risk (we don't recalc exactly; simple approach)
        risk_amount = position.r_risk()
        if risk_amount > 0:
            account.open_risk = max(0.0, account.open_risk - risk_amount)
        # Update equity by realized PnL
        account.equity += trade_log.net_pnl
        account.cash += trade_log.net_pnl
        account.updated_at = datetime.utcnow()
        paper_store.upsert_account(account)

    return trade_log


@paper_router.delete("/positions/{position_id}", status_code=204)
def delete_paper_position(position_id: str):
    """
    Delete a position (cancel). Only allowed if it exists.
    """
    deleted = paper_store.delete_position(position_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Position not found")
    return None


@paper_router.get("/trades", response_model=List[PaperTradeLog])
def list_trades(account_id: str = "default"):
    return paper_store.get_trades_by_account(account_id)


app.include_router(paper_router)


# =============================================================================
# Auto Sim Lab Endpoints
# =============================================================================

from backend.auto_sim_lab import AutoSimConfig, AutoSimResult, run_auto_simulation


@app.post("/auto-simulate", response_model=AutoSimResult)
def auto_simulate(config: AutoSimConfig):
    """
    Run automated paper trading simulation using Final Signal.
    
    This endpoint uses the same signal generation logic as Live Signal,
    but applies it historically to simulate trades with configurable risk management.
    
    Args:
        config: Simulation configuration including symbol, timeframe, capital, risk
        
    Returns:
        AutoSimResult with equity curve, trades, and detailed decision log
    """
    try:
        result = run_auto_simulation(config)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Multi-Symbol Simulation
from backend.models.multi_sim import MultiSimConfig, MultiSimResult
from backend.multi_sim_engine import run_multi_simulation


@app.post("/multi-simulate", response_model=MultiSimResult)
def multi_simulate(config: MultiSimConfig):
    """
    Run multi-symbol simulation across multiple symbols.
    
    Runs Auto Sim Lab for each symbol and returns a ranked table of results.
    
    Ranking criteria:
    1. Total R (highest first)
    2. Avg R/Trade (if tie)
    3. Max Drawdown (lower is better, if tie)
    
    Args:
        config: Multi-sim configuration with list of symbols and settings
        
    Returns:
        MultiSimResult with ranked results for all symbols
    """
    try:
        result = run_multi_simulation(config)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Realtime Sim Lab Endpoints
# =============================================================================

import asyncio
from contextlib import asynccontextmanager
from backend.realtime_sim_manager import get_realtime_manager, RealtimeSimManager


class StartRealtimeRequest(BaseModel):
    symbol: str
    timeframe: str = "1m"
    initial_capital: float = 100000.0
    risk_per_trade: float = 0.01


class StopRealtimeRequest(BaseModel):
    session_id: str


# Background task handle
_background_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    global _background_task
    
    # Startup
    print("[Lifespan] Starting realtime simulation background loop...")
    manager = get_realtime_manager()
    _background_task = asyncio.create_task(
        manager.start_background_loop(interval_seconds=10.0)
    )
    
    yield
    
    # Shutdown
    print("[Lifespan] Stopping realtime simulation background loop...")
    manager.stop_background_loop()
    if _background_task:
        _background_task.cancel()
        try:
            await _background_task
        except asyncio.CancelledError:
            pass


# Re-register app with lifespan
app.router.lifespan_context = lifespan


@app.post("/realtime-sim/start")
def start_realtime_simulation(req: StartRealtimeRequest):
    """
    Start a new realtime paper trading simulation session.
    
    Args:
        req: Configuration including symbol, timeframe, capital, risk
        
    Returns:
        Dict with session_id
    """
    manager = get_realtime_manager()
    
    config = AutoSimConfig(
        symbol=req.symbol,
        timeframe=req.timeframe,
        initial_capital=req.initial_capital,
        risk_per_trade=req.risk_per_trade
    )
    
    session_id = manager.start_session(config)
    
    return {
        "session_id": session_id,
        "symbol": req.symbol,
        "timeframe": req.timeframe,
        "initial_capital": req.initial_capital,
        "status": "started"
    }


@app.post("/realtime-sim/stop")
def stop_realtime_simulation(req: StopRealtimeRequest):
    """
    Stop a running realtime simulation session.
    
    Args:
        req: Request with session_id
        
    Returns:
        Status message
    """
    manager = get_realtime_manager()
    
    success = manager.stop_session(req.session_id)
    
    if success:
        return {"session_id": req.session_id, "status": "stopped"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/realtime-sim/state")
def get_realtime_simulation_state(session_id: str):
    """
    Get the current state of a realtime simulation session.
    
    Args:
        session_id: Session ID to query
        
    Returns:
        Session state including equity curve, trades, decision log
    """
    manager = get_realtime_manager()
    
    state = manager.get_state(session_id)
    
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return state


@app.get("/realtime-sim/sessions")
def list_realtime_sessions():
    """
    List all active realtime simulation sessions.
    
    Returns:
        Dict of sessions with basic info
    """
    manager = get_realtime_manager()
    return manager.list_sessions()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8001, reload=False)


