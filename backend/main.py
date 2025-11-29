"""
FastAPI Backend for AI Signal Chart Backtest System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import pandas as pd

from backend import data_feed
from backend.backtester import BacktestEngine
from backend.strategies.ma_cross import MACrossStrategy
from backend.models.backtest import (
    BacktestRequest,
    BacktestResponse,
    BacktestStats,
    EquityCurvePoint,
    TradeSummary,
)
from backend.models.optimization import OptimizationRequest, OptimizationResponse


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
def simulate_backtest(request: BacktestRequest) -> BacktestResponse:
    """
    Run backtest simulation.

    Args:
        request: BacktestRequest with symbol, timeframe, strategy parameters

    Returns:
        BacktestResponse with equity curve, trades, and statistics

    Raises:
        HTTPException: If data cannot be fetched or backtest fails
    """
    try:
        # 1. Fetch historical market data
        candles: List[dict] = data_feed.get_chart_data(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start=request.start_date,
            end=request.end_date,
            limit=2000,
        )

        if not candles:
            raise HTTPException(
                status_code=400,
                detail="No market data returned for this query.",
            )

        # 2. Convert to DataFrame and set index
        df = pd.DataFrame(candles)
        
        # Ensure 'time' column exists and convert to datetime
        if "time" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail="Data feed did not return 'time' column.",
            )
        
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

        # Ensure 'close' column exists
        if "close" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail="Data feed did not return 'close' column.",
            )

        # 3. Initialize strategy based on request
        if request.strategy == "ma_cross":
            strategy = MACrossStrategy(
                short_window=request.short_window,
                long_window=request.long_window,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported strategy: {request.strategy}",
            )

        # 4. Initialize and run backtest engine
        engine = BacktestEngine(
            initial_capital=request.initial_capital,
            commission_rate=request.commission_rate,
            position_size=request.position_size,
            lot_size=1.0,
        )

        result = engine.run_backtest(df, strategy)

        # 5. Convert result to response models
        equity_curve = [
            EquityCurvePoint(**point) for point in result["equity_curve"]
        ]
        trades = [TradeSummary(**trade) for trade in result["trades"]]
        stats = BacktestStats(**result["stats"])

        # 6. Build and return response
        response = BacktestResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=str(strategy),
            equity_curve=equity_curve,
            trades=trades,
            metrics=stats,
            data_points=len(df),
        )

        return response

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Catch any other errors and return 500
        raise HTTPException(
            status_code=400,
            detail=f"Data fetching or strategy execution failed: {e}",
        )


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


# `python -m backend.main` で起動できるように
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
