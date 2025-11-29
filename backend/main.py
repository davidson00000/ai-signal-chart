"""
FastAPI Backend for AI Signal Chart Backtest System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd

from backend.data_feed import get_chart_data
from backend.backtester import BacktestEngine
from backend.strategies.ma_cross import MACrossStrategy
from backend.models.backtest import (
    BacktestRequest,
    BacktestResponse,
    BacktestStats,
    EquityCurvePoint,
    TradeSummary,
)


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
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


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
        candles: List[dict] = get_chart_data(
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
            status_code=500,
            detail=f"Backtest failed: {str(e)}",
        )


# `python -m backend.main` で起動できるように
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
