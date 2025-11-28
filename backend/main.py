"""
EXITON Backend - Main FastAPI Application
"""
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from datetime import datetime
from typing import Optional

from backend.data_feed import get_chart_data as fetch_chart_data, get_latest_price
from backend.strategy import generate_signals_and_trades, generate_signal
from backend.paper_trade import PaperTrader
from backend.models.requests import PaperOrderRequest
from backend.models.responses import (
    HealthResponse,
    ChartDataResponse,
    SignalResponse,
    OrderResponse,
    PositionsResponse,
    TradesResponse,
    PnLResponse,
)

# Initialize FastAPI app
app = FastAPI(
    title="EXITON AI Trading System",
    description="AI-powered automated trading system backend",
    version="0.1.0"
)

# CORS middleware for local and Vercel deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global paper trader instance
trader = PaperTrader(initial_cash=100000.0)


# === API Endpoints ===

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/chart-data", response_model=ChartDataResponse)
def get_chart_data(
    symbol: str = Query(..., description="Crypto pair like BTC/USDT or stock like AAPL, 7203.T"),
    timeframe: str = Query("1m", description="1m,5m,15m,30m,1h,4h,1d"),
    limit: int = Query(200, ge=50, le=1000),
    short_window: int = Query(9, ge=1),
    long_window: int = Query(21, ge=2),
    tp_ratio: float = Query(0.01, gt=0),
    sl_ratio: float = Query(0.005, gt=0),
):
    """
    Get chart data with signals and trades
    
    This endpoint maintains backward compatibility with the existing frontend.
    Response format matches the original implementation to ensure the UI works correctly.
    """
    # Fetch candles
    candles = fetch_chart_data(symbol, timeframe, limit)
    
    # Check if we have enough data for MA calculation
    if len(candles) < max(short_window, long_window) + 5:
        raise HTTPException(
            status_code=400,
            detail="Not enough candles for MA calculation. Try smaller windows or different timeframe.",
        )
    
    # Generate signals and trades using MA cross strategy
    signal_result = generate_signals_and_trades(
        candles,
        short_window=short_window,
        long_window=long_window,
        tp_ratio=tp_ratio,
        sl_ratio=sl_ratio,
    )
    
    # Determine if crypto or stock
    is_crypto = "/" in symbol
    
    # Return in frontend-compatible format
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles,
        "shortMA": signal_result["shortMA"],
        "longMA": signal_result["longMA"],
        "signals": signal_result["signals"],
        "trades": signal_result["trades"],
        "stats": signal_result["stats"],
        "meta": {
            "isCrypto": is_crypto,
            "source": "ccxt" if is_crypto else "yfinance",
        },
    }


@app.get("/signal", response_model=SignalResponse)
def get_signal(
    symbol: str = Query(..., description="Symbol to analyze"),
    date: Optional[str] = Query(None, description="Date for signal (YYYY-MM-DD)"),
    strategy: str = Query("ma_cross", description="Strategy name"),
    timeframe: str = Query("1d", description="Timeframe"),
    limit: int = Query(200, ge=50, le=500),
    short_window: int = Query(9, ge=1),
    long_window: int = Query(21, ge=2),
    tp_ratio: float = Query(0.01, gt=0),
    sl_ratio: float = Query(0.005, gt=0),
):
    """
    Get trading signal for a symbol
    
    Per API_SPEC.md specification
    """
    # Fetch chart data
    candles = fetch_chart_data(symbol, timeframe, limit)
    
    if not candles:
        raise HTTPException(status_code=400, detail="No data available for symbol")
    
    # Generate signal
    signal = generate_signal(
        symbol=symbol,
        candles=candles,
        strategy=strategy,
        date=date,
        timeframe=timeframe,
        short_window=short_window,
        long_window=long_window,
        tp_ratio=tp_ratio,
        sl_ratio=sl_ratio,
    )
    
    return signal


@app.post("/paper-order", response_model=OrderResponse)
def place_paper_order(
    # JSON body (priority)
    body: Optional[PaperOrderRequest] = None,
    # Query parameters (fallback)
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    quantity: Optional[int] = None,
    price: Optional[float] = None,
    signal_id: Optional[str] = None,
    order_time: Optional[str] = None,
    mode: str = "market"
):
    """
    Place a paper trading order
    
    Per API_SPEC.md specification
    Now supports both JSON body and query parameters:
    - Priority 1: JSON body (PaperOrderRequest)
    - Priority 2: Query parameters (backward compatible)
    
    Automatically fetches market price when price not specified.
    """
    # Priority 1: Use JSON body if provided
    if body is not None:
        symbol = body.symbol
        side = body.side
        quantity = body.quantity
        price = body.price if body.price is not None else price
        signal_id = body.signal_id if body.signal_id is not None else signal_id
        order_time = body.order_time if body.order_time is not None else order_time
        mode = body.mode
    # Priority 2: Use query parameters
    elif symbol is None or side is None or quantity is None:
        raise HTTPException(
            status_code=422,
            detail="Either provide JSON body or symbol, side, and quantity as query parameters"
        )
    
    # Validate inputs
    if side not in ["BUY", "SELL"]:
        raise HTTPException(status_code=400, detail="side must be BUY or SELL")
    
    if quantity <= 0:
        raise HTTPException(status_code=400, detail="quantity must be positive")
    
    # If price not provided, fetch current market price
    if price is None:
        try:
            price = get_latest_price(symbol)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get market price for {symbol}: {e}"
            )
    
    # Execute order via paper trader
    result = trader.execute_order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        signal_id=signal_id,
        order_time=order_time,
        mode=mode
    )
    
    return result


@app.get("/positions", response_model=PositionsResponse)
def get_positions():
    """
    Get current positions with real-time pricing
    
    Per API_SPEC.md specification
    Now calculates unrealized P&L using current market prices
    """
    # Get positions with current prices and unrealized P&L
    positions = trader.get_positions(price_lookup_fn=get_latest_price)
    
    # Calculate total unrealized P&L
    total_unrealized = sum(
        p.get("unrealized_pnl", 0.0) 
        for p in positions 
        if p.get("unrealized_pnl") is not None
    )
    
    return {
        "positions": positions,
        "total_unrealized_pnl": total_unrealized
    }


@app.get("/trades", response_model=TradesResponse)
def get_trades(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of trades")
):
    """
    Get trade history
    
    Per API_SPEC.md specification
    """
    trades = trader.get_trades(
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
        limit=limit
    )
    
    return {"trades": trades}


@app.get("/pnl", response_model=PnLResponse)
def get_pnl(
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    mode: str = Query("daily", description="daily or monthly")
):
    """
    Get P&L summary
    
    Per API_SPEC.md specification
    
    Note: This is a basic implementation. A full implementation would
    calculate daily equity snapshots and track unrealized P&L over time.
    """
    total_pnl = trader.total_pnl()
    equity = trader.get_equity()
    
    # For now, return a simple summary
    # In production, we would track daily equity values
    return {
        "mode": mode,
        "from_date": from_date,
        "to_date": to_date,
        "pnl": [
            {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "realized": total_pnl,
                "unrealized": 0.0,
                "equity": equity
            }
        ]
    }


# === Frontend Static Files ===

# Serve frontend files
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.get("/")
def root():
    """Serve index.html at root"""
    return FileResponse("frontend/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
