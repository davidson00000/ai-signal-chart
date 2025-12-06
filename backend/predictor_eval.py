"""
Predictor Evaluation Module

Provides backtest-like evaluation for predictors (Stat, Rule v2, Buy & Hold).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta

from backend import data_feed
from backend import rule_predictor_v2
from backend.live.predictors import stat as stat_predictor


def _get_predictor_signal(df: pd.DataFrame, predictor_type: str, idx: int) -> str:
    """
    Get predictor signal ("UP", "DOWN", "FLAT") for a given bar index.
    Uses rolling window up to idx.
    """
    if predictor_type == "buy_and_hold":
        return "UP"  # Always long
    
    # Need at least 35 bars for indicators
    if idx < 35:
        return "FLAT"
    
    # Slice dataframe up to current bar (inclusive)
    df_slice = df.iloc[:idx+1].copy()
    
    if predictor_type == "stat":
        result = stat_predictor.predict(df_slice)
        direction = result.get("direction", "flat").upper()
        return direction
    
    elif predictor_type == "rule_v2":
        result = rule_predictor_v2.predict(df_slice)
        prob_up = result.get("prob_up", 0.5)
        prob_down = result.get("prob_down", 0.5)
        
        if prob_up > 0.55:
            return "UP"
        elif prob_down > 0.55:
            return "DOWN"
        else:
            return "FLAT"
    
    return "FLAT"


def run_predictor_backtest(
    symbol: str,
    timeframe: str = "1d",
    start_date: str = None,
    end_date: str = None,
    predictor_type: str = "rule_v2",
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    Run a simple predictor-driven backtest.
    
    Trading Rule:
    - If predictor says UP -> enter/hold LONG
    - If predictor says DOWN -> exit long (go flat)
    - If predictor says FLAT -> keep current position
    
    Args:
        symbol: Stock symbol
        timeframe: Timeframe (1d, 4h, etc.)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        predictor_type: "stat", "rule_v2", "buy_and_hold"
        initial_capital: Starting capital
        
    Returns:
        Dict with metrics and equity curve
    """
    # 1. Fetch data
    # Calculate lookback for warm-up period (need extra days for indicators)
    limit = 1000  # Max bars to fetch
    
    candles = data_feed.get_chart_data(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
        start=start_date,
        end=end_date
    )
    
    if not candles:
        return {"error": f"No data found for {symbol}"}
    
    df = pd.DataFrame(candles)
    
    # Normalize columns
    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"], unit="s")
    elif "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"])
    
    df = df.sort_values("date").reset_index(drop=True)
    
    # Filter by date range if specified
    if start_date:
        start_dt = pd.to_datetime(start_date)
        df = df[df["date"] >= start_dt].reset_index(drop=True)
    if end_date:
        end_dt = pd.to_datetime(end_date)
        df = df[df["date"] <= end_dt].reset_index(drop=True)
    
    if len(df) < 50:
        return {"error": f"Insufficient data for {symbol} (got {len(df)} bars)"}
    
    # 2. Run backtest simulation
    cash = initial_capital
    position = 0
    entry_price = 0.0
    
    trades: List[Dict] = []
    equity_curve: List[Dict] = []
    
    # Start trading after warm-up period (35 bars for indicators)
    start_idx = 35 if predictor_type != "buy_and_hold" else 0
    
    for idx in range(start_idx, len(df)):
        row = df.iloc[idx]
        price = float(row["close"])
        date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
        
        # Get predictor signal
        signal = _get_predictor_signal(df, predictor_type, idx)
        
        # Trading logic
        if signal == "UP" and position == 0:
            # Buy
            qty = cash // price
            if qty > 0:
                cost = qty * price
                cash -= cost
                position = qty
                entry_price = price
                trades.append({
                    "date": date_str,
                    "side": "BUY",
                    "price": price,
                    "quantity": qty,
                    "pnl": None
                })
        
        elif signal == "DOWN" and position > 0:
            # Sell
            qty = position
            revenue = qty * price
            pnl = revenue - (qty * entry_price)
            cash += revenue
            position = 0
            trades.append({
                "date": date_str,
                "side": "SELL",
                "price": price,
                "quantity": qty,
                "pnl": pnl
            })
        
        # Record equity
        equity = cash + position * price
        equity_curve.append({
            "date": date_str,
            "equity": equity
        })
    
    # Close any open position at end
    if position > 0:
        row = df.iloc[-1]
        price = float(row["close"])
        date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
        
        qty = position
        revenue = qty * price
        pnl = revenue - (qty * entry_price)
        cash += revenue
        position = 0
        trades.append({
            "date": date_str,
            "side": "SELL",
            "price": price,
            "quantity": qty,
            "pnl": pnl
        })
        equity_curve[-1]["equity"] = cash
    
    # 3. Calculate metrics
    final_equity = cash
    total_return = ((final_equity - initial_capital) / initial_capital) * 100
    
    # Win rate
    completed_trades = [t for t in trades if t.get("pnl") is not None]
    winning_trades = [t for t in completed_trades if t["pnl"] > 0]
    num_trades = len(completed_trades)
    win_rate = (len(winning_trades) / num_trades * 100) if num_trades > 0 else 0.0
    
    # Max drawdown
    max_drawdown = 0.0
    peak = initial_capital
    for point in equity_curve:
        equity = point["equity"]
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_drawdown:
            max_drawdown = dd
    max_drawdown_pct = max_drawdown * 100
    
    # Sharpe-like ratio
    sharpe = None
    if len(equity_curve) > 1:
        returns = []
        for i in range(1, len(equity_curve)):
            prev_eq = equity_curve[i-1]["equity"]
            curr_eq = equity_curve[i]["equity"]
            ret = (curr_eq - prev_eq) / prev_eq if prev_eq > 0 else 0.0
            returns.append(ret)
        
        if len(returns) > 0:
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            if std_ret > 0:
                sharpe = (mean_ret / std_ret) * np.sqrt(252)
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "predictor_type": predictor_type,
        "total_return": round(total_return, 2),
        "win_rate": round(win_rate, 2),
        "num_trades": num_trades,
        "max_drawdown": round(max_drawdown_pct, 2),
        "sharpe_ratio": round(sharpe, 2) if sharpe else None,
        "final_equity": round(final_equity, 2),
        "initial_capital": initial_capital,
        "equity_curve": equity_curve,
        "trades": trades
    }


def run_all_predictors_backtest(
    symbol: str,
    timeframe: str = "1d",
    start_date: str = None,
    end_date: str = None,
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    Run backtest for all predictors and return comparison.
    """
    predictors = ["buy_and_hold", "stat", "rule_v2"]
    results = {}
    
    for pred in predictors:
        try:
            result = run_predictor_backtest(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                predictor_type=pred,
                initial_capital=initial_capital
            )
            results[pred] = result
        except Exception as e:
            results[pred] = {"error": str(e)}
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "results": results
    }
