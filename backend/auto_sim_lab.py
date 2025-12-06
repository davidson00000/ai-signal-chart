"""
Auto Sim Lab - Automated Paper Trading Simulation Engine

This module implements an automated trading simulator that uses
the Final Signal from the Live Signal generator to make trading decisions.

Key features:
- Uses existing signal generator (Stat, Rule, ML predictors)
- Simulates paper trading with configurable risk management
- Records detailed Decision Log for each bar
- Returns equity curve, trades, and decision log
"""

from datetime import datetime, date
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from backend import data_feed
from backend.live.signal_generator import generate_live_signal
from backend.models.decision_log import DecisionEvent, DecisionLog


# =============================================================================
# Request/Response Models
# =============================================================================

class AutoSimConfig(BaseModel):
    """Configuration for Auto Sim Lab simulation."""
    symbol: str
    timeframe: str = "1d"
    initial_capital: float = 100000.0
    risk_per_trade: float = Field(0.01, description="Risk per trade as decimal (0.01 = 1%)")
    start_date: Optional[str] = None  # YYYY-MM-DD format
    end_date: Optional[str] = None    # YYYY-MM-DD format
    max_bars: Optional[int] = None    # Use last N bars if specified


class AutoSimResult(BaseModel):
    """Result of Auto Sim Lab simulation."""
    symbol: str
    timeframe: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    
    equity_curve: List[Dict[str, Any]]  # [{timestamp, equity}, ...]
    trades: List[Dict[str, Any]]        # [{entry_time, exit_time, ...}, ...]
    decision_log: List[Dict[str, Any]]  # List of DecisionEvent dicts
    
    summary: Dict[str, Any]  # Performance summary stats


# =============================================================================
# Signal Generation for Historical Data
# =============================================================================

def generate_signal_for_bar(df_slice: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
    """
    Generate signal for a single bar using historical data up to that point.
    
    This is a simplified version that analyzes the DataFrame slice directly
    without making API calls.
    
    Args:
        df_slice: DataFrame containing OHLCV data up to the current bar
        symbol: Stock symbol
        timeframe: Timeframe string
        
    Returns:
        Signal dict with action, confidence, and reason
    """
    if len(df_slice) < 20:
        return {
            "action": "hold",
            "confidence": 0.0,
            "reason": "Insufficient data for signal generation"
        }
    
    # Import predictors
    from backend.live.predictors import stat, rule, ml
    from backend import rule_predictor_v2
    
    # Call predictors
    stat_res = stat.predict(df_slice)
    v2_res = rule_predictor_v2.predict(df_slice)
    ml_res = ml.predict(df_slice)
    
    # Map v2 result
    prob_up = v2_res["prob_up"]
    prob_down = v2_res["prob_down"]
    confidence = max(prob_up, prob_down)
    
    rule_direction = "flat"
    if prob_up > 0.55:
        rule_direction = "up"
    elif prob_down > 0.55:
        rule_direction = "down"
        
    rule_res = {
        "direction": rule_direction,
        "score": v2_res["score"],
        "confidence": confidence,
    }
    
    predictions = {
        "stat": stat_res,
        "rule": rule_res,
        "ml": ml_res
    }
    
    # Combine Logic (same as signal_generator.py)
    scores = [p["score"] for p in predictions.values()]
    base_confidence = sum(scores) / len(scores)
    
    directions = [p["direction"] for p in predictions.values()]
    up_count = directions.count("up")
    down_count = directions.count("down")
    
    final_action = "hold"
    conf = base_confidence
    reason = ""
    
    if up_count == 3:
        final_action = "buy"
        reason = "Strong bullish consensus across all predictors."
    elif down_count == 3:
        final_action = "sell"
        reason = "Strong bearish consensus across all predictors."
    elif up_count >= 2:
        final_action = "buy"
        conf *= 0.85
        reason = "Bullish bias. Majority of predictors signal UP."
    elif down_count >= 2:
        final_action = "sell"
        conf *= 0.85
        reason = "Bearish bias. Majority of predictors signal DOWN."
    elif up_count == 1 and down_count == 1:
        final_action = "hold"
        conf *= 0.70
        reason = "Predictors disagree (Mixed signals)."
    else:
        final_action = "hold"
        reason = "Market is neutral or predictors are undecided."
    
    return {
        "action": final_action,
        "confidence": round(conf, 2),
        "reason": reason,
        "predictions": predictions
    }


# =============================================================================
# Auto Sim Engine
# =============================================================================

def run_auto_simulation(config: AutoSimConfig) -> AutoSimResult:
    """
    Run automated paper trading simulation using Final Signal.
    
    Trading Rules (Long-only, simplified):
    - BUY signal + flat position -> Enter long
    - SELL signal + long position -> Exit long
    - HOLD signal -> Do nothing
    
    Args:
        config: AutoSimConfig with simulation parameters
        
    Returns:
        AutoSimResult with equity curve, trades, and decision log
    """
    # 1. Fetch Historical Data
    candles = data_feed.get_chart_data(
        symbol=config.symbol,
        timeframe=config.timeframe,
        limit=500  # Get enough data
    )
    
    if not candles:
        return AutoSimResult(
            symbol=config.symbol,
            timeframe=config.timeframe,
            initial_capital=config.initial_capital,
            final_equity=config.initial_capital,
            total_return_pct=0.0,
            equity_curve=[],
            trades=[],
            decision_log=[],
            summary={"error": "No data available"}
        )
    
    df = pd.DataFrame(candles)
    
    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Parse timestamp
    if "time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    df = df.dropna(subset=["close", "timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Apply date filters
    if config.start_date:
        start_dt = pd.to_datetime(config.start_date)
        df = df[df["timestamp"] >= start_dt]
    
    if config.end_date:
        end_dt = pd.to_datetime(config.end_date)
        df = df[df["timestamp"] <= end_dt]
    
    if config.max_bars:
        df = df.tail(config.max_bars)
    
    df = df.reset_index(drop=True)
    
    if len(df) < 30:
        return AutoSimResult(
            symbol=config.symbol,
            timeframe=config.timeframe,
            initial_capital=config.initial_capital,
            final_equity=config.initial_capital,
            total_return_pct=0.0,
            equity_curve=[],
            trades=[],
            decision_log=[],
            summary={"error": "Insufficient data after filtering"}
        )
    
    # 2. Initialize Simulation State
    equity = config.initial_capital
    position_side = "flat"  # "flat" or "long"
    position_size = 0
    entry_price = 0.0
    entry_time = None
    
    equity_curve = []
    trades = []
    decision_log = DecisionLog()
    
    # Need at least 50 bars for indicator calculation
    warmup = 50
    
    # 3. Loop Through Bars
    for i in range(warmup, len(df)):
        bar = df.iloc[i]
        bar_time = bar["timestamp"]
        close_price = float(bar["close"])
        
        # Get data up to this bar for signal generation
        df_slice = df.iloc[:i+1].copy()
        
        # Generate signal
        signal = generate_signal_for_bar(df_slice, config.symbol, config.timeframe)
        final_action = signal["action"]
        
        # Calculate current equity (mark-to-market if in position)
        if position_side == "long":
            unrealized_pnl = (close_price - entry_price) * position_size
            current_equity = equity + unrealized_pnl
        else:
            current_equity = equity
        
        # Record signal decision
        decision_log.add(DecisionEvent(
            timestamp=bar_time,
            symbol=config.symbol,
            timeframe=config.timeframe,
            event_type="signal_decision",
            final_signal=final_action,
            raw_signals=signal.get("predictions"),
            position_side=position_side,
            position_size=position_size if position_side == "long" else None,
            price=close_price,
            equity_before=current_equity,
            equity_after=current_equity,
            risk_per_trade=config.risk_per_trade,
            reason=f"{signal['reason']} (confidence: {signal['confidence']:.2f})"
        ))
        
        # Trading Logic
        equity_before = current_equity
        
        if final_action == "buy" and position_side == "flat":
            # Enter Long
            risk_amount = equity * config.risk_per_trade
            position_size = int(risk_amount / close_price)
            
            if position_size > 0:
                position_side = "long"
                entry_price = close_price
                entry_time = bar_time
                
                decision_log.add(DecisionEvent(
                    timestamp=bar_time,
                    symbol=config.symbol,
                    timeframe=config.timeframe,
                    event_type="entry",
                    final_signal=final_action,
                    position_side="long",
                    position_size=position_size,
                    price=entry_price,
                    equity_before=equity_before,
                    equity_after=equity_before,  # No change on entry
                    risk_per_trade=config.risk_per_trade,
                    reason=f"Enter LONG @ ${entry_price:.2f}, size={position_size} shares. "
                           f"Risk amount: ${risk_amount:.2f} ({config.risk_per_trade*100:.1f}% of equity)"
                ))
        
        elif final_action == "sell" and position_side == "long":
            # Exit Long
            exit_price = close_price
            pnl = (exit_price - entry_price) * position_size
            equity += pnl
            
            trades.append({
                "entry_time": entry_time.isoformat() if hasattr(entry_time, 'isoformat') else str(entry_time),
                "exit_time": bar_time.isoformat() if hasattr(bar_time, 'isoformat') else str(bar_time),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "size": position_size,
                "pnl": round(pnl, 2),
                "return_pct": round((exit_price / entry_price - 1) * 100, 2)
            })
            
            decision_log.add(DecisionEvent(
                timestamp=bar_time,
                symbol=config.symbol,
                timeframe=config.timeframe,
                event_type="exit",
                final_signal=final_action,
                position_side="flat",
                position_size=position_size,
                price=exit_price,
                equity_before=equity_before,
                equity_after=equity,
                risk_per_trade=config.risk_per_trade,
                reason=f"Exit LONG @ ${exit_price:.2f}. PnL: ${pnl:+.2f} "
                       f"({(exit_price/entry_price-1)*100:+.2f}%)"
            ))
            
            position_side = "flat"
            position_size = 0
            entry_price = 0.0
            entry_time = None
        
        # Record equity curve
        if position_side == "long":
            unrealized_pnl = (close_price - entry_price) * position_size
            curve_equity = equity + unrealized_pnl
        else:
            curve_equity = equity
        
        equity_curve.append({
            "timestamp": bar_time.isoformat() if hasattr(bar_time, 'isoformat') else str(bar_time),
            "equity": round(curve_equity, 2),
            "position": position_side
        })
    
    # 4. Close any open position at the end
    if position_side == "long":
        exit_price = float(df.iloc[-1]["close"])
        pnl = (exit_price - entry_price) * position_size
        equity += pnl
        
        trades.append({
            "entry_time": entry_time.isoformat() if hasattr(entry_time, 'isoformat') else str(entry_time),
            "exit_time": df.iloc[-1]["timestamp"].isoformat() if hasattr(df.iloc[-1]["timestamp"], 'isoformat') else str(df.iloc[-1]["timestamp"]),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": position_size,
            "pnl": round(pnl, 2),
            "return_pct": round((exit_price / entry_price - 1) * 100, 2),
            "note": "Closed at end of simulation"
        })
    
    # 5. Calculate Summary Stats
    final_equity = equity
    total_return_pct = (final_equity / config.initial_capital - 1) * 100
    
    total_trades = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    
    total_pnl = sum(t["pnl"] for t in trades)
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
    
    summary = {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "best_trade": round(max((t["pnl"] for t in trades), default=0), 2),
        "worst_trade": round(min((t["pnl"] for t in trades), default=0), 2),
        "bars_analyzed": len(df) - warmup,
        "decision_events": len(decision_log)
    }
    
    return AutoSimResult(
        symbol=config.symbol,
        timeframe=config.timeframe,
        initial_capital=config.initial_capital,
        final_equity=round(final_equity, 2),
        total_return_pct=round(total_return_pct, 2),
        equity_curve=equity_curve,
        trades=trades,
        decision_log=decision_log.to_list(),
        summary=summary
    )
