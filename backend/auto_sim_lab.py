"""
Auto Sim Lab - Automated Paper Trading Simulation Engine

This module implements an automated trading simulator that uses
either the Final Signal from the Live Signal generator or a MA Crossover strategy
to make trading decisions.

Key features:
- Uses existing signal generator (Stat, Rule, ML predictors) OR MA Crossover strategy
- Simulates paper trading with configurable risk management and position sizing
- Records detailed Decision Log for each bar
- Returns equity curve, trades, and decision log

Strategy Modes:
- final_signal: Uses Live Signal predictors (default)
- ma_crossover: Uses Moving Average Crossover (same as Strategy Lab)

Position Sizing Modes:
- percent_of_equity: Risk % of equity per trade (default)
- full_equity: Use all available equity
- fixed_shares: Fixed number of shares
- fixed_dollar: Fixed dollar amount
"""

from datetime import datetime, date
from typing import List, Dict, Any, Optional, Literal, Tuple
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, model_validator

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
    
    # Strategy Mode
    strategy_mode: Literal["final_signal", "ma_crossover"] = "final_signal"
    
    # MA Crossover specific params
    ma_short_window: Optional[int] = None
    ma_long_window: Optional[int] = None
    
    # Position Sizing Mode
    position_sizing_mode: Literal[
        "percent_of_equity",
        "full_equity",
        "fixed_shares",
        "fixed_dollar"
    ] = "percent_of_equity"
    
    fixed_shares: Optional[int] = None
    fixed_dollar_amount: Optional[float] = None
    
    @model_validator(mode='after')
    def validate_config(self):
        """Validate configuration after all fields are set."""
        # Validate MA windows when ma_crossover mode is used
        if self.strategy_mode == 'ma_crossover':
            if self.ma_short_window is None:
                raise ValueError('ma_short_window required for ma_crossover mode')
            if self.ma_long_window is None:
                raise ValueError('ma_long_window required for ma_crossover mode')
            if self.ma_short_window >= self.ma_long_window:
                raise ValueError('ma_short_window must be < ma_long_window')
        
        # Validate fixed sizing parameters
        if self.position_sizing_mode == 'fixed_shares':
            if self.fixed_shares is None or self.fixed_shares <= 0:
                raise ValueError('fixed_shares must be > 0 for fixed_shares mode')
        if self.position_sizing_mode == 'fixed_dollar':
            if self.fixed_dollar_amount is None or self.fixed_dollar_amount <= 0:
                raise ValueError('fixed_dollar_amount must be > 0 for fixed_dollar mode')
        
        return self


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
# MA Crossover Signal Generation
# =============================================================================

def generate_ma_crossover_signals(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> pd.Series:
    """
    Generate MA Crossover signals.
    
    This is the CANONICAL implementation shared between Strategy Lab and Auto Sim Lab.
    Uses the same logic as backend/strategies/ma_cross.py:MACrossStrategy.
    
    Signal interpretation:
    - 1: Short MA > Long MA (bullish, enter long)
    - -1: Short MA < Long MA (bearish, exit long)
    
    Args:
        df: DataFrame with 'close' column
        short_window: Short MA period
        long_window: Long MA period
        
    Returns:
        Series of signals (1 or -1)
    """
    short_ma = df['close'].rolling(window=short_window).mean()
    long_ma = df['close'].rolling(window=long_window).mean()
    
    # Generate signals: 1 where short_ma > long_ma, -1 otherwise
    signals = np.where(short_ma > long_ma, 1, -1)
    
    return pd.Series(signals, index=df.index)


def generate_action_for_bar(
    config: AutoSimConfig,
    df: pd.DataFrame,
    idx: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate trading action for a single bar.
    
    Supports multiple strategy modes:
    - final_signal: Use Live Signal predictors
    - ma_crossover: Use MA Crossover strategy
    
    Args:
        config: Simulation configuration
        df: Full DataFrame with OHLCV data
        idx: Current bar index
        
    Returns:
        Tuple of (action, raw_info)
        - action: "buy" | "sell" | "hold"
        - raw_info: dict for DecisionLog
    """
    if config.strategy_mode == "ma_crossover":
        return _generate_ma_crossover_action(config, df, idx)
    else:
        return _generate_final_signal_action(config, df, idx)


def _generate_ma_crossover_action(
    config: AutoSimConfig,
    df: pd.DataFrame,
    idx: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate action using MA Crossover strategy.
    
    Matches the logic in Backend Strategy Lab's MA Crossover implementation.
    """
    short_window = config.ma_short_window
    long_window = config.ma_long_window
    
    # Need enough data for MA calculation
    if idx < long_window:
        return "hold", {
            "strategy": "ma_crossover",
            "reason": f"Insufficient data for MA calculation ({idx} < {long_window})"
        }
    
    # Get data up to current bar
    df_slice = df.iloc[:idx + 1].copy()
    
    # Calculate MAs
    short_ma = df_slice['close'].rolling(window=short_window).mean()
    long_ma = df_slice['close'].rolling(window=long_window).mean()
    
    current_short = short_ma.iloc[-1]
    current_long = long_ma.iloc[-1]
    
    # Prev bar MAs for crossover detection
    prev_short = short_ma.iloc[-2] if len(short_ma) > 1 else current_short
    prev_long = long_ma.iloc[-2] if len(long_ma) > 1 else current_long
    
    # Determine signal
    # Current position: short > long = bullish (1), short < long = bearish (-1)
    current_signal = 1 if current_short > current_long else -1
    prev_signal = 1 if prev_short > prev_long else -1
    
    # Detect crossover
    action = "hold"
    reason = ""
    
    # Golden cross: short crosses above long -> BUY
    if current_signal == 1 and prev_signal == -1:
        action = "buy"
        reason = f"MA Golden Cross: Short MA({short_window})={current_short:.2f} crossed above Long MA({long_window})={current_long:.2f}"
    # Death cross: short crosses below long -> SELL
    elif current_signal == -1 and prev_signal == 1:
        action = "sell"
        reason = f"MA Death Cross: Short MA({short_window})={current_short:.2f} crossed below Long MA({long_window})={current_long:.2f}"
    else:
        if current_signal == 1:
            reason = f"Bullish: Short MA({short_window})={current_short:.2f} > Long MA({long_window})={current_long:.2f}"
        else:
            reason = f"Bearish: Short MA({short_window})={current_short:.2f} < Long MA({long_window})={current_long:.2f}"
    
    raw_info = {
        "strategy": "ma_crossover",
        "short_ma": round(current_short, 2),
        "long_ma": round(current_long, 2),
        "signal": current_signal,
        "crossover": action != "hold",
        "reason": reason
    }
    
    return action, raw_info


def _generate_final_signal_action(
    config: AutoSimConfig,
    df: pd.DataFrame,
    idx: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate action using Final Signal (Stat, Rule, ML predictors).
    
    This is the original Auto Sim Lab behavior.
    """
    df_slice = df.iloc[:idx + 1].copy()
    
    if len(df_slice) < 20:
        return "hold", {
            "strategy": "final_signal",
            "reason": "Insufficient data for signal generation"
        }
    
    try:
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
        
        # Combine Logic
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
        
        raw_info = {
            "strategy": "final_signal",
            "predictions": predictions,
            "confidence": round(conf, 2),
            "reason": f"{reason} (confidence: {conf:.2f})"
        }
        
        return final_action, raw_info
        
    except Exception as e:
        return "hold", {
            "strategy": "final_signal",
            "error": str(e),
            "reason": f"Signal generation error: {e}"
        }


# =============================================================================
# Position Sizing
# =============================================================================

def calculate_position_size(
    equity: float,
    price: float,
    config: AutoSimConfig,
) -> int:
    """
    Calculate position size based on position sizing mode.
    
    Args:
        equity: Current equity
        price: Entry price
        config: Simulation configuration
        
    Returns:
        Number of shares to buy
    """
    if config.position_sizing_mode == "percent_of_equity":
        risk_amount = equity * config.risk_per_trade
        size = int(risk_amount / price)
        
    elif config.position_sizing_mode == "full_equity":
        # Use all available equity
        size = int(equity / price)
        
    elif config.position_sizing_mode == "fixed_shares":
        size = config.fixed_shares or 0
        
    elif config.position_sizing_mode == "fixed_dollar":
        dollar_amount = config.fixed_dollar_amount or 0
        size = int(dollar_amount / price)
        
    else:
        raise ValueError(f"Unknown position sizing mode: {config.position_sizing_mode}")
    
    return max(size, 0)


# =============================================================================
# Legacy Signal Generation (for backward compatibility)
# =============================================================================

def generate_signal_for_bar(df_slice: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, Any]:
    """
    Generate signal for a single bar using Final Signal predictors.
    
    This is the legacy function maintained for backward compatibility.
    New code should use generate_action_for_bar() instead.
    """
    if len(df_slice) < 20:
        return {
            "action": "hold",
            "confidence": 0.0,
            "reason": "Insufficient data for signal generation"
        }
    
    try:
        from backend.live.predictors import stat, rule, ml
        from backend import rule_predictor_v2
        
        stat_res = stat.predict(df_slice)
        v2_res = rule_predictor_v2.predict(df_slice)
        ml_res = ml.predict(df_slice)
        
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
    except Exception as e:
        return {
            "action": "hold",
            "confidence": 0.0,
            "reason": f"Error: {e}"
        }


# =============================================================================
# Auto Sim Engine
# =============================================================================

def run_auto_simulation(config: AutoSimConfig) -> AutoSimResult:
    """
    Run automated paper trading simulation.
    
    Supports multiple strategy modes:
    - final_signal: Uses Live Signal predictors (default)
    - ma_crossover: Uses Moving Average Crossover
    
    Trading Rules (Long-only, simplified):
    - BUY signal + flat position -> Enter long
    - SELL signal + long position -> Exit long
    - HOLD signal -> Do nothing
    
    Args:
        config: AutoSimConfig with simulation parameters
        
    Returns:
        AutoSimResult with equity curve, trades, and decision log
    """
    # Validate MA params if needed
    if config.strategy_mode == "ma_crossover":
        if not config.ma_short_window or not config.ma_long_window:
            return AutoSimResult(
                symbol=config.symbol,
                timeframe=config.timeframe,
                initial_capital=config.initial_capital,
                final_equity=config.initial_capital,
                total_return_pct=0.0,
                equity_curve=[],
                trades=[],
                decision_log=[],
                summary={"error": "ma_short_window and ma_long_window required for ma_crossover mode"}
            )
    
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
    
    # Determine warmup period
    if config.strategy_mode == "ma_crossover":
        warmup = config.ma_long_window + 5  # MA needs long_window bars
    else:
        warmup = 50  # Final signal needs ~50 bars for indicators
    
    if len(df) < warmup + 10:
        return AutoSimResult(
            symbol=config.symbol,
            timeframe=config.timeframe,
            initial_capital=config.initial_capital,
            final_equity=config.initial_capital,
            total_return_pct=0.0,
            equity_curve=[],
            trades=[],
            decision_log=[],
            summary={"error": f"Insufficient data after filtering (need {warmup + 10}, have {len(df)})"}
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
    
    # 3. Loop Through Bars
    for i in range(warmup, len(df)):
        bar = df.iloc[i]
        bar_time = bar["timestamp"]
        close_price = float(bar["close"])
        
        # Generate action using appropriate strategy
        action, raw_info = generate_action_for_bar(config, df, i)
        
        # Calculate current equity (mark-to-market if in position)
        if position_side == "long":
            unrealized_pnl = (close_price - entry_price) * position_size
            current_equity = equity + unrealized_pnl
        else:
            current_equity = equity
        
        # Build reason string
        reason = raw_info.get("reason", f"Strategy: {config.strategy_mode}")
        
        # Record signal decision
        decision_log.add(DecisionEvent(
            timestamp=bar_time,
            symbol=config.symbol,
            timeframe=config.timeframe,
            event_type="signal_decision",
            final_signal=action,
            raw_signals=raw_info,
            position_side=position_side,
            position_size=position_size if position_side == "long" else None,
            price=close_price,
            equity_before=current_equity,
            equity_after=current_equity,
            risk_per_trade=config.risk_per_trade,
            reason=reason
        ))
        
        # Trading Logic
        equity_before = current_equity
        
        if action == "buy" and position_side == "flat":
            # Enter Long
            position_size = calculate_position_size(equity, close_price, config)
            
            if position_size > 0:
                position_side = "long"
                entry_price = close_price
                entry_time = bar_time
                
                # Calculate risk amount for logging
                if config.position_sizing_mode == "percent_of_equity":
                    risk_amount = equity * config.risk_per_trade
                    sizing_info = f"({config.risk_per_trade*100:.1f}% of equity)"
                elif config.position_sizing_mode == "full_equity":
                    risk_amount = equity
                    sizing_info = "(full equity)"
                elif config.position_sizing_mode == "fixed_shares":
                    risk_amount = position_size * close_price
                    sizing_info = f"(fixed {config.fixed_shares} shares)"
                else:
                    risk_amount = config.fixed_dollar_amount or 0
                    sizing_info = f"(fixed ${risk_amount:.0f})"
                
                decision_log.add(DecisionEvent(
                    timestamp=bar_time,
                    symbol=config.symbol,
                    timeframe=config.timeframe,
                    event_type="entry",
                    final_signal=action,
                    position_side="long",
                    position_size=position_size,
                    price=entry_price,
                    equity_before=equity_before,
                    equity_after=equity_before,  # No change on entry
                    risk_per_trade=config.risk_per_trade,
                    reason=f"Enter LONG @ ${entry_price:.2f}, size={position_size} shares {sizing_info}"
                ))
        
        elif action == "sell" and position_side == "long":
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
                final_signal=action,
                position_side="flat",
                position_size=position_size,
                price=exit_price,
                equity_before=equity_before,
                equity_after=equity,
                risk_per_trade=config.risk_per_trade,
                reason=f"Exit LONG @ ${exit_price:.2f}. PnL: ${pnl:+.2f} ({(exit_price/entry_price-1)*100:+.2f}%)"
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
        "strategy_mode": config.strategy_mode,
        "position_sizing_mode": config.position_sizing_mode,
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
    
    # Add MA params to summary if using MA mode
    if config.strategy_mode == "ma_crossover":
        summary["ma_short_window"] = config.ma_short_window
        summary["ma_long_window"] = config.ma_long_window
    
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
