"""
Auto Sim Lab - Automated Paper Trading Simulation Engine

This module implements an automated trading simulator that uses
either the Final Signal from the Live Signal generator or a MA Crossover strategy
to make trading decisions.

Key features:
- Uses existing signal generator (Stat, Rule, ML predictors) OR MA Crossover strategy
- Simulates paper trading with configurable risk management and position sizing
- R-based risk management with virtual stops
- Execution modes (same_bar_close / next_bar_open)
- Loss control (max drawdown, max daily loss)
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
from backend.r_analytics import compute_r_analytics


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
    strategy_mode: Literal[
        "final_signal", 
        "ma_crossover", 
        "buy_and_hold", 
        "rsi_mean_reversion",
        "ema_crossover",
        "macd",
        "breakout",
        "bollinger"
    ] = "final_signal"
    
    # MA Crossover specific params
    ma_short_window: Optional[int] = None
    ma_long_window: Optional[int] = None
    
    # RSI Mean Reversion specific params
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    
    # EMA Strategy params
    ema_short: int = 12
    ema_long: int = 26
    
    # MACD Strategy params
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Breakout Strategy params
    breakout_window: int = 20
    exit_window: int = 10
    
    # Bollinger Strategy params
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Position Sizing Mode
    position_sizing_mode: Literal[
        "percent_of_equity",
        "full_equity",
        "fixed_shares",
        "fixed_dollar"
    ] = "percent_of_equity"
    
    fixed_shares: Optional[int] = None
    fixed_dollar_amount: Optional[float] = None
    
    # R-Management
    use_r_management: bool = False
    virtual_stop_method: Literal["atr", "percent"] = "percent"
    virtual_stop_atr_multiplier: float = 2.0
    virtual_stop_percent: float = 0.03  # 3%
    record_r_values: bool = True
    
    # Execution Mode
    execution_mode: Literal["same_bar_close", "next_bar_open"] = "same_bar_close"
    commission_percent: float = 0.0
    slippage_percent: float = 0.0
    
    # Loss Control
    max_drawdown_percent: Optional[float] = None
    max_daily_loss_r: Optional[float] = None
    
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
        
        # Validate RSI params when rsi_mean_reversion mode is used
        if self.strategy_mode == 'rsi_mean_reversion':
            if self.rsi_oversold >= self.rsi_overbought:
                raise ValueError('rsi_oversold must be < rsi_overbought')
        
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
    r_analytics: Optional[Dict[str, Any]] = None  # R-based analytics (when R management enabled)


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
    - buy_and_hold: Buy on first bar, hold forever
    - rsi_mean_reversion: Buy when RSI < oversold, sell when RSI > overbought
    - ema_crossover: EMA Crossover
    - macd: MACD Signal Line
    - breakout: Breakout Strategy
    - bollinger: Bollinger Mean Reversion
    
    Args:
        config: Simulation configuration
        df: Full DataFrame with OHLCV data
        idx: Current bar index
        
    Returns:
        Tuple of (action, raw_info)
        - action: "buy" | "sell" | "hold"
        - raw_info: dict for DecisionLog
    """
    mode = config.strategy_mode
    
    if mode == "ma_crossover":
        return _generate_ma_crossover_action(config, df, idx)
    elif mode == "buy_and_hold":
        return _generate_buy_and_hold_action(config, df, idx)
    elif mode == "rsi_mean_reversion":
        return _generate_rsi_action(config, df, idx)
    elif mode == "final_signal":
        return _generate_final_signal_action(config, df, idx)
    
    # New Strategies
    # Convert config to dict for strategy init
    config_dict = config.model_dump()
    
    if mode == "ema_crossover":
        from backend.strategies.ema_crossover import EmaCrossoverStrategy
        return EmaCrossoverStrategy(config_dict).generate_action(df, idx)
    elif mode == "macd":
        from backend.strategies.macd_strategy import MacdStrategy
        return MacdStrategy(config_dict).generate_action(df, idx)
    elif mode == "breakout":
        from backend.strategies.breakout_strategy import BreakoutStrategy
        return BreakoutStrategy(config_dict).generate_action(df, idx)
    elif mode == "bollinger":
        from backend.strategies.bollinger_strategy import BollingerStrategy
        return BollingerStrategy(config_dict).generate_action(df, idx)
        
    return "hold", {"reason": f"Unknown strategy mode: {mode}"}


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


def _generate_buy_and_hold_action(
    config: AutoSimConfig,
    df: pd.DataFrame,
    idx: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate action using Buy & Hold strategy.
    
    Logic:
    - Buy on the first opportunity (always return "buy")
    - The simulation will only execute if not already in position
    - Never sell (always return "hold" after first buy would execute)
    
    Note: The simulation logic handles the "already in position" case,
    so we always return "buy" to ensure entry on first bar.
    """
    # Always return "buy" - simulation handles position state
    # This ensures we buy on the first available bar
    action = "buy"
    reason = "Buy & Hold: Buy and hold position"
    
    raw_info = {
        "strategy": "buy_and_hold",
        "bar_index": idx,
        "reason": reason
    }
    
    return action, raw_info



def _generate_rsi_action(
    config: AutoSimConfig,
    df: pd.DataFrame,
    idx: int,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate action using RSI Mean Reversion strategy.
    
    Logic:
    - Buy when RSI < oversold
    - Sell when RSI > overbought
    """
    period = config.rsi_period
    oversold = config.rsi_oversold
    overbought = config.rsi_overbought
    
    # Need enough data for RSI calculation
    if idx < period:
        return "hold", {
            "strategy": "rsi_mean_reversion",
            "reason": f"Insufficient data for RSI calculation ({idx} < {period})"
        }
    
    # Calculate RSI
    close = df['close'].iloc[:idx+1]
    delta = close.diff()
    
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
    current_rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50
    
    # Determine action
    action = "hold"
    if current_rsi < oversold:
        action = "buy"
    elif current_rsi > overbought:
        action = "sell"
    
    reason = f"RSI={current_rsi:.1f} (oversold={oversold}, overbought={overbought})"
    
    raw_info = {
        "strategy": "rsi_mean_reversion",
        "rsi_period": period,
        "rsi_oversold": oversold,
        "rsi_overbought": overbought,
        "current_rsi": round(current_rsi, 2),
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
# ATR Calculation
# =============================================================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average True Range (ATR) for the given DataFrame."""
    if len(df) < period + 1:
        return 0.0
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    tr_list = []
    for i in range(1, len(df)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr_list.append(max(hl, hc, lc))
    
    if len(tr_list) < period:
        return np.mean(tr_list) if tr_list else 0.0
    
    return np.mean(tr_list[-period:])


# =============================================================================
# Virtual Stop Calculation
# =============================================================================

def calculate_virtual_stop(
    entry_price: float,
    atr_value: float,
    config: AutoSimConfig
) -> float:
    """
    Calculate virtual stop price for R-management.
    
    Args:
        entry_price: Entry price
        atr_value: Current ATR value
        config: Simulation configuration
        
    Returns:
        Virtual stop price
    """
    if config.virtual_stop_method == "atr":
        stop = entry_price - (atr_value * config.virtual_stop_atr_multiplier)
    else:  # percent
        stop = entry_price * (1 - config.virtual_stop_percent)
    
    return max(stop, 0.01)  # Ensure positive


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
# Execution Price Calculation
# =============================================================================

def calculate_execution_price(
    raw_price: float,
    is_buy: bool,
    config: AutoSimConfig
) -> Tuple[float, float, float]:
    """
    Calculate execution price with commission and slippage.
    
    Args:
        raw_price: Raw order price
        is_buy: True for buy, False for sell
        config: Simulation configuration
        
    Returns:
        Tuple of (execution_price, commission, slippage)
    """
    slippage = raw_price * config.slippage_percent
    commission = raw_price * config.commission_percent
    
    if is_buy:
        # Buy: price goes up with slippage
        execution_price = raw_price * (1 + config.slippage_percent)
    else:
        # Sell: price goes down with slippage
        execution_price = raw_price * (1 - config.slippage_percent)
    
    return execution_price, commission, slippage


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
    
    Enhanced features:
    - R-management with virtual stops
    - Execution modes (same_bar_close / next_bar_open)
    - Loss control (max drawdown, max daily loss)
    
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
    peak_equity = config.initial_capital
    position_side = "flat"  # "flat" or "long"
    position_size = 0
    entry_price = 0.0
    entry_time = None
    entry_stop_price = 0.0
    entry_risk_amount = 0.0
    entry_atr = 0.0
    
    equity_curve = []
    trades = []
    decision_log = DecisionLog()
    
    # Loss control state
    simulation_halted = False
    halt_reason = None
    daily_r_tracker: Dict[str, float] = {}  # date_str -> cumulative R
    
    # Trade ID tracking
    next_trade_id = 1
    current_trade_id: Optional[int] = None  # Set when trade is opened
    
    # Pending order state (for next_bar_open execution)
    pending_action = None  # "buy" or "sell"
    pending_signal_idx = None
    
    # 3. Loop Through Bars
    for i in range(warmup, len(df)):
        if simulation_halted:
            break
            
        bar = df.iloc[i]
        bar_time = bar["timestamp"]
        bar_date = bar_time.strftime("%Y-%m-%d") if hasattr(bar_time, 'strftime') else str(bar_time)[:10]
        close_price = float(bar["close"])
        open_price = float(bar["open"]) if "open" in bar else close_price
        
        # Calculate ATR for R-management
        current_atr = 0.0
        if config.use_r_management:
            df_slice = df.iloc[:i+1]
            current_atr = calculate_atr(df_slice)
        
        # Handle pending order (for next_bar_open execution)
        if pending_action and config.execution_mode == "next_bar_open":
            if pending_action == "buy" and position_side == "flat":
                # Execute pending buy at open
                raw_price = open_price
                exec_price, commission, slippage = calculate_execution_price(raw_price, True, config)
                
                position_size = calculate_position_size(equity, exec_price, config)
                
                if position_size > 0:
                    position_side = "long"
                    entry_price = exec_price
                    entry_time = bar_time
                    
                    # Assign trade ID
                    current_trade_id = next_trade_id
                    next_trade_id += 1
                    
                    # R-management
                    if config.use_r_management:
                        entry_atr = current_atr
                        entry_stop_price = calculate_virtual_stop(entry_price, entry_atr, config)
                        entry_risk_amount = (entry_price - entry_stop_price) * position_size
                    
                    # Build sizing info
                    if config.position_sizing_mode == "percent_of_equity":
                        sizing_info = f"({config.risk_per_trade*100:.1f}% of equity)"
                    elif config.position_sizing_mode == "full_equity":
                        sizing_info = "(full equity)"
                    elif config.position_sizing_mode == "fixed_shares":
                        sizing_info = f"(fixed {config.fixed_shares} shares)"
                    else:
                        sizing_info = f"(fixed ${config.fixed_dollar_amount:.0f})"
                    
                    decision_log.add(DecisionEvent(
                        timestamp=bar_time,
                        symbol=config.symbol,
                        timeframe=config.timeframe,
                        event_type="entry",
                        final_signal="buy",
                        position_side="long",
                        position_size=position_size,
                        price=entry_price,
                        execution_price=exec_price,
                        execution_mode="next_bar_open",
                        commission=commission,
                        slippage=slippage,
                        atr_value=entry_atr if config.use_r_management else None,
                        stop_price=entry_stop_price if config.use_r_management else None,
                        risk_amount=entry_risk_amount if config.use_r_management else None,
                        equity_before=equity,
                        equity_after=equity,
                        risk_per_trade=config.risk_per_trade,
                        trade_id=current_trade_id,
                        reason=f"Enter LONG @ ${entry_price:.2f}, size={position_size} shares {sizing_info}"
                    ))
                    
            elif pending_action == "sell" and position_side == "long":
                # Execute pending sell at open
                raw_price = open_price
                exec_price, commission, slippage = calculate_execution_price(raw_price, False, config)
                
                pnl = (exec_price - entry_price) * position_size
                
                # Calculate R-value
                r_value = None
                if config.use_r_management and entry_risk_amount > 0:
                    r_value = pnl / entry_risk_amount
                    
                    # Track daily R
                    if bar_date not in daily_r_tracker:
                        daily_r_tracker[bar_date] = 0.0
                    daily_r_tracker[bar_date] += r_value
                
                equity_before = equity
                equity += pnl
                
                trades.append({
                    "trade_id": current_trade_id,
                    "entry_time": entry_time.isoformat() if hasattr(entry_time, 'isoformat') else str(entry_time),
                    "exit_time": bar_time.isoformat() if hasattr(bar_time, 'isoformat') else str(bar_time),
                    "entry_price": entry_price,
                    "exit_price": exec_price,
                    "size": position_size,
                    "pnl": round(pnl, 2),
                    "return_pct": round((exec_price / entry_price - 1) * 100, 2),
                    "r_value": round(r_value, 2) if r_value is not None else None,
                    "stop_price": entry_stop_price if config.use_r_management else None,
                    "risk_amount": round(entry_risk_amount, 2) if config.use_r_management else None,
                    "execution_mode": "next_bar_open"
                })
                
                decision_log.add(DecisionEvent(
                    timestamp=bar_time,
                    symbol=config.symbol,
                    timeframe=config.timeframe,
                    event_type="exit",
                    final_signal="sell",
                    position_side="flat",
                    position_size=position_size,
                    price=exec_price,
                    execution_price=exec_price,
                    execution_mode="next_bar_open",
                    commission=commission,
                    slippage=slippage,
                    r_value=r_value,
                    stop_price=entry_stop_price if config.use_r_management else None,
                    risk_amount=entry_risk_amount if config.use_r_management else None,
                    equity_before=equity_before,
                    equity_after=equity,
                    risk_per_trade=config.risk_per_trade,
                    daily_r_loss=daily_r_tracker.get(bar_date),
                    trade_id=current_trade_id,
                    reason=f"Exit LONG @ ${exec_price:.2f}. PnL: ${pnl:+.2f} ({(exec_price/entry_price-1)*100:+.2f}%)"
                          + (f" [R: {r_value:+.2f}]" if r_value is not None else "")
                ))
                
                # Clear trade state
                position_side = "flat"
                position_size = 0
                entry_price = 0.0
                entry_time = None
                entry_stop_price = 0.0
                entry_risk_amount = 0.0
                current_trade_id = None  # Clear trade ID
            
            pending_action = None
            pending_signal_idx = None
        
        # Generate action using appropriate strategy
        action, raw_info = generate_action_for_bar(config, df, i)
        
        # Calculate current equity (mark-to-market if in position)
        if position_side == "long":
            unrealized_pnl = (close_price - entry_price) * position_size
            current_equity = equity + unrealized_pnl
        else:
            current_equity = equity
        
        # Update peak equity
        if current_equity > peak_equity:
            peak_equity = current_equity
        
        # Check max drawdown
        current_dd = 0.0
        if peak_equity > 0:
            current_dd = (peak_equity - current_equity) / peak_equity
        
        if config.max_drawdown_percent is not None and current_dd >= config.max_drawdown_percent:
            simulation_halted = True
            halt_reason = f"Max drawdown reached: {current_dd*100:.2f}% >= {config.max_drawdown_percent*100:.2f}%"
            
            decision_log.add(DecisionEvent(
                timestamp=bar_time,
                symbol=config.symbol,
                timeframe=config.timeframe,
                event_type="halt",
                position_side=position_side,
                price=close_price,
                equity_before=current_equity,
                equity_after=current_equity,
                current_drawdown=current_dd,
                halt_reason=halt_reason,
                reason=halt_reason
            ))
            break
        
        # Check max daily loss (R)
        trading_allowed_today = True
        if config.max_daily_loss_r is not None and bar_date in daily_r_tracker:
            if daily_r_tracker[bar_date] <= -config.max_daily_loss_r:
                trading_allowed_today = False
                
                decision_log.add(DecisionEvent(
                    timestamp=bar_time,
                    symbol=config.symbol,
                    timeframe=config.timeframe,
                    event_type="halt",
                    position_side=position_side,
                    price=close_price,
                    equity_before=current_equity,
                    equity_after=current_equity,
                    daily_r_loss=daily_r_tracker[bar_date],
                    halt_reason=f"max_daily_loss_reached: {daily_r_tracker[bar_date]:.2f}R <= -{config.max_daily_loss_r:.2f}R",
                    reason=f"Trading halted for the day: {daily_r_tracker[bar_date]:.2f}R loss (limit: -{config.max_daily_loss_r:.2f}R)"
                ))
        
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
            atr_value=current_atr if config.use_r_management else None,
            current_drawdown=current_dd if config.max_drawdown_percent else None,
            daily_r_loss=daily_r_tracker.get(bar_date) if config.max_daily_loss_r else None,
            equity_before=current_equity,
            equity_after=current_equity,
            risk_per_trade=config.risk_per_trade,
            trade_id=current_trade_id,  # Link to current trade if in position
            reason=reason
        ))
        
        # Skip trading if daily loss limit reached (but allow exits)
        if not trading_allowed_today and action == "buy":
            action = "hold"
        
        # Trading Logic
        equity_before = current_equity
        
        if config.execution_mode == "same_bar_close":
            # Execute on same bar close
            if action == "buy" and position_side == "flat":
                # Enter Long
                raw_price = close_price
                exec_price, commission, slippage = calculate_execution_price(raw_price, True, config)
                
                position_size = calculate_position_size(equity, exec_price, config)
                
                if position_size > 0:
                    position_side = "long"
                    entry_price = exec_price
                    entry_time = bar_time
                    
                    # Assign trade ID
                    current_trade_id = next_trade_id
                    next_trade_id += 1
                    
                    # R-management
                    if config.use_r_management:
                        entry_atr = current_atr
                        entry_stop_price = calculate_virtual_stop(entry_price, entry_atr, config)
                        entry_risk_amount = (entry_price - entry_stop_price) * position_size
                    
                    # Build sizing info
                    if config.position_sizing_mode == "percent_of_equity":
                        sizing_info = f"({config.risk_per_trade*100:.1f}% of equity)"
                    elif config.position_sizing_mode == "full_equity":
                        sizing_info = "(full equity)"
                    elif config.position_sizing_mode == "fixed_shares":
                        sizing_info = f"(fixed {config.fixed_shares} shares)"
                    else:
                        sizing_info = f"(fixed ${config.fixed_dollar_amount:.0f})"
                    
                    decision_log.add(DecisionEvent(
                        timestamp=bar_time,
                        symbol=config.symbol,
                        timeframe=config.timeframe,
                        event_type="entry",
                        final_signal=action,
                        position_side="long",
                        position_size=position_size,
                        price=entry_price,
                        execution_price=exec_price,
                        execution_mode="same_bar_close",
                        commission=commission,
                        slippage=slippage,
                        atr_value=entry_atr if config.use_r_management else None,
                        stop_price=entry_stop_price if config.use_r_management else None,
                        risk_amount=entry_risk_amount if config.use_r_management else None,
                        equity_before=equity_before,
                        equity_after=equity_before,
                        risk_per_trade=config.risk_per_trade,
                        trade_id=current_trade_id,
                        reason=f"Enter LONG @ ${entry_price:.2f}, size={position_size} shares {sizing_info}"
                    ))
            
            elif action == "sell" and position_side == "long":
                # Exit Long
                raw_price = close_price
                exec_price, commission, slippage = calculate_execution_price(raw_price, False, config)
                
                pnl = (exec_price - entry_price) * position_size
                
                # Calculate R-value
                r_value = None
                if config.use_r_management and entry_risk_amount > 0:
                    r_value = pnl / entry_risk_amount
                    
                    # Track daily R
                    if bar_date not in daily_r_tracker:
                        daily_r_tracker[bar_date] = 0.0
                    daily_r_tracker[bar_date] += r_value
                
                equity += pnl
                
                trades.append({
                    "trade_id": current_trade_id,
                    "entry_time": entry_time.isoformat() if hasattr(entry_time, 'isoformat') else str(entry_time),
                    "exit_time": bar_time.isoformat() if hasattr(bar_time, 'isoformat') else str(bar_time),
                    "entry_price": entry_price,
                    "exit_price": exec_price,
                    "size": position_size,
                    "pnl": round(pnl, 2),
                    "return_pct": round((exec_price / entry_price - 1) * 100, 2),
                    "r_value": round(r_value, 2) if r_value is not None else None,
                    "stop_price": entry_stop_price if config.use_r_management else None,
                    "risk_amount": round(entry_risk_amount, 2) if config.use_r_management else None,
                    "execution_mode": "same_bar_close"
                })
                
                decision_log.add(DecisionEvent(
                    timestamp=bar_time,
                    symbol=config.symbol,
                    timeframe=config.timeframe,
                    event_type="exit",
                    final_signal=action,
                    position_side="flat",
                    position_size=position_size,
                    price=exec_price,
                    execution_price=exec_price,
                    execution_mode="same_bar_close",
                    commission=commission,
                    slippage=slippage,
                    r_value=r_value,
                    stop_price=entry_stop_price if config.use_r_management else None,
                    risk_amount=entry_risk_amount if config.use_r_management else None,
                    equity_before=equity_before,
                    equity_after=equity,
                    risk_per_trade=config.risk_per_trade,
                    daily_r_loss=daily_r_tracker.get(bar_date),
                    trade_id=current_trade_id,
                    reason=f"Exit LONG @ ${exec_price:.2f}. PnL: ${pnl:+.2f} ({(exec_price/entry_price-1)*100:+.2f}%)"
                          + (f" [R: {r_value:+.2f}]" if r_value is not None else "")
                ))
                
                # Clear trade state
                position_side = "flat"
                position_size = 0
                entry_price = 0.0
                entry_time = None
                entry_stop_price = 0.0
                entry_risk_amount = 0.0
                current_trade_id = None  # Clear trade ID
        
        else:  # next_bar_open
            # Queue action for next bar
            if action in ["buy", "sell"]:
                pending_action = action
                pending_signal_idx = i
        
        # Record equity curve
        if position_side == "long":
            unrealized_pnl = (close_price - entry_price) * position_size
            curve_equity = equity + unrealized_pnl
        else:
            curve_equity = equity
        
        equity_curve.append({
            "timestamp": bar_time.isoformat() if hasattr(bar_time, 'isoformat') else str(bar_time),
            "equity": round(curve_equity, 2),
            "position": position_side,
            "drawdown": round(current_dd * 100, 2) if peak_equity > 0 else 0.0
        })
    
    # 4. Close any open position at the end
    if position_side == "long" and not simulation_halted:
        exit_price = float(df.iloc[-1]["close"])
        exit_time_ts = df.iloc[-1]["timestamp"]
        pnl = (exit_price - entry_price) * position_size
        
        r_value = None
        if config.use_r_management and entry_risk_amount > 0:
            r_value = pnl / entry_risk_amount
        
        equity_before = equity
        equity += pnl
        
        trades.append({
            "trade_id": current_trade_id,
            "entry_time": entry_time.isoformat() if hasattr(entry_time, 'isoformat') else str(entry_time),
            "exit_time": exit_time_ts.isoformat() if hasattr(exit_time_ts, 'isoformat') else str(exit_time_ts),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": position_size,
            "pnl": round(pnl, 2),
            "return_pct": round((exit_price / entry_price - 1) * 100, 2),
            "r_value": round(r_value, 2) if r_value is not None else None,
            "stop_price": entry_stop_price if config.use_r_management else None,
            "risk_amount": round(entry_risk_amount, 2) if config.use_r_management else None,
            "execution_mode": config.execution_mode,
            "note": "Closed at end of simulation"
        })
        
        # Add exit event to decision log
        decision_log.add(DecisionEvent(
            timestamp=exit_time_ts,
            symbol=config.symbol,
            timeframe=config.timeframe,
            event_type="exit",
            final_signal="sell",
            position_side="flat",
            position_size=position_size,
            price=exit_price,
            execution_price=exit_price,
            execution_mode=config.execution_mode,
            r_value=r_value,
            stop_price=entry_stop_price if config.use_r_management else None,
            risk_amount=entry_risk_amount if config.use_r_management else None,
            equity_before=equity_before,
            equity_after=equity,
            trade_id=current_trade_id,
            reason=f"Exit LONG @ ${exit_price:.2f} (end of simulation). PnL: ${pnl:+.2f}"
                  + (f" [R: {r_value:+.2f}]" if r_value is not None else "")
        ))
    
    # 5. Calculate Summary Stats
    final_equity = equity
    total_return_pct = (final_equity / config.initial_capital - 1) * 100
    
    total_trades = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    
    total_pnl = sum(t["pnl"] for t in trades)
    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
    
    # R-based stats
    r_values = [t.get("r_value") for t in trades if t.get("r_value") is not None]
    total_r = sum(r_values) if r_values else None
    avg_r = np.mean(r_values) if r_values else None
    
    summary = {
        "strategy_mode": config.strategy_mode,
        "position_sizing_mode": config.position_sizing_mode,
        "execution_mode": config.execution_mode,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "best_trade": round(max((t["pnl"] for t in trades), default=0), 2),
        "worst_trade": round(min((t["pnl"] for t in trades), default=0), 2),
        "bars_analyzed": len(df) - warmup,
        "decision_events": len(decision_log),
        "simulation_halted": simulation_halted,
        "halt_reason": halt_reason
    }
    
    # Add R stats if R-management enabled
    if config.use_r_management and r_values:
        summary["total_r"] = round(total_r, 2)
        summary["avg_r"] = round(avg_r, 2)
        summary["best_r"] = round(max(r_values), 2)
        summary["worst_r"] = round(min(r_values), 2)
    
    # Add MA params to summary if using MA mode
    if config.strategy_mode == "ma_crossover":
        summary["ma_short_window"] = config.ma_short_window
        summary["ma_long_window"] = config.ma_long_window
    
    # Compute R Analytics (only if R management enabled and we have trades with R values)
    r_analytics_dict = None
    if config.use_r_management:
        r_analytics = compute_r_analytics(trades)
        if r_analytics:
            r_analytics_dict = r_analytics.to_dict()
    
    return AutoSimResult(
        symbol=config.symbol,
        timeframe=config.timeframe,
        initial_capital=config.initial_capital,
        final_equity=round(final_equity, 2),
        total_return_pct=round(total_return_pct, 2),
        equity_curve=equity_curve,
        trades=trades,
        decision_log=decision_log.to_list(),
        summary=summary,
        r_analytics=r_analytics_dict
    )
