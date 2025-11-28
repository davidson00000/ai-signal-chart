"""
Strategy (Brain) Layer - Signal generation
Migrated from main_legacy.py
"""
from typing import List, Dict, Any, Optional, Literal
from backend.utils.indicators import simple_moving_average


def generate_signals_and_trades(
    candles: List[Dict[str, Any]],
    short_window: int,
    long_window: int,
    tp_ratio: float,
    sl_ratio: float,
) -> Dict[str, Any]:
    """
    Generate MA cross signals and simulated trades
    
    This function is migrated from main_legacy.py without logic changes.
    It implements:
    - Golden cross / Dead cross detection
    - TP/SL exit conditions
    - Reverse signal exit
    - Trade simulation with P&L calculation
    
    Args:
        candles: List of candle dictionaries with OHLCV data
        short_window: Short MA period
        long_window: Long MA period
        tp_ratio: Take profit ratio (e.g., 0.01 = 1%)
        sl_ratio: Stop loss ratio (e.g., 0.005 = 0.5%)
        
    Returns:
        Dictionary containing:
        - shortMA: List of short MA values
        - longMA: List of long MA values
        - signals: List of signal dictionaries
        - trades: List of trade dictionaries
        - stats: Trade statistics
    """
    closes = [c["close"] for c in candles]
    short_ma = simple_moving_average(closes, short_window)
    long_ma = simple_moving_average(closes, long_window)

    signals: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []

    position: Literal["long", "short", "flat"] = "flat"
    entry_price: Optional[float] = None
    entry_time: Optional[int] = None
    entry_index: Optional[int] = None
    trade_id = 0

    for i in range(1, len(candles)):
        c_prev = candles[i - 1]
        c = candles[i]
        ma_s_prev = short_ma[i - 1]
        ma_l_prev = long_ma[i - 1]
        ma_s = short_ma[i]
        ma_l = long_ma[i]

        # Skip if MA not yet calculated
        if (
            ma_s_prev is None
            or ma_l_prev is None
            or ma_s is None
            or ma_l is None
        ):
            continue

        # Detect crossovers
        # Golden cross: short MA crosses above long MA → BUY
        golden_cross = ma_s_prev <= ma_l_prev and ma_s > ma_l
        # Dead cross: short MA crosses below long MA → SELL
        dead_cross = ma_s_prev >= ma_l_prev and ma_s < ma_l

        # --- Entry Signals ---
        if position == "flat":
            if golden_cross:
                position = "long"
                entry_price = c["close"]
                entry_time = c["time"]
                entry_index = i
                trade_id += 1

                tp = entry_price * (1 + tp_ratio)
                sl = entry_price * (1 - sl_ratio)

                signals.append(
                    {
                        "id": trade_id,
                        "side": "BUY",
                        "time": entry_time,
                        "price": entry_price,
                        "tp": tp,
                        "sl": sl,
                        "index": i,
                    }
                )

            elif dead_cross:
                position = "short"
                entry_price = c["close"]
                entry_time = c["time"]
                entry_index = i
                trade_id += 1

                tp = entry_price * (1 - tp_ratio)
                sl = entry_price * (1 + sl_ratio)

                signals.append(
                    {
                        "id": trade_id,
                        "side": "SELL",
                        "time": entry_time,
                        "price": entry_price,
                        "tp": tp,
                        "sl": sl,
                        "index": i,
                    }
                )

        # --- Exit Logic ---
        else:
            assert entry_price is not None
            assert entry_time is not None
            assert entry_index is not None

            high = c["high"]
            low = c["low"]

            if position == "long":
                tp_price = entry_price * (1 + tp_ratio)
                sl_price = entry_price * (1 - sl_ratio)

                exit_reason = None
                exit_price: Optional[float] = None

                if low <= sl_price:
                    exit_reason = "SL"
                    exit_price = sl_price
                elif high >= tp_price:
                    exit_reason = "TP"
                    exit_price = tp_price
                elif dead_cross:
                    exit_reason = "Reverse"
                    exit_price = c["close"]

                if exit_reason is not None and exit_price is not None:
                    pnl = (exit_price - entry_price) / entry_price

                    trades.append(
                        {
                            "id": trade_id,
                            "side": "LONG",
                            "entry_time": entry_time,
                            "entry_price": entry_price,
                            "exit_time": c["time"],
                            "exit_price": exit_price,
                            "exit_reason": exit_reason,
                            "pnl": pnl,
                        }
                    )

                    position = "flat"
                    entry_price = None
                    entry_time = None
                    entry_index = None

            elif position == "short":
                tp_price = entry_price * (1 - tp_ratio)
                sl_price = entry_price * (1 + sl_ratio)

                exit_reason = None
                exit_price = None

                if high >= sl_price:
                    exit_reason = "SL"
                    exit_price = sl_price
                elif low <= tp_price:
                    exit_reason = "TP"
                    exit_price = tp_price
                elif golden_cross:
                    exit_reason = "Reverse"
                    exit_price = c["close"]

                if exit_reason is not None and exit_price is not None:
                    pnl = (entry_price - exit_price) / entry_price

                    trades.append(
                        {
                            "id": trade_id,
                            "side": "SHORT",
                            "entry_time": entry_time,
                            "entry_price": entry_price,
                            "exit_time": c["time"],
                            "exit_price": exit_price,
                            "exit_reason": exit_reason,
                            "pnl": pnl,
                        }
                    )

                    position = "flat"
                    entry_price = None
                    entry_time = None
                    entry_index = None

    # Calculate statistics
    total_trades = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    total_pnl = sum(t["pnl"] for t in trades)
    total_pnl_percent = total_pnl * 100.0

    stats = {
        "tradeCount": total_trades,
        "winRate": win_rate,
        "pnlPercent": total_pnl_percent,
    }

    return {
        "shortMA": short_ma,
        "longMA": long_ma,
        "signals": signals,
        "trades": trades,
        "stats": stats,
    }


def generate_signal(
    symbol: str,
    candles: List[Dict[str, Any]],
    strategy: str = "ma_cross",
    date: Optional[str] = None,
    timeframe: str = "1d",
    **params
) -> Dict[str, Any]:
    """
    Generate a single signal for API_SPEC.md /signal endpoint
    
    Args:
        symbol: Symbol string
        candles: List of candle data
        strategy: Strategy name (default: "ma_cross")
        date: Optional date for signal (defaults to latest)
        timeframe: Timeframe string
        **params: Strategy-specific parameters
        
    Returns:
        Signal dictionary per API_SPEC.md format
    """
    if not candles:
        return {
            "symbol": symbol,
            "date": date,
            "timeframe": timeframe,
            "strategy": strategy,
            "signal": "HOLD",
            "confidence": 0.0,
            "reason": "No data available",
            "price": 0.0,
        }
    
    # Use MA cross strategy by default
    short_window = params.get("short_window", 9)
    long_window = params.get("long_window", 21)
    
    result = generate_signals_and_trades(
        candles,
        short_window=short_window,
        long_window=long_window,
        tp_ratio=params.get("tp_ratio", 0.01),
        sl_ratio=params.get("sl_ratio", 0.005),
    )
    
    # Get the latest signal if any
    signals = result["signals"]
    if signals:
        latest_signal = signals[-1]
        return {
            "symbol": symbol,
            "date": date,
            "timeframe": timeframe,
            "strategy": strategy,
            "signal": latest_signal["side"],
            "confidence": 0.7,  # MA cross has moderate confidence
            "reason": f"MA cross signal: {latest_signal['side']}",
            "price": latest_signal["price"],
            "meta": {
                "short_ma": result["shortMA"][-1] if result["shortMA"] else None,
                "long_ma": result["longMA"][-1] if result["longMA"] else None,
                "tp": latest_signal.get("tp"),
                "sl": latest_signal.get("sl"),
            }
        }
    else:
        # No signal, return HOLD
        latest_candle = candles[-1]
        return {
            "symbol": symbol,
            "date": date,
            "timeframe": timeframe,
            "strategy": strategy,
            "signal": "HOLD",
            "confidence": 0.5,
            "reason": "No MA cross detected",
            "price": latest_candle["close"],
            "meta": {
                "short_ma": result["shortMA"][-1] if result["shortMA"] else None,
                "long_ma": result["longMA"][-1] if result["longMA"] else None,
            }
        }
