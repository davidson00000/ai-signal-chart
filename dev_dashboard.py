"""
EXITON Developer Dashboard
--------------------------

Streamlit-based internal dashboard for monitoring the paper-trading engine
and visualizing strategy behavior.

This is a *mock* / developer-facing dashboard, not the main user UI.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import plotly.graph_objects as go
import requests
import altair as alt
from datetime import datetime, timedelta
import uuid
import time
from strategy_guides import STRATEGY_GUIDES

# ============================================================================
# Global Constants
# ============================================================================

STRATEGY_TEMPLATES = {
    "ma_cross": {
        "label": "MA Cross",
        "supports_optimization": True,
        "usable": True,
    },
    "ema_cross": {
        "label": "EMA Cross",
        "supports_optimization": False,
        "usable": False,
    },
    "macd_trend": {
        "label": "MACD Trend",
        "supports_optimization": False,
        "usable": False,
    },
    "rsi_mean_reversion": {
        "label": "RSI Mean Reversion",
        "supports_optimization": True,
        "usable": True,
    },
    "stoch_oscillator": {
        "label": "Stochastic Oscillator",
        "supports_optimization": False,
        "usable": False,
    },
    "bollinger_mean_reversion": {
        "label": "Bollinger Mean Reversion",
        "supports_optimization": False,
        "usable": False,
    },
    "bollinger_breakout": {
        "label": "Bollinger Breakout",
        "supports_optimization": False,
        "usable": False,
    },
    "donchian_breakout": {
        "label": "Donchian Breakout",
        "supports_optimization": False,
        "usable": False,
    },
    "atr_trailing_stop": {
        "label": "ATR Trailing Stop",
        "supports_optimization": False,
        "usable": False,
    },
    "price_breakout": {
        "label": "Price Breakout",
        "supports_optimization": False,
        "usable": False,
    },
    "roc_momentum": {
        "label": "ROC Momentum",
        "supports_optimization": False,
        "usable": False,
    },
    "ema9_dip_buy": {
        "label": "EMA9 Dip Buy",
        "supports_optimization": True,
        "usable": True,
    },
}
import os
import json
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from backend.strategies import (
    MACrossStrategy, EMACrossStrategy, MACDTrendStrategy,
    RSIMeanReversionStrategy, StochasticOscillatorStrategy,
    BollingerMeanReversionStrategy, BollingerBreakoutStrategy,
    DonchianBreakoutStrategy, ATRTrailingMAStrategy, ROCMomentumStrategy,
    EMA9DipBuyStrategy
)

# ============================================================================
# URL Query Parameter Handling (Centralized State)
# ============================================================================

# Parse URL parameters at the very start
# We use st.query_params (Streamlit 1.30+) which behaves like a dict
query_params = st.query_params

mode_from_q = query_params.get("mode")
symbol_from_q = query_params.get("symbol")
timeframe_from_q = query_params.get("timeframe")
lookback_from_q = query_params.get("lookback")
universe_from_q = query_params.get("universe")

# Initialize session_state from query (only if not already set)
# This ensures that if a user bookmarks a URL, the app loads in the correct state.
if mode_from_q and "mode" not in st.session_state:
    st.session_state["mode"] = (
        "Market Scanner" if mode_from_q == "scanner"
        else "Live Signal" if mode_from_q == "live_signal"
        else "Developer Dashboard"
    )

if symbol_from_q and "symbol" not in st.session_state:
    st.session_state["symbol"] = symbol_from_q

if timeframe_from_q and "timeframe" not in st.session_state:
    st.session_state["timeframe"] = timeframe_from_q

if lookback_from_q and "lookback" not in st.session_state:
    try:
        st.session_state["lookback"] = int(lookback_from_q)
    except ValueError:
        pass

if universe_from_q and "universe" not in st.session_state:
    st.session_state["universe"] = universe_from_q

# ============================================================================
# Helpers
# ============================================================================

def ensure_paper_account(base_url: str, initial_equity: float = 100000.0) -> Optional[dict]:
    """
    Ensure that a default paper account exists.
    If it doesn't, create it with the given initial equity.
    Return the account dict on success, or None on failure.
    """
    try:
        # Try to fetch existing account
        resp = requests.get(f"{base_url}/paper/accounts/default", timeout=10)
        if resp.status_code == 200:
            return resp.json()

        # If not found, create a new account
        if resp.status_code == 404:
            create_resp = requests.post(
                f"{base_url}/paper/accounts",
                json={"account_id": "default", "initial_equity": initial_equity},
                timeout=10,
            )
            if create_resp.status_code == 200:
                return create_resp.json()

        return None
    except Exception as e:
        print(f"[PaperTrading] ensure_paper_account failed: {e}")
        return None

def render_predictor_card(title, pred):
    direction = pred.get("direction", "flat").upper()
    score = pred.get("score", 0.0)
    prob_up = pred.get("prob_up")
    prob_down = pred.get("prob_down")
    signals = pred.get("signals") or pred.get("details") # Handle both keys just in case
    
    # Icons & Colors
    icon = "➖"
    color_hex = "#7f8c8d" # Gray
    if direction == "UP": 
        icon = "📈"
        color_hex = "#2ecc71" # Green
    elif direction == "DOWN": 
        icon = "📉"
        color_hex = "#e74c3c" # Red
    
    # Probability Bars
    bars_html = ""
    if prob_up is not None and prob_down is not None:
        bars_html = f"""
<div style="margin-top: 10px; font-size: 12px; color: #ccc;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
        <span>Prob Up: {prob_up:.2f}</span>
        <span>Prob Down: {prob_down:.2f}</span>
    </div>
    <div style="display: flex; height: 6px; width: 100%; border-radius: 3px; overflow: hidden; background: #444;">
        <div style="width: {prob_up*100}%; background-color: #2ecc71;"></div>
        <div style="width: {prob_down*100}%; background-color: #e74c3c;"></div>
    </div>
</div>
"""
    else:
        # Fallback to single score bar
        bars_html = f"""
<div style="width: 100%; background-color: #444; height: 6px; border-radius: 3px; margin-top: 10px;">
    <div style="width: {abs(score)*100}%; background-color: {color_hex}; height: 6px; border-radius: 3px;"></div>
</div>
"""

    # Signals Breakdown (for Rule Predictor v2)
    signals_html = ""
    if signals:
        raw_signals = pred.get("raw_signals") or {}
        rows = ""
        for k, v in signals.items():
            s_color = "#7f8c8d"
            s_icon = "•"
            if v == 1: s_color = "#2ecc71"; s_icon = "▲"
            elif v == -1: s_color = "#e74c3c"; s_icon = "▼"
            
            # Tooltip content (continuous value)
            raw_val = raw_signals.get(k, 0.0)
            tooltip = f"{k.upper()}: {raw_val:.2f}"
            
            rows += f"""
<div style="display: flex; flex-direction: column; align-items: center; margin: 0 5px;" title="{tooltip}">
    <span style="font-size: 10px; color: #aaa;">{k.upper()}</span>
    <span style="font-size: 14px; color: {s_color}; font-weight: bold; cursor: help;">{s_icon}</span>
</div>
"""
        signals_html = f"""
<div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); display: flex; justify-content: center;">
    {rows}
</div>
"""

    # Card Style
    st.markdown(
        f"""
<div style="
    padding: 15px; 
    border-radius: 10px; 
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%); 
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 10px;
">
    <h4 style="margin:0; color: #aaa; font-size: 14px;">{title}</h4>
    <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 10px;">
        <span style="font-size: 24px;">{icon} {direction}</span>
        <span style="font-size: 18px; font-weight: bold; color: {color_hex};">{score:.2f}</span>
    </div>
    {bars_html}
    {signals_html}
</div>
""",
        unsafe_allow_html=True
    )

def render_timeframe_guidance(current_timeframe: str, mode: str = "live_signal"):
    """
    Render timeframe guidance for swing trading.
    mode: "live_signal" or "dashboard"
    """
    is_recommended = (current_timeframe == "1d")
    
    if mode == "live_signal":
        # Guidance box
        st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(52, 152, 219, 0.1) 0%, rgba(52, 152, 219, 0.05) 100%);
    border: 1px solid rgba(52, 152, 219, 0.3);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 15px;
">
    <div style="font-weight: 600; color: #3498db; margin-bottom: 8px;">
        📊 Recommended Timeframe (Swing Trading)
    </div>
    <div style="font-size: 13px; color: #bbb; line-height: 1.5;">
        <strong>Primary:</strong> 1d (daily) — for direction & signals<br>
        <strong>Secondary:</strong> 4h — for entry/exit fine-tuning
    </div>
</div>
""", unsafe_allow_html=True)
        
        # Timeframe hint based on selection
        if is_recommended:
            st.success("✅ **1d (Daily)** is the recommended timeframe for swing trading.")
        elif current_timeframe == "4h":
            st.info("ℹ️ **4h** is useful for entry/exit timing. Use with 1d for direction.")
        else:
            st.warning(f"⚠️ **{current_timeframe}** may be noisy for swing trading. Consider using **1d** for primary signals.")
    
    elif mode == "dashboard":
        # Compact tooltip-style hint for dashboard
        st.markdown("""
<div style="
    background: rgba(52, 152, 219, 0.08);
    border-left: 3px solid #3498db;
    padding: 8px 12px;
    margin-top: 10px;
    font-size: 12px;
    color: #aaa;
    border-radius: 0 4px 4px 0;
">
    <span style="color: #3498db; font-weight: 600;">ℹ️ Note:</span> 
    Rule Predictor v2 is optimized for <strong>daily (1d)</strong> candles and 
    <strong>swing trades</strong> (few days ~ few weeks). Intraday signals may be noisier.
</div>
""", unsafe_allow_html=True)

def apply_preset_callback(preset_params, strategy_type):
    """Callback to apply preset parameters to session state"""
    for p_key, p_val in preset_params.items():
        ss_key = None
        if strategy_type == "ma_cross":
            if p_key == "short_window": ss_key = "sl_ma_short"
            if p_key == "long_window": ss_key = "sl_ma_long"
        elif strategy_type == "rsi_mean_reversion":
            if p_key == "rsi_period": ss_key = "sl_rsi_period"
            if p_key == "oversold": ss_key = "sl_rsi_oversold"
            if p_key == "overbought": ss_key = "sl_rsi_overbought"
        elif strategy_type == "ema9_dip_buy":
            if p_key == "ema_fast": ss_key = "sl_ema9_fast"
            if p_key == "ema_slow": ss_key = "sl_ema9_slow"
            if p_key == "deviation_threshold": ss_key = "sl_ema9_dev"
            if p_key == "stop_buffer": ss_key = "sl_ema9_stop"
            if p_key == "risk_reward": ss_key = "sl_ema9_rr"
            if p_key == "lookback_volume": ss_key = "sl_ema9_vol"
        
        if ss_key:
            st.session_state[ss_key] = p_val
    
    st.session_state["preset_message"] = f"Preset applied successfully!"


# =============================================================================
# Configuration
# =============================================================================
BACKEND_URL = "http://localhost:8001"

# =============================================================================
# Universe Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
UNIVERSE_MODE = "mvp"  # 'mvp' or 'sp500'

if UNIVERSE_MODE == "mvp":
    UNIVERSE_CSV = PROJECT_ROOT / "tools" / "symbols_universe_mvp.csv"
elif UNIVERSE_MODE == "sp500":
    UNIVERSE_CSV = PROJECT_ROOT / "tools" / "symbols_universe_sp500.csv"
else:
    UNIVERSE_CSV = PROJECT_ROOT / "tools" / "symbols_universe.csv"

@st.cache_data
def load_symbol_universe(csv_path: Path) -> List[str]:
    """
    Load symbol list from the specified CSV file.
    Expected columns: 'symbol', 'note' (optional)
    """
    if not csv_path.exists():
        st.error(f"Universe CSV not found: {csv_path}")
        return []
    
    try:
        df = pd.read_csv(csv_path)
        if "symbol" not in df.columns:
            st.error(f"'symbol' column not found in {csv_path}")
            return []
        
        symbols = df["symbol"].astype(str).str.strip().tolist()
        return symbols
    except Exception as e:
        st.error(f"Failed to load symbol universe: {e}")
        return []


# =============================================================================
# Symbol Preset Management
# =============================================================================

SYMBOL_PRESET_PATH = Path("data/symbol_presets.json")

def load_symbol_presets() -> List[Dict[str, str]]:
    """Load symbol presets from JSON file."""
    if not SYMBOL_PRESET_PATH.exists():
        # Return default if file doesn't exist
        return [
            {"symbol": "AAPL", "label": "Apple"},
            {"symbol": "MSFT", "label": "Microsoft"},
            {"symbol": "TSLA", "label": "Tesla"},
        ]
    try:
        with SYMBOL_PRESET_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("symbols", [])
    except Exception as e:
        st.error(f"Failed to load symbol presets: {e}")
        return [{"symbol": "AAPL", "label": "Apple"}]

def save_symbol_presets(symbols: List[Dict[str, str]]) -> None:
    """Save symbol presets to JSON file."""
    try:
        SYMBOL_PRESET_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SYMBOL_PRESET_PATH.open("w", encoding="utf-8") as f:
            json.dump({"symbols": symbols}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed to save symbol presets: {e}")


# =============================================================================
# Mock data helpers
# =============================================================================


@dataclass
class MockPosition:
    symbol: str
    side: str  # "LONG" or "SHORT"
    quantity: float
    entry_price: float
    current_price: float

    @property
    def pnl(self) -> float:
        if self.side == "LONG":
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity


def generate_mock_price_data(
    periods: int = 200,
    start_price: float = 100.0,
    volatility: float = 0.02,
) -> pd.DataFrame:
    """
    Generate simple random-walk OHLCV data for demo purposes.
    """
    rng = np.random.default_rng(42)

    rets = rng.normal(loc=0.0005, scale=volatility, size=periods)
    price = start_price * np.exp(np.cumsum(rets))

    dates = pd.date_range(end=datetime.today(), periods=periods, freq="D")
    df = pd.DataFrame(index=dates)

    df["close"] = price
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + rng.normal(0.001, 0.005, size=periods))
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - rng.normal(0.001, 0.005, size=periods))
    df["volume"] = rng.integers(100_000, 1_000_000, size=periods)

    return df


def compute_mock_ma_signals(df: pd.DataFrame, short_window: int = 9, long_window: int = 21) -> pd.DataFrame:
    """
    Compute simple moving-average cross signals for visualization.

    1 = long, 0 = flat
    """
    df = df.copy()
    df["ma_short"] = df["close"].rolling(short_window, min_periods=1).mean()
    df["ma_long"] = df["close"].rolling(long_window, min_periods=1).mean()

    signal = np.where(df["ma_short"] > df["ma_long"], 1, 0)
    df["position"] = signal.astype(int)
    return df


def generate_mock_positions(symbol: str, price: float) -> List[MockPosition]:
    """
    Generate some fake open positions for the sidebar / table.
    """
    positions: List[MockPosition] = [
        MockPosition(symbol=symbol, side="LONG", quantity=100, entry_price=price * 0.9, current_price=price),
        MockPosition(symbol=symbol, side="SHORT", quantity=50, entry_price=price * 1.1, current_price=price),
    ]
    return positions


def generate_mock_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a simple trades table from the position signal.

    This is just for visual demo, NOT the real backtest results.
    """
    df_sig = compute_mock_ma_signals(df)
    df_sig["signal_change"] = df_sig["position"].diff().fillna(0)

    trades = []
    for ts, row in df_sig.iterrows():
        if row["signal_change"] == 1:  # BUY
            trades.append(
                {
                    "time": ts,
                    "side": "BUY",
                    "price": float(row["close"]),
                    "quantity": 100,
                    "commission": float(row["close"]) * 100 * 0.0005,
                    "pnl": 0.0,
                }
            )
        elif row["signal_change"] == -1:  # SELL
            trades.append(
                {
                    "time": ts,
                    "side": "SELL",
                    "price": float(row["close"]),
                    "quantity": 100,
                    "commission": float(row["close"]) * 100 * 0.0005,
                    "pnl": float(row["close"]) * 5,  # テキトー
                }
            )

    if not trades:
        return pd.DataFrame(columns=["time", "side", "price", "quantity", "commission", "pnl"])

    trades_df = pd.DataFrame(trades)
    trades_df.set_index("time", inplace=True)
    return trades_df


def generate_mock_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a simple equity curve from MA position.
    """
    df_sig = compute_mock_ma_signals(df)
    df_sig["daily_ret"] = df_sig["close"].pct_change().fillna(0.0)
    df_sig["strategy_ret"] = df_sig["daily_ret"] * df_sig["position"]
    df_sig["equity"] = (1 + df_sig["strategy_ret"]).cumprod() * 1_000_000
    df_sig["cash"] = 1_000_000  # モックなので固定
    return df_sig[["equity", "cash"]]


# =============================================================================
# Sidebar & layout helpers
# =============================================================================


def render_sidebar() -> Tuple[str, str, int, int, str, bool]:
    """
    Render sidebar controls and return selected values.

    Returns:
        (symbol, timeframe, limit, quantity, ma_type, refresh_flag)
    """
    st.sidebar.title("Controls")

    symbol = render_symbol_selector(key_prefix="dev", container=st.sidebar)
    timeframe = st.sidebar.selectbox("Timeframe", options=["1day", "4h", "1h"], index=0)
    limit = st.sidebar.slider("Lookback periods", min_value=50, max_value=500, value=200, step=10)
    quantity = st.sidebar.number_input("Default Quantity", min_value=1, max_value=10_000, value=100, step=10)

    st.sidebar.markdown("---")
    ma_type = st.sidebar.selectbox("MA Type (for chart)", options=["SMA", "EMA"], index=0)

    st.sidebar.markdown("---")
    refresh = st.sidebar.button("🔄 Refresh data")

    return symbol, timeframe, limit, quantity, ma_type, refresh


# =============================================================================
# Main page sections
# =============================================================================


def render_main_chart(df: pd.DataFrame, ma_type: str = "SMA"):
    st.subheader("📈 Price & MA Signals (Mock)")
    if df is None or df.empty:
        st.warning("No chart data available.")
        return

    df = compute_mock_ma_signals(df)
    chart_df = df[["close", "ma_short", "ma_long"]].copy()
    chart_df.columns = ["Close", "MA Short", "MA Long"]

    st.line_chart(chart_df)


def render_ma_signals(df: pd.DataFrame, selected_ma: str):
    st.subheader("⚙️ Strategy Signals (Demo)")
    if df is None or df.empty:
        st.info("Signals will appear once price data is loaded.")
        return

    df_sig = compute_mock_ma_signals(df)

    latest = df_sig.iloc[-1]
    st.metric("Latest Close", f"${latest['close']:.2f}")
    st.metric("MA Short", f"{latest['ma_short']:.2f}")
    st.metric("MA Long", f"{latest['ma_long']:.2f}")

    st.caption(f"MA type: {selected_ma}  |  Short={9}, Long={21} (hard-coded demo)")


def render_account_summary():
    st.subheader("💼 Account Summary (Mock)")
    df = st.session_state.get("chart_data")
    if df is None or df.empty:
        st.info("Account metrics will appear when price data is loaded.")
        return

    pnl_df = generate_mock_pnl(df)
    final_equity = pnl_df["equity"].iloc[-1]
    total_return = final_equity / 1_000_000 - 1.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Equity", f"¥{final_equity:,.0f}")
    col2.metric("Return", f"{total_return * 100:.2f}%")
    col3.metric("Max Drawdown", "-12.34%")  # モック値


def render_risk_metrics():
    st.subheader("📊 Risk Metrics (Mock)")
    df = st.session_state.get("chart_data")
    if df is None or df.empty:
        st.info("Risk metrics will appear when price data is loaded.")
        return

    pnl_df = generate_mock_pnl(df)
    rets = pnl_df["equity"].pct_change().dropna()
    vol = rets.std() * np.sqrt(252)
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0

    st.metric("Volatility (ann.)", f"{vol * 100:.2f}%")
    st.metric("Sharpe (mock)", f"{sharpe:.2f}")

    st.caption("These values are all based on mock data & simplified calculations.")


def render_positions_tab():
    st.subheader("Open Positions (Mock)")
    df = st.session_state.get("chart_data")
    if df is None or df.empty:
        st.info("No positions to show yet.")
        return

    current_price = df["close"].iloc[-1]
    positions = generate_mock_positions("AAPL", current_price)

    data = []
    for pos in positions:
        data.append(
            {
                "Symbol": pos.symbol,
                "Side": pos.side,
                "Qty": pos.quantity,
                "Entry": pos.entry_price,
                "Current": pos.current_price,
                "PnL": pos.pnl,
            }
        )

    pos_df = pd.DataFrame(data)
    st.dataframe(pos_df, use_container_width=True)


def render_trades_tab():
    st.subheader("Trade History (Mock)")
    df = st.session_state.get("chart_data")
    if df is None or df.empty:
        st.info("No trades yet.")
        return

    trades_df = generate_mock_trades(df)
    if trades_df.empty:
        st.info("No trade signals were generated by the mock MA strategy.")
        return

    st.dataframe(trades_df, use_container_width=True)


def render_pnl_tab():
    """Temporary stub: PnL tab is not implemented yet."""
    st.info("PnL tab (Profit & Loss) はまだ実装中です。今後のバージョンで有効になります。")



# =============================================================================
# Helper Functions
# =============================================================================

def render_symbol_selector(key_prefix: str = "sl", container: Any = st) -> str:
    """
    Renders a symbol selector with presets and custom input.
    Loads presets from data/symbol_presets.json.
    """
    # Load symbols from Universe CSV
    universe_symbols = load_symbol_universe(UNIVERSE_CSV)
    
    if not universe_symbols:
        # Fallback if universe load fails
        SYMBOL_PRESETS = ["AAPL", "MSFT", "TSLA", "Custom..."]
    else:
        SYMBOL_PRESETS = universe_symbols + ["Custom..."]
    
    # Initialize shared state if not present
    if "shared_symbol_preset" not in st.session_state:
        st.session_state["shared_symbol_preset"] = SYMBOL_PRESETS[0] if SYMBOL_PRESETS else "SMCI"
    if "shared_custom_symbol" not in st.session_state:
        st.session_state["shared_custom_symbol"] = ""

    current_preset = st.session_state["shared_symbol_preset"]
    if current_preset not in SYMBOL_PRESETS:
        current_preset = "Custom..." # Fallback if loaded symbol is not in presets
        st.session_state["shared_custom_symbol"] = st.session_state.get("sl_symbol", SYMBOL_PRESETS[0] if SYMBOL_PRESETS else "SMCI")

    def on_preset_change():
        st.session_state["shared_symbol_preset"] = st.session_state[f"{key_prefix}_preset_select"]
        
    symbol_preset = container.selectbox(
        "Symbol",
        options=SYMBOL_PRESETS,
        index=SYMBOL_PRESETS.index(current_preset) if current_preset in SYMBOL_PRESETS else 0,
        key=f"{key_prefix}_preset_select",
        help="よく使う銘柄のプリセットです。Custom... を選ぶと任意のシンボルを入力できます。",
        on_change=on_preset_change
    )

    effective_symbol = symbol_preset
    
    if symbol_preset == "Custom...":
        def on_custom_change():
            st.session_state["shared_custom_symbol"] = st.session_state[f"{key_prefix}_custom_input"]

        custom_symbol = container.text_input(
            "Custom Symbol",
            value=st.session_state["shared_custom_symbol"],
            key=f"{key_prefix}_custom_input",
            placeholder="例: 7203.T (トヨタ), 9984.T (ソフトバンクG) など",
            on_change=on_custom_change
        )
        effective_symbol = custom_symbol.strip() or (SYMBOL_PRESETS[0] if SYMBOL_PRESETS else "SMCI")
    
    return effective_symbol


# =============================================================================
# Backtest UI (NEW)
# =============================================================================


def render_backtest_ui():
    """
    Render the Backtest UI tab.
    Allows users to run simulations via the backend API.
    """
    st.title("🧪 Backtest Lab")
    st.caption("Run simulations using the backend engine.")

    # --- Sidebar Inputs ---
    st.sidebar.header("Backtest Settings")

    # Check for loaded strategy
    loaded_strat = st.session_state.get("loaded_strategy")
    default_symbol = "AAPL"
    default_short = 9
    default_long = 21
    
    if loaded_strat:
        st.sidebar.success(f"Loaded: {loaded_strat['name']}")
        default_symbol = loaded_strat.get("symbol", "AAPL")
        # Update shared state if loaded
        st.session_state["shared_symbol_preset"] = "Custom..." # Assume custom or we check if it's in preset
        st.session_state["shared_custom_symbol"] = default_symbol
        
        if loaded_strat.get("params"):
            default_short = loaded_strat["params"].get("short_window", 9)
            default_long = loaded_strat["params"].get("long_window", 21)

    # Use the shared symbol selector
    symbol = render_symbol_selector(key_prefix="bl", container=st.sidebar)
    
    timeframe = st.sidebar.selectbox("Timeframe", options=["1d", "1h", "5m"], index=0)
    
    start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime(2023, 12, 31))
    
    initial_capital = st.sidebar.number_input("Initial Capital", value=1_000_000, step=100_000)
    commission = st.sidebar.number_input("Commission Rate", value=0.001, step=0.0001, format="%.4f")
    
    st.sidebar.subheader("Strategy Parameters")
    
    # Strategy Template Selection
    STRATEGY_MAP = {
        "ma_cross": {"class": MACrossStrategy, "label": "MA Cross", "usable": True},
        "ema_cross": {"class": EMACrossStrategy, "label": "EMA Cross", "usable": False},
        "macd_trend": {"class": MACDTrendStrategy, "label": "MACD Trend", "usable": False},
        "rsi_mean_reversion": {"class": RSIMeanReversionStrategy, "label": "RSI Mean Reversion", "usable": True},
        "stoch_oscillator": {"class": StochasticOscillatorStrategy, "label": "Stochastic Oscillator", "usable": False},
        "bollinger_mean_reversion": {"class": BollingerMeanReversionStrategy, "label": "Bollinger Mean Reversion", "usable": False},
        "bollinger_breakout": {"class": BollingerBreakoutStrategy, "label": "Bollinger Breakout", "usable": False},
        "donchian_breakout": {"class": DonchianBreakoutStrategy, "label": "Donchian Breakout", "usable": False},
        "atr_trailing_ma": {"class": ATRTrailingMAStrategy, "label": "ATR Trailing MA", "usable": False},
        "roc_momentum": {"class": ROCMomentumStrategy, "label": "ROC Momentum", "usable": False},
    }

    def format_backtest_label(key: str) -> str:
        cfg = STRATEGY_MAP[key]
        base = cfg["label"]
        if cfg.get("usable", False):
            return f"✔ {base}"
        else:
            return f"✖ {base}"

    strategy_template = st.sidebar.selectbox(
        "Select Strategy Template",
        options=list(STRATEGY_MAP.keys()),
        format_func=format_backtest_label,
        index=0
    )
    
    # Dynamic Parameters
    strategy_cls = STRATEGY_MAP[strategy_template]["class"]
    schema = strategy_cls.get_params_schema()
    
    params = {}
    for param_name, config in schema.items():
        label = config.get("label", param_name)
        default = config.get("default")
        
        # Override default if loaded strategy exists and matches current template
        if loaded_strat and loaded_strat.get("params") and loaded_strat.get("strategy_type") == strategy_template:
             default = loaded_strat["params"].get(param_name, default)

        if config["type"] == "int":
            params[param_name] = st.sidebar.number_input(
                label, 
                min_value=config.get("min"), 
                max_value=config.get("max"), 
                value=int(default), 
                step=config.get("step", 1)
            )
        elif config["type"] == "float":
            params[param_name] = st.sidebar.number_input(
                label, 
                min_value=config.get("min"), 
                max_value=config.get("max"), 
                value=float(default), 
                step=config.get("step", 0.1),
                format="%.2f"
            )

    # Widgets will be created AFTER the load logic to avoid StreamlitAPIException
    
    # ==========================================
    # Main Area: Load from Strategy Library
    # ==========================================
    st.markdown("### 📚 Load from Strategy Library")
    st.caption("Load saved strategy parameters from Strategy Lab")
    
    lib = StrategyLibrary()
    strategies = lib.load_strategies()
    
    if strategies:
        # Create display options
        strategy_options = []
        for s in strategies:
            s_params = s.get("params", {})
            metrics = s.get("metrics", {})
            
            # Create a compact parameter string
            param_str = ",".join([f"{k}={v}" for k, v in s_params.items()])
            if len(param_str) > 20:
                param_str = param_str[:20] + "..."
                
            return_pct = metrics.get("return_pct", 0)
            label = f"{s['name']} | {s['symbol']} {s['timeframe']} | {s.get('strategy_type', 'ma_cross')} | Return: {return_pct:.2f}%"
            strategy_options.append(label)
        
        col_select, col_load = st.columns([3, 1])
        with col_select:
            selected_idx = st.selectbox(
                "Select Strategy",
                options=range(len(strategies)),
                format_func=lambda i: strategy_options[i],
                key="bt_strategy_select"
            )
        
        with col_load:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("📂 Load Parameters", key="bt_load_strategy_btn"):
                selected_strategy = strategies[selected_idx]
                loaded_params = selected_strategy.get("params", {})
                
                # Update session state
                st.session_state["shared_symbol_preset"] = selected_strategy.get("symbol", "AAPL")
                st.session_state["loaded_strategy"] = selected_strategy # Store full object
                
                # We can't easily update sidebar widgets for all strategies dynamically here 
                # without a complex rerun logic or using session state for every param.
                # For now, we just notify.
                
                st.success(f"✅ Loaded strategy: {selected_strategy['name']}")
                st.info(f"**Parameters:** {loaded_params}")
                st.rerun()
    else:
        st.info("No strategies saved yet. Go to Strategy Lab to save strategies from optimization results.")
    
    st.markdown("---")

    # ==========================================
    # Loaded Strategy Info Display
    # ==========================================
    if strategies:  # Only show if there are saved strategies
        loaded_strat_info = None
        # Check if a strategy was loaded from the section above
        # Simplified check: just check session state
        if "loaded_strategy" in st.session_state:
             loaded_strat_info = st.session_state["loaded_strategy"]
        
        if loaded_strat_info:
            with st.expander("📋 Currently Loaded Strategy", expanded=True):
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.write(f"**Name:** {loaded_strat_info['name']}")
                    st.write(f"**Symbol:** {loaded_strat_info['symbol']}")
                    st.write(f"**Timeframe:** {loaded_strat_info['timeframe']}")
                with col_info2:
                    l_params = loaded_strat_info.get("params", {})
                    metrics = loaded_strat_info.get("metrics", {})
                    st.write(f"**Type:** {loaded_strat_info.get('strategy_type', 'ma_cross')}")
                    st.write(f"**Parameters:** {l_params}")
                    st.write(f"**Return:** {metrics.get('return_pct', 0):.2f}%")
                
                if loaded_strat_info.get('description'):
                    st.caption(f"*Description: {loaded_strat_info['description']}*")
            
            st.markdown("---")
    
    # ==========================================
    # Strategy Comparison Section
    # ==========================================
    st.markdown("### 📊 Strategy Comparison")
    st.caption("Compare multiple saved strategies with the same conditions")
    
    if not strategies or len(strategies) < 2:
        st.info("Need at least 2 saved strategies to run comparison. Go to Strategy Lab to save more strategies.")
    else:
        # Create selection options
        strategy_options = []
        strategy_map = {}
        for s in strategies:
            s_params = s.get("params", {})
            metrics = s.get("metrics", {})
            return_pct = metrics.get("return_pct", 0)
            label = f"{s['name']} | {s['symbol']} {s['timeframe']} | {s.get('strategy_type', 'ma_cross')} | Return: {return_pct:.2f}%"
            strategy_options.append(label)
            strategy_map[label] = s
        
        # Multi-select for strategy selection
        selected_labels = st.multiselect(
            "Select Strategies to Compare (2 or more)",
            options=strategy_options,
            key="comparison_strategy_select"
        )
        
        selected_strategies = [strategy_map[label] for label in selected_labels]
        
        # Run Comparison Button
        col_comp1, col_comp2 = st.columns([1, 3])
        with col_comp1:
            run_comparison = st.button("🔬 Run Comparison", key="run_comparison_btn")
        
        # Validation and Execution
        if run_comparison:
            if len(selected_strategies) < 2:
                st.warning("Please select at least 2 strategies to compare.")
            else:
                # Validate strategies have same symbol and timeframe
                symbols = set(s["symbol"] for s in selected_strategies)
                timeframes = set(s["timeframe"] for s in selected_strategies)
                
                if len(symbols) > 1:
                    st.error(f"❌ All strategies must use the same symbol. Selected symbols: {', '.join(symbols)}")
                elif len(timeframes) > 1:
                    st.error(f"❌ All strategies must use the same timeframe. Selected timeframes: {', '.join(timeframes)}")
                else:
                    # Valid - run comparison
                    st.success(f"✅ Comparing {len(selected_strategies)} strategies with {list(symbols)[0]} / {list(timeframes)[0]}")
                    
                    with st.spinner("Running comparisons..."):
                        comparison_results = []
                        
                        for strategy in selected_strategies:
                            # requests is imported globally
                            
                            s_params = strategy.get("params", {})
                            payload = {
                                "symbol": strategy["symbol"],
                                "timeframe": strategy["timeframe"],
                                "start_date": start_date.isoformat(),
                                "end_date": end_date.isoformat(),
                                "initial_capital": initial_capital,
                                "commission": commission,
                                "strategy": strategy.get("strategy_type", "ma_cross"),
                                "params": s_params
                            }
                            
                            try:
                                response = requests.post(f"{BACKEND_URL}/simulate", json=payload, timeout=30)
                                response.raise_for_status()
                                result = response.json()
                                
                                comparison_results.append({
                                    "name": strategy["name"],
                                    "strategy": strategy,
                                    "result": result
                                })
                            except requests.exceptions.RequestException as e:
                                st.error(f"Failed to run backtest for '{strategy['name']}': {e}")
                        
                        # Display results if we have any successful runs
                        if comparison_results:
                            st.markdown("---")
                            st.markdown("#### 📈 Comparison Results")
                            
                            st.info("""
**Strategy Comparison**  
Comparing multiple strategies with the same symbol, timeframe, and date range.
- **Return (%)**: (Final Equity / Initial Capital - 1) × 100
- **Max Drawdown (%)**: Maximum peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return measure
- **Win Rate (%)**: Percentage of profitable trades
                            """)
                            
                            # Create comparison table
                            comparison_data = []
                            for cr in comparison_results:
                                metrics = cr['result']['metrics']
                                row = {
                                    "Name": cr["name"],
                                    "Symbol": cr["strategy"]["symbol"],
                                    "Timeframe": cr["strategy"]["timeframe"],
                                    "Short": cr["strategy"]["params"].get("short_window"),
                                    "Long": cr["strategy"]["params"].get("long_window"),
                                    "Return (%)": f"{metrics['return_pct']:.2f}",
                                    "Max DD (%)": f"{metrics['max_drawdown'] * 100:.2f}",
                                    "Sharpe": f"{metrics['sharpe_ratio']:.2f}",
                                    "Win Rate (%)": f"{metrics['win_rate'] * 100:.2f}",
                                    "Trades": metrics['trade_count']
                                }
                                comparison_data.append(row)
                            
                            df_comparison = pd.DataFrame(comparison_data)
                            
                            # Highlight best return
                            st.markdown("##### Comparison Table")
                            st.dataframe(df_comparison, use_container_width=True)
                            
                            # Find and display best performer
                            best_idx = df_comparison["Return (%)"].astype(float).idxmax()
                            best_name = df_comparison.loc[best_idx, "Name"]
                            best_return = df_comparison.loc[best_idx, "Return (%)"]
                            st.success(f"🏆 Best Performer: **{best_name}** with {best_return}% return")
                            # Best Parameters
                            # NOTE: Removed old top_results-based UI here.
                            # Strategy comparison now only uses df_comparison / best_idx.
                        # Equity Curve Overlay Chart
                        st.markdown("##### Equity Curve Comparison")
                        
                        # Prepare data for overlay
                        equity_data = []
                        for cr in comparison_results:
                            equity_curve = cr["result"].get("equity_curve", [])
                            for point in equity_curve:
                                equity_data.append({
                                    "date": point["date"],
                                    "equity": point["equity"],
                                    "strategy": cr["name"]
                                })
                            
                        if equity_data:
                            # import altair as alt (Moved to global scope)
                            
                            df_equity = pd.DataFrame(equity_data)
                            df_equity['date'] = pd.to_datetime(df_equity['date'])
                            
                            # Create Altair chart
                            chart = alt.Chart(df_equity).mark_line().encode(
                                x=alt.X('date:T', title='Date'),
                                y=alt.Y('equity:Q', title='Equity', scale=alt.Scale(zero=False)),
                                color=alt.Color('strategy:N', title='Strategy'),
                                tooltip=['strategy', 'date:T', 'equity:Q']
                            ).properties(
                                height=400,
                                title='Equity Curve Comparison'
                            ).interactive()
                            
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.warning("No equity curve data available for comparison.")
    
    st.markdown("---")
    
    # --- Run Backtest Button ---
    if not STRATEGY_MAP[strategy_template].get("usable", False):
         st.warning("This strategy is not implemented yet.")
         submitted = False
    else:
        # Input Form (for the submit button)
        with st.form("backtest_form"):
            st.markdown("---") # Separator for the button
            submitted = st.form_submit_button("▶ Run Backtest")

    if submitted:
        # API Call
        # requests is imported globally

        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": initial_capital,
            "commission_rate": commission,
            "position_size": 1.0,
            "strategy": strategy_template,
            "params": params,
        }

        with st.spinner("Running simulation..."):
            try:
                response = requests.post(f"{BACKEND_URL}/simulate", json=payload)
                response.raise_for_status()
                result = response.json()

                # Display Results
                st.success("Backtest completed!")
                
                # Metrics
                metrics = result["metrics"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Return", f"{metrics['return_pct']:.2f}%")
                m2.metric("Win Rate", f"{metrics['win_rate'] * 100:.1f}%")
                m3.metric("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")
                m4.metric("Trades", metrics["trade_count"])

                # Prepare Data
                price_series = result.get("price_series", [])
                equity_data = result.get("equity_curve", [])
                trades_data = result.get("trades", [])

                # --- Results (Vertical Layout) ---
                
                # 1. Price & Signals
                st.markdown("#### 📈 Price History & Trade Signals")
                
                price_chart = None
                equity_chart = None
                
                if price_series:
                    df_price = pd.DataFrame(price_series)
                    df_price["date"] = pd.to_datetime(df_price["date"])
                    
                    # Clean numeric cols
                    for col in ["close", "open", "high", "low"]:
                        if col in df_price.columns:
                            df_price[col] = pd.to_numeric(df_price[col], errors='coerce')
                    
                    # Drop rows with missing essential data
                    df_price = df_price.dropna(subset=["date", "close"])

                    # Base Chart
                    base = alt.Chart(df_price).encode(x=alt.X('date:T', title="Date"))
                    
                    # Dynamic Tooltip
                    tooltip_cols = ['date:T', 'close']
                    for col in ['open', 'high', 'low']:
                        if col in df_price.columns and df_price[col].notna().any():
                            tooltip_cols.append(col)

                    # Price Line
                    line = base.mark_line(color='#29b5e8').encode(
                        y=alt.Y('close:Q', title="Price", scale=alt.Scale(zero=False)),
                        tooltip=tooltip_cols
                    )
                    
                    layers = [line]
                    
                    # Add MA if available (optional, but good if present)
                    if "ma_short" in df_price.columns:
                        layers.append(base.mark_line(color='orange', opacity=0.8).encode(y='ma_short:Q', tooltip=['date:T', 'ma_short']))
                    if "ma_long" in df_price.columns:
                        layers.append(base.mark_line(color='blue', opacity=0.8).encode(y='ma_long:Q', tooltip=['date:T', 'ma_long']))

                    # Add Signals
                    if trades_data:
                        df_trades = pd.DataFrame(trades_data)
                        if "date" in df_trades.columns:
                            df_trades["date"] = pd.to_datetime(df_trades["date"])
                            
                            # Buy Signals
                            buys = df_trades[df_trades["side"] == "BUY"]
                            if not buys.empty:
                                buy_chart = alt.Chart(buys).mark_point(
                                    shape="triangle-up", filled=True, size=100, color="green"
                                ).encode(
                                    x="date:T",
                                    y="price:Q",
                                    tooltip=["date:T", "side", "price", "quantity"]
                                )
                                layers.append(buy_chart)
                            
                            # Sell Signals
                            sells = df_trades[df_trades["side"] == "SELL"]
                            if not sells.empty:
                                sell_chart = alt.Chart(sells).mark_point(
                                    shape="triangle-down", filled=True, size=100, color="red"
                                ).encode(
                                    x="date:T",
                                    y="price:Q",
                                    tooltip=["date:T", "side", "price", "quantity"]
                                )
                                layers.append(sell_chart)
                    
                    # Create Price Chart (not interactive yet)
                    price_chart = alt.layer(*layers).properties(
                        height=400,
                        title="Price History & Trade Signals"
                    )
                else:
                    st.warning("No price data available.")

                # 2. Equity Curve
                st.markdown("#### 💰 Equity Curve")
                if equity_data:
                    df_equity = pd.DataFrame(equity_data)
                    df_equity["date"] = pd.to_datetime(df_equity["date"])
                    df_equity["equity"] = pd.to_numeric(df_equity["equity"], errors='coerce')
                    
                    equity_chart = alt.Chart(df_equity).mark_line(color='#29b5e8').encode(
                        x=alt.X('date:T', title="Date"),
                        y=alt.Y('equity:Q', title="Equity", scale=alt.Scale(zero=False)),
                        tooltip=['date:T', 'equity']
                    ).properties(
                        height=250,
                        title="Equity Curve"
                    )
                else:
                    st.warning("No equity data available.")
                
                # Combine Charts
                if price_chart and equity_chart:
                    combined_chart = alt.vconcat(
                        price_chart,
                        equity_chart
                    ).resolve_scale(
                        x='shared'
                    ).interactive()
                    
                    st.altair_chart(combined_chart, use_container_width=True)
                elif price_chart:
                    st.altair_chart(price_chart.interactive(), use_container_width=True)
                elif equity_chart:
                    st.altair_chart(equity_chart.interactive(), use_container_width=True)

                # 3. Trade History
                st.markdown("#### 📜 Trade History")
                if trades_data:
                    df_trades_hist = pd.DataFrame(trades_data)
                    st.dataframe(df_trades_hist, use_container_width=True)
                    
                    # CSV Download
                    csv = df_trades_hist.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Trades CSV",
                        data=csv,
                        file_name=f"backtest_trades_{symbol}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No trades executed.")

            except requests.exceptions.RequestException as e:
                st.error(f"Backtest failed: {e}")
                if e.response is not None:
                    st.error(f"Details: {e.response.text}")




# =============================================================================
# Strategy Lab (v0.2)
# =============================================================================


def render_strategy_lab():
    """
    Render the Strategy Lab UI (v0.3 - Sectioned).
    Allows users to navigate between different sections via sidebar.
    """
    st.title("🧪 Strategy Lab")
    st.caption("Design and test algorithmic strategies.")
    
    # Get current section from sidebar (set via main.py sidebar)
    lab_section = st.session_state.get("sl_section", "Configuration")
    
    # Route to appropriate section
    if lab_section == "Configuration":
        render_configuration_section()
    elif lab_section == "Single Analysis":
        render_single_analysis_section()
    elif lab_section == "Parameter Optimization":
        render_parameter_optimization_section()
    elif lab_section == "Saved Strategies":
        render_saved_strategies_section()
    elif lab_section == "Symbol Preset Settings":
        render_symbol_preset_section()


# ============================================================================
# Strategy Lab Section Renderers
# ============================================================================

def render_configuration_section():
    """Configuration section - Market data, capital, strategy template selection"""
    st.header("⚙️ Configuration")
    st.caption("Select market, timeframe, capital settings and the strategy template to analyze.")

    # Check if a strategy was loaded
    loaded_strategy = st.session_state.get("strategy_lab_loaded_strategy")
    
    # Common Inputs
    with st.expander("📊 Market Data & Capital Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            # Use loaded strategy values as defaults
            symbol_default = loaded_strategy.get("symbol", "AAPL") if loaded_strategy else "AAPL"
            # Update shared symbol preset if loaded
            if loaded_strategy and "symbol" in loaded_strategy:
                st.session_state["shared_symbol_preset"] = symbol_default
            
            symbol = render_symbol_selector(key_prefix="sl", container=col1)
            
            # Timeframe with loaded strategy default
            timeframe_options = ["1d", "1h", "5m"]
            timeframe_default = loaded_strategy.get("timeframe", "1d") if loaded_strategy else "1d"
            timeframe_index = timeframe_options.index(timeframe_default) if timeframe_default in timeframe_options else 0
            
            timeframe = st.selectbox("Timeframe", options=timeframe_options, index=timeframe_index, key="sl_timeframe")
        with col2:
            start_date = st.date_input("Start Date", value=datetime(2025, 1, 1), key="sl_start")
            end_date = st.date_input("End Date", value=datetime(2025, 12, 31), key="sl_end")
        with col3:
            initial_capital = st.number_input("Initial Capital", value=1_000_000, step=100_000, key="sl_capital")
            commission = st.number_input("Commission Rate", value=0.001, step=0.0001, format="%.4f", key="sl_comm")

    st.markdown("---")

    # Strategy Selection
    
    strategy_keys = list(STRATEGY_TEMPLATES.keys())

    def format_strategy_label(key: str) -> str:
        cfg = STRATEGY_TEMPLATES[key]
        base = cfg["label"]
        return f"✔ {base}" if cfg.get("usable", False) else f"✖ {base}"

    # Determine default strategy to show
    default_strategy_key = st.session_state.get("persisted_strategy_type")

    # If user loaded a saved strategy (from Saved Strategies section), override
    loaded_strategy = st.session_state.get("strategy_lab_loaded_strategy")
    if loaded_strategy and loaded_strategy.get("strategy_type") in strategy_keys:
        default_strategy_key = loaded_strategy["strategy_type"]

    # Final fallback
    if default_strategy_key not in strategy_keys:
        default_strategy_key = "ma_cross"

    default_index = strategy_keys.index(default_strategy_key)

    selected_strategy_key = st.selectbox(
        "Select Strategy Template",
        options=strategy_keys,
        index=default_index,
        format_func=format_strategy_label,
        key="sl_strategy_type"
    )

    # Final strategy for use in Single Analysis / Optimization
    strategy_type = selected_strategy_key
    strategy_cfg = STRATEGY_TEMPLATES[strategy_type]

    # Persist for next time
    st.session_state["persisted_strategy_type"] = strategy_type
    st.session_state["sl_strategy_cfg"] = strategy_cfg
    st.session_state["sl_symbol"] = symbol
    
    # Info message
    st.info(f"✅ Configuration saved. Navigate to **Single Analysis** or **Parameter Optimization** to run backtests.")


def render_single_analysis_section():
    """Single Analysis section - Run single backtests with specific parameters"""
    st.header("📊 Single Analysis")
    st.caption("Run a single backtest with specific parameter values.")
    
    # Get configuration from session state (use persistent key)
    strategy_type = st.session_state.get("persisted_strategy_type", "ma_cross")
    symbol = st.session_state.get("sl_symbol", "AAPL")
    timeframe = st.session_state.get("sl_timeframe", "1d")
    start_date = st.session_state.get("sl_start", datetime(2025, 1, 1))
    end_date = st.session_state.get("sl_end", datetime(2025, 12, 31))
    initial_capital = st.session_state.get("sl_capital", 1_000_000)
    commission = st.session_state.get("sl_comm", 0.001)
    
    # Check if strategy is usable
    STRATEGY_TEMPLATES = {
        "ma_cross": {"label": "MA Cross", "usable": True},
        "rsi_mean_reversion": {"label": "RSI Mean Reversion", "usable": True},
        "ema9_dip_buy": {"label": "EMA9 Dip Buy", "usable": True},
    }
    strategy_cfg = STRATEGY_TEMPLATES.get(strategy_type, {"label": strategy_type, "usable": False})
    
    if not strategy_cfg.get("usable", False):
        st.warning("⚠️ This strategy is not implemented yet. Please select a different strategy in Configuration.")
        return
    
    # Quick Presets (Moved outside form to allow callbacks)
    st.markdown("#### ⚡ Quick Presets")
    
    # Get presets from STRATEGY_GUIDES
    guide = STRATEGY_GUIDES.get(strategy_type)
    if guide and guide.presets:
        cols = st.columns(len(guide.presets))
        for i, preset in enumerate(guide.presets):
            with cols[i]:
                st.button(
                    f"Apply {preset.label}", 
                    key=f"sl_preset_{strategy_type}_{i}",
                    help=getattr(preset, "description", None),
                    on_click=apply_preset_callback,
                    args=(preset.params, strategy_type)
                )
        
        if "preset_message" in st.session_state:
            st.success(st.session_state["preset_message"])
            del st.session_state["preset_message"]
    else:
        st.info("No presets available for this strategy.")
    
    st.markdown("---")
    
    # Strategy Parameters Form
    with st.form("strategy_form"):
        st.subheader("Strategy Parameters")
        
        strategy_params = {}
        submitted_single = False
        
        # MA Cross Strategy
        if strategy_type == "ma_cross":
            st.markdown("**Moving Average Crossover**")
            st.caption("Buy when Short MA crosses above Long MA. Sell when Short MA crosses below Long MA.")
            
            col1, col2 = st.columns(2)
            with col1:
                short_window = st.number_input("Short Window", min_value=1, max_value=200, value=st.session_state.get("sl_ma_short", 9), key="sl_ma_short")
            with col2:
                long_window = st.number_input("Long Window", min_value=1, max_value=400, value=st.session_state.get("sl_ma_long", 21), key="sl_ma_long")
            
            strategy_params = {
                "short_window": short_window,
                "long_window": long_window
            }
        
        # RSI Mean Reversion
        elif strategy_type == "rsi_mean_reversion":
            st.markdown("**RSI Mean Reversion**")
            st.caption("Buy when RSI crosses below Oversold. Sell when RSI crosses above Overbought.")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_period = st.number_input("RSI Period", min_value=2, value=st.session_state.get("sl_rsi_period", 14), key="sl_rsi_period")
            with col2:
                oversold = st.number_input("Oversold Level", min_value=1, max_value=49, value=st.session_state.get("sl_rsi_oversold", 30), key="sl_rsi_oversold")
            with col3:
                overbought = st.number_input("Overbought Level", min_value=51, max_value=99, value=st.session_state.get("sl_rsi_overbought", 70), key="sl_rsi_overbought")
            
            strategy_params = {
                "rsi_period": rsi_period,
                "oversold": oversold,
                "overbought": overbought
            }
        
        # EMA9 Dip Buy
        elif strategy_type == "ema9_dip_buy":
            st.markdown("**EMA9 Dip Buy**")
            st.caption("Long-only pullback strategy. Buys dips near 9EMA in strong uptrends with volume confirmation.")
            
            col1, col2 = st.columns(2)
            with col1:
                ema_fast = st.number_input("Fast EMA Period", min_value=5, max_value=20, value=st.session_state.get("sl_ema9_fast", 9), key="sl_ema9_fast")
                deviation_threshold = st.number_input("Deviation Threshold %", min_value=0.5, max_value=5.0, value=st.session_state.get("sl_ema9_dev", 2.0), step=0.5, key="sl_ema9_dev")
                stop_buffer = st.number_input("Stop Loss Buffer %", min_value=0.1, max_value=2.0, value=st.session_state.get("sl_ema9_stop", 0.5), step=0.1, key="sl_ema9_stop")
            with col2:
                ema_slow = st.number_input("Slow EMA Period", min_value=10, max_value=50, value=st.session_state.get("sl_ema9_slow", 21), key="sl_ema9_slow")
                risk_reward = st.number_input("Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=st.session_state.get("sl_ema9_rr", 2.0), step=0.5, key="sl_ema9_rr")
                lookback_volume = st.number_input("Volume Lookback", min_value=10, max_value=50, value=st.session_state.get("sl_ema9_vol", 20), step=5, key="sl_ema9_vol")
            
            strategy_params = {
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "deviation_threshold": deviation_threshold,
                "stop_buffer": stop_buffer,
                "risk_reward": risk_reward,
                "lookback_volume": lookback_volume
            }
        
        else:
            st.error(f"Unknown strategy: {strategy_type}")
            return
        
        st.markdown("---")
        submitted_single = st.form_submit_button("🚀 Run Strategy Analysis")
    
    # Execute Single Run
    if submitted_single:
        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date),
            "end_date": end_date.isoformat() if hasattr(end_date, 'isoformat') else str(end_date),
            "initial_capital": initial_capital,
            "commission_rate": commission,
            "position_size": 1.0,
            "strategy": strategy_type,
            **strategy_params
        }
        
        with st.spinner(f"Running {strategy_cfg['label']} Analysis..."):
            try:
                response = requests.post(f"{BACKEND_URL}/simulate", json=payload)
                response.raise_for_status()
                result = response.json()
                
                st.success("✅ Analysis Completed!")
                
                # Metrics
                metrics = result.get("metrics", {})
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return", f"{metrics.get('return_pct', 0):.2f}%")
                col2.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                col3.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%")
                col4.metric("Trades", metrics.get('trade_count', 0))
                
                # Prepare Data
                price_series = result.get("price_series", [])
                equity_data = result.get("equity_curve", [])
                trades_data = result.get("trades", [])
                
                # --- Results (Vertical Layout) ---
                
                # 1. Price & Signals
                st.markdown("#### 📈 Price History & Trade Signals")
                
                price_chart = None
                equity_chart = None
                
                if price_series:
                    df_price = pd.DataFrame(price_series)
                    df_price["date"] = pd.to_datetime(df_price["date"])
                    
                    # Clean numeric cols
                    for col in ["close", "open", "high", "low"]:
                        if col in df_price.columns:
                            df_price[col] = pd.to_numeric(df_price[col], errors='coerce')
                    
                    # Drop rows with missing essential data
                    df_price = df_price.dropna(subset=["date", "close"])

                    # Base Chart
                    base = alt.Chart(df_price).encode(x=alt.X('date:T', title="Date"))
                    
                    # Dynamic Tooltip
                    tooltip_cols = ['date:T', 'close']
                    for col in ['open', 'high', 'low']:
                        if col in df_price.columns and df_price[col].notna().any():
                            tooltip_cols.append(col)

                    # Price Line
                    line = base.mark_line(color='#29b5e8').encode(
                        y=alt.Y('close:Q', title="Price", scale=alt.Scale(zero=False)),
                        tooltip=tooltip_cols
                    )
                    
                    layers = [line]
                    
                    # Add MA if available (optional, but good if present)
                    if "ma_short" in df_price.columns:
                        layers.append(base.mark_line(color='orange', opacity=0.8).encode(y='ma_short:Q', tooltip=['date:T', 'ma_short']))
                    if "ma_long" in df_price.columns:
                        layers.append(base.mark_line(color='blue', opacity=0.8).encode(y='ma_long:Q', tooltip=['date:T', 'ma_long']))

                    # Add Signals
                    if trades_data:
                        df_trades = pd.DataFrame(trades_data)
                        if "date" in df_trades.columns:
                            df_trades["date"] = pd.to_datetime(df_trades["date"])
                            
                            # Buy Signals
                            buys = df_trades[df_trades["side"] == "BUY"]
                            if not buys.empty:
                                buy_chart = alt.Chart(buys).mark_point(
                                    shape="triangle-up", filled=True, size=100, color="green"
                                ).encode(
                                    x="date:T",
                                    y="price:Q",
                                    tooltip=["date:T", "side", "price", "quantity"]
                                )
                                layers.append(buy_chart)
                            
                            # Sell Signals
                            sells = df_trades[df_trades["side"] == "SELL"]
                            if not sells.empty:
                                sell_chart = alt.Chart(sells).mark_point(
                                    shape="triangle-down", filled=True, size=100, color="red"
                                ).encode(
                                    x="date:T",
                                    y="price:Q",
                                    tooltip=["date:T", "side", "price", "quantity"]
                                )
                                layers.append(sell_chart)
                    
                    # Create Price Chart (not interactive yet)
                    price_chart = alt.layer(*layers).properties(
                        height=400,
                        title="Price History & Trade Signals"
                    )
                else:
                    st.warning("No price data available.")

                # 2. Equity Curve
                st.markdown("#### 💰 Equity Curve")
                if equity_data:
                    df_equity = pd.DataFrame(equity_data)
                    df_equity["date"] = pd.to_datetime(df_equity["date"])
                    df_equity["equity"] = pd.to_numeric(df_equity["equity"], errors='coerce')
                    
                    equity_chart = alt.Chart(df_equity).mark_line(color='#29b5e8').encode(
                        x=alt.X('date:T', title="Date"),
                        y=alt.Y('equity:Q', title="Equity", scale=alt.Scale(zero=False)),
                        tooltip=['date:T', 'equity']
                    ).properties(
                        height=250,
                        title="Equity Curve"
                    )
                else:
                    st.warning("No equity data available.")
                
                # Combine Charts
                if price_chart and equity_chart:
                    combined_chart = alt.vconcat(
                        price_chart,
                        equity_chart
                    ).resolve_scale(
                        x='shared'
                    ).interactive()
                    
                    st.altair_chart(combined_chart, use_container_width=True)
                elif price_chart:
                    st.altair_chart(price_chart.interactive(), use_container_width=True)
                elif equity_chart:
                    st.altair_chart(equity_chart.interactive(), use_container_width=True)

                # --- Tab 3: Trade History ---
                st.markdown("#### 📜 Trade History")
                if trades_data:
                    df_trades_hist = pd.DataFrame(trades_data)
                    st.dataframe(df_trades_hist, use_container_width=True)
                    
                    # CSV Download
                    csv = df_trades_hist.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Trades CSV",
                        data=csv,
                        file_name=f"strategy_trades_{symbol}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No trades executed.")
                
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Analysis failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    st.error(f"Details: {e.response.text}")


def render_parameter_optimization_section():
    """Parameter Optimization section - Run grid search optimization"""
    import numpy as np
    
    st.header("🔬 Parameter Optimization")
    st.caption("Run grid search to find optimal parameter combinations.")
    
    # Get configuration from session state
    strategy_type = st.session_state.get("persisted_strategy_type", "ma_cross")
    symbol = st.session_state.get("sl_symbol", "AAPL")
    timeframe = st.session_state.get("sl_timeframe", "1d")
    start_date = st.session_state.get("sl_start", datetime(2025, 1, 1))
    end_date = st.session_state.get("sl_end", datetime(2025, 12, 31))
    initial_capital = st.session_state.get("sl_capital", 1_000_000)
    commission = st.session_state.get("sl_comm", 0.001)
    
    strategy_cfg = STRATEGY_TEMPLATES.get(strategy_type, {"label": strategy_type, "usable": False})
    
    if not strategy_cfg.get("supports_optimization", False):
        st.info(
            f"Optimization is not yet available for **{strategy_cfg['label']}**.\n"
            "Please select a different strategy in Configuration."
        )
        return

    # Optimization Configuration Map
    OPTIMIZATION_CONFIG = {
        "ma_cross": {
            "x_param": "short_window", "y_param": "long_window",
            "x_label": "Short Window", "y_label": "Long Window",
            "x_range": [5, 50, 5], "y_range": [20, 200, 10],
            "fixed": {}
        },
        "rsi_mean_reversion": {
            "x_param": "oversold", "y_param": "overbought",
            "x_label": "Oversold Level", "y_label": "Overbought Level",
            "x_range": [20, 45, 5], "y_range": [55, 80, 5],
            "fixed": {"rsi_period": 14}
        },
        "ema9_dip_buy": {
            "x_param": "deviation_threshold", "y_param": "risk_reward",
            "x_label": "Deviation Threshold %", "y_label": "Risk/Reward Ratio",
            "x_range": [1.0, 3.0, 0.5], "y_range": [1.5, 3.0, 0.5],
            "fixed": {"ema_fast": 9, "ema_slow": 21, "stop_buffer": 0.5, "lookback_volume": 20}
        }
    }
    
    if strategy_type not in OPTIMIZATION_CONFIG:
        st.warning(f"Optimization configuration missing for {strategy_type}")
        return

    opt_config = OPTIMIZATION_CONFIG[strategy_type]
    x_param = opt_config["x_param"]
    y_param = opt_config["y_param"]
    
    # Optimizer Settings UI
    with st.expander("Optimizer Settings", expanded=True):
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            st.markdown(f"**{opt_config['x_label']} (X-Axis)**")
            x_min = st.number_input(f"Min {opt_config['x_label']}", value=opt_config["x_range"][0], key="opt_x_min")
            x_max = st.number_input(f"Max {opt_config['x_label']}", value=opt_config["x_range"][1], key="opt_x_max")
            x_step = st.number_input(f"Step {opt_config['x_label']}", value=opt_config["x_range"][2], key="opt_x_step")
        
        with col_opt2:
            st.markdown(f"**{opt_config['y_label']} (Y-Axis)**")
            y_min = st.number_input(f"Min {opt_config['y_label']}", value=opt_config["y_range"][0], key="opt_y_min")
            y_max = st.number_input(f"Max {opt_config['y_label']}", value=opt_config["y_range"][1], key="opt_y_max")
            y_step = st.number_input(f"Step {opt_config['y_label']}", value=opt_config["y_range"][2], key="opt_y_step")
            
        # Calculate combinations
        try:
            x_values = np.arange(x_min, x_max + x_step, x_step)
            y_values = np.arange(y_min, y_max + y_step, y_step)
            total_combinations = len(x_values) * len(y_values)
            st.caption(f"Total Combinations: {total_combinations}")
            if total_combinations > 100:
                st.warning("⚠️ High number of combinations. Optimization may be slow.")
        except Exception:
            pass

    # Run Optimization
    if st.button("🚀 Run Optimization"):
        with st.spinner("Running grid search..."):
            try:
                # Generate parameter lists
                if isinstance(x_step, float) or isinstance(opt_config["x_range"][2], float):
                     x_values = np.arange(x_min, x_max + x_step/100, x_step)
                else:
                     x_values = np.arange(x_min, x_max + 1, x_step)
                     
                if isinstance(y_step, float) or isinstance(opt_config["y_range"][2], float):
                     y_values = np.arange(y_min, y_max + y_step/100, y_step)
                else:
                     y_values = np.arange(y_min, y_max + 1, y_step)

                # Convert to python types
                x_list = [float(x) if isinstance(x, (float, np.floating)) else int(x) for x in x_values]
                y_list = [float(y) if isinstance(y, (float, np.floating)) else int(y) for y in y_values]

                # Build payload
                payload = {
                    "strategy_type": strategy_type,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start_date": start_date.strftime("%Y-%m-%d"),
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "initial_capital": initial_capital,
                    "commission": commission,
                    "param_grid": {
                        x_param: x_list,
                        y_param: y_list
                    },
                    "fixed_params": opt_config["fixed"]
                }
                
                # Call backend
                # Use BACKEND_URL instead of hardcoded localhost:8000
                response = requests.post(f"{BACKEND_URL}/optimize/generic", json=payload, timeout=300)
                response.raise_for_status()
                results = response.json()
                
                # Store results in session state
                st.session_state["opt_results"] = results
                st.session_state["opt_strategy_type"] = strategy_type # Track which strategy results belong to
                
                # Try to compute how many combinations were analyzed
                if isinstance(results, dict):
                    analyzed = results.get("total_combinations")
                    if analyzed is None:
                        if "results" in results and isinstance(results["results"], list):
                            analyzed = len(results["results"])
                        else:
                            # Fallback: search for first list value
                            analyzed = len(next((v for v in results.values() if isinstance(v, list)), []))
                else:
                    analyzed = len(results)
                    
                st.success(f"Optimization complete! Analyzed {analyzed} combinations.")
                
            except requests.exceptions.RequestException as e:
                st.error(f"Optimization failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    st.error(f"Details: {e.response.text}")

    # Display Results
    results = st.session_state.get("opt_results")
    result_strategy = st.session_state.get("opt_strategy_type")
    
    # Only show results if they match the current strategy
    if results and result_strategy == strategy_type:
        st.markdown("---")
        st.subheader("Optimization Results")
        
        # --- Normalize results into a list of dicts ---
        if isinstance(results, dict):
            # Common pattern: {"results": [...], "total_combinations": ...}
            if "results" in results and isinstance(results["results"], list):
                results_list = results["results"]
            elif "items" in results and isinstance(results["items"], list):
                results_list = results["items"]
            else:
                # Fallback: search the first list value in the dict
                results_list = None
                for v in results.values():
                    if isinstance(v, list):
                        results_list = v
                        break
                if results_list is None:
                    st.error("Optimization results format not recognized.")
                    return
        else:
            # Already a list
            results_list = results
            
        # Flatten results for DataFrame
        flat_results = []
        if results_list:
            for r in results_list:
                if not isinstance(r, dict):
                    continue
                params = r.get("params", {}) or {}
                metrics = r.get("metrics", {}) or {}
                row = {}
                row.update(params)
                row.update(metrics)
                flat_results.append(row)
            
        if not flat_results:
            st.error("No optimization results to display.")
            return

        df_results = pd.DataFrame(flat_results)
        
        # Ensure score is numeric
        if "score" in df_results.columns:
            df_results["score"] = pd.to_numeric(df_results["score"], errors="coerce")
        
        # Best Parameters
        if not df_results.empty and "score" in df_results.columns:
            best_row_series = df_results.sort_values("score", ascending=False).iloc[0]
            best_row = best_row_series.to_dict()
        else:
            st.error("No valid results to display.")
            return
        
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Best Score", f"{best_row.get('score', 0):.2f}")
        col_res2.metric("Return %", f"{best_row.get('return_pct', 0):.2f}%")
        col_res3.metric("Sharpe Ratio", f"{best_row.get('sharpe_ratio', 0):.2f}")
        
        st.success(f"**Best Parameters:** {opt_config['x_label']}={best_row.get(x_param)}, {opt_config['y_label']}={best_row.get(y_param)}")
        
        # Save best params to session state for the Apply button
        st.session_state["opt_best_params"] = best_row
        
        # Apply to Single Analysis Button
        st.caption("Apply the best parameters to the Single Analysis form.")
        
        def apply_best_to_single():
            best_params = st.session_state.get("opt_best_params", {})
            apply_strategy_params_to_session(strategy_type, best_params)
            
        st.button("Apply to Single Analysis", on_click=apply_best_to_single, key="opt_apply_to_single")
        st.info("✅ Applied! Open the **Single Analysis** section to run a backtest with these parameters.")
        
        # Heatmap
        st.subheader("🔥 Score Heatmap")
        try:
            # FIX: Ensure data types are strictly numeric for Altair
            df_heatmap = df_results.copy()
            df_heatmap[x_param] = pd.to_numeric(df_heatmap[x_param], errors='coerce')
            df_heatmap[y_param] = pd.to_numeric(df_heatmap[y_param], errors='coerce')
            df_heatmap["score"] = pd.to_numeric(df_heatmap["score"], errors='coerce')
            
            # Drop invalid rows
            df_heatmap = df_heatmap.dropna(subset=[x_param, y_param, "score"])
            
            chart = alt.Chart(df_heatmap).mark_rect().encode(
                x=alt.X(f'{x_param}:O', title=opt_config['x_label']),
                y=alt.Y(f'{y_param}:O', title=opt_config['y_label']),
                color=alt.Color('score:Q', title='Score', scale=alt.Scale(scheme='viridis')),
                tooltip=[x_param, y_param, 'return_pct', 'sharpe_ratio', 'max_drawdown', 'score']
            ).properties(
                title="Optimization Score Heatmap"
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Heatmap error: {e}")
            
        # Top Results Table
        st.subheader("Top 10 Configurations")
        top_10 = df_results.sort_values("score", ascending=False).head(10)
        st.dataframe(
            top_10[[x_param, y_param, "score", "return_pct", "sharpe_ratio", "max_drawdown", "trade_count"]],
            use_container_width=True
        )
        
        # Save Strategy
        st.markdown("---")
        with st.expander("💾 Save Best Parameters as Strategy"):
            with st.form("save_best_strategy_form"):
                default_name = f"{symbol}_{timeframe}_{strategy_type}_Best"
                strategy_name = st.text_input("Strategy Name", value=default_name)
                strategy_desc = st.text_area("Description", value=f"Grid Search Result. Return: {best_row['return_pct']:.2f}%")
            
                if st.form_submit_button("💾 Save Strategy"):
                    if not strategy_name:
                        st.error("Strategy Name is required.")
                    else:
                        lib = StrategyLibrary()
                        new_strategy = {
                            "id": str(uuid.uuid4()),
                            "name": strategy_name,
                            "description": strategy_desc,
                            "created_at": datetime.now().isoformat(),
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "strategy_type": strategy_type,
                            "params": {
                                x_param: float(best_row[x_param]), # Ensure native float/int
                                y_param: float(best_row[y_param]),
                                **opt_config["fixed"]
                            },
                            "metrics": best_row
                        }
                        lib.save_strategy(new_strategy)
                        st.success(f"Strategy '{strategy_name}' saved successfully!")
        
# ============================================================================
# Helper Functions
# ============================================================================

def apply_strategy_params_to_session(strategy_type: str, params: Dict[str, Any]):
    """
    Helper to apply strategy parameters to Single Analysis session state.
    """
    # Set the strategy type for Single Analysis
    st.session_state["sl_strategy_type"] = strategy_type
    
    if strategy_type == "ma_cross":
        st.session_state["sl_ma_short"] = int(params.get("short_window", 0))
        st.session_state["sl_ma_long"] = int(params.get("long_window", 0))
        
    elif strategy_type == "rsi_mean_reversion":
        if "rsi_period" in params:
            st.session_state["sl_rsi_period"] = int(params["rsi_period"])
        st.session_state["sl_rsi_oversold"] = int(params.get("oversold_level", 20))
        st.session_state["sl_rsi_overbought"] = int(params.get("overbought_level", 80))
        
    elif strategy_type == "ema9_dip_buy":
        st.session_state["sl_ema9_dev"] = float(params.get("deviation_threshold", 0.0))
        st.session_state["sl_ema9_rr"] = float(params.get("risk_reward", 0.0))
        
        # Handle fixed parameters
        opt_cfg = OPTIMIZATION_CONFIG.get("ema9_dip_buy", {})
        fixed = opt_cfg.get("fixed", {})
        
        def get_or_fixed(name, default=None):
            if name in params:
                return params[name]
            if name in fixed:
                return fixed[name]
            return default
        
        st.session_state["sl_ema9_stop"] = float(get_or_fixed("stop_buffer", 0.5))
        st.session_state["sl_ema9_vol"] = int(get_or_fixed("lookback_volume", 20))
        st.session_state["sl_ema9_fast"] = int(get_or_fixed("ema_fast", 9))
        st.session_state["sl_ema9_slow"] = int(get_or_fixed("ema_slow", 21))


# ... (existing code) ...




def render_saved_strategies_section():
    """Saved Strategies section - View and manage saved strategies"""
    # Saved Strategies Section
    # ==========================================
    st.markdown("---")
    st.subheader("📚 Saved Strategies")
    
    lib = StrategyLibrary()
    strategies = lib.load_strategies()
    
    # Ensure all strategies have favorite field for backward compatibility
    for s in strategies:
        if "favorite" not in s:
            s["favorite"] = False
    
    if not strategies:
        st.info("No strategies saved yet.")
    else:
        # Filters
        st.markdown("#### Filters & Sorting")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        
        with col_f1:
            # Symbol filter
            unique_symbols = sorted(set(s["symbol"] for s in strategies))
            symbol_filter = st.selectbox("Symbol", ["All"] + unique_symbols, key="strat_filter_symbol")
        
        with col_f2:
            # Timeframe filter
            unique_timeframes = sorted(set(s["timeframe"] for s in strategies))
            timeframe_filter = st.selectbox("Timeframe", ["All"] + unique_timeframes, key="strat_filter_timeframe")
        
        with col_f3:
            # Favorite filter
            favorite_only = st.checkbox("⭐ Favorites only", key="strat_filter_favorite")
        
        with col_f4:
            # Sorting
            sort_options = ["Return (%)", "Max Drawdown", "Created (newest)", "Created (oldest)", "Name"]
            sort_by = st.selectbox("Sort by", sort_options, key="strat_sort")
        
        # Apply filters
        filtered_strategies = strategies
        if symbol_filter != "All":
            filtered_strategies = [s for s in filtered_strategies if s["symbol"] == symbol_filter]
        if timeframe_filter != "All":
            filtered_strategies = [s for s in filtered_strategies if s["timeframe"] == timeframe_filter]
        if favorite_only:
            filtered_strategies = [s for s in filtered_strategies if s.get("favorite", False)]
        
        # Apply sorting
        if sort_by == "Return (%)":
            filtered_strategies = sorted(filtered_strategies, key=lambda s: s.get("metrics", {}).get("return_pct", 0), reverse=True)
        elif sort_by == "Max Drawdown":
            filtered_strategies = sorted(filtered_strategies, key=lambda s: s.get("metrics", {}).get("max_drawdown", 0))
        elif sort_by == "Created (newest)":
            filtered_strategies = sorted(filtered_strategies, key=lambda s: s.get("created_at", ""), reverse=True)
        elif sort_by == "Created (oldest)":
            filtered_strategies = sorted(filtered_strategies, key=lambda s: s.get("created_at", ""))
        elif sort_by == "Name":
            filtered_strategies = sorted(filtered_strategies, key=lambda s: s.get("name", ""))
        
        if not filtered_strategies:
            st.info("No strategies match the current filters.")
        else:
            # Display table
            st.markdown(f"**Found {len(filtered_strategies)} strategies**")
            
            # Create display dataframe
            strat_rows = []
            for s in filtered_strategies:
                row = {
                    "ID": s["id"],
                    "⭐": "⭐" if s.get("favorite", False) else "☆",
                    "Name": s["name"],
                    "Symbol": s["symbol"],
                    "Timeframe": s["timeframe"],
                    "Type": s["strategy_type"],
                    "Return (%)": f"{s.get('metrics', {}).get('return_pct', 0):.2f}%",
                    "Created": s["created_at"][:16].replace("T", " ")
                }
                strat_rows.append(row)
            
            st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)
            
            # Management Section
            st.markdown("---")
            st.subheader("🛠️ Manage Strategy")
            
            selected_strat_name = st.selectbox(
                "Select Strategy to Manage",
                options=[s["name"] for s in filtered_strategies],
                key="strat_select_manage"
            )
            
            # Find selected strategy object
            selected_strat = next((s for s in filtered_strategies if s["name"] == selected_strat_name), None)
            
            if selected_strat:
                col_m1, col_m2 = st.columns([2, 1])
                
                with col_m1:
                    st.markdown(f"**Description:** {selected_strat.get('description', 'No description')}")
                    st.json(selected_strat.get("params", {}), expanded=False)
                    
                    # Load Button
                    def load_selected_strategy():
                        apply_strategy_params_to_session(selected_strat["strategy_type"], selected_strat["params"])
                        # Set other session state vars if needed (symbol, timeframe)
                        st.session_state["sl_symbol"] = selected_strat["symbol"]
                        st.session_state["sl_timeframe"] = selected_strat["timeframe"]
                        
                    st.button("📂 Load to Single Analysis", on_click=load_selected_strategy, key=f"btn_load_{selected_strat['id']}")
                    st.caption("Loads parameters and switches context to Single Analysis.")

                with col_m2:
                    # Favorite Toggle
                    is_fav = selected_strat.get("favorite", False)
                    if st.button("Unfavorite" if is_fav else "Favorite", key=f"btn_fav_{selected_strat['id']}"):
                        lib.update_strategy(selected_strat["id"], {"favorite": not is_fav})
                        st.rerun()
                        
                    # Delete Button
                    if st.button("🗑️ Delete Strategy", key=f"btn_del_{selected_strat['id']}", type="primary"):
                        lib.delete_strategy(selected_strat["id"])
                        st.success(f"Strategy '{selected_strat['name']}' deleted.")
                        st.rerun()
            
            # Actions
            st.markdown("#### Actions")
            col_action1, col_action2 = st.columns(2)
            
            with col_action1:
                selected_strat_name = st.selectbox(
                    "Select Strategy", 
                    options=[s["name"] for s in filtered_strategies],
                    key="strat_action_selector"
                )
            
            with col_action2:
                action_cols = st.columns(4)
                
                selected_strat = next((s for s in filtered_strategies if s["name"] == selected_strat_name), None)
                
                if selected_strat:
                    with action_cols[0]:
                        if st.button("⭐ Toggle Favorite", key="strat_action_favorite"):
                            lib.toggle_favorite(selected_strat["id"])
                            st.success(f"Toggled favorite for '{selected_strat_name}'")
                            st.rerun()
                    
                    with action_cols[1]:
                        if st.button("📂 Load", key="strat_action_load"):
                            st.session_state["strategy_lab_loaded_strategy"] = selected_strat
                            st.success(f"Loaded '{selected_strat_name}'. Parameters will be applied on next render.")
                            st.rerun()
                    
                    with action_cols[2]:
                        if st.button("✏️ Rename", key="strat_action_rename"):
                            st.session_state["rename_strategy_id"] = selected_strat["id"]
                            st.session_state["rename_strategy_name"] = selected_strat["name"]
                    
                    with action_cols[3]:
                        if st.button("🗑️ Delete", key="strat_action_delete"):
                            st.session_state["delete_strategy_id"] = selected_strat["id"]
                            st.session_state["delete_strategy_name"] = selected_strat["name"]
            
            # Rename dialog
            if "rename_strategy_id" in st.session_state:
                st.markdown("---")
                st.markdown("### ✏️ Rename Strategy")
                new_name = st.text_input(
                    "New Name", 
                    value=st.session_state.get("rename_strategy_name", ""),
                    key="rename_input"
                )
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    if st.button("💾 Save", key="rename_save"):
                        if new_name.strip():
                            lib.update_strategy(st.session_state["rename_strategy_id"], {"name": new_name.strip()})
                            st.success(f"Renamed to '{new_name}'")
                            del st.session_state["rename_strategy_id"]
                            del st.session_state["rename_strategy_name"]
                            st.rerun()
                        else:
                            st.warning("Name cannot be empty")
                with col_r2:
                    if st.button("❌ Cancel", key="rename_cancel"):
                        del st.session_state["rename_strategy_id"]
                        del st.session_state["rename_strategy_name"]
                        st.rerun()
            
            # Delete confirmation
            if "delete_strategy_id" in st.session_state:
                st.markdown("---")
                st.warning(f"⚠️ Are you sure you want to delete '{st.session_state.get('delete_strategy_name')}'?")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    if st.button("🗑️ Confirm Delete", key="delete_confirm"):
                        lib.delete_strategy(st.session_state["delete_strategy_id"])
                        st.success(f"Deleted '{st.session_state['delete_strategy_name']}'")
                        del st.session_state["delete_strategy_id"]
                        del st.session_state["delete_strategy_name"]
                        st.rerun()
                with col_d2:
                    if st.button("❌ Cancel", key="delete_cancel"):
                        del st.session_state["delete_strategy_id"]
                        del st.session_state["delete_strategy_name"]
                        st.rerun()


def render_symbol_preset_section():
    """Symbol Preset Settings section - Developer settings"""
    st.header("🔧 Symbol Preset Settings")
    st.caption("Developer-only settings for managing symbol presets.")
    
    # TODO: Implement in Phase 5
    st.warning("🚧 This section is under construction. Will be implemented in Phase 5.")
# ============================================================================
# Old Strategy Lab Code (to be extracted in phases)
# ============================================================================

def _old_strategy_lab_code():
    """
    This function contains the old Strategy Lab code that will be 
    progressively moved to section functions in Phases 2-5.
    DO NOT CALL THIS FUNCTION - it's just a holder for code to be moved.
    """
    # Strategy Parameters section starts here
    st.subheader("Strategy Parameters")

    if not strategy_cfg.get("usable", False):
        st.warning("This strategy is not implemented yet.")
        return

    
    # Single Column Layout
    # 1. Strategy Parameters Form
    # 2. Quick Presets
    # 3. Strategy Guide (Collapsible)

    # Dynamic Form based on selection
    # Dynamic Form based on selection
    # We use a form for parameters and analysis execution
    with st.form("strategy_form"):
        # Default values for params - use loaded strategy if available
        if loaded_strategy and loaded_strategy.get("params"):
            params = loaded_strategy.get("params", {})
            short_window_default = params.get("short_window", 9)
            long_window_default = params.get("long_window", 21)
        else:
            short_window_default = 9
            long_window_default = 21
            
        rsi_period = 14
        oversold = 30
        overbought = 70
        lookback_window = 20
        threshold = 1.0
        
        # Dictionary to hold parameters for the selected strategy
        strategy_params = {}
        
        submitted_single = False
        submitted_opt = False

        if strategy_type == "ma_cross":
            # --- Manage Optimizer State Persistence ---
            current_config = (symbol, timeframe, start_date, end_date)
            last_config = st.session_state.get("ma_cross_last_config")
            
            if last_config != current_config:
                # Config changed, clear optimizer results
                st.session_state["ma_cross_opt_results"] = None
                st.session_state["ma_cross_best_params"] = None
                st.session_state["ma_cross_last_config"] = current_config
            
            st.markdown("**Moving Average Crossover**")
            st.caption("Buy when Short MA crosses above Long MA. Sell when Short MA crosses below Long MA.")
            
            tab_single, tab_opt = st.tabs(["Single Run", "Parameter Optimization"])
            
            with tab_single:
                # Initialize session state for Single Run inputs if not present
                if "single_run_short_window" not in st.session_state:
                    st.session_state["single_run_short_window"] = short_window_default
                if "single_run_long_window" not in st.session_state:
                    st.session_state["single_run_long_window"] = long_window_default

                col1, col2 = st.columns(2)
                with col1:
                    short_window = st.number_input("Short Window", min_value=1, max_value=200, key="single_run_short_window")
                with col2:
                    long_window = st.number_input("Long Window", min_value=1, max_value=400, key="single_run_long_window")
                
                strategy_params = {
                    "short_window": short_window,
                    "long_window": long_window
                }
                
                st.markdown("---")
                # Trigger analysis if button clicked OR auto-trigger flag is set
                submitted_single = st.form_submit_button("🚀 Run Strategy Analysis") or st.session_state.get("trigger_single_run", False)
                
                # Reset trigger after use (will be handled in the execution block)
            
            with tab_opt:
                st.markdown("#### Grid Search Optimizer（設定）")
                st.caption("このブロックでは「どのパラメータ範囲で」「どのスコアロジックで」最適化するかを設定します。実行結果はページ下部の「Parameter Optimization Results」に表示されます。")
                st.caption("ℹ️ **Score Logic**: `Score = Sharpe * 2.0 + Return * 0.5`. Penalties applied for < 5 trades or > 70% Max DD (Score = -1e9).")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("Short Window Range")
                    short_min = st.number_input("Min", value=5, key="s_min")
                    short_max = st.number_input("Max", value=20, key="s_max")
                    short_step = st.number_input("Step", value=2, key="s_step")
                with c2:
                    st.markdown("Long Window Range")
                    long_min = st.number_input("Min", value=20, key="l_min")
                    long_max = st.number_input("Max", value=60, key="l_max")
                    long_step = st.number_input("Step", value=5, key="l_step")
                
                # Calculate combinations
                import math
                if short_step > 0 and long_step > 0 and short_max >= short_min and long_max >= long_min:
                    short_count = math.floor((short_max - short_min) / short_step) + 1
                    long_count = math.floor((long_max - long_min) / long_step) + 1
                    total_combinations = short_count * long_count
                else:
                    total_combinations = 0
                    
                # Display with color coding
                st.markdown("---")
                color = "green" if 0 < total_combinations <= 400 else "red"
                st.markdown(f"**Total combinations: <span style='color:{color}'>{total_combinations}</span> (limit: 400)**", unsafe_allow_html=True)
                
                if total_combinations > 400:
                    st.caption("⚠️ Over limit! Please reduce parameter ranges.")
                elif total_combinations == 0:
                    st.caption("⚠️ Invalid range settings.")

                # Result Filters
                st.markdown("##### Result Filters")
                f1, f2 = st.columns(2)
                with f1:
                    min_trades_filter = st.slider("Min Trades (filter)", 1, 30, 3, key="opt_min_trades")
                with f2:
                    max_dd_filter = st.slider("Max Drawdown % (filter)", 10, 80, 60, key="opt_max_dd")
                
                # Disable button if invalid
                opt_disabled = total_combinations == 0 or total_combinations > 400
                submitted_opt = st.form_submit_button("🔍 Run Optimization", disabled=opt_disabled)





        elif strategy_type == "rsi_mean_reversion":
            st.markdown("**RSI Mean Reversion**")
            st.caption("Buy when RSI crosses below Oversold. Sell when RSI crosses above Overbought.")
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_period = st.number_input("RSI Period", min_value=2, value=14, key="sl_rsi_period")
            with col2:
                oversold = st.number_input("Oversold Level", min_value=1, max_value=49, value=30, key="sl_rsi_oversold")
            with col3:
                overbought = st.number_input("Overbought Level", min_value=51, max_value=99, value=70, key="sl_rsi_overbought")
            
            strategy_params = {
                "rsi_period": rsi_period,
                "oversold": oversold,
                "overbought": overbought
            }
            
            st.markdown("---")
            submitted_single = st.form_submit_button("🚀 Run Strategy Analysis")
            submitted_opt = False # Optimization handled by generic section if supported



        elif strategy_type == "ema9_dip_buy":
            st.markdown("**EMA9 Dip Buy**")
            st.caption("Long-only pullback strategy. Buys dips near 9EMA in strong uptrends with volume confirmation.")
            
            # Single Run only (Parameter Optimization is in the common section below)
            col1, col2 = st.columns(2)
            with col1:
                ema_fast = st.number_input("Fast EMA Period", min_value=5, max_value=20, value=9, key="sl_ema9_fast")
                deviation_threshold = st.number_input("Deviation Threshold %", min_value=0.5, max_value=5.0, value=2.0, step=0.5, key="sl_ema9_dev")
                stop_buffer = st.number_input("Stop Loss Buffer %", min_value=0.1, max_value=2.0, value=0.5, step=0.1, key="sl_ema9_stop")
            with col2:
                ema_slow = st.number_input("Slow EMA Period", min_value=10, max_value=50, value=21, key="sl_ema9_slow")
                risk_reward = st.number_input("Risk/Reward Ratio", min_value=1.0, max_value=5.0, value=2.0, step=0.5, key="sl_ema9_rr")
                lookback_volume = st.number_input("Volume Lookback", min_value=10, max_value=50, value=20, step=5, key="sl_ema9_vol")
            
            strategy_params = {
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
                "deviation_threshold": deviation_threshold,
                "stop_buffer": stop_buffer,
                "risk_reward": risk_reward,
                "lookback_volume": lookback_volume
            }
            
            st.markdown("---")
            submitted_single = st.form_submit_button("🚀 Run Strategy Analysis")
            # ✅ submitted_opt is defined in the Parameter Optimization tab, don't overwrite it



        elif strategy_type == "price_breakout":
            st.markdown("**Price Breakout Strategy**")
            st.caption("Buy when price breaks above N-period high. Sell when price breaks below N-period low.")
            col1, col2 = st.columns(2)
            with col1:
                lookback_window = st.number_input("Lookback Window", min_value=1, value=20)
            with col2:
                # threshold is not used in simple PriceBreakout but kept for compatibility or future use
                threshold = st.number_input("Threshold Multiplier", min_value=1.0, value=1.0, step=0.1)
            
            strategy_params = {
                "lookback": int(lookback_window)
            }
            
            st.markdown("---")
            submitted_single = st.form_submit_button("🚀 Run Strategy Analysis")
            submitted_opt = False
            
        else:
            # Generic placeholder for other strategies
            st.info(f"**{strategy_cfg['label']}** selected.")
            st.warning("Optimization for this strategy is coming soon. Please use the Backtest Lab (Sidebar) for single runs.")
            submitted_single = False
            submitted_opt = False

    # Fetch Guide Data (Markdown + Presets)
    guide_markdown = None
    guide_presets = None
    
    try:
        backend_id = strategy_type
        response = requests.get(f"{BACKEND_URL}/strategies/{backend_id}/doc", timeout=5)
        if response.status_code == 200:
            data = response.json()
            guide_markdown = data.get("markdown")
            guide_presets = data.get("presets")
    except Exception as e:
        st.error(f"Failed to load guide: {e}")

    # ⚡ Quick Presets Section
    st.divider()
    st.markdown("#### ⚡ Quick Presets")
    
    if guide_presets:
        st.caption("Select a preset to automatically load recommended parameters.")
        
        # Convert presets dict to list for column layout
        preset_items = list(guide_presets.items())
        cols = st.columns(len(preset_items))
        
        for i, (key, preset) in enumerate(preset_items):
            with cols[i]:
                label = preset.get("label", key)
                desc = preset.get("description", "")
                params = preset.get("params", {})
                
                if st.button(f"Apply {label}", 
                            key=f"preset_{strategy_type}_{i}",
                            help=desc,
                            on_click=apply_preset_callback,
                            args=(params, strategy_type)):
                    pass
        
        # Show message if set
        if "preset_message" in st.session_state:
            st.success(st.session_state["preset_message"])
            del st.session_state["preset_message"]
    else:
        # Fallback for strategies without presets in frontmatter
        if strategy_type in STRATEGY_GUIDES:
             # Legacy fallback
             guide = STRATEGY_GUIDES[strategy_type]
             st.caption("Apply recommended parameter sets (Legacy)")
             cols = st.columns(len(guide.presets))
             for i, preset in enumerate(guide.presets):
                with cols[i]:
                    st.button(f"Apply {preset.label}", 
                              key=f"preset_{strategy_type}_{i}",
                              on_click=apply_preset_callback,
                              args=(preset.params, strategy_type))
             if "preset_message" in st.session_state:
                st.success(st.session_state["preset_message"])
                del st.session_state["preset_message"]
        else:
             st.info("Presets not available for this strategy.")

    # 📘 Strategy Guide (Collapsible)
    st.divider()
    with st.expander("📘 Trading Strategy Guide (Markdown)"):
        if guide_markdown:
            st.markdown(guide_markdown)
        else:
            st.warning(f"Guide not found for {strategy_type}")
            if strategy_type in STRATEGY_GUIDES:
                 st.info("Showing local summary:")
                 st.markdown(STRATEGY_GUIDES[strategy_type].overview)

    # Handle Actions
    
    # Unified Strategy Analysis (Single Run)
    if submitted_single:
        # Prepare payload
        # Use session state to ensure latest values
        s_date = st.session_state.get("sl_start", start_date)
        e_date = st.session_state.get("sl_end", end_date)

        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": s_date.isoformat(),
            "end_date": e_date.isoformat(),
            "initial_capital": initial_capital,
            "commission_rate": commission,
            "position_size": 1.0,
            "strategy": strategy_type,
            **strategy_params
        }

        with st.spinner(f"Running {strategy_cfg['label']} Analysis..."):
            try:
                response = requests.post(f"{BACKEND_URL}/simulate", json=payload)
                response.raise_for_status()
                result = response.json()

                st.success("Analysis Completed!")

                # Metrics
                metrics = result["metrics"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Return", f"{metrics['return_pct']:.2f}%")
                m2.metric("Win Rate", f"{metrics['win_rate'] * 100:.1f}%")
                m3.metric("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")
                m4.metric("Trades", metrics["trade_count"])

                
                # --- Price & Trade Signals Chart ---
                if "price_series" in result and result["price_series"]:
                    # Prepare DataFrames
                    df_price = pd.DataFrame(result["price_series"])
                    df_price["date"] = pd.to_datetime(df_price["date"])
                    
                    df_trades = pd.DataFrame(result["trades"])
                    if not df_trades.empty:
                        df_trades["date"] = pd.to_datetime(df_trades["date"])
                    
                    # Equity Data for Shared Scale
                    equity_data = result["equity_curve"]
                    df_equity = pd.DataFrame(equity_data) if equity_data else pd.DataFrame()
                    if not df_equity.empty:
                        df_equity["date"] = pd.to_datetime(df_equity["date"])
                        
                    # Define Shared Scale
                    if not df_equity.empty:
                        x_min = df_equity["date"].min()
                        x_max = df_equity["date"].max()
                        # Add a small buffer or just use min/max
                        x_scale = alt.Scale(domain=[x_min, x_max])
                    else:
                        x_scale = alt.Scale() # Default auto

                    # Base Chart with Shared Scale
                    base = alt.Chart(df_price).encode(
                        x=alt.X("date:T", title="Date", scale=x_scale)
                    )
                    
                    # Price Line
                    price_line = base.mark_line().encode(
                        y=alt.Y("close:Q", title="Price", scale=alt.Scale(zero=False)),
                        tooltip=["date:T", "close:Q"]
                    )
                    
                    chart_layers = [price_line]
                    
                    # Optional MA Lines
                    if "ma_short" in df_price.columns:
                        ma_short_line = base.mark_line(strokeDash=[4, 2]).encode(
                            y=alt.Y("ma_short:Q"),
                            color=alt.value("#8888ff"),
                            tooltip=["date:T", "ma_short:Q"]
                        )
                        chart_layers.append(ma_short_line)
                        
                    if "ma_long" in df_price.columns:
                        ma_long_line = base.mark_line(strokeDash=[2, 2]).encode(
                            y=alt.Y("ma_long:Q"),
                            color=alt.value("#ff88ff"),
                            tooltip=["date:T", "ma_long:Q"]
                        )
                        chart_layers.append(ma_long_line)
                    
                    # Trade Markers
                    if not df_trades.empty:
                        # Buy Markers
                        buy_trades = df_trades[df_trades["side"] == "BUY"]
                        if not buy_trades.empty:
                            buy_markers = alt.Chart(buy_trades).mark_point(
                                shape="triangle-up",
                                size=100,
                                filled=True,
                                color="green"
                            ).encode(
                                x=alt.X("date:T", scale=x_scale),
                                y="price:Q",
                                tooltip=["date:T", "price:Q", "side:N", "quantity:Q"]
                            )
                            chart_layers.append(buy_markers)
                        
                        # Sell Markers
                        sell_trades = df_trades[df_trades["side"] == "SELL"]
                        if not sell_trades.empty:
                            sell_markers = alt.Chart(sell_trades).mark_point(
                                shape="triangle-down",
                                size=100,
                                filled=True,
                                color="red"
                            ).encode(
                                x=alt.X("date:T", scale=x_scale),
                                y="price:Q",
                                tooltip=["date:T", "price:Q", "side:N", "quantity:Q", "pnl:Q"]
                            )
                            chart_layers.append(sell_markers)
                    
                    # Combine Price Chart Layers
                    price_chart = alt.layer(*chart_layers).properties(
                        title="Price & Trade Signals",
                        height=300
                    )
                    
                    # Equity Chart with Shared Scale
                    if not df_equity.empty:
                        equity_chart = alt.Chart(df_equity).mark_line(color="green").encode(
                            x=alt.X("date:T", title="Date", scale=x_scale),
                            y=alt.Y("equity:Q", title="Equity ($)"),
                            tooltip=["date:T", "equity:Q"]
                        ).properties(
                            title="Equity Curve",
                            height=200
                        )
                        
                        # Combine Vertically and Sync Scale
                        combined_chart = alt.vconcat(
                            price_chart,
                            equity_chart
                        ).resolve_scale(
                            x='shared'
                        ).interactive()
                        
                        st.altair_chart(combined_chart, use_container_width=True)
                    else:
                        # Fallback if no equity data (shouldn't happen on success)
                        st.altair_chart(price_chart.interactive(), use_container_width=True)
                
                # --- Metrics & Trades ---
                metrics = result["metrics"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Return", f"{metrics['return_pct']:.2f}%")
                m2.metric("Win Rate", f"{metrics['win_rate'] * 100:.1f}%")
                m3.metric("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")
                m4.metric("Trades", metrics["trade_count"])
                
                st.subheader("Trade History")
                trades_data = result["trades"]
                if trades_data:
                    df_trades = pd.DataFrame(trades_data)
                    st.dataframe(df_trades, use_container_width=True)

                    # CSV Download
                    csv = df_trades.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Trades CSV",
                        data=csv,
                        file_name=f"strategy_lab_trades_{symbol}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No trades executed.")

            except requests.exceptions.RequestException as e:
                st.error(f"Backtest failed: {e}")
                if e.response is not None:
                    st.error(f"Details: {e.response.text}")
            
            # Reset auto-trigger flag
            if st.session_state.get("trigger_single_run", False):
                st.session_state["trigger_single_run"] = False
                st.rerun()

    # MA Cross Specific Optimization (Legacy/Specific Endpoint)
    if strategy_type == "ma_cross":
        # 1. Execution Logic
        if submitted_opt:
            # API Call for MA Cross Optimization
            payload = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "initial_capital": initial_capital,
                "commission_rate": commission, # Added back commission_rate
                "short_min": short_min,
                "short_max": short_max,
                "short_step": short_step,
                "long_min": long_min,
                "long_max": long_max,
                "long_step": long_step,
            }
            
            with st.spinner("Running MA Cross Optimization..."):
                try:
                    response = requests.post(f"{BACKEND_URL}/optimize/ma_cross", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    
                    st.success(f"Optimization Completed! Tested {data['total_combinations']} combinations.")
                    
                    # Store results in session state
                    st.session_state["ma_cross_opt_results"] = data
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Optimization failed: {e}")
        
        # 2. Display Logic (from Session State)
        if st.session_state.get("ma_cross_opt_results"):
            data = st.session_state["ma_cross_opt_results"]
            top_results = data.get("top_results", [])
            
            if not top_results:
                st.warning("No valid results found.")
            else:
                # Create DataFrame from all results
                rows = []
                for r in top_results:
                    row = r["params"].copy()
                    row.update(r["metrics"])
                    rows.append(row)
                df_results = pd.DataFrame(rows)

                # --- Apply Filters for Table & Best Params ---
                # Filter by Min Trades
                df_filtered = df_results[df_results["trade_count"] >= min_trades_filter]
                # Filter by Max Drawdown (convert decimal to %)
                df_filtered = df_filtered[df_filtered["max_drawdown"] * 100 <= max_dd_filter]
                # Filter out penalized scores
                df_filtered = df_filtered[df_filtered["score"] > -1e8]
                
                # Sort by Score
                if "score" in df_filtered.columns:
                    df_filtered = df_filtered.sort_values(by=["score", "sharpe_ratio"], ascending=[False, False])

                # --- Display Best Parameters (Filtered) ---
                if not df_filtered.empty:
                    best_row = df_filtered.iloc[0]
                    
                    st.markdown("### 🏆 Best Parameters (Filtered)")
                    st.caption("Best based on composite score (Sharpe + Return with DD / trade penalties)")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Short Window", int(best_row["short_window"]))
                    col2.metric("Long Window", int(best_row["long_window"]))
                    col3.metric("Total Return", f"{best_row['return_pct']:.2f}%")
                    col4.metric("Score", f"{best_row['score']:.2f}")
                    
                    # Update strategy save default values
                    best_metrics = best_row # For save form below
                    
                    # --- Apply to Single Run Button ---
                    st.caption("上の Best Parameters を Single Run の短期・長期ウィンドウに反映し、Single Run バックテストを自動実行します。")
                    
                    def apply_best_params_callback():
                        st.session_state["single_run_short_window"] = int(best_row["short_window"])
                        st.session_state["single_run_long_window"] = int(best_row["long_window"])
                        st.session_state["trigger_single_run"] = True
                    
                    st.button("Apply to Single Run", on_click=apply_best_params_callback, key="apply_best_btn")
                    
                    if st.session_state.get("trigger_single_run"):
                        st.success("Parameters applied! Switch to 'Single Run' tab to see results.")
                    best_metrics = best_row # For save form below
                else:
                    st.warning("⚠️ No strategies matched your filters. Please relax the Min Trades or Max Drawdown constraints.")
                    best_metrics = {} # Handle empty case

                # --- Heatmap (Unfiltered - Full View) ---
                st.subheader("🔥 Score by Parameter Combination (All Results)")
                try:
                    # import altair as alt (Moved to global scope)
                    chart = alt.Chart(df_results).mark_rect().encode(
                        x=alt.X('short_window:O', title='Short Window'),
                        y=alt.Y('long_window:O', title='Long Window'),
                        color=alt.Color('score:Q', title='Score', scale=alt.Scale(scheme='viridis')),
                        tooltip=['short_window', 'long_window', 'return_pct', 'sharpe_ratio', 'max_drawdown', 'score']
                    ).properties(
                        title="Optimization Score Heatmap"
                    )
                    st.altair_chart(chart, use_container_width=True)
                except Exception as e:
                    st.error(f"Heatmap error: {e}")
                    
                # --- Top Results Table (Filtered) ---
                st.subheader("Top Results (Filtered & Sorted)")
                
                if not df_filtered.empty:
                    st.dataframe(
                        df_filtered[[
                            "short_window", "long_window", "score", 
                            "return_pct", "sharpe_ratio", "max_drawdown", "trade_count"
                        ]], 
                        use_container_width=True,
                        column_config={
                            "score": st.column_config.NumberColumn("Score", format="%.2f"),
                            "return_pct": st.column_config.NumberColumn("Return %", format="%.2f%%"),
                            "sharpe_ratio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                            "max_drawdown": st.column_config.NumberColumn("Max DD", format="%.2f"),
                        }
                    )
                else:
                    st.info("No results match the current filters.")

                # Anti Summary: Restored Equity Curve by re-simulating top 5 results to get equity data (missing in optimization response).
                # --- Equity Curve Comparison (Top 5) ---
                if not df_filtered.empty:
                    st.markdown("##### Equity Curve Comparison (Top 5)")
                    
                    with st.spinner("Fetching equity curves for top results..."):
                        equity_data = []
                        top_5 = df_filtered.head(5)
                        
                        for idx, row in top_5.iterrows():
                            # Re-run backtest to get equity curve
                            s_params = {
                                "short_window": int(row["short_window"]),
                                "long_window": int(row["long_window"])
                            }
                            
                            payload = {
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "start_date": start_date.isoformat(),
                                "end_date": end_date.isoformat(),
                                "initial_capital": initial_capital,
                                "commission": commission,
                                "strategy": "ma_cross",
                                "params": s_params
                            }
                            
                            try:
                                # Use the simulation endpoint which returns full details including equity curve
                                res = requests.post(f"{BACKEND_URL}/simulate", json=payload, timeout=10)
                                if res.status_code == 200:
                                    bk_data = res.json()
                                    curve = bk_data.get("equity_curve", [])
                                    
                                    # Downsample if too large for plotting performance
                                    if len(curve) > 1000:
                                        step = len(curve) // 1000
                                        curve = curve[::step]
                                    
                                    for point in curve:
                                        equity_data.append({
                                            "date": point["date"],
                                            "equity": point["equity"],
                                            "strategy": f"S:{int(row['short_window'])}/L:{int(row['long_window'])}"
                                        })
                            except Exception:
                                pass # Skip if failed
                        
                        if equity_data:
                            df_equity = pd.DataFrame(equity_data)
                            df_equity['date'] = pd.to_datetime(df_equity['date'])
                            
                            chart = alt.Chart(df_equity).mark_line().encode(
                                x=alt.X('date:T', title='Date'),
                                y=alt.Y('equity:Q', title='Equity'),
                                color=alt.Color('strategy:N', title='Config'),
                                tooltip=['date:T', 'equity:Q', 'strategy:N']
                            ).properties(
                                height=400,
                                title='Top 5 Strategies Equity Curve'
                            ).interactive()
                            
                            st.altair_chart(chart, use_container_width=True)
                        else:
                            st.warning("Could not load equity curves.")

                        
                # Save Strategy Logic
                if not df_filtered.empty:
                    st.markdown("---")
                    st.subheader("💾 Save to Strategy Library")
                    with st.expander("Save Best Parameters as New Strategy", expanded=False):
                        with st.form("save_best_strategy_form_ma"):
                            default_name = f"{symbol}_{timeframe}_MACross_Best"
                            strategy_name = st.text_input("Strategy Name", value=default_name)
                            strategy_desc = st.text_area("Description", value=f"Grid Search Result. Return: {best_metrics['return_pct']:.2f}%")
                        
                            if st.form_submit_button("💾 Save Strategy"):
                                if not strategy_name:
                                    st.error("Strategy Name is required.")
                                else:
                                    lib = StrategyLibrary()
                                    new_strategy = {
                                        "id": str(uuid.uuid4()),
                                        "name": strategy_name,
                                        "description": strategy_desc,
                                        "created_at": datetime.now().isoformat(),
                                        "symbol": symbol,
                                        "timeframe": timeframe,
                                        "strategy_type": "ma_cross",
                                        "params": {
                                            "short_window": int(best_row["short_window"]),
                                            "long_window": int(best_row["long_window"])
                                        },
                                        "metrics": best_metrics.to_dict() if hasattr(best_metrics, 'to_dict') else best_metrics
                                    }
                                    lib.save_strategy(new_strategy)
                                    st.success(f"Strategy '{strategy_name}' saved successfully!")


    # ==========================================
    # Optimization Logic Removed
    # ==========================================
    # The optimization logic has been moved to render_parameter_optimization_section().
    # This placeholder is kept to maintain file structure until full migration.



    # --- Live Trading Setup (Outside Form) ---
    if strategy_type == "ma_cross":
        st.divider()
        st.subheader("Live Trading Setup")
        st.caption("Save the current MA Crossover settings as the live trading strategy.")

        position_shares = 0
        position_amount_jpy = 0
        
        position_mode = st.radio(
            "Position sizing mode",
            options=["Fixed shares", "Fixed amount (JPY)"],
            index=1,
            key="ma_position_mode",
            horizontal=True
        )
        
        if position_mode == "Fixed shares":
            position_shares = st.number_input(
                "Fixed Position Size (shares)", min_value=1, max_value=10000, value=100, step=1, key="ma_fixed_shares"
            )
        else:
            position_amount_jpy = st.number_input(
                "Amount per trade (JPY)", min_value=10000, max_value=1000000, value=100000, step=10000, key="ma_fixed_amount"
            )

        if st.button("Set as Live Strategy 🚀", key="set_live_ma"):
            risk_config = {}
            if position_mode == "Fixed shares":
                risk_config = {
                    "position_mode": "fixed_shares",
                    "position_value": float(position_shares),
                }
            else:
                risk_config = {
                    "position_mode": "fixed_amount_jpy",
                    "position_value": float(position_amount_jpy),
                }
                
            payload = {
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy_name": "MA Crossover",
                "strategy_type": "ma_cross",
                "params": {
                    "short_window": int(short_window),
                    "long_window": int(long_window),
                },
                "risk": risk_config,
            }

            try:
                res = requests.post(f"{BACKEND_URL}/live-strategy", json=payload, timeout=10)
                if res.status_code == 200:
                    st.success("Live strategy saved successfully! 🎯")
                else:
                    st.error(f"Failed to save live strategy: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"Error while saving live strategy: {e}")

    elif strategy_type == "rsi_mean_reversion":
        st.divider()
        st.subheader("Live Trading Setup")
        st.caption("Save the current RSI settings as the live trading strategy.")

        position_shares = 0
        position_amount_jpy = 0
        
        position_mode = st.radio(
            "Position sizing mode",
            options=["Fixed shares", "Fixed amount (JPY)"],
            index=1,
            key="rsi_position_mode",
            horizontal=True
        )
        
        if position_mode == "Fixed shares":
            position_shares = st.number_input(
                "Fixed Position Size (shares)", min_value=1, max_value=10000, value=100, step=1, key="rsi_fixed_shares"
            )
        else:
            position_amount_jpy = st.number_input(
                "Amount per trade (JPY)", min_value=10000, max_value=1000000, value=100000, step=10000, key="rsi_fixed_amount"
            )

        if st.button("Set as Live Strategy 🚀", key="set_live_rsi"):
            risk_config = {}
            if position_mode == "Fixed shares":
                risk_config = {
                    "position_mode": "fixed_shares",
                    "position_value": float(position_shares),
                }
            else:
                risk_config = {
                    "position_mode": "fixed_amount_jpy",
                    "position_value": float(position_amount_jpy),
                }

            payload = {
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy_name": "RSI Mean Reversion",
                "strategy_type": "rsi_mean_reversion",
                "params": {
                    "rsi_period": int(rsi_period),
                    "oversold_level": int(oversold),
                    "overbought_level": int(overbought),
                },
                "risk": risk_config,
            }

            try:
                res = requests.post(f"{BACKEND_URL}/live-strategy", json=payload, timeout=10)
                if res.status_code == 200:
                    st.success("Live strategy saved successfully! 🎯")
                else:
                    st.error(f"Failed to save live strategy: {res.status_code} - {res.text}")
            except Exception as e:
                st.error(f"Error while saving live strategy: {e}")
                st.subheader("📊 Top Results")
                if df_results is not None:
                    display_df = df_results.copy()
                    # Rename for display
                    display_df = display_df.rename(columns={
                        x_param: opt_config["x_label"],
                        y_param: opt_config["y_label"],
                        "return_pct": "Return (%)",
                        "max_drawdown": "Max DD (%)",
                        "sharpe_ratio": "Sharpe",
                        "win_rate": "Win Rate",
                        "trade_count": "Trades"
                    })
                    
                    st.dataframe(display_df, use_container_width=True)
                
                # Save Strategy
                st.markdown("---")
                st.subheader("💾 Save to Strategy Library")
                with st.expander("Save Best Parameters as New Strategy", expanded=False):
                    with st.form("save_best_strategy_form_opt"): # Changed key to avoid conflict
                        default_name = f"{symbol}_{timeframe}_{strategy_type}_Best"
                        strategy_name = st.text_input("Strategy Name", value=default_name)
                        strategy_desc = st.text_area("Description", value=f"Grid Search Result. Return: {best_metrics['return_pct']:.2f}%")
                        
                        if st.form_submit_button("💾 Save Strategy"):
                            if not strategy_name:
                                st.error("Strategy Name is required.")
                            else:
                                lib = StrategyLibrary()
                                new_strategy = {
                                    "id": str(uuid.uuid4()),
                                    "name": strategy_name,
                                    "description": strategy_desc,
                                    "created_at": datetime.now().isoformat(),
                                    "symbol": symbol,
                                    "timeframe": timeframe,
                                    "strategy_type": strategy_type,
                                    "params": best_params,
                                    "metrics": best_metrics
                                }
                                lib.save_strategy(new_strategy)
                                st.success(f"Strategy '{strategy_name}' saved successfully!")

    if submitted_single and strategy_type not in ["ma_cross", "price_breakout", "ema9_dip_buy", "rsi_mean_reversion"]: # For other strategies
        # Placeholder for other strategies
        st.info(f"**{strategy_cfg['label']}** selected.")
        st.warning("この戦略タイプの自動バックテストは v0.3 以降で実装予定です。")
        st.write("Parameters captured (for future use):")
        if strategy_type == "rsi_mean_reversion":
            st.json({"period": rsi_period, "buy_level": oversold, "sell_level": overbought})
            
            # Run single backtest for RSI
            # requests is imported globally
            payload = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "initial_capital": initial_capital,
                "commission": commission,
                "strategy": "rsi_mean_reversion",
                "params": {
                    "period": int(rsi_period),
                    "buy_level": int(oversold),
                    "sell_level": int(overbought)
                }
            }
            try:
                with st.spinner("Running RSI backtest..."):
                    response = requests.post(f"{BACKEND_URL}/simulate", json=payload, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                
                # Display results
                st.success("✅ Backtest completed!")
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("Total Return", f"{result['metrics']['return_pct']:.2f}%")
                with col_r2:
                    st.metric("Sharpe Ratio", f"{result['metrics']['sharpe_ratio']:.2f}")
                with col_r3:
                    st.metric("Max Drawdown", f"{result['metrics']['max_drawdown'] * 100:.2f}%")
                
                # Equity Curve
                if "equity_curve" in result and result["equity_curve"]:
                    equity_df = pd.DataFrame(result["equity_curve"])
                    df_equity = equity_df.copy()
                    df_equity["date"] = pd.to_datetime(df_equity["date"])
                    df_equity.set_index("date", inplace=True)
                    st.line_chart(df_equity["equity"])
            except requests.exceptions.RequestException as e:
                st.error(f"Backtest failed: {e}")

        elif strategy_type == "price_breakout":
            # Already handled in the block above
            pass
            
    # ==========================================
    # ==========================================
    # Symbol Preset Settings (Developer Tools)
    # ==========================================
    # This section was moved to its own function: render_symbol_preset_section()


def render_symbol_preset_section():
    """
    Symbol Preset Settings section - Manage symbol presets (Developer Tools).
    Ported from the old 'Symbol Preset Settings (Developer Tools)' block.
    Persists data to 'data/symbol_presets.json' via save_symbol_presets().
    """
    st.header("🔧 Symbol Preset Settings")
    st.caption("Manage default settings and recommended parameters for each symbol. (Developer Mode)")
    
    st.markdown("### Current Presets")
    symbols = load_symbol_presets()

    if symbols:
        df = pd.DataFrame(symbols)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No symbol presets found.")

    st.markdown("### Add New Preset")
    with st.form("add_preset_form"):
        col_add1, col_add2 = st.columns(2)
        with col_add1:
            new_symbol = st.text_input("Symbol", key="new_symbol_input", placeholder="e.g. NVDA, 7203.T")
        with col_add2:
            new_label = st.text_input("Label (optional)", key="new_symbol_label", placeholder="e.g. NVIDIA")
        
        submitted = st.form_submit_button("➕ Add Preset")
        if submitted:
            new_symbol_val = new_symbol.strip().upper()
            if not new_symbol_val:
                st.warning("Symbol cannot be empty.")
            elif any(s["symbol"] == new_symbol_val for s in symbols):
                st.warning(f"Symbol '{new_symbol_val}' already exists in presets.")
            else:
                new_entry = {
                    "symbol": new_symbol_val,
                    "label": new_label.strip() or new_symbol_val
                }
                symbols.append(new_entry)
                save_symbol_presets(symbols)
                st.success(f"✅ Added preset: {new_symbol_val}")
                st.rerun()

    st.markdown("### Delete Preset")
    if symbols:
        with st.form("delete_preset_form"):
            delete_target = st.selectbox(
                "Select symbol to delete",
                options=[s["symbol"] for s in symbols],
                key="delete_symbol_select"
            )
            if st.form_submit_button("🗑️ Delete Preset"):
                updated = [s for s in symbols if s["symbol"] != delete_target]
                if not updated:
                    st.warning("At least one symbol must remain.")
                else:
                    save_symbol_presets(updated)
                    st.success(f"✅ Deleted preset: {delete_target}")
                    st.rerun()
    else:
        st.info("No presets to delete.")


class StrategyLibrary:
    """
    Simple file-based strategy library manager.
    """
    FILE_PATH = "data/strategies.json"

    def __init__(self):
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists(self.FILE_PATH):
            with open(self.FILE_PATH, "w") as f:
                json.dump({"strategies": []}, f)

    def load_strategies(self) -> List[Dict]:
        try:
            with open(self.FILE_PATH, "r") as f:
                data = json.load(f)
                return data.get("strategies", [])
        except Exception:
            return []

    def save_strategy(self, strategy: Dict):
        strategies = self.load_strategies()
        # Ensure favorite field exists (for backward compatibility)
        if "favorite" not in strategy:
            strategy["favorite"] = False
        strategies.append(strategy)
        with open(self.FILE_PATH, "w") as f:
            json.dump({"strategies": strategies}, f, indent=2)

    def get_strategy(self, strategy_id: str) -> Optional[Dict]:
        strategies = self.load_strategies()
        for s in strategies:
            if s["id"] == strategy_id:
                return s
        return None
    
    def update_strategy(self, strategy_id: str, updates: Dict) -> bool:
        """Update specific fields of a strategy"""
        strategies = self.load_strategies()
        for i, s in enumerate(strategies):
            if s["id"] == strategy_id:
                strategies[i].update(updates)
                with open(self.FILE_PATH, "w") as f:
                    json.dump({"strategies": strategies}, f, indent=2)
                return True
        return False
    
    def delete_strategy(self, strategy_id: str) -> bool:
        """Delete a strategy by ID"""
        strategies = self.load_strategies()
        original_len = len(strategies)
        strategies = [s for s in strategies if s["id"] != strategy_id]
        if len(strategies) < original_len:
            with open(self.FILE_PATH, "w") as f:
                json.dump({"strategies": strategies}, f, indent=2)
            return True
        return False
    
    def toggle_favorite(self, strategy_id: str) -> bool:
        """Toggle favorite status of a strategy"""
        strategies = self.load_strategies()
        for i, s in enumerate(strategies):
            if s["id"] == strategy_id:
                current_favorite = s.get("favorite", False)
                strategies[i]["favorite"] = not current_favorite
                with open(self.FILE_PATH, "w") as f:
                    json.dump({"strategies": strategies}, f, indent=2)
                return True
        return False


# =============================================================================
# Main App Logic
# =============================================================================


def fetch_chart_data(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """
    For now this just calls the mock generator, but in future we can
    call the FastAPI backend to fetch real OHLCV data from yfinance or Alpaca.
    """
    # TODO: connect to backend/data_feed.py
    _ = symbol, timeframe, limit  # unused for now
    return generate_mock_price_data(periods=limit)


def main():
    """Main application with mode switch"""
    # Page Config
    st.set_page_config(
        page_title="AI Signal Chart - Dev Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize Session State
    if "selected_strategy_label" not in st.session_state:
        st.session_state["selected_strategy_label"] = "MA Cross"
    if "selected_strategy_type" not in st.session_state:
        st.session_state["selected_strategy_type"] = "ma_cross"
    if "single_run_params" not in st.session_state:
        st.session_state["single_run_params"] = {}
    if "optimization_params" not in st.session_state:
        st.session_state["optimization_params"] = {}

    # Sidebar mode switch
    # Use session state to control default selection if navigated from URL
    # We bind directly to "mode" key so st.session_state["mode"] is automatically updated
    
    # Ensure mode is in options, otherwise default to Dashboard
    options = ["Developer Dashboard", "Backtest Lab", "Strategy Lab", "Live Signal", "Predictor Backtest Lab", "Market Scanner", "Auto Sim Lab"]
    if st.session_state.get("mode") not in options:
        st.session_state["mode"] = "Developer Dashboard"

    mode = st.sidebar.selectbox(
        "Mode",
        options=options,
        key="mode"
    )
    
    mode_map_rev = {
        "Developer Dashboard": "dashboard",
        "Backtest Lab": "backtest",
        "Strategy Lab": "strategy",
        "Live Signal": "live_signal",
        "Predictor Backtest Lab": "predictor",
        "Market Scanner": "scanner",
        "Auto Sim Lab": "auto_sim"
    }
    
    # Update URL params whenever mode changes
    # We can do this by checking if the URL param matches the current mode
    current_url_mode = st.query_params.get("mode")
    target_url_mode = mode_map_rev.get(mode, "dashboard")
    
    if current_url_mode != target_url_mode:
        params = st.query_params.to_dict()
        params["mode"] = target_url_mode
        st.query_params.clear()
        st.query_params.update(params)
    
    # Strategy Lab Section Navigator (shown only when in Strategy Lab mode)
    lab_section = None
    if mode == "Strategy Lab":
        st.sidebar.markdown("---")
        lab_section = st.sidebar.radio(
            "📑 Strategy Lab Sections",
            [
                "Configuration",
                "Single Analysis",
                "Parameter Optimization",
                "Saved Strategies",
                "Symbol Preset Settings"
            ],
            key="sl_section"
        )

    if mode == "Developer Dashboard":
        # Header
        st.title("📊 EXITON Developer Dashboard")
        st.caption("Real-time monitoring and control for paper trading system")

        # Sidebar (controls below the mode selector)
        symbol, timeframe, limit, quantity, selected_ma, refresh = render_sidebar()

        # Store in session state
        if "symbol" not in st.session_state or refresh:
            st.session_state.symbol = symbol
        
        # Mock Data & Chart
        df = fetch_chart_data(symbol, timeframe, limit)
        render_main_chart(df, selected_ma)
        render_ma_signals(df, selected_ma)
        
        # Additional metrics (optional)
        render_account_summary()
        render_risk_metrics()

        # Main content area - 2 columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
             # Rule Predictor v2 Section
             st.markdown("### 🔮 Rule Predictor v2")
             with st.expander("Show Rule Predictor v2 Details", expanded=True):
                 try:
                     # Call Backend
                     resp = requests.get(f"{BACKEND_URL}/rule_predictor_v2?symbol={symbol}")
                     if resp.status_code == 200:
                         data = resp.json()
                         
                         # Construct pred object for render_predictor_card
                         pred = {
                             "score": data["score"],
                             "prob_up": data["prob_up"],
                             "prob_down": data["prob_down"],
                             "signals": data["signals"],
                             "raw_signals": data.get("raw_signals"),
                             "direction": "flat"
                         }
                         
                         # Calculate direction
                         if pred["prob_up"] > 0.55: pred["direction"] = "UP"
                         elif pred["prob_down"] > 0.55: pred["direction"] = "DOWN"
                         
                         render_predictor_card("Rule Predictor v2", pred)
                         
                         # Timeframe Guidance for Dashboard
                         render_timeframe_guidance("1d", mode="dashboard")
                     else:
                         st.error(f"Failed to fetch prediction: {resp.text}")
                 except Exception as e:
                     st.error(f"Error connecting to backend: {e}")

        with col2:
            render_account_summary()
            render_risk_metrics()

        # Tabs for detailed views
        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📋 Positions", "📜 Trades", "💰 P&L"])

        with tab1:
            render_positions_tab()

        with tab2:
            render_trades_tab()

        with tab3:
            render_pnl_tab()

        # Footer
        st.markdown("---")
        st.caption("EXITON Developer Dashboard | Powered by Streamlit & FastAPI")

    elif mode == "Backtest Lab":
        render_backtest_ui()

    elif mode == "Strategy Lab":
        render_strategy_lab()

    elif mode == "Live Signal":
        render_live_signal()
    
    elif mode == "Predictor Backtest Lab":
        render_predictor_backtest_lab()
        
    elif mode == "Market Scanner":
        render_market_scanner()
    
    elif mode == "Auto Sim Lab":
        render_auto_sim_lab()




# =============================================================================
# Market Scanner UI
# =============================================================================

def render_market_scanner():
    st.title("📡 EXITON Market Scanner")
    st.caption("Scan the universe for swing-trade opportunities using Rule Predictor v2.")
    
    # Inputs
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col1:
        universe = st.selectbox("Universe", ["default", "sp500", "mvp"], index=0, key="scan_universe")
    
    with col2:
        timeframe = st.selectbox("Timeframe", ["1d"], index=0, key="scan_timeframe", disabled=True)
    
    with col3:
        lookback = st.slider("Lookback", min_value=100, max_value=300, value=200, step=10, key="scan_lookback")
    
    with col4:
        limit = st.slider("Top N", min_value=10, max_value=100, value=50, step=10, key="scan_limit")
        
    with col5:
        st.write("") # Spacer
        st.write("")
        run_btn = st.button("🚀 Run Market Scan", type="primary", use_container_width=True)
        
    # Session state for results
    if "scan_results" not in st.session_state:
        st.session_state["scan_results"] = None
        
    if run_btn:
        try:
            with st.spinner(f"Scanning {universe} universe..."):
                params = {
                    "universe": universe,
                    "timeframe": timeframe,
                    "lookback": lookback,
                    "limit": limit
                }
                
                response = requests.get(f"{BACKEND_URL}/scan_market", params=params, timeout=120)
                response.raise_for_status()
                data = response.json()
                st.session_state["scan_results"] = data
                
        except Exception as e:
            st.error(f"Error running scan: {e}")
            return
            
    # Display Results
    if st.session_state["scan_results"]:
        data = st.session_state["scan_results"]
        results = data.get("results", [])
        
        st.markdown("---")
        st.subheader(f"📊 Scan Results ({len(results)} symbols)")
        st.caption(f"Sorted by Bullish Score (Combined Rule v2 + Stat). Timeframe: {data.get('timeframe')}")
        
        if not results:
            st.warning("No results found.")
            return
            
        # Prepare table data for display
        table_data = []
        for res in results:
            signal = res.get("final_signal", "FLAT")
            signal_icon = "➖"
            if "UP" in signal: signal_icon = "📈"
            elif "DOWN" in signal: signal_icon = "📉"
            
            table_data.append({
                "Symbol": res.get("symbol"),
                "Price": f"${res.get('latest_price', 0):.2f}",
                "Signal": f"{signal_icon} {signal}",
                "Score": f"{res.get('combined_score', 0):.2f}",
                "Rule": f"{res.get('rule_direction')} ({res.get('rule_score'):.2f})",
                "Stat": f"{res.get('stat_direction')} ({res.get('stat_conf'):.2f})",
            })
            
        df = pd.DataFrame(table_data)
        
        # Custom Styling
        def highlight_signal(row):
            styles = [""] * len(row)
            signal = row["Signal"]
            if "STRONG_UP" in signal:
                return ["background-color: rgba(46, 204, 113, 0.2)"] * len(row)
            elif "UP" in signal:
                return ["background-color: rgba(46, 204, 113, 0.1)"] * len(row)
            elif "STRONG_DOWN" in signal:
                return ["background-color: rgba(231, 76, 60, 0.2)"] * len(row)
            elif "DOWN" in signal:
                return ["background-color: rgba(231, 76, 60, 0.1)"] * len(row)
            return styles
            
        st.dataframe(df.style.apply(highlight_signal, axis=1), use_container_width=True, hide_index=True)
        
        # Navigation buttons for each symbol
        st.markdown("---")
        st.subheader("🧭 Quick Actions")
        st.caption("Click a symbol to open it in Live Signal with current settings.")
        
        # Display buttons in grid (4 per row)
        import urllib.parse
        
        cols = st.columns(len(results))
        # Limit to first 8 results to avoid clutter if list is long, or just use grid
        # The user's snippet implies a row per result or columns matching results.
        # Let's stick to the grid layout but use the link logic.
        
        # Actually, the user snippet used `cols = st.columns(len(results))` which implies a single row.
        # But previous code had a grid. Let's adapt the user's snippet to the grid structure if possible,
        # or just use the user's snippet if results are few.
        # The previous code handled pagination/grid.
        # Let's use a grid layout but render HTML links inside.
        
        cols_per_row = 4
        for i in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < len(results):
                    res = results[i + j]
                    symbol_name = res.get("symbol")
                    signal = res.get("final_signal", "FLAT")
                    score = res.get("combined_score", 0)
                    
                    # Signal emoji
                    signal_emoji = "➖"
                    if "STRONG_UP" in signal: signal_emoji = "🟢"
                    elif "UP" in signal: signal_emoji = "🔵"
                    elif "STRONG_DOWN" in signal: signal_emoji = "🔴"
                    elif "DOWN" in signal: signal_emoji = "🟠"
                    
                    # Construct URL
                    live_params = {
                        "mode": "live_signal",
                        "symbol": symbol_name,
                        "timeframe": timeframe,
                        "lookback": str(lookback),
                        "universe": universe,
                    }
                    live_query = urllib.parse.urlencode(live_params)
                    live_href = f"?{live_query}"
                    
                    with col:
                        # Render as HTML link styled like a button
                        st.markdown(
                            f"""
                            <a href="{live_href}" target="_self" style="text-decoration: none;">
                              <div style="
                                  width: 100%;
                                  padding: 6px 10px;
                                  border-radius: 8px;
                                  border: 1px solid #444;
                                  background-color: {'#113B5C' if score >= 0 else '#5C1111'};
                                  color: white;
                                  font-size: 14px;
                                  text-align: center;
                                  cursor: pointer;
                                  display: block;
                              ">
                                {signal_emoji} {symbol_name} ({score:+.2f})
                              </div>
                            </a>
                            """,
                            unsafe_allow_html=True,
                        )
        
        st.info("💡 Tip: Click a symbol button above to instantly open **Live Signal** with that ticker preloaded.")


# =============================================================================
# Predictor Backtest Lab UI
# =============================================================================

def render_predictor_backtest_lab():
    st.title("🔬 Predictor Backtest Lab")
    st.caption("Compare predictors on historical data to evaluate performance.")
    
    # Inputs
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        universe = load_symbol_universe(UNIVERSE_CSV)
        default_symbols = ["NVDA", "SMCI", "TSLA", "COIN", "MSTR", "AAPL", "MSFT"]
        options = universe if universe else default_symbols
        symbol = st.selectbox("Symbol", options=options, index=0, key="pbl_symbol")
    
    with col2:
        timeframe = st.selectbox("Timeframe", ["1d"], index=0, key="pbl_timeframe", 
                                  help="Currently only 1d is supported")
    
    with col3:
        date_presets = {
            "Last 1 Year": 365,
            "Last 2 Years": 730,
            "Last 3 Years": 1095,
            "Max Available": 0
        }
        preset = st.selectbox("Date Range", list(date_presets.keys()), index=2, key="pbl_preset")
    
    with col4:
        st.write("")  # Spacer
        st.write("")
        run_btn = st.button("🚀 Run Predictor Backtest", type="primary", use_container_width=True)
    
    # Calculate dates
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime("%Y-%m-%d")
    days = date_presets[preset]
    if days > 0:
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    else:
        start_date = None  # Max available
    
    # Session state for results
    if "pbl_results" not in st.session_state:
        st.session_state["pbl_results"] = None
    
    if run_btn:
        try:
            with st.spinner(f"Running predictor backtest for {symbol}..."):
                params = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                }
                if start_date:
                    params["start"] = start_date
                params["end"] = end_date
                
                response = requests.get(f"{BACKEND_URL}/predictor_backtest", params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                st.session_state["pbl_results"] = data
                
        except Exception as e:
            st.error(f"Error running backtest: {e}")
            return
    
    # Display results
    if st.session_state["pbl_results"]:
        data = st.session_state["pbl_results"]
        results = data.get("results", {})
        
        st.markdown("---")
        st.subheader("📊 Predictor Comparison Results")
        st.caption(f"Symbol: {data.get('symbol')} | Timeframe: {data.get('timeframe')} | Period: {data.get('start_date', 'Max')} to {data.get('end_date', 'Now')}")
        
        # Build comparison table
        table_data = []
        predictor_names = {
            "buy_and_hold": "📈 Buy & Hold (Baseline)",
            "stat": "📊 Stat Predictor",
            "rule_v2": "🔮 Rule Predictor v2"
        }
        
        best_return = -float("inf")
        worst_return = float("inf")
        
        for pred_key, pred_name in predictor_names.items():
            res = results.get(pred_key, {})
            if "error" in res:
                table_data.append({
                    "Predictor": pred_name,
                    "Total Return": "Error",
                    "Win Rate": "-",
                    "# Trades": "-",
                    "Max DD": "-",
                    "Sharpe": "-"
                })
            else:
                total_return = res.get("total_return", 0)
                if total_return > best_return:
                    best_return = total_return
                if total_return < worst_return:
                    worst_return = total_return
                    
                table_data.append({
                    "Predictor": pred_name,
                    "Total Return": f"{total_return:.2f}%",
                    "Win Rate": f"{res.get('win_rate', 0):.1f}%",
                    "# Trades": res.get("num_trades", 0),
                    "Max DD": f"{res.get('max_drawdown', 0):.2f}%",
                    "Sharpe": res.get("sharpe_ratio", "-") if res.get("sharpe_ratio") else "-"
                })
        
        # Display table
        df_table = pd.DataFrame(table_data)
        
        # Custom styling for table
        def highlight_returns(row):
            styles = [""] * len(row)
            val = row["Total Return"]
            if val != "Error" and val != "-":
                val_num = float(val.replace("%", ""))
                if val_num == best_return:
                    styles[1] = "background-color: rgba(46, 204, 113, 0.3); color: #2ecc71; font-weight: bold;"
                elif val_num == worst_return:
                    styles[1] = "background-color: rgba(231, 76, 60, 0.2); color: #e74c3c;"
            return styles
        
        styled_df = df_table.style.apply(highlight_returns, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Equity Curve Chart
        st.markdown("---")
        st.subheader("📈 Equity Curves")
        
        # Prepare chart data
        chart_data = []
        for pred_key, pred_name in predictor_names.items():
            res = results.get(pred_key, {})
            if "error" not in res and "equity_curve" in res:
                for point in res["equity_curve"]:
                    chart_data.append({
                        "Date": point["date"],
                        "Equity": point["equity"],
                        "Predictor": pred_name.split(" ")[-1] if "(" in pred_name else pred_name.split(" ")[1]  # Simplify name
                    })
        
        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            chart_df["Date"] = pd.to_datetime(chart_df["Date"])
            
            # Use Altair for multi-line chart
            chart = alt.Chart(chart_df).mark_line().encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Equity:Q", title="Equity ($)"),
                color=alt.Color("Predictor:N", legend=alt.Legend(title="Predictor")),
                strokeDash=alt.condition(
                    alt.datum.Predictor == "Hold",
                    alt.value([5, 5]),
                    alt.value([0])
                )
            ).properties(
                height=400
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
        
        # Additional metrics
        st.markdown("---")
        st.subheader("📋 Detailed Metrics")
        
        cols = st.columns(3)
        for i, (pred_key, pred_name) in enumerate(predictor_names.items()):
            res = results.get(pred_key, {})
            with cols[i]:
                st.markdown(f"**{pred_name}**")
                if "error" not in res:
                    st.metric("Final Equity", f"${res.get('final_equity', 0):,.2f}")
                    st.metric("Total Return", f"{res.get('total_return', 0):.2f}%")
                else:
                    st.error(res.get("error", "Unknown error"))


# =============================================================================
# Auto Sim Lab UI
# =============================================================================

def render_auto_sim_lab():
    """
    Auto Sim Lab - Automated Paper Trading Simulation using Final Signal.
    
    This UI allows users to:
    - Configure simulation parameters (symbol, timeframe, capital, risk)
    - Run automated trading simulation (Historical or Realtime mode)
    - View equity curve, trades, and decision log
    """
    st.title("🤖 Auto Sim Lab")
    st.caption("Automated Paper Trading Simulation using Final Signal predictors")
    
    # Mode selector
    sim_mode = st.radio(
        "Simulation Mode",
        ["📊 Historical (Backtest)", "🔴 Realtime (Live)"],
        horizontal=True,
        key="auto_sim_mode"
    )
    
    is_realtime = sim_mode == "🔴 Realtime (Live)"
    
    if is_realtime:
        render_auto_sim_realtime()
    else:
        render_auto_sim_historical()


def render_auto_sim_historical():
    """Render Historical (backtest) Auto Sim Lab UI."""
    st.markdown("""
    **Historical Mode** simulates automated trading on historical data using your chosen strategy.
    Select between **Final Signal** (using Live Signal predictors) or **MA Crossover** (matching Strategy Lab).
    """)
    
    st.markdown("---")
    
    # Strategy Mode Selection
    st.subheader("🎯 Strategy Mode")
    
    strategy_mode = st.radio(
        "Select Strategy",
        ["Final Signal (Default)", "MA Crossover (Match Strategy Lab)"],
        horizontal=True,
        key="auto_sim_hist_strategy_mode"
    )
    
    is_ma_mode = strategy_mode == "MA Crossover (Match Strategy Lab)"
    
    # MA Parameters (shown only for MA mode)
    ma_short_window = None
    ma_long_window = None
    
    if is_ma_mode:
        st.info("📊 **MA Crossover Mode**: Uses the same Moving Average Crossover logic as Strategy Lab for consistency verification.")
        col_ma1, col_ma2 = st.columns(2)
        with col_ma1:
            ma_short_window = st.number_input(
                "Short MA Window",
                min_value=1,
                max_value=200,
                value=50,
                key="auto_sim_hist_ma_short"
            )
        with col_ma2:
            ma_long_window = st.number_input(
                "Long MA Window",
                min_value=2,
                max_value=400,
                value=60,
                key="auto_sim_hist_ma_long"
            )
        
        if ma_short_window >= ma_long_window:
            st.warning("⚠️ Short Window must be less than Long Window")
    
    st.markdown("---")
    
    # Configuration Section
    st.subheader("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = render_symbol_selector(key_prefix="auto_sim_hist", container=st)
        timeframe = st.selectbox(
            "Timeframe",
            options=["1d", "1h", "4h", "1wk"],
            index=0,
            key="auto_sim_hist_timeframe"
        )
    
    with col2:
        initial_capital = st.number_input(
            "Initial Capital ($)",
            min_value=1000.0,
            max_value=10000000.0,
            value=100000.0,
            step=10000.0,
            key="auto_sim_hist_capital"
        )
    
    # Position Sizing Mode
    st.markdown("---")
    st.subheader("📐 Position Sizing")
    
    position_sizing_mode = st.selectbox(
        "Position Sizing Mode",
        ["Percent of Equity", "Full Equity", "Fixed Shares", "Fixed Dollar Amount"],
        key="auto_sim_hist_pos_mode"
    )
    
    # Map display names to API values
    pos_mode_map = {
        "Percent of Equity": "percent_of_equity",
        "Full Equity": "full_equity",
        "Fixed Shares": "fixed_shares",
        "Fixed Dollar Amount": "fixed_dollar"
    }
    pos_mode_value = pos_mode_map[position_sizing_mode]
    
    # Show relevant inputs based on mode
    risk_pct = 1.0
    fixed_shares = None
    fixed_dollar_amount = None
    
    if position_sizing_mode == "Percent of Equity":
        risk_pct = st.slider(
            "Risk per Trade (%)",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.5,
            key="auto_sim_hist_risk",
            help="Percentage of equity to risk per trade"
        )
    elif position_sizing_mode == "Full Equity":
        st.caption("Uses 100% of available equity for each trade (maximum position size).")
    elif position_sizing_mode == "Fixed Shares":
        fixed_shares = st.number_input(
            "Number of Shares",
            min_value=1,
            max_value=10000,
            value=100,
            key="auto_sim_hist_fixed_shares"
        )
    elif position_sizing_mode == "Fixed Dollar Amount":
        fixed_dollar_amount = st.number_input(
            "Dollar Amount ($)",
            min_value=100.0,
            max_value=1000000.0,
            value=10000.0,
            step=1000.0,
            key="auto_sim_hist_fixed_dollar"
        )
    
    st.markdown("---")
    
    # R-Management Panel
    st.subheader("📊 R-Management (Optional)")
    
    use_r_management = st.checkbox(
        "Enable R-Management",
        value=False,
        key="auto_sim_hist_use_r",
        help="Track trades in R (risk) multiples using virtual stops"
    )
    
    virtual_stop_method = "percent"
    virtual_stop_atr_multiplier = 2.0
    virtual_stop_percent = 0.03
    
    if use_r_management:
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            virtual_stop_method = st.selectbox(
                "Virtual Stop Method",
                ["percent", "atr"],
                key="auto_sim_hist_stop_method",
                help="How to calculate the virtual stop for R calculation"
            )
        
        with col_r2:
            if virtual_stop_method == "atr":
                virtual_stop_atr_multiplier = st.number_input(
                    "ATR Multiplier",
                    min_value=0.5,
                    max_value=10.0,
                    value=2.0,
                    step=0.5,
                    key="auto_sim_hist_atr_mult",
                    help="Stop = Entry - (ATR × Multiplier)"
                )
            else:
                virtual_stop_percent = st.number_input(
                    "Stop Percent (%)",
                    min_value=0.5,
                    max_value=20.0,
                    value=3.0,
                    step=0.5,
                    key="auto_sim_hist_stop_pct",
                    help="Stop = Entry × (1 - Stop%)"
                ) / 100.0
    
    st.markdown("---")
    
    # Execution Mode Panel
    st.subheader("⚡ Execution Mode")
    
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        execution_mode = st.selectbox(
            "Execution Mode",
            ["same_bar_close", "next_bar_open"],
            key="auto_sim_hist_exec_mode",
            help="same_bar_close = execute at signal bar close. next_bar_open = execute at next bar open (more realistic)"
        )
    
    with col_ex2:
        commission_percent = st.number_input(
            "Commission (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            key="auto_sim_hist_commission",
            help="Commission as percentage of trade value"
        ) / 100.0
    
    with col_ex3:
        slippage_percent = st.number_input(
            "Slippage (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            key="auto_sim_hist_slippage",
            help="Slippage as percentage of trade price"
        ) / 100.0
    
    st.markdown("---")
    
    # Loss Control Panel
    st.subheader("🛡️ Loss Control (Optional)")
    
    col_lc1, col_lc2 = st.columns(2)
    with col_lc1:
        use_max_dd = st.checkbox("Enable Max Drawdown Limit", value=False, key="auto_sim_hist_use_max_dd")
        max_drawdown_percent = None
        if use_max_dd:
            max_drawdown_percent = st.number_input(
                "Max Drawdown (%)",
                min_value=1.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                key="auto_sim_hist_max_dd",
                help="Halt simulation if drawdown exceeds this percentage"
            ) / 100.0
    
    with col_lc2:
        use_max_daily_r = st.checkbox("Enable Max Daily Loss (R)", value=False, key="auto_sim_hist_use_daily_r")
        max_daily_loss_r = None
        if use_max_daily_r:
            max_daily_loss_r = st.number_input(
                "Max Daily Loss (R)",
                min_value=0.5,
                max_value=10.0,
                value=3.0,
                step=0.5,
                key="auto_sim_hist_max_daily_r",
                help="Stop trading for the day if cumulative R loss exceeds this"
            )
    
    st.markdown("---")
    
    # Date Range / Max Bars
    col3, col4 = st.columns(2)
    with col3:
        use_date_range = st.checkbox("Use Date Range", value=False, key="auto_sim_hist_use_date")
        if use_date_range:
            start_date = st.date_input("Start Date", key="auto_sim_hist_start")
            end_date = st.date_input("End Date", key="auto_sim_hist_end")
        else:
            start_date = None
            end_date = None
    
    with col4:
        max_bars = st.number_input(
            "Max Bars (0 = all available)",
            min_value=0,
            max_value=1000,
            value=200,
            step=50,
            key="auto_sim_hist_max_bars"
        )
    
    st.markdown("---")
    
    # Run Simulation
    btn_label = "🚀 Run Historical Simulation"
    if is_ma_mode:
        btn_label = f"🚀 Run MA Crossover ({ma_short_window}/{ma_long_window})"
    
    if st.button(btn_label, type="primary", use_container_width=True, key="auto_sim_hist_run"):
        # Validation
        if is_ma_mode and ma_short_window >= ma_long_window:
            st.error("Short Window must be less than Long Window")
            return
        
        with st.spinner("Running simulation... This may take a moment."):
            try:
                payload = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "initial_capital": initial_capital,
                    "risk_per_trade": risk_pct / 100.0,
                    "strategy_mode": "ma_crossover" if is_ma_mode else "final_signal",
                    "position_sizing_mode": pos_mode_value,
                    "execution_mode": execution_mode,
                    "commission_percent": commission_percent,
                    "slippage_percent": slippage_percent,
                }
                
                # Add MA params
                if is_ma_mode:
                    payload["ma_short_window"] = ma_short_window
                    payload["ma_long_window"] = ma_long_window
                
                # Add position sizing params
                if fixed_shares:
                    payload["fixed_shares"] = fixed_shares
                if fixed_dollar_amount:
                    payload["fixed_dollar_amount"] = fixed_dollar_amount
                
                # Add R-management params
                if use_r_management:
                    payload["use_r_management"] = True
                    payload["virtual_stop_method"] = virtual_stop_method
                    if virtual_stop_method == "atr":
                        payload["virtual_stop_atr_multiplier"] = virtual_stop_atr_multiplier
                    else:
                        payload["virtual_stop_percent"] = virtual_stop_percent
                
                # Add loss control params
                if max_drawdown_percent is not None:
                    payload["max_drawdown_percent"] = max_drawdown_percent
                if max_daily_loss_r is not None:
                    payload["max_daily_loss_r"] = max_daily_loss_r
                
                # Add date range
                if use_date_range and start_date and end_date:
                    payload["start_date"] = start_date.isoformat()
                    payload["end_date"] = end_date.isoformat()
                
                if max_bars > 0:
                    payload["max_bars"] = max_bars
                
                response = requests.post(f"{BACKEND_URL}/auto-simulate", json=payload, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state["auto_sim_hist_result"] = result
                    
                    # Show strategy info in success message
                    strategy_info = "Final Signal" if not is_ma_mode else f"MA Crossover ({ma_short_window}/{ma_long_window})"
                    
                    # Check if halted
                    if result.get("summary", {}).get("simulation_halted"):
                        halt_reason = result["summary"].get("halt_reason", "Unknown")
                        st.warning(f"⚠️ Simulation halted: {halt_reason}")
                    
                    st.success(f"✅ Simulation complete! [{strategy_info}] Final Equity: ${result['final_equity']:,.2f} ({result['total_return_pct']:+.2f}%)")
                else:
                    st.error(f"Simulation failed: {response.text}")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Display Results
    if "auto_sim_hist_result" in st.session_state:
        result = st.session_state["auto_sim_hist_result"]
        _render_auto_sim_results(result, "hist")


def render_auto_sim_realtime():
    """Render Realtime Auto Sim Lab UI with live updates."""
    
    st.markdown("""
    **Realtime Mode** runs a live paper trading simulation using real-time price feeds.
    The simulation processes market data and generates trading signals continuously.
    
    ⚠️ **Note**: Price updates occur every ~10 seconds. This is for demonstration purposes.
    """)
    
    st.markdown("---")
    
    # Check for active session
    active_session = st.session_state.get("realtime_session_id")
    
    if active_session:
        # Active session - show status and controls
        st.subheader("🔴 Live Session Active")
        
        # Stop button
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("⏹️ Stop", type="secondary", key="realtime_stop_btn"):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/realtime-sim/stop",
                        json={"session_id": active_session},
                        timeout=10
                    )
                    if resp.status_code == 200:
                        st.session_state["realtime_session_id"] = None
                        st.toast("✅ Session stopped")
                        st.rerun()
                    else:
                        st.error(f"Failed to stop: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col2:
            st.info(f"Session ID: `{active_session}`")
        
        # Fetch and display state
        try:
            state_resp = requests.get(
                f"{BACKEND_URL}/realtime-sim/state",
                params={"session_id": active_session},
                timeout=10
            )
            
            if state_resp.status_code == 200:
                state = state_resp.json()
                
                # Status metrics
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Symbol", state.get("symbol", "N/A"))
                with col2:
                    st.metric("Ticks", state.get("tick_count", 0))
                with col3:
                    st.metric("Current Equity", f"${state.get('current_equity', 0):,.2f}")
                with col4:
                    ret = state.get("total_return_pct", 0)
                    st.metric("Return", f"{ret:+.2f}%")
                
                # Position info
                pos = state.get("position", {})
                if pos.get("side") == "long":
                    st.success(f"📈 **LONG** {pos.get('size', 0)} shares @ ${pos.get('entry_price', 0):.2f}")
                else:
                    st.info("📊 **FLAT** - No open position")
                
                # Equity Curve
                st.subheader("📈 Equity Curve (Live)")
                if state.get("equity_curve"):
                    eq_df = pd.DataFrame(state["equity_curve"])
                    eq_df['timestamp'] = pd.to_datetime(eq_df['timestamp'])
                    
                    eq_chart = alt.Chart(eq_df).mark_line(
                        strokeWidth=2,
                        color="#FF6B6B"
                    ).encode(
                        x=alt.X('timestamp:T', title='Time'),
                        y=alt.Y('equity:Q', title='Equity ($)', scale=alt.Scale(zero=False))
                    ).properties(
                        height=250
                    )
                    st.altair_chart(eq_chart, use_container_width=True)
                
                # Trades
                if state.get("trades"):
                    st.subheader("📋 Trades")
                    trades_df = pd.DataFrame(state["trades"])
                    st.dataframe(trades_df, use_container_width=True)
                
                # Decision Log (collapsible)
                with st.expander("📝 Decision Log (Last 50)", expanded=False):
                    log = state.get("decision_log", [])
                    if log:
                        log_df = pd.DataFrame(log)
                        display_cols = ['timestamp', 'event_type', 'final_signal', 'price', 'reason']
                        display_cols = [c for c in display_cols if c in log_df.columns]
                        st.dataframe(log_df[display_cols], use_container_width=True, height=300)
                    else:
                        st.info("No events yet.")
                
                # Auto-refresh
                st.caption(f"Last update: {state.get('last_tick_time', 'N/A')}")
                time.sleep(2)  # Small delay before rerun
                st.rerun()
                
            elif state_resp.status_code == 404:
                st.warning("Session not found. It may have been stopped.")
                st.session_state["realtime_session_id"] = None
                st.rerun()
            else:
                st.error(f"Failed to get state: {state_resp.text}")
                
        except Exception as e:
            st.error(f"Error fetching state: {e}")
    
    else:
        # No active session - show configuration
        st.subheader("⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = render_symbol_selector(key_prefix="auto_sim_rt", container=st)
            timeframe = st.selectbox(
                "Timeframe",
                options=["1m"],  # Only 1m for realtime (for now)
                index=0,
                key="auto_sim_rt_timeframe",
                help="Currently only 1-minute timeframe is supported for realtime simulation"
            )
        
        with col2:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000.0,
                max_value=10000000.0,
                value=100000.0,
                step=10000.0,
                key="auto_sim_rt_capital"
            )
            risk_pct = st.slider(
                "Risk per Trade (%)",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.5,
                key="auto_sim_rt_risk"
            )
        
        st.markdown("---")
        
        # Start button
        if st.button("🔴 Start Realtime Simulation", type="primary", use_container_width=True, key="realtime_start_btn"):
            with st.spinner("Starting realtime simulation..."):
                try:
                    payload = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "initial_capital": initial_capital,
                        "risk_per_trade": risk_pct / 100.0
                    }
                    
                    resp = requests.post(f"{BACKEND_URL}/realtime-sim/start", json=payload, timeout=10)
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        st.session_state["realtime_session_id"] = result["session_id"]
                        st.success(f"Started! Session ID: {result['session_id']}")
                        st.rerun()
                    else:
                        st.error(f"Failed to start: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")


def _render_auto_sim_results(result: dict, key_suffix: str):
    """Render Auto Sim results (shared by Historical and Realtime)."""
    st.markdown("---")
    st.subheader("📊 Results")
    
    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Equity", f"${result['final_equity']:,.2f}")
    with col2:
        st.metric("Total Return", f"{result['total_return_pct']:+.2f}%")
    with col3:
        st.metric("Total Trades", result['summary']['total_trades'])
    with col4:
        st.metric("Win Rate", f"{result['summary']['win_rate']:.1f}%")
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Total PnL", f"${result['summary']['total_pnl']:,.2f}")
    with col6:
        st.metric("Avg PnL", f"${result['summary']['avg_pnl']:.2f}")
    with col7:
        st.metric("Best Trade", f"${result['summary']['best_trade']:.2f}")
    with col8:
        st.metric("Worst Trade", f"${result['summary']['worst_trade']:.2f}")
    
    st.markdown("---")
    
    # Equity Curve
    st.subheader("📈 Equity Curve")
    if result['equity_curve']:
        eq_df = pd.DataFrame(result['equity_curve'])
        eq_df['timestamp'] = pd.to_datetime(eq_df['timestamp'])
        
        eq_chart = alt.Chart(eq_df).mark_line(
            strokeWidth=2,
            color="#00D4AA"
        ).encode(
            x=alt.X('timestamp:T', title='Date'),
            y=alt.Y('equity:Q', title='Equity ($)', scale=alt.Scale(zero=False))
        ).properties(
            height=300
        )
        
        st.altair_chart(eq_chart, use_container_width=True)
    else:
        st.info("No equity curve data available.")
    
    # Trades Table
    st.subheader("📋 Trades")
    if result['trades']:
        trades_df = pd.DataFrame(result['trades'])
        st.dataframe(
            trades_df,
            use_container_width=True,
            column_config={
                "pnl": st.column_config.NumberColumn("PnL ($)", format="$%.2f"),
                "return_pct": st.column_config.NumberColumn("Return (%)", format="%.2f%%"),
                "entry_price": st.column_config.NumberColumn("Entry", format="$%.2f"),
                "exit_price": st.column_config.NumberColumn("Exit", format="$%.2f"),
            }
        )
    else:
        st.info("No trades executed during simulation.")
    
    # Decision Log
    with st.expander("📝 Decision Log", expanded=False):
        st.caption(f"Total events: {len(result['decision_log'])}")
        
        event_types = ["all", "signal_decision", "entry", "exit"]
        selected_type = st.selectbox(
            "Filter by event type:",
            event_types,
            key=f"decision_log_filter_{key_suffix}"
        )
        
        log_events = result['decision_log']
        if selected_type != "all":
            log_events = [e for e in log_events if e['event_type'] == selected_type]
        
        if log_events:
            log_df = pd.DataFrame(log_events)
            display_cols = ['timestamp', 'event_type', 'final_signal', 'position_side', 'price', 'reason']
            display_cols = [c for c in display_cols if c in log_df.columns]
            st.dataframe(log_df[display_cols], use_container_width=True, height=400)
        else:
            st.info("No events match the selected filter.")


# =============================================================================
# Live Signal UI
# =============================================================================

def render_live_signal():
    col_title, col_back = st.columns([3, 1])
    with col_title:
        st.title("📡 Live Signal (v1)")
        st.caption("Real-time trading signals based on EXITON v1 architecture.")
    with col_back:
        st.write("")
        st.write("")
        
        # Back to Scanner Link
        import urllib.parse
        
        scanner_universe = st.session_state.get("universe", "default")
        scanner_timeframe = st.session_state.get("timeframe", "1d")
        scanner_lookback = st.session_state.get("lookback", 200)
        current_symbol = st.session_state.get("symbol", "SMCI")
        
        scanner_params = {
            "mode": "scanner",
            "symbol": current_symbol,
            "timeframe": scanner_timeframe,
            "lookback": str(scanner_lookback),
            "universe": scanner_universe,
        }
        scanner_query = urllib.parse.urlencode(scanner_params)
        scanner_href = f"?{scanner_query}"
        
        st.markdown(
            f"""
            <div style="text-align: right; margin-bottom: 8px;">
              <a href="{scanner_href}" target="_self" style="text-decoration: none;">
                <button style="
                    padding: 6px 12px;
                    border-radius: 8px;
                    border: 1px solid #444;
                    background-color: #333;
                    color: white;
                    font-size: 14px;
                    cursor: pointer;
                ">
                  ⬅ Back to Scanner
                </button>
              </a>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # =========================================================================
    # TASK D: Ensure Live Signal Loads Navigation State
    # =========================================================================
    # With the new URL-based system, state is already loaded into st.session_state
    # by the centralized logic at the top of the file.
    # We just need to ensure we use those values.
    
    nav_symbol = st.session_state.get("symbol")
    nav_timeframe = st.session_state.get("timeframe")
    nav_lookback = st.session_state.get("lookback")
    
    # These will be used to set defaults below
    
    # Session state for preset tracking
    if "live_preset" not in st.session_state:
        st.session_state["live_preset"] = None
    if "live_preset_modified" not in st.session_state:
        st.session_state["live_preset_modified"] = False
    if "live_last_symbol" not in st.session_state:
        st.session_state["live_last_symbol"] = None
    
    # Load symbols from universe
    universe = load_symbol_universe(UNIVERSE_CSV)
    default_symbols = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD", "INTC", "SMCI"]
    options = universe if universe else default_symbols
    
    # Determine default symbol index (prioritize nav, then session state, then first)
    default_symbol_index = 0
    if nav_symbol and nav_symbol in options:
        default_symbol_index = options.index(nav_symbol)
    
    # Symbol selection (separate so we can detect changes)
    col_sym = st.columns([1, 3])[0]
    with col_sym:
        symbol = st.selectbox("Symbol", options=options, index=default_symbol_index, key="live_symbol")
        
        # Update URL with symbol
        params = st.query_params.to_dict()
        params["symbol"] = symbol
        st.query_params.update(params)
    
    # Fetch preset when symbol changes
    if symbol != st.session_state["live_last_symbol"]:
        try:
            resp = requests.get(f"{BACKEND_URL}/symbol_preset", params={"symbol": symbol}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                preset = data.get("preset", {})
                st.session_state["live_preset"] = preset
                st.session_state["live_preset_modified"] = False
                
                # Set default values based on preset
                st.session_state["live_timeframe_default"] = preset.get("timeframe", "1d")
                st.session_state["live_lookback_default"] = preset.get("lookback", 200)
        except Exception:
            st.session_state["live_preset"] = None
        
        st.session_state["live_last_symbol"] = symbol
    
    # Get preset values for defaults
    preset = st.session_state.get("live_preset") or {}
    preset_tf = preset.get("timeframe", "1d")
    preset_lookback = preset.get("lookback", 200)
    
    # Navigation values override preset defaults
    effective_tf = nav_timeframe if nav_timeframe else preset_tf
    effective_lookback = nav_lookback if nav_lookback else preset_lookback
    
    # Inputs
    col_in2, col_in3, col_btn = st.columns([1, 1, 1])
    with col_in2:
        tf_options = ["1d", "1h", "4h", "15m"]
        tf_index = tf_options.index(effective_tf) if effective_tf in tf_options else 0
        timeframe = st.selectbox("Timeframe", tf_options, index=tf_index, key="live_timeframe")
        
        # Check if modified
        if timeframe != preset_tf:
            st.session_state["live_preset_modified"] = True
    
    with col_in3:
        lookback = st.number_input("Lookback", min_value=50, max_value=500, value=effective_lookback, step=10, key="live_lookback")
        
        # Check if modified
        if lookback != preset_lookback:
            st.session_state["live_preset_modified"] = True
    
    with col_btn:
        st.write("")  # Spacer
        st.write("")
        run_btn = st.button("🚀 Run Live Analysis", type="primary", use_container_width=True)
    
    # Preset Info Box
    if preset:
        predictor_names = {"rule_v2": "Rule Predictor v2", "stat": "Stat Predictor", "ml_v1": "ML Predictor v1"}
        pred_name = predictor_names.get(preset.get("predictor", "rule_v2"), preset.get("predictor", "?"))
        modified_label = " <span style='color: #f39c12; font-weight: bold;'>(Modified)</span>" if st.session_state["live_preset_modified"] else ""
        
        st.markdown(f"""
<div style="
    background: linear-gradient(135deg, rgba(155, 89, 182, 0.1) 0%, rgba(155, 89, 182, 0.05) 100%);
    border: 1px solid rgba(155, 89, 182, 0.3);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 10px;
    font-size: 13px;
">
    <span style="color: #9b59b6; font-weight: 600;">⚙️ Preset for {symbol}:</span>{modified_label}<br>
    <span style="color: #bbb;">
        {pred_name} | {preset.get("timeframe", "1d")} timeframe | Lookback {preset.get("lookback", 200)} | 
        Min hold {preset.get("min_hold_days", 3)} days | Risk {preset.get("position_risk_pct", 0.05)*100:.0f}%
    </span>
</div>
""", unsafe_allow_html=True)
    
    # Timeframe Guidance
    render_timeframe_guidance(timeframe, mode="live_signal")
    
    # Update URL with timeframe
    params = st.query_params.to_dict()
    params["timeframe"] = timeframe
    st.query_params.update(params)
        
    # Initialize session state for Live Signal
    if "live_signal_data" not in st.session_state:
        st.session_state["live_signal_data"] = None
    if "live_chart_data" not in st.session_state:
        st.session_state["live_chart_data"] = None
    if "live_signal_params" not in st.session_state:
        st.session_state["live_signal_params"] = {}

    if run_btn:
        try:
            with st.spinner(f"Analyzing {symbol}..."):
                # Call API
                params = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "lookback": lookback
                }
                response = requests.get(f"{BACKEND_URL}/live/signal", params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                # Fetch Chart Data
                chart_res = requests.get(
                    f"{BACKEND_URL}/chart-data", 
                    params={"symbol": symbol, "interval": timeframe, "limit": lookback, "with_ma": "true"}
                )
                chart_data = []
                if chart_res.status_code == 200:
                    chart_data = chart_res.json().get("data", [])
                
                # Store in Session State
                st.session_state["live_signal_data"] = data
                st.session_state["live_chart_data"] = chart_data
                st.session_state["live_signal_params"] = params
                
        except Exception as e:
            st.error(f"Error running analysis: {e}")
            return

    # Render UI if data exists (Persistent)
    if st.session_state["live_signal_data"]:
        data = st.session_state["live_signal_data"]
        chart_data = st.session_state["live_chart_data"]
        
        # Parse Response
        latest_price = data.get("latest_price", 0.0)
        price_time = data.get("price_time", "")
        final_signal = data.get("final_signal", {})
        predictions = data.get("predictions", {})
        
        # --- UI Layout ---
        
        # 1. Top Metrics (Price & Final Signal)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Latest Price", f"${latest_price:,.2f}", help=f"Time: {price_time}")
        with m2:
            action = final_signal.get("action", "HOLD").upper()
            conf = final_signal.get("confidence", 0.0)
            color = "off"
            if action == "BUY": color = "normal" 
            elif action == "SELL": color = "inverse"
            
            st.metric("Final Signal", action, f"Conf: {conf:.2f}")
        with m3:
            st.info(f"**Reason:** {final_signal.get('reason', 'N/A')}")
            
        st.markdown("---")
        
        # =====================================================================
        # 🧪 Paper Trading Panel
        # =====================================================================
        with st.container():
            st.markdown("### 🧪 Paper Trading (Experimental)")
            
            # Use current Live Signal symbol and latest price
            paper_symbol = symbol
            paper_price = latest_price
            
            col_pt_info, col_pt_input = st.columns([1, 2])
            with col_pt_info:
                st.write(f"**Symbol:** {paper_symbol}")
                st.write(f"**Price:** ${paper_price:,.2f}")
                
            with col_pt_input:
                pt_size = st.number_input(
                    "Position Size (units)",
                    min_value=0.1,
                    max_value=100000.0,
                    value=1.0,
                    step=0.1,
                    key="paper_trade_size",
                )
            
            col_long, col_short = st.columns(2)
            
            with col_long:
                if st.button("✅ Open LONG (Paper)", key="paper_open_long", use_container_width=True):
                    account = ensure_paper_account(BACKEND_URL)
                    if not account:
                        st.error("Failed to initialize paper account.")
                    else:
                        payload = {
                            "account_id": "default",
                            "symbol": paper_symbol,
                            "direction": "LONG",
                            "size": float(pt_size),
                            "entry_price": float(paper_price),
                            "stop_price": None,
                            "target_price": None,
                            "tags": ["live_signal"],
                        }
                        try:
                            resp = requests.post(f"{BACKEND_URL}/paper/positions", json=payload, timeout=15)
                            if resp.status_code == 200:
                                pos = resp.json()
                                st.success(f"Opened LONG: {pos.get('position_id')} @ {paper_price:.2f}")
                            else:
                                st.error(f"Failed: {resp.status_code} {resp.text}")
                        except Exception as e:
                            st.error(f"Error: {e}")

            with col_short:
                if st.button("❌ Open SHORT (Paper)", key="paper_open_short", use_container_width=True):
                    account = ensure_paper_account(BACKEND_URL)
                    if not account:
                        st.error("Failed to initialize paper account.")
                    else:
                        payload = {
                            "account_id": "default",
                            "symbol": paper_symbol,
                            "direction": "SHORT",
                            "size": float(pt_size),
                            "entry_price": float(paper_price),
                            "stop_price": None,
                            "target_price": None,
                            "tags": ["live_signal"],
                        }
                        try:
                            resp = requests.post(f"{BACKEND_URL}/paper/positions", json=payload, timeout=15)
                            if resp.status_code == 200:
                                pos = resp.json()
                                st.success(f"Opened SHORT: {pos.get('position_id')} @ {paper_price:.2f}")
                            else:
                                st.error(f"Failed: {resp.status_code} {resp.text}")
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            # =================================================================
            # Open Paper Positions (Improved)
            # =================================================================
            try:
                resp = requests.get(f"{BACKEND_URL}/paper/positions/open", params={"account_id": "default"}, timeout=5)
                if resp.status_code == 200:
                    positions = resp.json()
                    symbol_positions = [p for p in positions if p.get("symbol") == paper_symbol]
                    
                    if symbol_positions:
                        st.markdown("#### 🟢 Open Paper Positions")
                        
                        # Header
                        h1, h2, h3, h4, h5, h6, h7, h8 = st.columns([1.5, 0.8, 0.8, 1, 1, 1, 1, 2])
                        h1.markdown("**ID**")
                        h2.markdown("**Dir**")
                        h3.markdown("**Size**")
                        h4.markdown("**Entry**")
                        h5.markdown("**Current**")
                        h6.markdown("**PnL ($)**")
                        h7.markdown("**R (Risk)**")
                        h8.markdown("**Actions**")
                        
                        for p in symbol_positions:
                            pos_id = p["position_id"]
                            direction = p["direction"]
                            size = p["size"]
                            entry = p["entry_price"]
                            current = paper_price
                            
                            # Calc PnL
                            if direction == "LONG":
                                pnl = (current - entry) * size
                            else:
                                pnl = (entry - current) * size
                                
                            # Calc R (approximate if stop not set, use 1R = 1% equity or just display raw risk if available)
                            # Here we use the backend's r_risk() logic if available, but we don't have it in the response directly unless we added it.
                            # The backend model has r_risk() method but it's not a field in JSON unless computed.
                            # We'll compute R-multiple based on a hypothetical risk or just show PnL.
                            # User asked for "Unrealized R-multiple".
                            # r_multiple = pnl / (risk_value). We need risk_value.
                            # We can try to estimate risk from stop_price if available.
                            stop_price = p.get("stop_price")
                            risk_val = 0.0
                            if stop_price:
                                if direction == "LONG":
                                    risk_val = (entry - stop_price) * size
                                else:
                                    risk_val = (stop_price - entry) * size
                            
                            r_mult_str = "N/A"
                            if risk_val > 0:
                                r_mult = pnl / risk_val
                                r_mult_str = f"{r_mult:+.2f}R"
                            
                            # Color for PnL
                            pnl_color = "green" if pnl >= 0 else "red"
                            
                            c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1.5, 0.8, 0.8, 1, 1, 1, 1, 2])
                            
                            c1.caption(pos_id.split("-")[-1]) # Short ID
                            c2.write(direction)
                            c3.write(f"{size}")
                            c4.write(f"{entry:.2f}")
                            c5.write(f"{current:.2f}")
                            c6.markdown(f":{pnl_color}[{pnl:+.2f}]")
                            c7.write(r_mult_str)
                            
                            with c8:
                                ac1, ac2 = st.columns(2)
                                with ac1:
                                    # Unique key for Close button
                                    close_key = f"paper_close_default_{pos_id}"
                                    if st.button("Close", key=close_key, type="primary", use_container_width=True):
                                        # Close Position
                                        try:
                                            cl_resp = requests.post(
                                                f"{BACKEND_URL}/paper/positions/{pos_id}/close",
                                                json={"exit_price": current},
                                                timeout=10
                                            )
                                            if cl_resp.status_code == 200:
                                                st.toast(f"✅ Closed {pos_id}")
                                                time.sleep(0.5) # Brief pause to let toast be seen/processed
                                                st.rerun()
                                            else:
                                                st.error("Failed to close")
                                        except Exception as e:
                                            st.error(f"Err: {e}")
                                            
                                with ac2:
                                    # Unique key for Delete button
                                    del_key = f"paper_del_default_{pos_id}"
                                    if st.button("×", key=del_key, help="Delete (Cancel)", use_container_width=True):
                                        # Delete Position
                                        try:
                                            del_resp = requests.delete(f"{BACKEND_URL}/paper/positions/{pos_id}", timeout=10)
                                            if del_resp.status_code == 204:
                                                st.toast(f"⚠️ Deleted {pos_id}")
                                                time.sleep(0.5)
                                                st.rerun()
                                            else:
                                                st.error("Failed to delete")
                                        except Exception as e:
                                            st.error(f"Err: {e}")

            except Exception as e:
                st.error(f"Error fetching positions: {e}")

            # =================================================================
            # Closed Paper Trades (History)
            # =================================================================
            try:
                t_resp = requests.get(f"{BACKEND_URL}/paper/trades", params={"account_id": "default"}, timeout=5)
                if t_resp.status_code == 200:
                    trades = t_resp.json()
                    # Filter by symbol if desired, or show all? User asked for "Closed Trades section to Live Signal".
                    # Usually context is symbol-specific. Let's filter by symbol.
                    symbol_trades = [t for t in trades if t.get("symbol") == paper_symbol]
                    
                    if symbol_trades:
                        st.markdown("#### 🏁 Closed Paper Trades")
                        
                        # Sort by closed_at desc
                        symbol_trades.sort(key=lambda x: x.get("closed_at") or "", reverse=True)
                        
                        t_data = []
                        for t in symbol_trades:
                            t_data.append({
                                "Time": t.get("closed_at", "")[:16].replace("T", " "),
                                "Dir": t["direction"],
                                "Size": t["size"],
                                "Entry": t["entry_price"],
                                "Exit": t["exit_price"],
                                "PnL ($)": t["net_pnl"],
                                "R": t["r_multiple"]
                            })
                        
                        st.dataframe(
                            t_data, 
                            use_container_width=True,
                            column_config={
                                "PnL ($)": st.column_config.NumberColumn(format="$%.2f"),
                                "R": st.column_config.NumberColumn(format="%.2fR"),
                            }
                        )
            except Exception:
                pass

            # =================================================================
            # Paper Account Summary
            # =================================================================
            st.markdown("---")
            try:
                acct_resp = requests.get(f"{BACKEND_URL}/paper/accounts/default", timeout=5)
                if acct_resp.status_code == 200:
                    acct = acct_resp.json()
                    equity = acct.get("equity", 0.0)
                    cash = acct.get("cash", 0.0)
                    open_risk = acct.get("open_risk", 0.0)
                    risk_pct = (open_risk / equity) if equity > 0 else 0.0
                    
                    st.markdown("### 📊 Paper Account Summary")
                    cols = st.columns(4)
                    cols[0].metric("Equity", f"${equity:,.2f}")
                    cols[1].metric("Cash", f"${cash:,.2f}")
                    cols[2].metric("Open Risk", f"${open_risk:,.2f}")
                    cols[3].metric("Risk %", f"{risk_pct*100:.1f}%")
                elif acct_resp.status_code == 404:
                    st.caption("📊 No paper account yet. Open a position to create one.")
            except Exception as e:
                st.warning(f"Could not fetch paper account: {e}")

            # =================================================================
            # Paper Trading Performance Summary (All Trades)
            # =================================================================
            try:
                perf_resp = requests.get(f"{BACKEND_URL}/paper/trades", params={"account_id": "default"}, timeout=5)
                if perf_resp.status_code == 200:
                    all_trades = perf_resp.json()
                    
                    if all_trades:
                        st.markdown("### 📈 Paper Trading Performance")
                        
                        # Compute stats
                        total_trades = len(all_trades)
                        r_values = [t.get("r_multiple", 0.0) for t in all_trades]
                        wins = sum(1 for r in r_values if r > 0)
                        losses = total_trades - wins
                        win_rate = wins / total_trades if total_trades > 0 else 0.0
                        avg_r = sum(r_values) / total_trades if total_trades > 0 else 0.0
                        total_r = sum(r_values)
                        best_r = max(r_values) if r_values else 0.0
                        worst_r = min(r_values) if r_values else 0.0
                        
                        # Summary line
                        st.write(
                            f"**Total Trades:** {total_trades}  |  "
                            f"**Win Rate:** {win_rate*100:.1f}%  |  "
                            f"**Avg R:** {avg_r:.2f}  |  "
                            f"**Total R:** {total_r:.2f}"
                        )
                        st.write(
                            f"**Best R:** {best_r:.2f}  |  "
                            f"**Worst R:** {worst_r:.2f}"
                        )
                        
                        # Full trade history table
                        all_trades.sort(key=lambda x: x.get("closed_at") or "", reverse=True)
                        
                        perf_data = []
                        for t in all_trades:
                            perf_data.append({
                                "Symbol": t.get("symbol", ""),
                                "Dir": t.get("direction", ""),
                                "Entry": t.get("entry_price", 0.0),
                                "Exit": t.get("exit_price", 0.0),
                                "R": t.get("r_multiple", 0.0),
                                "PnL ($)": t.get("net_pnl", 0.0),
                                "Closed": t.get("closed_at", "")[:16].replace("T", " ") if t.get("closed_at") else ""
                            })
                        
                        st.dataframe(
                            perf_data,
                            use_container_width=True,
                            column_config={
                                "PnL ($)": st.column_config.NumberColumn(format="$%.2f"),
                                "R": st.column_config.NumberColumn(format="%.2fR"),
                            }
                        )
                    else:
                        st.caption("📈 No closed paper trades yet.")
            except Exception as e:
                st.warning(f"Could not fetch paper trades: {e}")

        st.markdown("---")
        
        # 2. Predictor Breakdown
        st.subheader("🤖 Predictor Breakdown")
        p1, p2, p3 = st.columns(3)
        


        with p1:
            render_predictor_card("Stat Predictor", predictions.get("stat", {}))
        with p2:
            render_predictor_card("Rule Predictor v2", predictions.get("rule", {}))
        with p3:
            render_predictor_card("ML Predictor (v1)", predictions.get("ml", {}))
            
        st.markdown("---")
        
        # 3. Chart (Price + MA)
        # Anti Summary: Simplified chart logic. Removed "Fit Y-Axis" checkbox. Fixed data types and auto-scaling (zero=False).
        st.subheader("📈 Price Chart")

        if chart_data:
            df_chart = pd.DataFrame(chart_data)
            
            # Normalize time column
            if "timestamp" in df_chart.columns:
                df_chart["time"] = pd.to_datetime(df_chart["timestamp"])
            elif "time" in df_chart.columns:
                df_chart["time"] = pd.to_datetime(df_chart["time"])
            else:
                st.warning("No time column found in chart data.")
                return

            # Ensure numeric types for scaling
            numeric_cols = ["open", "high", "low", "close"]
            if "ma_short" in df_chart.columns: numeric_cols.append("ma_short")
            if "ma_long" in df_chart.columns: numeric_cols.append("ma_long")
            
            # Check if we have valid data
            has_valid_data = False
            for col in numeric_cols:
                df_chart[col] = pd.to_numeric(df_chart[col], errors='coerce')
                if df_chart[col].notna().any():
                    has_valid_data = True
            
            if not has_valid_data:
                st.warning("No valid price data to display.")
                return

            # Simple Auto-Scale (Zero=False)
            y_scale = alt.Scale(zero=False)

            base = alt.Chart(df_chart).encode(
                x=alt.X('time:T', axis=alt.Axis(title="Time", format="%m/%d %H:%M"))
            )
            
            # Tooltip setup
            tooltip = [
                alt.Tooltip('time:T', title='Time', format='%Y-%m-%d %H:%M'),
                alt.Tooltip('open:Q', title='Open', format=',.2f'),
                alt.Tooltip('high:Q', title='High', format=',.2f'),
                alt.Tooltip('low:Q', title='Low', format=',.2f'),
                alt.Tooltip('close:Q', title='Close', format=',.2f'),
            ]
            if "ma_short" in df_chart.columns:
                tooltip.append(alt.Tooltip('ma_short:Q', title='MA Short', format=',.2f'))
            if "ma_long" in df_chart.columns:
                tooltip.append(alt.Tooltip('ma_long:Q', title='MA Long', format=',.2f'))

            line = base.mark_line(color='#bdc3c7').encode(
                y=alt.Y('close:Q', scale=y_scale, title="Price"),
                tooltip=tooltip
            )
            
            layers = [line]
            
            if "ma_short" in df_chart.columns:
                ma_s = base.mark_line(color='#f39c12').encode(
                    y='ma_short:Q',
                    tooltip=tooltip
                )
                # Label at the end
                last_val = df_chart["ma_short"].iloc[-1]
                if pd.notna(last_val):
                    label_s = base.mark_text(align='left', dx=5, color='#f39c12').encode(
                        x=alt.value(600), # Approximate right edge
                        y=alt.datum(last_val),
                        text=alt.value(f"MA Short: {last_val:.2f}")
                    )
                    layers.append(ma_s)
                
            if "ma_long" in df_chart.columns:
                ma_l = base.mark_line(color='#3498db').encode(
                    y='ma_long:Q',
                    tooltip=tooltip
                )
                layers.append(ma_l)
                
            st.altair_chart(alt.layer(*layers).interactive(), use_container_width=True)
        else:
            if st.session_state["live_signal_data"]: # Only show warning if we have signal but no chart
                 st.warning("Could not load chart data.")




if __name__ == "__main__":
    main()
