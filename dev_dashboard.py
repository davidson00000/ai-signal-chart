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
from datetime import datetime, timedelta
import uuid
import os
import json
from typing import Any, Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


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


def render_main_chart():
    st.subheader("📈 Price & MA Signals (Mock)")
    df = st.session_state.get("chart_data")
    if df is None or df.empty:
        st.warning("No chart data available.")
        return

    df = compute_mock_ma_signals(df)
    chart_df = df[["close", "ma_short", "ma_long"]].copy()
    chart_df.columns = ["Close", "MA Short", "MA Long"]

    st.line_chart(chart_df)


def render_ma_signals(selected_ma: str):
    st.subheader("⚙️ Strategy Signals (Demo)")
    df = st.session_state.get("chart_data")
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
    # Load presets from JSON
    presets = load_symbol_presets()
    SYMBOL_PRESETS = [p["symbol"] for p in presets] + ["Custom..."]
    
    # Initialize shared state if not present
    if "shared_symbol_preset" not in st.session_state:
        st.session_state["shared_symbol_preset"] = SYMBOL_PRESETS[0] if SYMBOL_PRESETS else "AAPL"
    if "shared_custom_symbol" not in st.session_state:
        st.session_state["shared_custom_symbol"] = ""

    current_preset = st.session_state["shared_symbol_preset"]
    if current_preset not in SYMBOL_PRESETS:
        current_preset = "Custom..." # Fallback if loaded symbol is not in presets
        st.session_state["shared_custom_symbol"] = st.session_state.get("sl_symbol", SYMBOL_PRESETS[0] if SYMBOL_PRESETS else "AAPL")

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
        effective_symbol = custom_symbol.strip() or (SYMBOL_PRESETS[0] if SYMBOL_PRESETS else "AAPL")
    
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
    
    st.sidebar.subheader("Strategy Parameters (MA Cross)")
    short_window = st.sidebar.number_input("Short Window", min_value=1, value=default_short)
    long_window = st.sidebar.number_input("Long Window", min_value=1, value=default_long)

    # Input Form (for the submit button)
    with st.form("backtest_form"):
        st.markdown("---") # Separator for the button
        submitted = st.form_submit_button("▶ Run Backtest")

    if submitted:
        # API Call
        import requests

        payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": initial_capital,
            "commission_rate": commission,
            "position_size": 1.0,
            "strategy": "ma_cross",
            "short_window": short_window,
            "long_window": long_window,
        }

        with st.spinner("Running simulation..."):
            try:
                response = requests.post("http://localhost:8000/simulate", json=payload)
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

                # Equity Curve
                st.subheader("Equity Curve")
                equity_data = result["equity_curve"]
                if equity_data:
                    df_equity = pd.DataFrame(equity_data)
                    df_equity["date"] = pd.to_datetime(df_equity["date"])
                    df_equity.set_index("date", inplace=True)
                    st.line_chart(df_equity["equity"])
                else:
                    st.warning("No equity data returned.")

                # Trades Table
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
    Render the Strategy Lab UI (v0.2).
    Allows users to select strategy templates, input parameters, and run backtests (MA Cross only).
    """
    st.title("🧪 Strategy Lab")
    st.caption("Design and test algorithmic strategies.")

    # Common Inputs
    with st.expander("📊 Market Data & Capital Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = render_symbol_selector(key_prefix="sl", container=col1)
            timeframe = st.selectbox("Timeframe", options=["1d", "1h", "5m"], index=0, key="sl_timeframe")
        with col2:
            start_date = st.date_input("Start Date", value=datetime(2020, 1, 1), key="sl_start")
            end_date = st.date_input("End Date", value=datetime(2023, 12, 31), key="sl_end")
        with col3:
            initial_capital = st.number_input("Initial Capital", value=1_000_000, step=100_000, key="sl_capital")
            commission = st.number_input("Commission Rate", value=0.001, step=0.0001, format="%.4f", key="sl_comm")

    st.markdown("---")

    # Strategy Selection
    strategy_type = st.selectbox(
        "Select Strategy Template",
        options=["MA Cross", "RSI Reversal", "Breakout"],
        index=0
    )

    st.subheader("Strategy Parameters")

    # Dynamic Form based on selection
    with st.form("strategy_form"):
        # Default values for params
        short_window = 9
        long_window = 21
        rsi_period = 14
        oversold = 30
        overbought = 70
        lookback_window = 20
        threshold = 1.0

        if strategy_type == "MA Cross":
            st.markdown("**Moving Average Crossover**")
            st.caption("Buy when Short MA crosses above Long MA. Sell when Short MA crosses below Long MA.")
            
            tab_single, tab_opt = st.tabs(["Single Run", "Parameter Optimization"])
            
            with tab_single:
                col1, col2 = st.columns(2)
                with col1:
                    short_window = st.number_input("Short Window", min_value=1, value=9)
                with col2:
                    long_window = st.number_input("Long Window", min_value=1, value=21)
                
                submitted_single = st.form_submit_button("🚀 Run Single Analysis")
            
            with tab_opt:
                st.markdown("#### Grid Search Optimizer")
                c1, c2, c3 = st.columns(3)
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
                with c3:
                    st.info("Total Combinations must be <= 400")
                
                submitted_opt = st.form_submit_button("🔍 Run Optimization")

        elif strategy_type == "RSI Reversal":
            # ... (existing RSI code)
            st.markdown("**RSI Reversal**")
            st.caption("Buy when RSI crosses below Oversold. Sell when RSI crosses above Overbought.")
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_period = st.number_input("RSI Period", min_value=2, value=14)
            with col2:
                oversold = st.number_input("Oversold Level", min_value=1, max_value=49, value=30)
            with col3:
                overbought = st.number_input("Overbought Level", min_value=51, max_value=99, value=70)
            submitted_single = st.form_submit_button("🚀 Run Strategy Analysis")
            submitted_opt = False

        elif strategy_type == "Breakout":
            # ... (existing Breakout code)
            st.markdown("**Breakout Strategy**")
            st.caption("Buy when price breaks above N-period high. Sell when price breaks below N-period low.")
            col1, col2 = st.columns(2)
            with col1:
                lookback_window = st.number_input("Lookback Window", min_value=1, value=20)
            with col2:
                threshold = st.number_input("Threshold Multiplier", min_value=1.0, value=1.0, step=0.1)
            submitted_single = st.form_submit_button("🚀 Run Strategy Analysis")
            submitted_opt = False

    # Handle Actions
    if strategy_type == "MA Cross":
        import requests
        
        if submitted_single:
            # API Call for MA Cross Single Run
            payload = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "initial_capital": initial_capital,
                "commission_rate": commission,
                "position_size": 1.0,
                "strategy": "ma_cross",
                "short_window": short_window,
                "long_window": long_window,
            }

            with st.spinner("Running MA Cross Backtest..."):
                try:
                    response = requests.post("http://localhost:8000/simulate", json=payload)
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

                    # Equity Curve
                    st.subheader("Equity Curve")
                    equity_data = result["equity_curve"]
                    if equity_data:
                        df_equity = pd.DataFrame(equity_data)
                        df_equity["date"] = pd.to_datetime(df_equity["date"])
                        df_equity.set_index("date", inplace=True)
                        st.line_chart(df_equity["equity"])
                    else:
                        st.warning("No equity data returned.")

                    # Trades Table
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
        
        elif submitted_opt:
            # API Call for Optimization
            payload = {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "initial_capital": initial_capital,
                "commission_rate": commission,
                "short_min": short_min,
                "short_max": short_max,
                "short_step": short_step,
                "long_min": long_min,
                "long_max": long_max,
                "long_step": long_step
            }
            
            with st.spinner("Running Optimization..."):
                try:
                    response = requests.post("http://localhost:8000/optimize/ma_cross", json=payload)
                    response.raise_for_status()
                    data = response.json()
                    
                    st.success(f"Optimization Completed! Tested {data['total_combinations']} combinations.")
                    
                    top_results = data["top_results"]
                    if not top_results:
                        st.warning("No valid results found.")
                    else:
                        best = top_results[0]
                        best_params = best["params"]
                        best_metrics = best["metrics"]
                        
                        # Store in session state for persistence
                        st.session_state["ma_grid_best_params"] = best_params
                        st.session_state["ma_grid_best_metrics"] = best_metrics
                        st.session_state["ma_grid_top_results"] = top_results
                        
                        # Prepare Data for Visualization (and store in session state)
                        rows = []
                        for r in top_results:
                            row = r["params"].copy()
                            row.update(r["metrics"])
                            rows.append(row)
                        df_results = pd.DataFrame(rows)
                        st.session_state["ma_grid_results_df"] = df_results
                        
                except requests.exceptions.RequestException as e:
                    st.error(f"Optimization failed: {e}")
                    if e.response is not None:
                        st.error(f"Details: {e.response.text}")

    # ==========================================
    # Display Results (from Session State)
    # ==========================================
    # Check if we have results in session state to display
    if "ma_grid_best_params" in st.session_state and "ma_grid_best_metrics" in st.session_state:
        best_params = st.session_state["ma_grid_best_params"]
        best_metrics = st.session_state["ma_grid_best_metrics"]
        df_results = st.session_state.get("ma_grid_results_df")
        
        # Best Params Card
        st.markdown("### 🏆 Best Parameters")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "Short Window", 
            best_params["short_window"],
            help="短期の移動平均線の期間（バー数）です。値が小さいほど価格の変化に敏感になります。"
        )
        col2.metric(
            "Long Window", 
            best_params["long_window"],
            help="長期の移動平均線の期間（バー数）です。値が大きいほどゆっくりとしたトレンドを捉えます。"
        )
        col3.metric(
            "Total Return", 
            f"{best_metrics['return_pct']:.2f}%",
            help="バックテスト期間で、初期資産に対して最終的にどれだけ増えたかの割合です。100%なら資産が2倍、200%なら3倍になったことを意味します。"
        )
        col4.metric(
            "Sharpe Ratio", 
            f"{best_metrics['sharpe_ratio']:.2f}",
            help="リスク（リターンのブレ）に対する効率の良さを表す指標です。一般的には 1.0 以上で良好、2.0 以上で非常に優秀とされています。"
        )
        
        # Heatmap
        st.subheader("🔥 Performance Heatmap")
        st.caption(
            "横軸が Short Window、縦軸が Long Window、色が Total Return (%) を表します。"
            "明るい色ほど成績が良く、暗い色ほど悪い組み合わせです。"
        )
        try:
            import altair as alt
            
            if df_results is not None:
                chart = alt.Chart(df_results).mark_rect().encode(
                    x=alt.X('short_window:O', title='Short Window'),
                    y=alt.Y('long_window:O', title='Long Window'),
                    color=alt.Color('return_pct:Q', title='Return %', scale=alt.Scale(scheme='viridis')),
                    tooltip=['short_window', 'long_window', 'return_pct', 'max_drawdown', 'trade_count']
                ).properties(
                    title="Return % by Parameter Combination"
                )
                st.altair_chart(chart, use_container_width=True)
        except ImportError:
            st.warning("Altair not installed. Skipping heatmap.")
        except Exception as e:
            st.warning(f"Could not render heatmap: {e}")
            
        # Results Table
        st.subheader("📊 Top Results")
        
        if df_results is not None:
            # Prepare Display DataFrame
            # We use the raw numerical values (df_results) but rename columns for matching config
            display_df = df_results.copy()
            display_df = display_df.rename(columns={
                "short_window": "Short",
                "long_window": "Long",
                "total_pnl": "Total PnL",
                "return_pct": "Total Return (%)",
                "sharpe_ratio": "Sharpe",
                "max_drawdown": "Max Drawdown (%)",
                "win_rate": "Win Rate (%)",
                "trade_count": "Trades"
            })
            
            # Select specific columns to display
            cols_to_show = [
                "Short", "Long", "Total Return (%)", "Sharpe", 
                "Max Drawdown (%)", "Win Rate (%)", "Trades", "Total PnL"
            ]
            # Filter only existing columns (just in case)
            existing_cols = [c for c in cols_to_show if c in display_df.columns]
            
            st.dataframe(
                display_df[existing_cols],
                column_config={
                    "Short": st.column_config.NumberColumn(
                        "Short",
                        help="短期の移動平均線の期間（バー数）です。値が小さいほど価格変動に敏感になります。",
                        format="%d"
                    ),
                    "Long": st.column_config.NumberColumn(
                        "Long",
                        help="長期の移動平均線の期間（バー数）です。値が大きいほどゆっくりとしたトレンドを捉えます。",
                        format="%d"
                    ),
                    "Total Return (%)": st.column_config.NumberColumn(
                        "Total Return (%)",
                        help="バックテスト期間において、初期資産に対してどれだけ増えたかの割合です。100%なら資産が2倍、200%なら3倍です。",
                        format="%.2f%%"
                    ),
                    "Sharpe": st.column_config.NumberColumn(
                        "Sharpe",
                        help="リスク（リターンのブレ）に対する効率の良さを表す指標です。一般的には 1.0 以上で良好、2.0 以上で非常に優秀とされます。",
                        format="%.2f"
                    ),
                    "Max Drawdown (%)": st.column_config.NumberColumn(
                        "Max Drawdown (%)",
                        help="バックテスト期間中の資産曲線が、ピークからどれだけ大きく落ち込んだか（最大下落率）です。数値が小さいほど安全です。",
                        format="%.2f%%"
                    ),
                    "Win Rate (%)": st.column_config.NumberColumn(
                        "Win Rate (%)",
                        help="全トレードのうち、利益が出たトレードの割合です。高いほど勝ちトレードが多いことを意味しますが、リスクリワードとのバランスも重要です。",
                        format="%.2f%%"
                    ),
                    "Trades": st.column_config.NumberColumn(
                        "Trades",
                        help="バックテスト期間中に実行されたトレードの回数です。",
                        format="%d"
                    ),
                    "Total PnL": st.column_config.NumberColumn(
                        "Total PnL",
                        help="バックテスト期間全体での最終損益（Profit and Loss）です。通貨単位で表示されます。",
                        format="%d" # Simple integer format, or could use currency symbol if desired
                    ),
                },
                use_container_width=True
            )
        
        # ==========================================
        # Strategy Library Integration (Save Best)
        # ==========================================
        st.markdown("---")
        st.subheader("💾 Save to Strategy Library")
        
        with st.expander("Save Best Parameters as New Strategy", expanded=False):
            with st.form("save_best_strategy_form"):
                default_name = f"{symbol}_{timeframe}_MA_{best_params['short_window']}-{best_params['long_window']}_Best"
                strategy_name = st.text_input("Strategy Name", value=default_name)
                strategy_desc = st.text_area("Description", value=f"Grid Search Result. Return: {best_metrics['return_pct']:.2f}%")
                
                submitted_save = st.form_submit_button("💾 Save Strategy")
                
                if submitted_save:
                    if not strategy_name:
                        st.error("Strategy Name is required.")
                    else:
                        # Use session state values for saving
                        current_best_params = st.session_state.get("ma_grid_best_params")
                        current_best_metrics = st.session_state.get("ma_grid_best_metrics")
                        
                        if not current_best_params or not current_best_metrics:
                             st.error("Grid Search results missing. Please run optimization first.")
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
                                "params": current_best_params,
                                "metrics": current_best_metrics
                            }
                            lib.save_strategy(new_strategy)
                            st.success(f"Strategy '{strategy_name}' saved successfully!")

    elif submitted_single: # For other strategies
        # Placeholder for other strategies
        st.info(f"**{strategy_type}** selected.")
        st.warning("この戦略タイプの自動バックテストは v0.3 以降で実装予定です。")
        st.write("Parameters captured (for future use):")
        if strategy_type == "RSI Reversal":
            st.json({"rsi_period": rsi_period, "oversold": oversold, "overbought": overbought})
        elif strategy_type == "Breakout":
            st.json({"lookback_window": lookback_window, "threshold": threshold})
            
    # ==========================================
    # Strategy Library List & Load
    # ==========================================
    st.markdown("---")
    st.subheader("📚 Saved Strategies")
    
    lib = StrategyLibrary()
    strategies = lib.load_strategies()
    
    if not strategies:
        st.info("No strategies saved yet.")
    else:
        # Create display dataframe
        strat_rows = []
        for s in strategies:
            row = {
                "ID": s["id"],
                "Name": s["name"],
                "Symbol": s["symbol"],
                "Timeframe": s["timeframe"],
                "Type": s["strategy_type"],
                "Short": s["params"].get("short_window"),
                "Long": s["params"].get("long_window"),
                "Return (%)": f"{s.get('metrics', {}).get('return_pct', 0):.2f}%",
                "Created": s["created_at"][:16].replace("T", " ")
            }
            strat_rows.append(row)
        
        df_strats = pd.DataFrame(strat_rows)
        
        # Display as table with selection
        # Using st.dataframe with selection is available in newer Streamlit, 
        # but for compatibility we can use a selectbox or buttons.
        # Let's use a selectbox for "Load" action.
        
        st.dataframe(df_strats.drop(columns=["ID"]), use_container_width=True)
        
        col_load1, col_load2 = st.columns([3, 1])
        with col_load1:
            selected_strat_name = st.selectbox(
                "Select Strategy to Load", 
                options=[s["name"] for s in strategies],
                key="strat_selector"
            )
        with col_load2:
            st.write("") # Spacer
            st.write("") # Spacer
            if st.button("📂 Load Strategy"):
                # Find selected strategy
                selected_strat = next((s for s in strategies if s["name"] == selected_strat_name), None)
                if selected_strat:
                    # Update Session State for Strategy Lab
                    st.session_state["sl_symbol"] = selected_strat["symbol"]
                    st.session_state["sl_timeframe"] = selected_strat["timeframe"]
                    
                    # Store for Backtest Lab
                    st.session_state["loaded_strategy"] = selected_strat
                    
                    st.success(f"Loaded '{selected_strat_name}'. Please check parameters.")
                    st.info(f"**Loaded Parameters:** Short={selected_strat['params']['short_window']}, Long={selected_strat['params']['long_window']}")
                    st.caption("Go to 'Backtest Lab' to use these parameters instantly.")

    # ==========================================
    # Symbol Preset Settings (Developer Tools)
    # ==========================================
    st.markdown("---")
    with st.expander("⚙️ Symbol Preset Settings (開発者向け)", expanded=False):
        st.markdown("### Current Presets")
        symbols = load_symbol_presets()

        if symbols:
            df = pd.DataFrame(symbols)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No symbol presets found.")

        st.markdown("### Add New Preset")
        col_add1, col_add2 = st.columns(2)
        with col_add1:
            new_symbol = st.text_input("Symbol", key="new_symbol_input", placeholder="例: NVDA, 7203.T")
        with col_add2:
            new_label = st.text_input("Label (optional)", key="new_symbol_label", placeholder="例: NVIDIA")

        if st.button("➕ Add Preset"):
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
            delete_target = st.selectbox(
                "Select symbol to delete",
                options=[s["symbol"] for s in symbols],
                key="delete_symbol_select"
            )
            if st.button("🗑️ Delete Preset"):
                updated = [s for s in symbols if s["symbol"] != delete_target]
                if not updated:
                    st.warning("少なくとも1件のシンボルは残してください。")
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
        strategies.append(strategy)
        with open(self.FILE_PATH, "w") as f:
            json.dump({"strategies": strategies}, f, indent=2)

    def get_strategy(self, strategy_id: str) -> Optional[Dict]:
        strategies = self.load_strategies()
        for s in strategies:
            if s["id"] == strategy_id:
                return s
        return None


# =============================================================================
# Main app
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
    # Page config
    st.set_page_config(
        page_title="EXITON Developer Dashboard",
        page_icon="📊",
        layout="wide",
    )

    # Sidebar mode switch
    mode = st.sidebar.selectbox(
        "Mode",
        options=["Developer Dashboard", "Backtest Lab", "Strategy Lab"],
        index=0,
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
            st.session_state.timeframe = timeframe
            st.session_state.chart_data = fetch_chart_data(symbol, timeframe, limit)

        # Main content area - 2 columns
        col1, col2 = st.columns([2, 1])

        with col1:
            render_main_chart()
            render_ma_signals(selected_ma)

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


if __name__ == "__main__":
    main()
