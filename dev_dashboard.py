"""
EXITON Developer Dashboard
--------------------------

Streamlit-based internal dashboard for monitoring the paper-trading engine
and visualizing strategy behavior.

This is a *mock* / developer-facing dashboard, not the main user UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st


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

    symbol = st.sidebar.selectbox("Symbol", options=["AAPL", "MSFT", "TSLA", "NVDA"], index=0)
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
    st.subheader("Equity Curve (Mock)")
    df = st.session_state.get("chart_data")
    if df is None or df.empty:
        st.info("No P&L yet.")
        return

    pnl_df = generate_mock_pnl(df)
    st.line_chart(pnl_df)


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

    # Input Form
    with st.form("backtest_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input("Symbol", value="AAPL")
            timeframe = st.selectbox("Timeframe", options=["1d", "1h", "5m"], index=0)
        with col2:
            start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
            end_date = st.date_input("End Date", value=datetime(2023, 12, 31))
        with col3:
            initial_capital = st.number_input("Initial Capital", value=1_000_000, step=100_000)
            commission = st.number_input("Commission Rate", value=0.001, step=0.0001, format="%.4f")

        st.markdown("### Strategy Parameters (MA Cross)")
        col4, col5 = st.columns(2)
        with col4:
            short_window = st.slider("Short Window", 5, 50, 9)
        with col5:
            long_window = st.slider("Long Window", 20, 200, 21)

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
# Strategy Lab (v0.1)
# =============================================================================


def render_strategy_lab():
    """
    Render the Strategy Lab UI (v0.1).
    Allows users to select strategy templates and input parameters.
    """
    st.title("🧪 Strategy Lab")
    st.caption("Design and test algorithmic strategies.")

    # Strategy Selection
    strategy_type = st.selectbox(
        "Select Strategy Template",
        options=["MA Cross", "RSI Reversal", "Breakout"],
        index=0
    )

    st.markdown("---")
    st.subheader("Strategy Parameters")

    # Dynamic Form based on selection
    with st.form("strategy_form"):
        if strategy_type == "MA Cross":
            st.markdown("**Moving Average Crossover**")
            st.caption("Buy when Short MA crosses above Long MA. Sell when Short MA crosses below Long MA.")
            col1, col2 = st.columns(2)
            with col1:
                short_window = st.number_input("Short Window", min_value=1, value=9)
            with col2:
                long_window = st.number_input("Long Window", min_value=1, value=21)

        elif strategy_type == "RSI Reversal":
            st.markdown("**RSI Reversal**")
            st.caption("Buy when RSI crosses below Oversold. Sell when RSI crosses above Overbought.")
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_period = st.number_input("RSI Period", min_value=2, value=14)
            with col2:
                oversold = st.number_input("Oversold Level", min_value=1, max_value=49, value=30)
            with col3:
                overbought = st.number_input("Overbought Level", min_value=51, max_value=99, value=70)

        elif strategy_type == "Breakout":
            st.markdown("**Breakout Strategy**")
            st.caption("Buy when price breaks above N-period high. Sell when price breaks below N-period low.")
            col1, col2 = st.columns(2)
            with col1:
                lookback_window = st.number_input("Lookback Window", min_value=1, value=20)
            with col2:
                threshold = st.number_input("Threshold Multiplier", min_value=1.0, value=1.0, step=0.1)

        st.markdown("---")
        submitted = st.form_submit_button("🚀 Run Strategy Analysis")

    if submitted:
        st.info(f"**{strategy_type}** selected. (v0.1 のため実行機能は未実装です)")
        st.write("Parameters captured:")
        if strategy_type == "MA Cross":
            st.json({"short_window": short_window, "long_window": long_window})
        elif strategy_type == "RSI Reversal":
            st.json({"rsi_period": rsi_period, "oversold": oversold, "overbought": overbought})
        elif strategy_type == "Breakout":
            st.json({"lookback_window": lookback_window, "threshold": threshold})


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
