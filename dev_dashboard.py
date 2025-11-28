"""
EXITON Developer Dashboard
A Streamlit-based monitoring and control interface for the EXITON paper trading system.

Run with: streamlit run dev_dashboard.py
"""
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

import plotly.graph_objects as go  # ★これを追加


# API Configuration
API_BASE = "http://127.0.0.1:8000"

# Page Configuration
st.set_page_config(
    page_title="EXITON Developer Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        max-width: 100%;
    }
    .st-emotion-cache-1y4p8pa {
        padding: 1rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# === API Functions ===

def check_health() -> Dict:
    """Check if backend is healthy"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def fetch_chart_data(symbol: str, timeframe: str = "1d", limit: int = 100) -> Optional[Dict]:
    """Fetch chart data from backend"""
    try:
        params = {
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit
        }
        response = requests.get(f"{API_BASE}/api/chart-data", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch chart data: {e}")
        return None


def fetch_signal(symbol: str, timeframe: str = "1d") -> Optional[Dict]:
    """Fetch trading signal"""
    try:
        params = {"symbol": symbol, "timeframe": timeframe}
        response = requests.get(f"{API_BASE}/signal", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch signal: {e}")
        return None


def place_order(symbol: str, side: str, quantity: int) -> Optional[Dict]:
    """Place paper trading order"""
    try:
        payload = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity
        }
        response = requests.post(
            f"{API_BASE}/paper-order",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to place order: {e}")
        return None


def fetch_positions() -> Optional[Dict]:
    """Fetch current positions"""
    try:
        response = requests.get(f"{API_BASE}/positions", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch positions: {e}")
        return None


def fetch_trades() -> Optional[Dict]:
    """Fetch trade history"""
    try:
        response = requests.get(f"{API_BASE}/trades", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch trades: {e}")
        return None


def fetch_pnl() -> Optional[Dict]:
    """Fetch P&L data"""
    try:
        response = requests.get(f"{API_BASE}/pnl", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch P&L: {e}")
        return None


# === Helper Functions ===

def align_to_step(v: float, min_v: float, max_v: float, step: float) -> float:
    """
    Align a value to slider's min/max/step constraints
    
    Args:
        v: Value to align
        min_v: Minimum allowed value
        max_v: Maximum allowed value
        step: Step size
    
    Returns:
        Value clipped to [min_v, max_v] and aligned to step grid
    """
    # Clip to range
    v = max(min_v, min(max_v, v))
    # Align to step grid
    n = round((v - min_v) / step)
    return float(min_v + n * step)


# === UI Components ===

def render_sidebar():
    """Render sidebar with controls"""
    st.sidebar.title("📊 EXITON Dashboard")
    
    # Health check
    health = check_health()
    if health.get("status") == "ok":
        st.sidebar.success("✅ Backend Connected")
        st.sidebar.caption(f"v{health.get('version', 'N/A')}")
    else:
        st.sidebar.error("❌ Backend Offline")
        st.sidebar.caption(health.get("message", ""))
    
    st.sidebar.markdown("---")
    
    # Symbol input
    symbol = st.sidebar.text_input(
        "Symbol",
        value="AAPL",
        help="Enter symbol (e.g., AAPL, MSFT, BTC/USDT)"
    )
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        index=6,  # Default to 1d
        help="Select chart timeframe"
    )
    
    # Data size
    limit = st.sidebar.slider(
        "Data Points",
        min_value=50,
        max_value=500,
        value=100,
        step=50,
        help="Number of candles to fetch"
    )
    
    # Order quantity
    quantity = st.sidebar.number_input(
        "Order Quantity",
        min_value=1,
        max_value=1000,
        value=10,
        help="Number of shares/units for orders"
    )
    
    # Refresh button
    refresh = st.sidebar.button("🔄 Refresh Data", type="primary", use_container_width=True)
    
    return symbol, timeframe, limit, quantity, refresh


def render_chart(data: Dict):
    """Render interactive Plotly chart with MA toggle and Y-axis zoom"""
    st.subheader("📈 Price Chart")

    # --- Safety check ---
    if not data or "candles" not in data:
        st.warning("No chart data available")
        return

    candles = data["candles"]
    if not candles:
        st.warning("No candles data")
        return

    # --- DataFrame 化 ---
    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["time"], unit="s")

    # MA を DataFrame に載せる（長さが合わない場合は切り詰め）
    if "shortMA" in data:
        short_ma = data["shortMA"][: len(df)]
        df["shortMA"] = short_ma
    else:
        df["shortMA"] = None

    if "longMA" in data:
        long_ma = data["longMA"][: len(df)]
        df["longMA"] = long_ma
    else:
        df["longMA"] = None

    # ====== Y軸レンジのベース値計算 ======
    price_min = float(df["close"].min())
    price_max = float(df["close"].max())

    # 価格が全部同じなどでレンジがゼロの場合
    if price_max == price_min:
        price_max = price_min + 1.0

    # データの min/max から 5% 余白をつけた「デフォルト FIT レンジ」
    base_span = price_max - price_min
    padding = max(base_span * 0.12, 1.0)  # 最低 0.5 は余白をつける

    default_y_min = price_min - padding
    default_y_max = price_max + padding

    # スライダー全体の min/max（デフォルトより広めに）
    slider_min = price_min - padding * 3
    slider_max = price_max + padding * 3

    # 価格がマイナスにならないように軽くガード（株価用）
    if slider_min < 0:
        slider_min = 0.0

    # データが変わったかどうかを判定するための簡易ハッシュ
    data_hash = hash(tuple(round(v, 4) for v in df["close"].tolist()))

    if "y_range" not in st.session_state:
        st.session_state["y_range"] = (default_y_min, default_y_max)
        st.session_state["y_range_data_hash"] = data_hash
    else:
        # シンボルやタイムフレーム変更でデータが変わったらリセット
        if st.session_state.get("y_range_data_hash") != data_hash:
            st.session_state["y_range"] = (default_y_min, default_y_max)
            st.session_state["y_range_data_hash"] = data_hash

    # ====== UI（MAトグル & FITボタン） ======
    col_left, col_right = st.columns([3, 1])

    with col_left:
        c1, c2 = st.columns(2)
        with c1:
            show_short_ma = st.checkbox("Show Short\nMA", value=True)
        with c2:
            show_long_ma = st.checkbox("Show Long\nMA", value=True)

    with col_right:
        fit_clicked = st.button("FIT", use_container_width=True)

    # FIT が押されたら、y_range をデフォルトレンジにリセット
    if fit_clicked:
        st.session_state["y_range"] = (default_y_min, default_y_max)
        st.rerun()  # 強制的に再描画

    # ====== Y軸スライダー ======
    current_low, current_high = st.session_state["y_range"]

    # スライダーの min/max の外に飛んでいたらクリップ
    current_low = max(slider_min, min(current_low, slider_max))
    current_high = max(slider_min, min(current_high, slider_max))

    if current_low >= current_high:
        current_low, current_high = default_y_min, default_y_max

    # ステップ幅（雑に 200 分割くらい）
    step = (slider_max - slider_min) / 200.0
    if step <= 0:
        step = 0.1

    y_min, y_max = st.slider(
        "Price Range (Y-axis Zoom)",
        min_value=float(slider_min),
        max_value=float(slider_max),
        value=(float(current_low), float(current_high)),
        step=float(step),
        format="%.2f",
        key="y_range",  # session_stateと同じ名前で自動同期
    )

    # Note: key="y_range" により自動的に st.session_state["y_range"] と同期される

    # ====== Plotly チャート描画 ======
    fig = go.Figure()

    # Close price
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["close"],
            mode="lines",
            name="Close",
            line=dict(width=2),
        )
    )

    # 短期 MA
    if show_short_ma and df["shortMA"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["shortMA"],
                mode="lines",
                name="Short MA",
                line=dict(width=1.5, dash="dot"),
            )
        )

    # 長期 MA
    if show_long_ma and df["longMA"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["longMA"],
                mode="lines",
                name="Long MA",
                line=dict(width=1.5, dash="dash"),
            )
        )

    # Y軸レンジをスライダーと完全同期
    fig.update_yaxes(range=[y_min, y_max], title_text="Price")
    fig.update_xaxes(title_text="Time")

    fig.update_layout(
        margin=dict(l=40, r=20, t=30, b=40),
        height=420,
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ====== 下のメトリクス類 ======
    cols = st.columns(4)

    if "stats" in data:
        stats = data["stats"]
        cols[0].metric("Trades", stats.get("tradeCount", 0))
        cols[1].metric("Win Rate", f"{stats.get('winRate', 0):.1f}%")
        cols[2].metric("Total P&L", f"{stats.get('pnlPercent', 0):.2f}%")

    if not df.empty:
        latest_close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2] if len(df) > 1 else latest_close
        change = ((latest_close - prev_close) / prev_close * 100) if prev_close else 0
        cols[3].metric("Latest Close", f"${latest_close:.2f}", f"{change:+.2f}%")



def render_signal_and_orders(symbol: str, timeframe: str, quantity: int):
    """Render signal display and order buttons"""
    st.subheader("🎯 Signal & Orders")
    
    # Fetch signal
    signal_data = fetch_signal(symbol, timeframe)
    
    if signal_data:
        # Display signal
        signal = signal_data.get("signal", "HOLD")
        price = signal_data.get("price", 0)
        confidence = signal_data.get("confidence", 0)
        reason = signal_data.get("reason", "N/A")
        
        # Color code signal
        if signal == "BUY":
            st.success(f"🟢 **{signal}** @ ${price:.2f}")
        elif signal == "SELL":
            st.error(f"🔴 **{signal}** @ ${price:.2f}")
        else:
            st.info(f"⚪ **{signal}** @ ${price:.2f}")
        
        st.caption(f"Confidence: {confidence:.0%} | {reason}")
    else:
        st.warning("No signal available")
    
    st.markdown("---")
    
    # Order buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🟢 BUY (Paper Trade)", type="primary", use_container_width=True):
            result = place_order(symbol, "BUY", quantity)
            if result:
                st.success(f"✅ BUY Order Executed")
                st.json({
                    "Order ID": result.get("order_id"),
                    "Quantity": result.get("quantity"),
                    "Price": f"${result.get('executed_price', 0):.2f}",
                    "Time": result.get("executed_at")
                })
                st.rerun()
    
    with col2:
        if st.button("🔴 SELL (Paper Trade)", type="secondary", use_container_width=True):
            result = place_order(symbol, "SELL", quantity)
            if result:
                st.success(f"✅ SELL Order Executed")
                st.json({
                    "Order ID": result.get("order_id"),
                    "Quantity": result.get("quantity"),
                    "Price": f"${result.get('executed_price', 0):.2f}",
                    "Time": result.get("executed_at")
                })
                st.rerun()


def render_positions_tab():
    """Render positions table"""
    positions_data = fetch_positions()
    
    if not positions_data:
        st.warning("Failed to fetch positions")
        return
    
    positions = positions_data.get("positions", [])
    total_unrealized = positions_data.get("total_unrealized_pnl", 0)
    
    if not positions:
        st.info("No open positions")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(positions)
    
    # Format columns
    df["avg_price"] = df["avg_price"].apply(lambda x: f"${x:.2f}")
    df["current_price"] = df["current_price"].apply(
        lambda x: f"${x:.2f}" if x is not None else "N/A"
    )
    df["unrealized_pnl"] = df["unrealized_pnl"].apply(
        lambda x: f"${x:.2f}" if x is not None else "N/A"
    )
    
    # Rename columns
    df = df.rename(columns={
        "symbol": "Symbol",
        "quantity": "Qty",
        "avg_price": "Avg Price",
        "current_price": "Current Price",
        "unrealized_pnl": "Unrealized P&L"
    })
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.metric("Total Unrealized P&L", f"${total_unrealized:.2f}")


def render_trades_tab():
    """Render trades history table"""
    trades_data = fetch_trades()
    
    if not trades_data:
        st.warning("Failed to fetch trades")
        return
    
    trades = trades_data.get("trades", [])
    
    if not trades:
        st.info("No trade history")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(trades)
    
    # Sort by most recent first
    if "executed_at" in df.columns:
        df = df.sort_values("executed_at", ascending=False)
    
    # Format columns
    if "price" in df.columns:
        df["price"] = df["price"].apply(lambda x: f"${x:.2f}")
    if "pnl" in df.columns:
        df["pnl"] = df["pnl"].apply(lambda x: f"${x:.2f}")
    
    # Rename columns
    column_mapping = {
        "order_id": "Order ID",
        "symbol": "Symbol",
        "side": "Side",
        "quantity": "Qty",
        "price": "Price",
        "executed_at": "Executed At",
        "pnl": "P&L"
    }
    df = df.rename(columns=column_mapping)
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(f"Total Trades: {len(trades)}")


def render_pnl_tab():
    """Render P&L chart and table"""
    pnl_data = fetch_pnl()
    
    if not pnl_data:
        st.warning("Failed to fetch P&L")
        return
    
    pnl_entries = pnl_data.get("pnl", [])
    
    if not pnl_entries:
        st.info("No P&L data")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(pnl_entries)
    
    # Display as table
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # If there's equity data, show a chart
    if "equity" in df.columns and len(df) > 1:
        st.subheader("Equity Curve")
        equity_df = df[["date", "equity"]].copy()
        equity_df.set_index("date", inplace=True)
        st.line_chart(equity_df)


# === Main App ===

def main():
    """Main application"""
    # Header
    st.title("📊 EXITON Developer Dashboard")
    st.caption("Real-time monitoring and control for paper trading system")
    
    # Sidebar
    symbol, timeframe, limit, quantity, refresh = render_sidebar()
    
    # Store in session state
    if "symbol" not in st.session_state or refresh:
        st.session_state.symbol = symbol
        st.session_state.timeframe = timeframe
        st.session_state.chart_data = fetch_chart_data(symbol, timeframe, limit)
    
    # Main content area - 2 columns
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # Chart
        if st.session_state.get("chart_data"):
            render_chart(st.session_state.chart_data)
        else:
            st.warning("No chart data available. Click 'Refresh Data' to load.")
    
    with col_right:
        # Signal and Orders
        render_signal_and_orders(symbol, timeframe, quantity)
    
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


if __name__ == "__main__":
    main()
