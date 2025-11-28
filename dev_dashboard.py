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
    """Render candlestick chart with MA lines"""
    st.subheader("📈 Price Chart")
    
    if not data or "candles" not in data:
        st.warning("No chart data available")
        return
    
    # Convert candles to DataFrame
    candles = data["candles"]
    if not candles:
        st.warning("No candles data")
        return
    
    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    
    # Prepare chart data
    chart_data = pd.DataFrame({
        "Close": df["close"],
    })
    
    # Add MA lines if available
    if "shortMA" in data and data["shortMA"]:
        short_ma = [x for x in data["shortMA"] if x is not None]
        if short_ma:
            chart_data["Short MA"] = data["shortMA"][-len(df):]
    
    if "longMA" in data and data["longMA"]:
        long_ma = [x for x in data["longMA"] if x is not None]
        if long_ma:
            chart_data["Long MA"] = data["longMA"][-len(df):]
    
    # Display chart
    st.line_chart(chart_data, height=400)
    
    # Display stats
    cols = st.columns(4)
    if "stats" in data:
        stats = data["stats"]
        cols[0].metric("Trades", stats.get("tradeCount", 0))
        cols[1].metric("Win Rate", f"{stats.get('winRate', 0):.1f}%")
        cols[2].metric("Total P&L", f"{stats.get('pnlPercent', 0):.2f}%")
    
    # Latest price
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
