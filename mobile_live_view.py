"""
EXITON Live Signals - Mobile View

スマホでの閲覧に特化したシンプルなLive Signalビューです。
"""

import os
import streamlit as st
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
import time

# ===============================================================================
# Configuration
# ===============================================================================

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8001")
LIVE_SIGNALS_ENDPOINT = f"{BACKEND_URL}/live-signals"
AUTO_REFRESH_INTERVAL = 60  # seconds

# デフォルト設定
DEFAULT_SYMBOLS = "AAPL,MSFT,GOOGL,TSLA,NVDA"
DEFAULT_STRATEGY = "ma_cross"

# ===============================================================================
# Helper Functions
# ===============================================================================

def fetch_live_signals(symbols: str, strategy: str) -> Optional[List[Dict[str, Any]]]:
    """
    バックエンドからLive Signalを取得する
    
    Args:
        symbols: カンマ区切りのシンボルリスト
        strategy: 戦略名
        
    Returns:
        シグナルのリスト、エラー時はNone
    """
    try:
        response = requests.get(
            LIVE_SIGNALS_ENDPOINT,
            params={"symbols": symbols, "strategy": strategy},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"🔌 バックエンドに接続できません: {BACKEND_URL}")
        st.info("バックエンドが起動していることを確認してください。")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ タイムアウト: バックエンドの応答が遅すぎます")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTPエラー: {e}")
        return None
    except Exception as e:
        st.error(f"❌ エラー: {str(e)}")
        return None


def format_datetime(iso_string: str) -> str:
    """ISO8601文字列をローカルタイムに変換"""
    try:
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return iso_string


def get_side_color(side: str) -> str:
    """サイドに応じた色を返す"""
    if side == "BUY":
        return "#22c55e"  # Green
    elif side == "SELL":
        return "#ef4444"  # Red
    else:
        return "#6b7280"  # Gray


def get_confidence_color(confidence: float) -> str:
    """信頼度に応じた色を返す"""
    if confidence >= 0.7:
        return "#22c55e"  # Green
    elif confidence >= 0.5:
        return "#eab308"  # Yellow
    else:
        return "#ef4444"  # Red

# ===============================================================================
# UI Components
# ===============================================================================

def render_signal_card(signal: Dict[str, Any]):
    """シグナルカードを表示"""
    symbol = signal.get("symbol", "N/A")
    side = signal.get("side", "HOLD")
    price = signal.get("price", 0.0)
    time_str = signal.get("time", "")
    confidence = signal.get("confidence", 0.5)
    reason_summary = signal.get("reason_summary", "")
    explain = signal.get("explain", {})
    
    # カードのスタイル
    side_color = get_side_color(side)
    confidence_color = get_confidence_color(confidence)
    
    # カードコンテナ
    with st.container():
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-left: 4px solid {side_color};
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h3 style="margin: 0; color: #fff; font-size: 1.5rem;">{symbol}</h3>
                <span style="
                    background-color: {side_color};
                    color: white;
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 0.9rem;
                    font-weight: 600;
                ">{side}</span>
            </div>
            <div style="color: #cbd5e1; margin-bottom: 8px;">
                <span style="font-size: 1.2rem; font-weight: 500;">${price:.2f}</span>
                <span style="margin-left: 12px; font-size: 0.85rem; color: #94a3b8;">{format_datetime(time_str)}</span>
            </div>
            <div style="color: #e2e8f0; font-size: 0.9rem; margin-bottom: 8px;">
                {reason_summary}
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="flex: 1; background: #334155; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="
                        width: {confidence * 100}%;
                        height: 100%;
                        background: {confidence_color};
                        transition: width 0.3s ease;
                    "></div>
                </div>
                <span style="color: {confidence_color}; font-size: 0.85rem; font-weight: 600;">
                    {confidence * 100:.0f}%
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 詳細情報（expander）
        with st.expander("📊 詳細情報"):
            # Indicators
            st.markdown("**指標値**")
            indicators = explain.get("indicators", {})
            if indicators:
                cols = st.columns(2)
                for idx, (key, value) in enumerate(indicators.items()):
                    col = cols[idx % 2]
                    with col:
                        st.metric(
                            label=key.replace("_", " ").title(),
                            value=f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                        )
            else:
                st.info("指標情報がありません")
            
            st.markdown("---")
            
            # Conditions
            st.markdown("**発火条件**")
            conditions = explain.get("conditions_triggered", [])
            if conditions:
                for condition in conditions:
                    st.markdown(f"✓ {condition}")
            else:
                st.info("条件情報がありません")
            
            st.markdown("---")
            
            # Confidence
            st.markdown(f"**信頼度スコア**: `{explain.get('confidence', 0.5):.2f}`")


# ===============================================================================
# Main App
# ===============================================================================

def main():
    """メインアプリケーション"""
    
    # ページ設定
    st.set_page_config(
        page_title="EXITON Live Signals (Mobile)",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # カスタムCSS
    st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        h1 {
            color: #38bdf8;
            margin-bottom: 0.5rem;
        }
        .stExpander {
            background-color: #1e293b;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # ヘッダー
    st.title("📱 EXITON Live Signals")
    st.markdown(f"*自動更新: {AUTO_REFRESH_INTERVAL}秒ごと*")
    
    # 設定（サイドバー）
    with st.sidebar:
        st.header("⚙️ 設定")
        symbols_input = st.text_input(
            "銘柄 (カンマ区切り)",
            value=DEFAULT_SYMBOLS,
            help="例: AAPL,MSFT,GOOGL"
        )
        strategy_input = st.selectbox(
            "戦略",
            options=["ma_cross", "rsi_mean_reversion", "macd_trend"],
            index=0
        )
        
        st.markdown("---")
        st.markdown(f"**バックエンドURL**")
        st.code(BACKEND_URL, language="text")
        
        if st.button("🔄 手動更新", use_container_width=True):
            st.rerun()
    
    # シグナル取得
    with st.spinner("シグナルを取得中..."):
        signals = fetch_live_signals(symbols_input, strategy_input)
    
    # シグナル表示
    if signals is None:
        st.warning("⚠️ シグナルを取得できませんでした")
        st.stop()
    
    if len(signals) == 0:
        st.info("ℹ️ 現在アクティブなシグナルはありません")
    else:
        st.success(f"✅ {len(signals)}件のシグナルを取得しました")
        
        # シグナルカードを表示
        for signal in signals:
            render_signal_card(signal)
    
    # 最終更新時刻
    st.markdown("---")
    st.caption(f"最終更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 自動リロード
    time.sleep(AUTO_REFRESH_INTERVAL)
    st.rerun()


if __name__ == "__main__":
    main()
