from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import ccxt
import yfinance as yf
from typing import List, Literal, Dict, Any, Optional
from datetime import datetime
import math

app = FastAPI(title="AI Signal Chart System")

# CORS（ローカル & Vercel 両対応用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要なら後で絞る
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 取引所（仮想通貨用） ---
# Binance は Vercel から 451 を返すので、クラウド側は Bybit 等にするのが安全
CRYPTO_EXCHANGE_ID = "bybit"  # ローカルだけなら "binance" でもOK

def get_exchange():
    exchange_class = getattr(ccxt, CRYPTO_EXCHANGE_ID)
    return exchange_class({"enableRateLimit": True})


# === ユーティリティ ===

def ohlcv_to_candles(ohlcv: List[List[float]]) -> List[Dict[str, Any]]:
    """
    ccxt 形式の OHLCV: [timestamp_ms, open, high, low, close, volume]
    をフロント用 dict に変換
    """
    candles = []
    for t, o, h, l, c, v in ohlcv:
        candles.append(
            {
                "time": int(t / 1000),  # 秒に変換（Lightweight Charts 用）
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v),
            }
        )
    return candles


def df_to_candles(df, limit: int) -> List[Dict[str, Any]]:
    """
    yfinance の DataFrame をフロント用の candles に変換。
    index: DatetimeIndex
    columns: Open, High, Low, Close, Volume
    """
    candles: List[Dict[str, Any]] = []
    df = df.dropna().tail(limit)

    for ts, row in df.iterrows():
        candles.append(
            {
                "time": int(ts.timestamp()),  # 秒
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0.0)),
            }
        )
    return candles


def simple_moving_average(values: List[float], window: int) -> List[Optional[float]]:
    """
    単純移動平均。
    window に満たない最初の部分は None を返す（NaN は JSON でエラーになるため使用しない）。
    """
    if window <= 0:
        raise ValueError("window must be positive")
    ma: List[Optional[float]] = []
    sum_val = 0.0
    for i, v in enumerate(values):
        sum_val += v
        if i >= window:
            sum_val -= values[i - window]
        if i >= window - 1:
            ma.append(sum_val / window)
        else:
            ma.append(None)
    return ma


def generate_signals_and_trades(
    candles: List[Dict[str, Any]],
    short_window: int,
    long_window: int,
    tp_ratio: float,
    sl_ratio: float,
) -> Dict[str, Any]:
    """
    MA クロスから BUY / SELL シグナルとトレード結果を生成
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

        # MA がまだ計算できていない部分（None）はスキップ
        if (
            ma_s_prev is None
            or ma_l_prev is None
            or ma_s is None
            or ma_l is None
        ):
            continue

        # クロス検知
        # ゴールデンクロス: short が long を下から上抜け → BUY
        golden_cross = ma_s_prev <= ma_l_prev and ma_s > ma_l
        # デッドクロス: short が long を上から下抜け → SELL
        dead_cross = ma_s_prev >= ma_l_prev and ma_s < ma_l

        # --- エントリーシグナル ---
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

        # --- エグジットロジック ---
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

    # stats 計算
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


# === データ取得（仮想通貨） ===

def fetch_crypto_candles(
    symbol: str,
    timeframe: str,
    limit: int,
) -> List[Dict[str, Any]]:
    exchange = get_exchange()
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch crypto data: {e}")

    if not ohlcv:
        raise HTTPException(status_code=400, detail="No OHLCV data for crypto symbol")

    return ohlcv_to_candles(ohlcv)


# === データ取得（株価：Yahoo Finance） ===

def fetch_stock_candles(
    symbol: str,
    timeframe: str,
    limit: int,
) -> List[Dict[str, Any]]:
    """
    yfinance で株価を取得。
    timeframe はフロントの指定をそのまま受けつつ、yfinance の interval にマッピング。
    """
    tf_to_interval = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "60m",
        "4h": "60m",   # 4h は 60m をまとめて見るイメージ
        "1d": "1d",
    }

    if timeframe not in tf_to_interval:
        raise HTTPException(status_code=400, detail=f"Unsupported timeframe for stocks: {timeframe}")

    interval = tf_to_interval[timeframe]

    # 期間はざっくり多めに取る（limitより多めに取り、後ろから limit 件だけ使う）
    if timeframe in ("1m", "5m", "15m", "30m", "1h", "4h"):
        period = "7d"   # intraday は直近 7 日程度
    else:
        period = "2y"   # 日足は 2 年分くらい

    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch stock data: {e}")

    if df is None or df.empty:
        raise HTTPException(status_code=400, detail=f"No stock data for symbol: {symbol}")

    return df_to_candles(df, limit)


# === API エンドポイント ===

@app.get("/api/chart-data")
def get_chart_data(
    symbol: str = Query(..., description="Crypto pair like BTC/USDT or stock like AAPL, 7203.T"),
    timeframe: str = Query("1m", description="1m,5m,15m,30m,1h,4h,1d"),
    limit: int = Query(200, ge=50, le=1000),
    short_window: int = Query(9, ge=1),
    long_window: int = Query(21, ge=2),
    tp_ratio: float = Query(0.01, gt=0),
    sl_ratio: float = Query(0.005, gt=0),
):
    """
    symbol に "/" がある → 仮想通貨（ccxt）
    symbol に "/" がない → 株価（yfinance）
    """

    # どちらのモードか判定
    is_crypto = "/" in symbol

    if is_crypto:
        candles = fetch_crypto_candles(symbol, timeframe, limit)
    else:
        candles = fetch_stock_candles(symbol, timeframe, limit)

    if len(candles) < max(short_window, long_window) + 5:
        raise HTTPException(
            status_code=400,
            detail="Not enough candles for MA calculation. Try smaller windows or different timeframe.",
        )

    signal_result = generate_signals_and_trades(
        candles,
        short_window=short_window,
        long_window=long_window,
        tp_ratio=tp_ratio,
        sl_ratio=sl_ratio,
    )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles,
        "shortMA": signal_result["shortMA"],
        "longMA": signal_result["longMA"],
        "signals": signal_result["signals"],
        "trades": signal_result["trades"],
        "stats": signal_result["stats"],
        "meta": {
            "isCrypto": is_crypto,
            "source": "ccxt" if is_crypto else "yfinance",
        },
    }


# フロントエンドの静的ファイルを配信
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
def root():
    # ルートアクセスでそのまま index.html を返す
    return FileResponse("frontend/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
