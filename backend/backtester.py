"""
Backtest Engine - Core backtesting logic
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

from backend.strategies.base import StrategyBase


class BacktestEngine:
    """
    シンプルなバックテストエンジン。

    戦略からシグナルを受け取り、売買をシミュレートして損益を計算する。
    """

    def __init__(
        self,
        initial_capital: float,
        position_size: float = 1.0,
        commission_rate: float = 0.0,
        lot_size: float = 1.0,
    ):
        """
        Initialize BacktestEngine.

        Args:
            initial_capital: Starting capital
            position_size: Position size multiplier (0.0-1.0)
            commission_rate: Commission rate per trade (e.g., 0.001 = 0.1%)
            lot_size: Lot size for position calculation
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission_rate = commission_rate
        self.lot_size = lot_size

    def _iso(self, ts) -> str:
        """Convert timestamp to ISO8601 string."""
        if hasattr(ts, 'tzinfo'):
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
        return ts.isoformat().replace("+00:00", "Z")

    def run_backtest(self, candles: pd.DataFrame, strategy: StrategyBase, start_date: pd.Timestamp = None) -> Dict[str, Any]:
        """
        Run backtest with given candles and strategy.

        Args:
            candles: DataFrame with OHLCV data (must have 'close' column)
            strategy: Strategy instance that implements generate_signals()
            start_date: Optional start date for trading (data before this is used for warm-up)

        Returns:
            Dict containing:
                - equity_curve: List[Dict] with timestamp and equity
                - trades: List[Dict] with trade details
                - stats: Dict with performance metrics
        """
        # 1. Generate signals from strategy
        signals_df = strategy.generate_signals(candles)
        if isinstance(signals_df, pd.DataFrame) and "signal" in signals_df.columns:
            signals = signals_df["signal"]
        elif isinstance(signals_df, pd.Series):
            signals = signals_df
        else:
            signals = pd.Series(0, index=candles.index)

        # 2. Initialize state
        cash = self.initial_capital
        position = 0
        entry_price = 0.0

        trades: List[Dict] = []
        equity_curve: List[Dict] = []

        # Ensure start_date is timezone-aware if candles index is
        if start_date and candles.index.tz is not None and start_date.tzinfo is None:
            start_date = start_date.tz_localize("UTC")

        # 3. Iterate through candles
        for idx, (ts, row) in enumerate(candles.iterrows()):
            # Skip if before start_date (warm-up period)
            if start_date and ts < start_date:
                continue

            price = float(row["close"])
            # Get signal for this timestamp
            # signals is a Series with same index as df
            # We need to access by label (timestamp)
            try:
                signal = int(signals.loc[ts])
            except (KeyError, ValueError, TypeError):
                signal = 0

            # Buy signal (1) and no position
            if signal == 1 and position == 0:
                # Calculate quantity to buy
                qty = (cash * self.position_size) // price
                if qty > 0:
                    cost = qty * price
                    commission = cost * self.commission_rate

                    cash -= (cost + commission)
                    position = qty
                    entry_price = price

                    trades.append({
                        "date": self._iso(ts),
                        "side": "BUY",
                        "price": price,
                        "quantity": qty,
                        "commission": commission,
                        "pnl": None,
                        "cash_after": cash,
                    })

            # Sell signal (-1) and have position
            elif signal == -1 and position > 0:
                qty = position
                revenue = qty * price
                commission = revenue * self.commission_rate

                # Calculate PnL
                pnl = revenue - commission - (qty * entry_price)

                cash += (revenue - commission)
                position = 0

                trades.append({
                    "date": self._iso(ts),
                    "side": "SELL",
                    "price": price,
                    "quantity": qty,
                    "commission": commission,
                    "pnl": pnl,
                    "cash_after": cash,
                })

            # Record equity
            equity = cash + position * price
            equity_curve.append({
                "date": self._iso(ts),
                "equity": equity,
                "cash": cash,
            })

        # 4. Force close position at end if still open
        if position > 0:
            ts = candles.index[-1]
            price = float(candles.iloc[-1]["close"])

            qty = position
            revenue = qty * price
            commission = revenue * self.commission_rate
            pnl = revenue - commission - (qty * entry_price)

            cash += (revenue - commission)
            position = 0

            trades.append({
                "date": self._iso(ts),
                "side": "SELL",
                "price": price,
                "quantity": qty,
                "commission": commission,
                "pnl": pnl,
                "cash_after": cash,
            })

            # Update final equity point
            equity_curve.append({
                "date": self._iso(ts),
                "equity": cash,
                "cash": cash,
            })

        # 5. Calculate statistics
        final_equity = cash
        total_pnl = final_equity - self.initial_capital
        return_pct = (total_pnl / self.initial_capital) * 100

        # Calculate trade statistics
        winning_trades = [t for t in trades if t.get("pnl") is not None and t["pnl"] > 0]
        losing_trades = [t for t in trades if t.get("pnl") is not None and t["pnl"] < 0]
        completed_trades = [t for t in trades if t.get("pnl") is not None]

        trade_count = len(completed_trades)
        win_count = len(winning_trades)
        lose_count = len(losing_trades)
        win_rate = (win_count / trade_count) if trade_count > 0 else 0.0

        # Calculate max drawdown
        max_drawdown = 0.0
        peak = self.initial_capital
        for point in equity_curve:
            equity = point["equity"]
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak if peak > 0 else 0.0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Calculate Sharpe ratio (simplified)
        sharpe_ratio = None
        if len(equity_curve) > 1:
            returns = []
            for i in range(1, len(equity_curve)):
                prev_equity = equity_curve[i - 1]["equity"]
                curr_equity = equity_curve[i]["equity"]
                ret = (curr_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
                returns.append(ret)

            if len(returns) > 0:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    sharpe_ratio = (mean_return / std_return) * np.sqrt(252)  # Annualized

        stats = {
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "total_pnl": total_pnl,
            "return_pct": return_pct,
            "trade_count": trade_count,
            "winning_trades": win_count,
            "losing_trades": lose_count,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
        }

        # Prepare price series for charting
        price_series = []
        # Reset index to get timestamp as column if it's the index
        df_reset = candles.reset_index()
        for _, row in df_reset.iterrows():
            # Handle different index names or columns
            ts = row.get("timestamp", row.get("date", row.get("index")))
            if pd.isna(ts):
                continue
                
            price_series.append({
                "date": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume")
            })

        return {
            "equity_curve": equity_curve,
            "trades": trades,
            "stats": stats,
            "price_series": price_series
        }
