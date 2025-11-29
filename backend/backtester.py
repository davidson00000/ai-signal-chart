# backend/backtester.py
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .strategies.base import BaseStrategy


class BacktestEngine:
    """
    シンプルなフルイン・フルアウト型バックテストエンジン。

    tests/test_backtester.py が期待している動作：
    - metrics に以下が含まれる：
        * initial_capital
        * final_equity
        * total_pnl
        * return_pct
        * max_drawdown
        * win_rate
        * trade_count
        * winning_trades
        * losing_trades
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission: float = 0.0005,
        position_size: float = 1.0,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.commission = float(commission)
        self.position_size = float(position_size)

        self._reset()

    # ---- internal state ----
    def _reset(self) -> None:
        self.cash: float = self.initial_capital
        self.position: int = 0  # 保有株数
        self.equity: float = self.initial_capital
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self._entry_price: float | None = None

    # ---- main entry ----
    def run(self, df: pd.DataFrame, strategy: BaseStrategy) -> Dict[str, Any]:
        """
        Run backtest simulation
        """
        # 戦略側のバリデーション
        strategy.validate_dataframe(df)

        # 状態リセット
        self._reset()

        # シグナル生成（Series 想定）
        signals = strategy.generate_signals(df)
        if not isinstance(signals, pd.Series):
            raise ValueError("Strategy.generate_signals must return a pandas Series.")

        # 日毎ループ
        for date, row in df.iterrows():
            price = float(row["close"])
            signal = int(signals.loc[date])

            # トレード前のマークトゥマーケット
            self.equity = self.cash + self.position * price

            # シグナルに応じた売買
            # signal: 1 → ロング、0 → ノーポジ
            if signal == 1 and self.position == 0:
                self._execute_buy(date, price)
            elif signal == 0 and self.position > 0:
                self._execute_sell(date, price)

            # トレード後のエクイティ
            self.equity = self.cash + self.position * price
            self.equity_curve.append(
                {
                    "date": date,
                    "equity": self.equity,
                    "cash": self.cash,
                    "position_value": self.position * price,
                }
            )

        # 期末にポジションが残っていたら、最後の価格でクローズ
        if self.position > 0:
            last_date = df.index[-1]
            last_price = float(df.iloc[-1]["close"])
            self._execute_sell(last_date, last_price)

            # 最後の equity_curve を更新（ノーポジ状態に）
            self.equity = self.cash
            if self.equity_curve:
                self.equity_curve[-1].update(
                    {
                        "equity": self.equity,
                        "cash": self.cash,
                        "position_value": 0.0,
                    }
                )

        metrics = self._calculate_metrics()

        return {
            "strategy": str(strategy),
            "metrics": metrics,
            "trades": self.trades,
            "equity_curve": self.equity_curve,
        }

    # ---- order execution ----
    def _execute_buy(self, date: pd.Timestamp, price: float) -> None:
        # 資金の position_size 割合でフルイン
        max_shares = int((self.cash * self.position_size) // price)
        if max_shares <= 0:
            return

        cost = max_shares * price
        commission = cost * self.commission
        total_cost = cost + commission

        self.cash -= total_cost
        self.position += max_shares
        self._entry_price = price

        self.trades.append(
            {
                "date": date,
                "side": "BUY",
                "price": price,
                "quantity": max_shares,
                "commission": commission,
                "cash_after": self.cash,
                "position": self.position,
                "pnl": 0.0,
            }
        )

    def _execute_sell(self, date: pd.Timestamp, price: float) -> None:
        if self.position <= 0:
            return

        shares = self.position
        proceeds = shares * price
        commission = proceeds * self.commission
        net_proceeds = proceeds - commission

        self.cash += net_proceeds

        entry_price = self._entry_price if self._entry_price is not None else price
        trade_pnl = (price - entry_price) * shares - commission

        self.position = 0
        self._entry_price = None

        self.trades.append(
            {
                "date": date,
                "side": "SELL",
                "price": price,
                "quantity": shares,
                "commission": commission,
                "cash_after": self.cash,
                "position": self.position,
                "pnl": trade_pnl,
            }
        )

    # ---- metrics ----
    def _calculate_metrics(self) -> Dict[str, Any]:
        if self.equity_curve:
            final_equity = float(self.equity_curve[-1]["equity"])
        else:
            final_equity = self.initial_capital

        total_pnl = final_equity - self.initial_capital
        return_pct = (
            (total_pnl / self.initial_capital) * 100.0 if self.initial_capital else 0.0
        )

        max_dd = self._calculate_max_drawdown()
        trade_stats = self._calculate_trade_stats()

        metrics: Dict[str, Any] = {
            "initial_capital": self.initial_capital,
            "final_equity": final_equity,
            "total_pnl": total_pnl,
            "return_pct": return_pct,
            "max_drawdown": max_dd,
        }
        metrics.update(trade_stats)
        return metrics

    def _calculate_max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0

        equity_series = np.array(
            [point["equity"] for point in self.equity_curve], dtype=float
        )
        peaks = np.maximum.accumulate(equity_series)
        drawdowns = (equity_series - peaks) / peaks  # 負の値

        if len(drawdowns) == 0:
            return 0.0

        return float(drawdowns.min()) * 100.0  # ％表示

    def _calculate_trade_stats(self) -> Dict[str, Any]:
        """
        勝率計算と、勝ち負け・トレード数の集計。
        - trade_count: 決済トレード数（SELL の数）
        - winning_trades: pnl > 0 の SELL 数
        - losing_trades: pnl <= 0 の SELL 数
        - win_rate: winning_trades / trade_count * 100
        """
        sells = [t for t in self.trades if t["side"] == "SELL"]
        trade_count = len(sells)

        if trade_count == 0:
            return {
                "win_rate": 0.0,
                "trade_count": 0,
                "winning_trades": 0,
                "losing_trades": 0,
            }

        winning_trades = sum(1 for t in sells if t["pnl"] > 0)
        losing_trades = trade_count - winning_trades
        win_rate = (winning_trades / trade_count) * 100.0

        return {
            "win_rate": win_rate,
            "trade_count": trade_count,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
        }
