"""
Realtime Simulation Engine for Auto Sim Lab

This module implements the core simulation engine for realtime paper trading.
It processes live price ticks, generates signals, and executes trades.

Key features:
- Real-time price processing
- Signal generation using existing predictors
- Position management (long-only for now)
- Decision logging for audit trail
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import asyncio

from backend.realtime_data_source import RealtimeDataSource, RealtimeCandle
from backend.models.decision_log import DecisionEvent, DecisionLog
from backend.auto_sim_lab import AutoSimConfig


class RealtimeSimEngine:
    """
    Core engine for realtime paper trading simulation.
    
    This engine processes price ticks, generates trading signals,
    and manages positions in real-time.
    """
    
    def __init__(
        self,
        session_id: str,
        config: AutoSimConfig,
        data_source: RealtimeDataSource
    ):
        """
        Initialize the realtime simulation engine.
        
        Args:
            session_id: Unique identifier for this simulation session
            config: Simulation configuration
            data_source: Data source for price feeds
        """
        self.session_id = session_id
        self.config = config
        self.data_source = data_source
        
        # State
        self.symbol = config.symbol
        self.timeframe = config.timeframe or "1m"
        self.equity = config.initial_capital
        self.initial_equity = config.initial_capital
        self.risk_per_trade = config.risk_per_trade
        
        # Position state
        self.position_side: str = "flat"  # "flat" or "long"
        self.position_size: float = 0.0
        self.entry_price: float = 0.0
        self.entry_time: Optional[datetime] = None
        
        # Historical data buffer for signal generation
        self.candle_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = 100  # Keep last 100 candles for indicators
        
        # Results
        self.equity_curve: List[Dict[str, Any]] = []
        self.trades: List[Dict[str, Any]] = []
        self.decision_log = DecisionLog()
        
        # Control
        self.is_running = True
        self.last_tick_time: Optional[datetime] = None
        self.tick_count = 0
        
        # Record initial state
        self.equity_curve.append({
            "timestamp": datetime.utcnow().isoformat(),
            "equity": self.equity,
            "position": self.position_side
        })
    
    async def tick(self) -> bool:
        """
        Process one tick of realtime data.
        
        This method:
        1. Fetches latest price/candle
        2. Updates candle buffer
        3. Generates signal
        4. Executes trading logic
        5. Updates equity curve
        
        Returns:
            True if tick was processed successfully, False otherwise
        """
        if not self.is_running:
            return False
        
        try:
            self.tick_count += 1
            now = datetime.utcnow()
            self.last_tick_time = now
            
            # 1. Fetch latest candle
            candle = await self.data_source.get_or_update_candle(
                self.symbol, self.timeframe
            )
            
            if candle is None:
                return False
            
            close_price = candle.close
            
            # 2. Update candle buffer
            self._update_candle_buffer(candle)
            
            # 3. Generate signal
            signal = self._generate_signal()
            final_action = signal.get("action", "hold")
            
            # Calculate current equity (mark-to-market)
            if self.position_side == "long":
                unrealized_pnl = (close_price - self.entry_price) * self.position_size
                current_equity = self.equity + unrealized_pnl
            else:
                current_equity = self.equity
            
            # 4. Log signal decision
            self.decision_log.add(DecisionEvent(
                timestamp=now,
                symbol=self.symbol,
                timeframe=self.timeframe,
                event_type="signal_decision",
                final_signal=final_action,
                raw_signals=signal.get("predictions"),
                position_side=self.position_side,
                position_size=self.position_size if self.position_side == "long" else None,
                price=close_price,
                equity_before=current_equity,
                equity_after=current_equity,
                risk_per_trade=self.risk_per_trade,
                reason=f"[Tick {self.tick_count}] {signal.get('reason', 'No reason')} "
                       f"(price: ${close_price:.2f})"
            ))
            
            # 5. Execute trading logic
            if final_action == "buy" and self.position_side == "flat":
                await self._enter_long(close_price, now)
            elif final_action == "sell" and self.position_side == "long":
                await self._exit_long(close_price, now)
            
            # 6. Update equity curve
            if self.position_side == "long":
                unrealized_pnl = (close_price - self.entry_price) * self.position_size
                curve_equity = self.equity + unrealized_pnl
            else:
                curve_equity = self.equity
            
            self.equity_curve.append({
                "timestamp": now.isoformat(),
                "equity": round(curve_equity, 2),
                "position": self.position_side,
                "price": close_price
            })
            
            return True
            
        except Exception as e:
            print(f"[RealtimeSimEngine] Tick error: {e}")
            return False
    
    def _update_candle_buffer(self, candle: RealtimeCandle) -> None:
        """Add new candle to buffer, maintaining max size."""
        candle_dict = {
            "time": int(candle.timestamp.timestamp()),
            "timestamp": candle.timestamp.isoformat(),
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume
        }
        
        # Replace last candle if same timestamp, otherwise append
        if self.candle_buffer and self.candle_buffer[-1].get("time") == candle_dict["time"]:
            self.candle_buffer[-1] = candle_dict
        else:
            self.candle_buffer.append(candle_dict)
        
        # Trim buffer
        if len(self.candle_buffer) > self.max_buffer_size:
            self.candle_buffer = self.candle_buffer[-self.max_buffer_size:]
    
    def _generate_signal(self) -> Dict[str, Any]:
        """
        Generate trading signal using existing predictors.
        
        Reuses the signal generation logic from Auto Sim Lab.
        """
        if len(self.candle_buffer) < 20:
            return {
                "action": "hold",
                "confidence": 0.0,
                "reason": f"Insufficient data ({len(self.candle_buffer)}/20 candles)"
            }
        
        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.candle_buffer)
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Import and use signal generation logic
            from backend.auto_sim_lab import generate_signal_for_bar
            return generate_signal_for_bar(df, self.symbol, self.timeframe)
            
        except Exception as e:
            return {
                "action": "hold",
                "confidence": 0.0,
                "reason": f"Signal generation error: {e}"
            }
    
    async def _enter_long(self, price: float, timestamp: datetime) -> None:
        """
        Enter a long position.
        
        Args:
            price: Entry price
            timestamp: Entry timestamp
        """
        risk_amount = self.equity * self.risk_per_trade
        position_size = int(risk_amount / price)
        
        if position_size <= 0:
            self.decision_log.add(DecisionEvent(
                timestamp=timestamp,
                symbol=self.symbol,
                timeframe=self.timeframe,
                event_type="signal_decision",
                final_signal="buy",
                position_side="flat",
                price=price,
                equity_before=self.equity,
                equity_after=self.equity,
                risk_per_trade=self.risk_per_trade,
                reason=f"Entry skipped: position size would be 0 (risk=${risk_amount:.2f}, price=${price:.2f})"
            ))
            return
        
        self.position_side = "long"
        self.position_size = position_size
        self.entry_price = price
        self.entry_time = timestamp
        
        self.decision_log.add(DecisionEvent(
            timestamp=timestamp,
            symbol=self.symbol,
            timeframe=self.timeframe,
            event_type="entry",
            final_signal="buy",
            position_side="long",
            position_size=position_size,
            price=price,
            equity_before=self.equity,
            equity_after=self.equity,
            risk_per_trade=self.risk_per_trade,
            reason=f"ENTER LONG @ ${price:.2f}, size={position_size} shares. "
                   f"Risk: ${risk_amount:.2f} ({self.risk_per_trade*100:.1f}% of ${self.equity:,.2f})"
        ))
    
    async def _exit_long(self, price: float, timestamp: datetime) -> None:
        """
        Exit the current long position.
        
        Args:
            price: Exit price
            timestamp: Exit timestamp
        """
        if self.position_side != "long":
            return
        
        pnl = (price - self.entry_price) * self.position_size
        self.equity += pnl
        
        # Record trade
        self.trades.append({
            "entry_time": self.entry_time.isoformat() if self.entry_time else "",
            "exit_time": timestamp.isoformat(),
            "entry_price": self.entry_price,
            "exit_price": price,
            "size": self.position_size,
            "pnl": round(pnl, 2),
            "return_pct": round((price / self.entry_price - 1) * 100, 2)
        })
        
        self.decision_log.add(DecisionEvent(
            timestamp=timestamp,
            symbol=self.symbol,
            timeframe=self.timeframe,
            event_type="exit",
            final_signal="sell",
            position_side="flat",
            position_size=self.position_size,
            price=price,
            equity_before=self.equity - pnl,  # Before this trade
            equity_after=self.equity,
            risk_per_trade=self.risk_per_trade,
            reason=f"EXIT LONG @ ${price:.2f}. "
                   f"PnL: ${pnl:+,.2f} ({(price/self.entry_price-1)*100:+.2f}%)"
        ))
        
        # Reset position
        self.position_side = "flat"
        self.position_size = 0
        self.entry_price = 0.0
        self.entry_time = None
    
    def stop(self) -> None:
        """Stop the simulation engine."""
        self.is_running = False
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current state of the simulation.
        
        Returns:
            Dict with equity curve, trades, decision log, and position info
        """
        return {
            "session_id": self.session_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "is_running": self.is_running,
            "tick_count": self.tick_count,
            "last_tick_time": self.last_tick_time.isoformat() if self.last_tick_time else None,
            "initial_equity": self.initial_equity,
            "current_equity": round(self.equity, 2),
            "total_return_pct": round((self.equity / self.initial_equity - 1) * 100, 2),
            "position": {
                "side": self.position_side,
                "size": self.position_size,
                "entry_price": self.entry_price,
                "entry_time": self.entry_time.isoformat() if self.entry_time else None
            },
            "equity_curve": self.equity_curve[-100:],  # Last 100 points
            "trades": self.trades,
            "decision_log": self.decision_log.to_list()[-50:],  # Last 50 events
            "summary": {
                "total_trades": len(self.trades),
                "wins": sum(1 for t in self.trades if t["pnl"] > 0),
                "losses": sum(1 for t in self.trades if t["pnl"] <= 0),
                "total_pnl": round(sum(t["pnl"] for t in self.trades), 2)
            }
        }
