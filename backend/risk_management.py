"""
Risk Management Engine - Position Sizing and Stop Loss Calculation

This module provides core risk management functionality for the EXITON
semi-automated trading system.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from math import floor


class RiskManager:
    """
    Risk Manager for calculating position sizes and stop losses
    
    Key Concepts:
    - 1R: The amount of capital risked on a single trade
    - Position Size: Number of shares to buy based on risk tolerance
    - Stop Loss: Price level at which to exit to limit losses
    """
    
    def __init__(
        self,
        default_account_size: float = 10000,
        default_risk_pct: float = 1.0
    ):
        """
        Initialize Risk Manager
        
        Args:
            default_account_size: Default account size in dollars
            default_risk_pct: Default risk per trade as percentage (1.0 = 1%)
        """
        self.default_account_size = default_account_size
        self.default_risk_pct = default_risk_pct
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        account_size: Optional[float] = None,
        risk_per_trade_pct: Optional[float] = None,
        min_position_size: int = 0,
    ) -> Dict[str, Any]:
        """
        Calculate position size based on risk management principles
        
        Formula:
            risk_amount = account_size * (risk_per_trade_pct / 100)
            risk_per_share = abs(entry_price - stop_price)
            position_size = floor(risk_amount / risk_per_share)
        
        Args:
            entry_price: Entry price per share
            stop_price: Stop loss price per share
            account_size: Total account size (uses default if None)
            risk_per_trade_pct: Risk percentage per trade (uses default if None)
            min_position_size: Minimum position size (for lot sizes, etc.)
        
        Returns:
            Dict containing:
                - risk_amount (float): 1R in dollars
                - risk_per_share (float): Risk per share
                - position_size (int): Recommended number of shares
                - warnings (list[str]): Warning messages if any
                - account_size (float): Account size used
                - risk_per_trade_pct (float): Risk % used
                - entry_price (float): Entry price
                - stop_price (float): Stop loss price
        """
        # Use defaults if not provided
        if account_size is None:
            account_size = self.default_account_size
        if risk_per_trade_pct is None:
            risk_per_trade_pct = self.default_risk_pct
        
        warnings = []
        
        # Validate inputs
        if account_size <= 0:
            warnings.append("Account size must be positive")
            return {
                "risk_amount": 0,
                "risk_per_share": 0,
                "position_size": 0,
                "warnings": warnings,
                "account_size": account_size,
                "risk_per_trade_pct": risk_per_trade_pct,
                "entry_price": entry_price,
                "stop_price": stop_price
            }
        
        if risk_per_trade_pct <= 0 or risk_per_trade_pct > 100:
            warnings.append("Risk percentage must be between 0 and 100")
            return {
                "risk_amount": 0,
                "risk_per_share": 0,
                "position_size": 0,
                "warnings": warnings,
                "account_size": account_size,
                "risk_per_trade_pct": risk_per_trade_pct,
                "entry_price": entry_price,
                "stop_price": stop_price
            }
        
        # Calculate 1R (risk amount)
        risk_amount = account_size * (risk_per_trade_pct / 100.0)
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_price)
        
        # Check if risk per share is valid
        if risk_per_share <= 0:
            warnings.append(
                "Entry price equals stop price - no risk per share. "
                "Cannot calculate position size."
            )
            return {
                "risk_amount": risk_amount,
                "risk_per_share": risk_per_share,
                "position_size": 0,
                "warnings": warnings,
                "account_size": account_size,
                "risk_per_trade_pct": risk_per_trade_pct,
                "entry_price": entry_price,
                "stop_price": stop_price
            }
        
        # Calculate position size
        position_size = floor(risk_amount / risk_per_share)
        
        # Apply minimum position size
        if position_size < min_position_size:
            warnings.append(
                f"Calculated position size ({position_size}) is less than "
                f"minimum ({min_position_size}). Setting to 0."
            )
            # Note: For lot-based trading (e.g., 100 shares per lot),
            # this can be extended to round to nearest lot
            position_size = 0
        
        # Warn if position size is very large
        max_shares = floor(account_size / entry_price)
        if position_size > max_shares:
            warnings.append(
                f"Position size ({position_size}) exceeds maximum buyable "
                f"with account size ({max_shares}). Consider wider stop loss."
            )
            position_size = max_shares
        
        # Warn if risking more than recommended
        if risk_per_trade_pct > 2.0:
            warnings.append(
                f"Risk per trade ({risk_per_trade_pct}%) exceeds recommended "
                f"maximum (2%). Consider reducing risk percentage."
            )
        
        return {
            "risk_amount": round(risk_amount, 2),
            "risk_per_share": round(risk_per_share, 2),
            "position_size": int(position_size),
            "warnings": warnings,
            "account_size": account_size,
            "risk_per_trade_pct": risk_per_trade_pct,
            "entry_price": entry_price,
            "stop_price": stop_price
        }
    
    def suggest_stop_price_from_atr(
        self,
        df: pd.DataFrame,
        current_price: Optional[float] = None,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        direction: str = "LONG",
    ) -> Dict[str, Any]:
        """
        Suggest stop loss price based on ATR (Average True Range)
        
        Args:
            df: DataFrame with OHLC data
            current_price: Current/entry price (uses last close if None)
            atr_period: Period for ATR calculation
            atr_multiplier: Multiplier for ATR (2.0 = 2x ATR)
            direction: "LONG" or "SHORT"
        
        Returns:
            Dict containing:
                - suggested_stop (float | None): Suggested stop price
                - atr_value (float | None): Calculated ATR value
                - current_price (float | None): Price used for calculation
                - warnings (list[str]): Warning messages if any
        """
        warnings = []
        
        # Validate direction
        if direction not in ["LONG", "SHORT"]:
            warnings.append(f"Invalid direction '{direction}'. Use 'LONG' or 'SHORT'.")
            return {
                "suggested_stop": None,
                "atr_value": None,
                "current_price": None,
                "warnings": warnings
            }
        
        # Check if DataFrame has required columns
        required_cols = ['high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            warnings.append(
                f"DataFrame missing required columns: {missing_cols}"
            )
            return {
                "suggested_stop": None,
                "atr_value": None,
                "current_price": None,
                "warnings": warnings
            }
        
        # Check if we have enough data
        if len(df) < atr_period + 1:
            warnings.append(
                f"Insufficient data for ATR calculation. Need at least "
                f"{atr_period + 1} bars, got {len(df)}."
            )
            return {
                "suggested_stop": None,
                "atr_value": None,
                "current_price": None,
                "warnings": warnings
            }
        
        try:
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=atr_period).mean()
            
            # Get the latest ATR value
            atr_value = atr.iloc[-1]
            
            if pd.isna(atr_value):
                warnings.append("ATR calculation resulted in NaN. Insufficient data.")
                return {
                    "suggested_stop": None,
                    "atr_value": None,
                    "current_price": None,
                    "warnings": warnings
                }
            
            # Determine current price
            if current_price is None:
                current_price = df['close'].iloc[-1]
            
            # Calculate suggested stop
            if direction == "LONG":
                suggested_stop = current_price - (atr_multiplier * atr_value)
            else:  # SHORT
                suggested_stop = current_price + (atr_multiplier * atr_value)
            
            # Warn if stop is too tight (< 1% from entry)
            stop_distance_pct = abs(current_price - suggested_stop) / current_price * 100
            if stop_distance_pct < 1.0:
                warnings.append(
                    f"Suggested stop is very tight ({stop_distance_pct:.2f}% from entry). "
                    f"Consider using larger ATR multiplier."
                )
            
            return {
                "suggested_stop": round(suggested_stop, 2),
                "atr_value": round(atr_value, 2),
                "current_price": round(current_price, 2),
                "warnings": warnings
            }
            
        except Exception as e:
            warnings.append(f"Error calculating ATR: {str(e)}")
            return {
                "suggested_stop": None,
                "atr_value": None,
                "current_price": None,
                "warnings": warnings
            }
    
    def calculate_risk_reward_ratio(
        self,
        entry_price: float,
        stop_price: float,
        target_price: float
    ) -> Dict[str, Any]:
        """
        Calculate risk/reward ratio for a trade
        
        Args:
            entry_price: Entry price
            stop_price: Stop loss price
            target_price: Target/take-profit price
        
        Returns:
            Dict containing:
                - risk: Distance from entry to stop
                - reward: Distance from entry to target
                - ratio: Reward/Risk ratio
                - warnings: Warning messages
        """
        warnings = []
        
        risk = abs(entry_price - stop_price)
        reward = abs(target_price - entry_price)
        
        if risk <= 0:
            warnings.append("Risk is zero or negative. Cannot calculate ratio.")
            return {
                "risk": risk,
                "reward": reward,
                "ratio": None,
                "warnings": warnings
            }
        
        ratio = reward / risk
        
        if ratio < 1.0:
            warnings.append(
                f"Risk/Reward ratio ({ratio:.2f}) is less than 1:1. "
                f"Consider wider target or tighter stop."
            )
        
        return {
            "risk": round(risk, 2),
            "reward": round(reward, 2),
            "ratio": round(ratio, 2),
            "warnings": warnings
        }
