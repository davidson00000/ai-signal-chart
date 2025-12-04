"""
Strategy Package - Trading strategy abstractions
"""
from backend.strategies.base import StrategyBase
from backend.strategies.ma_cross import MACrossStrategy
from backend.strategies.ema_cross import EMACrossStrategy
from backend.strategies.macd_trend import MACDTrendStrategy
from backend.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from backend.strategies.stoch_oscillator import StochasticOscillatorStrategy
from backend.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from backend.strategies.bollinger_breakout import BollingerBreakoutStrategy
from backend.strategies.donchian_breakout import DonchianBreakoutStrategy
from backend.strategies.atr_trailing_ma import ATRTrailingMAStrategy
from backend.strategies.roc_momentum import ROCMomentumStrategy
from backend.strategies.ema9_dip_buy import EMA9DipBuyStrategy

__all__ = [
    "StrategyBase",
    "MACrossStrategy",
    "EMACrossStrategy",
    "MACDTrendStrategy",
    "RSIMeanReversionStrategy",
    "StochasticOscillatorStrategy",
    "BollingerMeanReversionStrategy",
    "BollingerBreakoutStrategy",
    "DonchianBreakoutStrategy",
    "ATRTrailingMAStrategy",
    "ROCMomentumStrategy",
    "EMA9DipBuyStrategy"
]
