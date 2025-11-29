"""
Strategy Package - Trading strategy abstractions
"""
from backend.strategies.base import BaseStrategy
from backend.strategies.ma_cross import MACrossStrategy

__all__ = ["BaseStrategy", "MACrossStrategy"]
