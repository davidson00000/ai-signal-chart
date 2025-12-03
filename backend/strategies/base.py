from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class StrategyBase(ABC):
    """
    Abstract base class for all trading strategies.
    """

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on the strategy logic.
        
        Args:
            df (pd.DataFrame): Historical price data with columns like 'open', 'high', 'low', 'close', 'volume'.
                               Index should be datetime or compatible.
        
        Returns:
            pd.Series: A series of signals (1 for Buy, 0 for Hold/Exit, -1 for Sell if applicable, though spec says 1/0).
                       The spec says: "Entry condition (BUY = 1), EXIT condition (0)".
        """
        pass

    @classmethod
    @abstractmethod
    def get_params_schema(cls) -> Dict[str, Any]:
        """
        Return the schema for the strategy parameters to generate UI inputs.
        
        Returns:
            Dict[str, Any]: A dictionary defining the parameters.
            Example:
            {
                "short_window": {"type": "int", "min": 1, "max": 200, "default": 10, "step": 1, "label": "Short Window"},
                "threshold": {"type": "float", "min": 0.0, "max": 10.0, "default": 1.0, "step": 0.1, "label": "Threshold"}
            }
        """
        pass

    @classmethod
    def get_optimization_config(cls) -> Dict[str, Any]:
        """
        Return the default configuration for parameter optimization (2D grid search).
        
        Returns:
            Dict[str, Any]: Configuration for X and Y axes.
            Example:
            {
                "x_param": "short_window",
                "y_param": "long_window",
                "x_range": {"min": 5, "max": 50, "step": 5},
                "y_range": {"min": 20, "max": 200, "step": 10}
            }
        """
        return {}
