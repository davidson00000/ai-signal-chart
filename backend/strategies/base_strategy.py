from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import pandas as pd

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize strategy with configuration parameters.
        
        Args:
            config: Dictionary containing strategy parameters
        """
        self.config = config

    @abstractmethod
    def generate_action(self, df: pd.DataFrame, idx: int) -> Tuple[str, Dict[str, Any]]:
        """
        Generate trading action for a specific bar.
        
        Args:
            df: DataFrame containing price data (must include 'open', 'high', 'low', 'close')
            idx: Index of the current bar to analyze
            
        Returns:
            Tuple of (action, info_dict)
            action: "buy", "sell", or "hold"
            info_dict: Dictionary containing strategy-specific details (indicators, reason, etc.)
        """
        pass
