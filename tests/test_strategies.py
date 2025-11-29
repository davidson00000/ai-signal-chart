"""
Tests for trading strategies
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backend.strategies.base import BaseStrategy
from backend.strategies.ma_cross import MACrossStrategy


# Test data fixture
@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        'open': close_prices + np.random.randn(100) * 0.5,
        'high': close_prices + np.abs(np.random.randn(100)) * 1.5,
        'low': close_prices - np.abs(np.random.randn(100)) * 1.5,
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    return df


class TestBaseStrategy:
    """Test BaseStrategy interface"""
    
    def test_base_strategy_cannot_instantiate(self):
        """BaseStrategy should not be instantiable"""
        with pytest.raises(TypeError):
            BaseStrategy()
    
    def test_validate_dataframe_missing_columns(self, sample_ohlcv_data):
        """Should raise error if DataFrame missing columns"""
        strategy = MACrossStrategy()
        
        # Remove close column
        df_bad = sample_ohlcv_data.drop(columns=['close'])
        
        with pytest.raises(ValueError, match="missing required columns"):
            strategy.validate_dataframe(df_bad)
    
    def test_validate_dataframe_empty(self):
        """Should raise error if DataFrame is empty"""
        strategy = MACrossStrategy()
        df_empty = pd.DataFrame()
        
        with pytest.raises(ValueError, match="empty"):
            strategy.validate_dataframe(df_empty)


class TestMACrossStrategy:
    """Test MA Cross Strategy"""
    
    def test_initialization_default(self):
        """Test default initialization"""
        strategy = MACrossStrategy()
        assert strategy.params['short_window'] == 9
        assert strategy.params['long_window'] == 21
    
    def test_initialization_custom(self):
        """Test custom parameters"""
        strategy = MACrossStrategy(short_window=5, long_window=20)
        assert strategy.params['short_window'] == 5
        assert strategy.params['long_window'] == 20
    
    def test_initialization_invalid_windows(self):
        """Test invalid window sizes"""
        with pytest.raises(ValueError):
            MACrossStrategy(short_window=0, long_window=10)
        
        with pytest.raises(ValueError):
            MACrossStrategy(short_window=20, long_window=10)
    
    def test_generate_signals(self, sample_ohlcv_data):
        """Test signal generation"""
        strategy = MACrossStrategy(short_window=5, long_window=20)
        signals = strategy.generate_signals(sample_ohlcv_data)
        
        # Check output type and length
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_ohlcv_data)
        
        # Check signal values are valid (0 or 1)
        assert signals.isin([0, 1]).all()
        
        # First long_window-1 signals should be 0 (no MA data yet)
        assert (signals.iloc[:19] == 0).all()
    
    def test_insufficient_data(self):
        """Test with insufficient data"""
        strategy = MACrossStrategy(short_window=5, long_window=20)
        
        # Create small dataset
        df_small = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100, 101, 102],
            'volume': [1000, 1000, 1000]
        }, index=pd.date_range('2023-01-01', periods=3, freq='D'))
        
        with pytest.raises(ValueError, match="Insufficient data"):
            strategy.generate_signals(df_small)
    
    def test_get_ma_values(self, sample_ohlcv_data):
        """Test MA value extraction"""
        strategy = MACrossStrategy(short_window=5, long_window=20)
        short_ma, long_ma = strategy.get_ma_values(sample_ohlcv_data)
        
        assert isinstance(short_ma, pd.Series)
        assert isinstance(long_ma, pd.Series)
        assert len(short_ma) == len(sample_ohlcv_data)
        assert len(long_ma) == len(sample_ohlcv_data)
