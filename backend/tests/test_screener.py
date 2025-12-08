"""
Tests for Symbol Screener functionality
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.signals.screener import SymbolScreener, SymbolScoreResult, screen_symbols


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    # Generate realistic price data with trend
    base_price = 100
    prices = []
    for i in range(100):
        # Add trend + noise
        trend = i * 0.5
        noise = np.random.randn() * 2
        price = base_price + trend + noise
        prices.append(max(price, 1))  # Ensure positive
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': [1000000 + np.random.randint(-100000, 100000) for _ in range(100)]
    }, index=dates)
    
    return df


class TestSymbolScreener:
    """Test cases for SymbolScreener"""
    
    def test_score_symbol_basic(self, sample_ohlcv_data):
        """Test basic symbol scoring functionality"""
        screener = SymbolScreener()
        result = screener.score_symbol(sample_ohlcv_data, "TEST")
        
        assert result is not None
        assert result.symbol == "TEST"
        assert 0 <= result.score <= 100
        assert isinstance(result.factors, dict)
        assert isinstance(result.as_of, datetime)
    
    def test_score_components_in_range(self, sample_ohlcv_data):
        """Test that all score components are in valid range"""
        screener = SymbolScreener()
        result = screener.score_symbol(sample_ohlcv_data, "TEST")
        
        assert result is not None
        
        # Check all factors are in 0-100 range
        for factor_name, factor_value in result.factors.items():
            assert 0 <= factor_value <= 100, f"{factor_name} out of range: {factor_value}"
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        screener = SymbolScreener()
        
        # Create small dataset (< 60 bars)
        small_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000000, 1100000, 1050000]
        })
        
        result = screener.score_symbol(small_df, "TEST")
        assert result is None  # Should return None for insufficient data
    
    def test_trend_score_calculation(self, sample_ohlcv_data):
        """Test trend score calculation logic"""
        screener = SymbolScreener()
        trend_score = screener._calculate_trend_score(sample_ohlcv_data)
        
        assert 0 <= trend_score <= 100
        assert isinstance(trend_score, float)
    
    def test_volatility_score_calculation(self, sample_ohlcv_data):
        """Test volatility score calculation logic"""
        screener = SymbolScreener()
        volatility_score = screener._calculate_volatility_score(sample_ohlcv_data)
        
        assert 0 <= volatility_score <= 100
        assert isinstance(volatility_score, float)
    
    def test_momentum_score_calculation(self, sample_ohlcv_data):
        """Test momentum score calculation logic"""
        screener = SymbolScreener()
        momentum_score = screener._calculate_momentum_score(sample_ohlcv_data)
        
        assert 0 <= momentum_score <= 100
        assert isinstance(momentum_score, float)
    
    def test_oversold_score_calculation(self, sample_ohlcv_data):
        """Test oversold score calculation logic"""
        screener = SymbolScreener()
        oversold_score = screener._calculate_oversold_score(sample_ohlcv_data)
        
        assert 0 <= oversold_score <= 100
        assert isinstance(oversold_score, (int, float))
    
    def test_volume_spike_score_calculation(self, sample_ohlcv_data):
        """Test volume spike score calculation logic"""
        screener = SymbolScreener()
        volume_spike_score = screener._calculate_volume_spike_score(sample_ohlcv_data)
        
        assert 0 <= volume_spike_score <= 100
        assert isinstance(volume_spike_score, float)
    
    def test_to_dict_serialization(self, sample_ohlcv_data):
        """Test that result can be serialized to dict"""
        screener = SymbolScreener()
        result = screener.score_symbol(sample_ohlcv_data, "TEST")
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "symbol" in result_dict
        assert "score" in result_dict
        assert "factors" in result_dict
        assert result_dict["symbol"] == "TEST"
        
        # Check that scores are rounded
        assert isinstance(result_dict["score"], (int, float))
        for factor_value in result_dict["factors"].values():
            assert isinstance(factor_value, (int, float))


class TestScreenSymbols:
    """Test cases for screen_symbols function"""
    
    def test_screen_multiple_symbols(self, sample_ohlcv_data):
        """Test screening multiple symbols"""
        
        def mock_data_fetcher(symbol, timeframe):
            # Return sample data for all symbols
            return sample_ohlcv_data
        
        symbols = ["AAPL", "MSFT", "GOOGL"]
        results = screen_symbols(
            symbols=symbols,
            data_fetcher=mock_data_fetcher,
            timeframe="1d",
            limit=10
        )
        
        assert isinstance(results, list)
        assert len(results) <= 3
        assert len(results) <= 10
        
        # Check structure
        if results:
            assert "symbol" in results[0]
            assert "score" in results[0]
            assert "factors" in results[0]
    
    def test_screen_with_limit(self, sample_ohlcv_data):
        """Test that limit parameter works correctly"""
        
        def mock_data_fetcher(symbol, timeframe):
            return sample_ohlcv_data
        
        symbols = ["SYM1", "SYM2", "SYM3", "SYM4", "SYM5"]
        results = screen_symbols(
            symbols=symbols,
            data_fetcher=mock_data_fetcher,
            timeframe="1d",
            limit=3
        )
        
        assert len(results) <= 3
    
    def test_screen_handles_errors(self):
        """Test that screening handles errors gracefully"""
        
        def failing_data_fetcher(symbol, timeframe):
            if symbol == "BAD":
                raise Exception("Data fetch failed")
            return None
        
        symbols = ["GOOD", "BAD", "GOOD"]
        results = screen_symbols(
            symbols=symbols,
            data_fetcher=failing_data_fetcher,
            timeframe="1d",
            limit=10
        )
        
        # Should not crash, should return empty or partial results
        assert isinstance(results, list)
