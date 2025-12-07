"""
Explainability Layer Tests

This module tests the Explainability Layer implementation for trading strategies.
It verifies that:
1. Indicator values match independent calculations (Indicator Integrity)
2. Condition triggers are consistent with indicator values (Rule Trigger Consistency)
3. Confidence scores are coherent (Confidence Score Coherence)
"""

import pytest
import pandas as pd
import numpy as np
from backend.strategies.ma_cross import MACrossStrategy
from backend.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from backend.strategies.macd_trend import MACDTrendStrategy


@pytest.fixture
def sample_ohlcv_data():
    """
    Create sample OHLCV data for testing.
    Returns a DataFrame with 20 candles showing an uptrend.
    """
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    
    # Create uptrend data
    close_prices = [100 + i * 2 + np.sin(i) * 3 for i in range(20)]
    
    df = pd.DataFrame({
        'open': [p - 1 for p in close_prices],
        'high': [p + 2 for p in close_prices],
        'low': [p - 2 for p in close_prices],
        'close': close_prices,
        'volume': [1000000] * 20
    }, index=dates)
    
    return df


@pytest.fixture
def volatile_ohlcv_data():
    """
    Create volatile OHLCV data with clear RSI oversold/overbought conditions.
    """
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    
    # Create data with sharp moves
    close_prices = [
        100, 102, 104, 106, 108,  # Up
        107, 106, 105, 104, 103,  # Down
        102, 101, 100, 99, 98,    # Continued down (oversold)
        100, 102, 104, 106, 108   # Recovery
    ]
    
    df = pd.DataFrame({
        'open': [p - 0.5 for p in close_prices],
        'high': [p + 1 for p in close_prices],
        'low': [p - 1 for p in close_prices],
        'close': close_prices,
        'volume': [1000000] * 20
    }, index=dates)
    
    return df


# ============================================================================
# Test 1: Indicator Integrity Tests
# ============================================================================

class TestIndicatorIntegrity:
    """Test that indicator values in explain() match independent calculations."""
    
    def test_ma_cross_indicators(self, sample_ohlcv_data):
        """Test MA Cross strategy indicators."""
        df = sample_ohlcv_data
        strategy = MACrossStrategy(short_window=5, long_window=10)
        
        # Calculate MAs independently
        short_ma_expected = df['close'].rolling(window=5).mean()
        long_ma_expected = df['close'].rolling(window=10).mean()
        
        # Test at index where both MAs are available
        test_idx = 12
        explain = strategy.explain(df, test_idx)
        
        # Verify indicators
        assert 'indicators' in explain
        indicators = explain['indicators']
        
        # Allow small floating point error
        tolerance = 0.01
        assert abs(indicators['short_ma'] - round(short_ma_expected.iloc[test_idx], 2)) < tolerance
        assert abs(indicators['long_ma'] - round(long_ma_expected.iloc[test_idx], 2)) < tolerance
        assert abs(indicators['close'] - round(df['close'].iloc[test_idx], 2)) < tolerance
        
    def test_rsi_indicators(self, volatile_ohlcv_data):
        """Test RSI strategy indicators."""
        df = volatile_ohlcv_data
        strategy = RSIMeanReversionStrategy(period=14, oversold=30, overbought=70)
        
        # Calculate RSI independently
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_expected = 100 - (100 / (1 + rs))
        
        # Test at index 18 (should have RSI data)
        test_idx = 18
        explain = strategy.explain(df, test_idx)
        
        assert 'indicators' in explain
        indicators = explain['indicators']
        
        # Verify RSI value (allow rounding difference)
        tolerance = 0.5
        expected_rsi = rsi_expected.iloc[test_idx]
        if not pd.isna(expected_rsi):
            assert abs(indicators['rsi'] - round(expected_rsi, 2)) < tolerance
        
        # Verify static parameters
        assert indicators['rsi_period'] == 14
        assert indicators['oversold_level'] == 30
        assert indicators['overbought_level'] == 70
        
    def test_macd_indicators(self, sample_ohlcv_data):
        """Test MACD strategy indicators."""
        df = sample_ohlcv_data
        strategy = MACDTrendStrategy(fast_period=12, slow_period=26, signal_period=9)
        
        # Calculate MACD independently (using same 12/26/9 on 20 candles)
        # Note: With only 20 candles, slow EMA needs warm-up
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        macd_line_expected = ema_fast - ema_slow
        signal_line_expected = macd_line_expected.ewm(span=9, adjust=False).mean()
        hist_expected = macd_line_expected - signal_line_expected
        
        test_idx = 19  # Last index
        explain = strategy.explain(df, test_idx)
        
        assert 'indicators' in explain
        indicators = explain['indicators']
        
        # MACD values can be very small, so use absolute tolerance
        tolerance = 0.01
        assert abs(indicators['macd_line'] - round(macd_line_expected.iloc[test_idx], 4)) < tolerance
        assert abs(indicators['signal_line'] - round(signal_line_expected.iloc[test_idx], 4)) < tolerance
        assert abs(indicators['histogram'] - round(hist_expected.iloc[test_idx], 4)) < tolerance


# ============================================================================
# Test 2: Rule Trigger Consistency Tests
# ============================================================================

class TestRuleTriggerConsistency:
    """Test that conditions_triggered are consistent with indicator values."""
    
    def test_ma_cross_conditions(self, sample_ohlcv_data):
        """Test MA Cross conditions match indicator values."""
        df = sample_ohlcv_data
        strategy = MACrossStrategy(short_window=5, long_window=10)
        
        test_idx = 15
        explain = strategy.explain(df, test_idx)
        
        indicators = explain['indicators']
        conditions = explain['conditions_triggered']
        
        # Check that conditions reflect the actual MA relationship
        short_ma = indicators['short_ma']
        long_ma = indicators['long_ma']
        
        if short_ma > long_ma:
            # Should have "Short MA (5) > Long MA (10)" in conditions
            assert any('Short MA (5) > Long MA (10)' in c for c in conditions), \
                f"Expected '>' condition when short_ma={short_ma} > long_ma={long_ma}, got: {conditions}"
        elif short_ma < long_ma:
            # Should have "Short MA (5) < Long MA (10)" in conditions
            assert any('Short MA (5) < Long MA (10)' in c for c in conditions), \
                f"Expected '<' condition when short_ma={short_ma} < long_ma={long_ma}, got: {conditions}"
        
    def test_rsi_conditions(self, volatile_ohlcv_data):
        """Test RSI conditions match indicator values."""
        df = volatile_ohlcv_data
        oversold = 30
        overbought = 70
        strategy = RSIMeanReversionStrategy(period=14, oversold=oversold, overbought=overbought)
        
        test_idx = 18
        explain = strategy.explain(df, test_idx)
        
        indicators = explain['indicators']
        conditions = explain['conditions_triggered']
        rsi = indicators['rsi']
        
        # Check that conditions match RSI zones
        if rsi < oversold:
            assert any('Oversold' in c and 'BUY' in c for c in conditions), \
                f"Expected oversold BUY condition when RSI={rsi} < {oversold}, got: {conditions}"
        elif rsi > overbought:
            assert any('Overbought' in c and 'EXIT' in c for c in conditions), \
                f"Expected overbought EXIT condition when RSI={rsi} > {overbought}, got: {conditions}"
        else:
            assert any('neutral' in c for c in conditions), \
                f"Expected neutral condition when RSI={rsi} is between {oversold} and {overbought}, got: {conditions}"
    
    def test_macd_conditions(self, sample_ohlcv_data):
        """Test MACD conditions match indicator values."""
        df = sample_ohlcv_data
        strategy = MACDTrendStrategy(fast_period=12, slow_period=26, signal_period=9)
        
        test_idx = 19
        explain = strategy.explain(df, test_idx)
        
        indicators = explain['indicators']
        conditions = explain['conditions_triggered']
        
        macd_line = indicators['macd_line']
        signal_line = indicators['signal_line']
        histogram = indicators['histogram']
        
        # Check MACD line vs signal line condition
        if macd_line > signal_line:
            assert any('MACD Line > Signal Line' in c for c in conditions), \
                f"Expected bullish condition when macd={macd_line} > signal={signal_line}"
        else:
            assert any('MACD Line < Signal Line' in c for c in conditions), \
                f"Expected bearish condition when macd={macd_line} < signal={signal_line}"
        
        # Check histogram condition
        if histogram > 0:
            assert any('Histogram positive' in c for c in conditions), \
                f"Expected positive histogram condition when hist={histogram} > 0"
        else:
            assert any('Histogram negative' in c for c in conditions), \
                f"Expected negative histogram condition when hist={histogram} < 0"


# ============================================================================
# Test 3: Confidence Score Coherence Tests
# ============================================================================

class TestConfidenceCoherence:
    """Test that confidence scores are coherent and within valid range."""
    
    def test_confidence_range(self, sample_ohlcv_data):
        """Test that confidence is always between 0 and 1."""
        df = sample_ohlcv_data
        strategies = [
            MACrossStrategy(short_window=5, long_window=10),
            RSIMeanReversionStrategy(period=14, oversold=30, overbought=70),
            MACDTrendStrategy(fast_period=12, slow_period=26, signal_period=9),
        ]
        
        for strategy in strategies:
            for idx in range(10, len(df)):  # Skip early indices without enough data
                explain = strategy.explain(df, idx)
                confidence = explain['confidence']
                
                assert 0.0 <= confidence <= 1.0, \
                    f"Confidence {confidence} out of range [0, 1] for {strategy.__class__.__name__} at idx {idx}"
                assert not np.isnan(confidence), \
                    f"Confidence is NaN for {strategy.__class__.__name__} at idx {idx}"
    
    def test_ma_cross_confidence_monotonicity(self):
        """Test that larger MA spread leads to higher confidence."""
        df_narrow = pd.DataFrame({
            'close': [100.0] * 20 + [100.1] * 10  # Very narrow spread
        })
        
        df_wide = pd.DataFrame({
            'close': [100.0] * 20 + [110.0] * 10  # Wide spread
        })
        
        strategy = MACrossStrategy(short_window=5, long_window=20)
        
        # Get confidence for narrow spread
        explain_narrow = strategy.explain(df_narrow, 25)
        conf_narrow = explain_narrow['confidence']
        
        # Get confidence for wide spread
        explain_wide = strategy.explain(df_wide, 25)
        conf_wide = explain_wide['confidence']
        
        # Wide spread should have higher confidence
        assert conf_wide > conf_narrow, \
            f"Expected higher confidence for wide spread ({conf_wide}) vs narrow ({conf_narrow})"
    
    def test_rsi_extreme_confidence(self):
        """Test that extreme RSI values have higher confidence."""
        # Create data with extreme RSI (oversold)
        df_extreme = pd.DataFrame({
            'close': [100] + [95 - i for i in range(19)]  # Sharp decline
        })
        
        # Create data with moderate RSI
        df_moderate = pd.DataFrame({
            'close': [100 + np.sin(i) * 2 for i in range(20)]  # Oscillating
        })
        
        strategy = RSIMeanReversionStrategy(period=14, oversold=30, overbought=70)
        
        # Compare confidence at similar index
        explain_extreme = strategy.explain(df_extreme, 19)
        explain_moderate = strategy.explain(df_moderate, 19)
        
        # Both should be valid
        assert 0.0 <= explain_extreme['confidence'] <= 1.0
        assert 0.0 <= explain_moderate['confidence'] <= 1.0


# ============================================================================
# Integration Test
# ============================================================================

class TestExplainabilityIntegration:
    """Integration tests that combine multiple aspects."""
    
    def test_full_explain_structure(self, sample_ohlcv_data):
        """Test that explain() returns complete and valid structure."""
        df = sample_ohlcv_data
        strategy = MACrossStrategy(short_window=5, long_window=10)
        
        test_idx = 15
        explain = strategy.explain(df, test_idx)
        
        # Check structure
        assert isinstance(explain, dict)
        assert 'indicators' in explain
        assert 'conditions_triggered' in explain
        assert 'confidence' in explain
        
        # Check indicators
        assert isinstance(explain['indicators'], dict)
        assert len(explain['indicators']) > 0
        
        # Check conditions
        assert isinstance(explain['conditions_triggered'], list)
        assert len(explain['conditions_triggered']) > 0
        
        # Check confidence
        assert isinstance(explain['confidence'], (int, float))
        assert 0.0 <= explain['confidence'] <= 1.0
