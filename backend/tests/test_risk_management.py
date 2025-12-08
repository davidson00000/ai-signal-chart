"""
Tests for Risk Management functionality
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backend.risk_management import RiskManager


@pytest.fixture
def risk_manager():
    """Create a RiskManager instance with default settings"""
    return RiskManager(default_account_size=10000, default_risk_pct=1.0)


@pytest.fixture
def sample_ohlc_data():
    """Generate sample OHLC data for ATR calculation"""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    # Generate realistic price data
    np.random.seed(42)
    close_prices = 100 + np.cumsum(np.random.randn(30) * 2)
    
    df = pd.DataFrame({
        'open': close_prices * 0.99,
        'high': close_prices * 1.02,
        'low': close_prices * 0.98,
        'close': close_prices,
        'volume': np.random.randint(1000000, 2000000, 30)
    }, index=dates)
    
    return df


class TestPositionSizeCalculation:
    """Test position size calculations"""
    
    def test_basic_position_size(self, risk_manager):
        """Test basic position size calculation"""
        result = risk_manager.calculate_position_size(
            entry_price=100,
            stop_price=95,
            account_size=10000,
            risk_per_trade_pct=1.0
        )
        
        # risk_amount = 10000 * 0.01 = 100
        # risk_per_share = 100 - 95 = 5
        # position_size = floor(100 / 5) = 20
        
        assert result['risk_amount'] == 100.0
        assert result['risk_per_share'] == 5.0
        assert result['position_size'] == 20
        assert len(result['warnings']) == 0
    
    def test_default_parameters(self, risk_manager):
        """Test using default account size and risk percentage"""
        result = risk_manager.calculate_position_size(
            entry_price=100,
            stop_price=95
        )
        
        # Should use defaults: 10000 account, 1% risk
        assert result['account_size'] == 10000
        assert result['risk_per_trade_pct'] == 1.0
        assert result['risk_amount'] == 100.0
    
    def test_zero_risk_per_share(self, risk_manager):
        """Test when entry price equals stop price"""
        result = risk_manager.calculate_position_size(
            entry_price=100,
            stop_price=100,  # Same as entry
            account_size=10000,
            risk_per_trade_pct=1.0
        )
        
        assert result['position_size'] == 0
        assert len(result['warnings']) > 0
        assert "Entry price equals stop price" in result['warnings'][0]
    
    def test_minimum_position_size(self, risk_manager):
        """Test minimum position size enforcement"""
        result = risk_manager.calculate_position_size(
            entry_price=100,
            stop_price=99.9,  # Very tight stop -> small position
            account_size=10000,
            risk_per_trade_pct=1.0,
            min_position_size=50  # Require at least 50 shares
        )
        
        # With tight stop, calculated position will be small
        # Should be clipped to 0 if below minimum
        if result['position_size'] > 0:
            assert result['position_size'] >= 50
        else:
            assert len(result['warnings']) > 0
    
    def test_high_risk_percentage_warning(self, risk_manager):
        """Test warning for high risk percentage"""
        result = risk_manager.calculate_position_size(
            entry_price=100,
            stop_price=95,
            account_size=10000,
            risk_per_trade_pct=5.0  # 5% is high
        )
        
        warnings_text = ' '.join(result['warnings'])
        assert 'exceeds recommended' in warnings_text or len(result['warnings']) > 0
    
    def test_negative_account_size(self, risk_manager):
        """Test invalid account size"""
        result = risk_manager.calculate_position_size(
            entry_price=100,
            stop_price=95,
            account_size=-1000,
            risk_per_trade_pct=1.0
        )
        
        assert result['position_size'] == 0
        assert len(result['warnings']) > 0
    
    def test_invalid_risk_percentage(self, risk_manager):
        """Test invalid risk percentage"""
        result = risk_manager.calculate_position_size(
            entry_price=100,
            stop_price=95,
            account_size=10000,
            risk_per_trade_pct=150  # > 100%
        )
        
        assert result['position_size'] == 0
        assert len(result['warnings']) > 0


class TestATRCalculation:
    """Test ATR-based stop loss calculations"""
    
    def test_basic_atr_calculation(self, risk_manager, sample_ohlc_data):
        """Test basic ATR calculation for LONG position"""
        result = risk_manager.suggest_stop_price_from_atr(
            df=sample_ohlc_data,
            atr_period=14,
            atr_multiplier=2.0,
            direction="LONG"
        )
        
        assert result['suggested_stop'] is not None
        assert result['atr_value'] is not None
        assert result['current_price'] is not None
        assert result['suggested_stop'] < result['current_price']  # Stop below entry for LONG
    
    def test_atr_short_position(self, risk_manager, sample_ohlc_data):
        """Test ATR calculation for SHORT position"""
        result = risk_manager.suggest_stop_price_from_atr(
            df=sample_ohlc_data,
            direction="SHORT"
        )
        
        if result['suggested_stop'] is not None:
            # Stop above entry for SHORT
            assert result['suggested_stop'] > result['current_price']
    
    def test_atr_insufficient_data(self, risk_manager):
        """Test ATR with insufficient data"""
        # Create very short DataFrame
        short_df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102]
        })
        
        result = risk_manager.suggest_stop_price_from_atr(
            df=short_df,
            atr_period=14
        )
        
        assert result['suggested_stop'] is None
        assert len(result['warnings']) > 0
        assert 'Insufficient data' in result['warnings'][0]
    
    def test_atr_missing_columns(self, risk_manager):
        """Test ATR with missing required columns"""
        bad_df = pd.DataFrame({
            'close': [100, 101, 102]
            # Missing 'high' and 'low'
        })
        
        result = risk_manager.suggest_stop_price_from_atr(
            df=bad_df
        )
        
        assert result['suggested_stop'] is None
        assert len(result['warnings']) > 0
    
    def test_atr_invalid_direction(self, risk_manager, sample_ohlc_data):
        """Test ATR with invalid direction"""
        result = risk_manager.suggest_stop_price_from_atr(
            df=sample_ohlc_data,
            direction="INVALID"
        )
        
        assert result['suggested_stop'] is None
        assert len(result['warnings']) > 0
        assert 'Invalid direction' in result['warnings'][0]
    
    def test_atr_custom_price(self, risk_manager, sample_ohlc_data):
        """Test ATR with custom current price"""
        custom_price = 150.0
        
        result = risk_manager.suggest_stop_price_from_atr(
            df=sample_ohlc_data,
            current_price=custom_price,
            direction="LONG"
        )
        
        assert result['current_price'] == custom_price


class TestRiskRewardRatio:
    """Test risk/reward ratio calculations"""
    
    def test_basic_risk_reward(self, risk_manager):
        """Test basic risk/reward calculation"""
        result = risk_manager.calculate_risk_reward_ratio(
            entry_price=100,
            stop_price=95,  # 5 points risk
            target_price=115  # 15 points reward
        )
        
        assert result['risk'] == 5.0
        assert result['reward'] == 15.0
        assert result['ratio'] == 3.0  # 15/5 = 3:1
    
    def test_poor_risk_reward(self, risk_manager):
        """Test poor risk/reward ratio (< 1:1)"""
        result = risk_manager.calculate_risk_reward_ratio(
            entry_price=100,
            stop_price=95,  # 5 points risk
            target_price=102  # Only 2 points reward
        )
        
        assert result['ratio'] < 1.0
        assert len(result['warnings']) > 0
    
    def test_zero_risk(self, risk_manager):
        """Test with zero risk (invalid)"""
        result = risk_manager.calculate_risk_reward_ratio(
            entry_price=100,
            stop_price=100,  # No risk
            target_price=110
        )
        
        assert result['ratio'] is None
        assert len(result['warnings']) > 0


class TestIntegration:
    """Integration tests combining multiple components"""
    
    def test_full_risk_workflow(self, risk_manager, sample_ohlc_data):
        """Test complete risk management workflow"""
        # 1. Get suggested stop from ATR
        atr_result = risk_manager.suggest_stop_price_from_atr(
            df=sample_ohlc_data,
            direction="LONG"
        )
        
        assert atr_result['suggested_stop'] is not None
        
        # 2. Calculate position size with ATR stop
        position_result = risk_manager.calculate_position_size(
            entry_price=atr_result['current_price'],
            stop_price=atr_result['suggested_stop'],
            account_size=10000,
            risk_per_trade_pct=1.0
        )
        
        assert position_result['position_size'] > 0
        assert position_result['risk_amount'] <= 100  # 1% of 10000
        
        # 3. Verify calculations
        expected_risk = position_result['position_size'] * position_result['risk_per_share']
        assert abs(expected_risk - position_result['risk_amount']) < position_result['risk_per_share']
