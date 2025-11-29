"""
Tests for backtest engine
"""
import pytest
import pandas as pd
import numpy as np
from backend.backtester import BacktestEngine
from backend.strategies.ma_cross import MACrossStrategy


@pytest.fixture
def trending_up_data():
    """Create uptrending price data"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    close_prices = 100 + np.arange(100) * 0.5  # Steady uptrend
    
    df = pd.DataFrame({
        'open': close_prices - 0.2,
        'high': close_prices + 0.5,
        'low': close_prices - 0.5,
        'close': close_prices,
        'volume': [1000000] * 100
    }, index=dates)
    
    return df


@pytest.fixture
def ranging_data():
    """Create ranging (sideways) price data"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    close_prices = 100 + np.sin(np.arange(100) * 0.2) * 5  # Oscillating
    
    df = pd.DataFrame({
        'open': close_prices - 0.2,
        'high': close_prices + 0.5,
        'low': close_prices - 0.5,
        'close': close_prices,
        'volume': [1000000] * 100
    }, index=dates)
    
    return df


class TestBacktestEngine:
    """Test BacktestEngine"""
    
    def test_initialization(self):
        """Test engine initialization"""
        engine = BacktestEngine(
            initial_capital=1000000,
            commission=0.0005,
            position_size=1.0
        )
        
        assert engine.initial_capital == 1000000
        assert engine.commission == 0.0005
        assert engine.position_size == 1.0
    
    def test_run_uptrend(self, trending_up_data):
        """Test backtest on uptrending data"""
        strategy = MACrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(initial_capital=1000000)
        
        results = engine.run(trending_up_data, strategy)
        
        # Check structure
        assert 'metrics' in results
        assert 'trades' in results
        assert 'equity_curve' in results
        
        # Check metrics
        metrics = results['metrics']
        assert metrics['initial_capital'] == 1000000
        assert metrics['final_equity'] > 0
        
        # Uptrend should be profitable
        assert metrics['total_pnl'] > 0
        assert metrics['return_pct'] > 0
        
        # Should have some trades
        assert metrics['trade_count'] > 0
    
    def test_run_ranging(self, ranging_data):
        """Test backtest on ranging data"""
        strategy = MACrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(initial_capital=1000000)
        
        results = engine.run(ranging_data, strategy)
        
        # Should complete without error
        assert results is not None
        
        # May be profitable or not in ranging market
        metrics = results['metrics']
        assert 'total_pnl' in metrics
        assert 'return_pct' in metrics
    
    def test_trade_execution(self, trending_up_data):
        """Test trade execution logic"""
        strategy = MACrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(initial_capital=1000000)
        
        results = engine.run(trending_up_data, strategy)
        trades = results['trades']
        
        # Should have both BUY and SELL trades
        buy_trades = [t for t in trades if t['side'] == 'BUY']
        sell_trades = [t for t in trades if t['side'] == 'SELL']
        
        assert len(buy_trades) > 0
        assert len(sell_trades) > 0
        
        # Each trade should have required fields
        for trade in trades:
            assert 'date' in trade
            assert 'side' in trade
            assert 'price' in trade
            assert 'quantity' in trade
            assert 'commission' in trade
    
    def test_equity_curve(self, trending_up_data):
        """Test equity curve generation"""
        strategy = MACrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(initial_capital=1000000)
        
        results = engine.run(trending_up_data, strategy)
        equity_curve = results['equity_curve']
        
        # Should have equity point for each day
        assert len(equity_curve) == len(trending_up_data)
        
        # Each point should have required fields
        for point in equity_curve:
            assert 'date' in point
            assert 'equity' in point
            assert 'cash' in point
            assert 'position_value' in point
        
        # First equity should equal initial capital
        assert equity_curve[0]['equity'] == 1000000
    
    def test_commission_calculation(self, trending_up_data):
        """Test commission is calculated correctly"""
        strategy = MACrossStrategy(short_window=5, long_window=20)
        
        # Test with no commission
        engine_no_comm = BacktestEngine(initial_capital=1000000, commission=0.0)
        results_no_comm = engine_no_comm.run(trending_up_data, strategy)
        
        # Test with commission
        engine_with_comm = BacktestEngine(initial_capital=1000000, commission=0.001)
        results_with_comm = engine_with_comm.run(trending_up_data, strategy)
        
        # With commission should have lower final equity
        assert results_with_comm['metrics']['final_equity'] < results_no_comm['metrics']['final_equity']
    
    def test_max_drawdown_calculation(self, trending_up_data):
        """Test maximum drawdown calculation"""
        strategy = MACrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(initial_capital=1000000)
        
        results = engine.run(trending_up_data, strategy)
        max_dd = results['metrics']['max_drawdown']
        
        # Max DD should be negative or zero
        assert max_dd <= 0
    
    def test_win_rate_calculation(self, trending_up_data):
        """Test win rate calculation"""
        strategy = MACrossStrategy(short_window=5, long_window=20)
        engine = BacktestEngine(initial_capital=1000000)
        
        results = engine.run(trending_up_data, strategy)
        
        metrics = results['metrics']
        win_rate = metrics['win_rate']
        
        # Win rate should be between 0 and 100
        assert 0 <= win_rate <= 100
        
        # Check winning/losing trades add up
        assert metrics['winning_trades'] + metrics['losing_trades'] == metrics['trade_count']
