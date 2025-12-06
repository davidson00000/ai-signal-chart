"""
Test suite for EMA9 Dip Buy strategy
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from backend.strategies.ema9_dip_buy import EMA9DipBuyStrategy

def test_strategy_instantiation():
    """Test that the strategy can be instantiated with default parameters"""
    strategy = EMA9DipBuyStrategy()
    assert strategy.ema_fast == 9
    assert strategy.ema_slow == 21
    assert strategy.deviation_threshold == 2.0
    assert strategy.stop_buffer == 0.5
    assert strategy.risk_reward == 2.0
    assert strategy.lookback_volume == 20
    print("✓ Strategy instantiation test passed")

def test_strategy_with_custom_params():
    """Test that the strategy can be instantiated with custom parameters"""
    strategy = EMA9DipBuyStrategy(
        ema_fast=12,
        ema_slow=26,
        deviation_threshold=3.0,
        stop_buffer=1.0,
        risk_reward=3.0,
        lookback_volume=30
    )
    assert strategy.ema_fast == 12
    assert strategy.ema_slow == 26
    assert strategy.deviation_threshold == 3.0
    assert strategy.stop_buffer == 1.0
    assert strategy.risk_reward == 3.0
    assert strategy.lookback_volume == 30
    print("✓ Custom parameters test passed")

def test_generate_signals():
    """Test signal generation with sample data"""
    # Create sample price data with an uptrend and pullback
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Create uptrend with pullback
    prices = np.linspace(100, 150, 100) + np.random.randn(100) * 2
    # Add a pullback around day 60-70
    prices[60:70] = prices[60:70] - 5
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    strategy = EMA9DipBuyStrategy()
    signals = strategy.generate_signals(df)
    
    # Verify signals are generated
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(df)
    assert signals.isin([-1, 0, 1]).all()
    
    # Check that we have some buy signals
    buy_signals = (signals == 1).sum()
    print(f"✓ Signal generation test passed (generated {buy_signals} buy signals)")

def test_params_schema():
    """Test that params schema is correctly defined"""
    schema = EMA9DipBuyStrategy.get_params_schema()
    
    assert "ema_fast" in schema
    assert "ema_slow" in schema
    assert "deviation_threshold" in schema
    assert "stop_buffer" in schema
    assert "risk_reward" in schema
    assert "lookback_volume" in schema
    
    # Verify parameter types
    assert schema["ema_fast"]["type"] == "int"
    assert schema["deviation_threshold"]["type"] == "float"
    assert schema["risk_reward"]["default"] == 2.0
    
    print("✓ Parameters schema test passed")

def test_optimization_config():
    """Test that optimization config is correctly defined"""
    config = EMA9DipBuyStrategy.get_optimization_config()
    
    assert "x_param" in config
    assert "y_param" in config
    assert "x_range" in config
    assert "y_range" in config
    
    assert config["x_param"] == "ema_fast"
    assert config["y_param"] == "risk_reward"
    
    print("✓ Optimization config test passed")

if __name__ == "__main__":
    print("\n=== Testing EMA9 Dip Buy Strategy ===\n")
    
    test_strategy_instantiation()
    test_strategy_with_custom_params()
    test_generate_signals()
    test_params_schema()
    test_optimization_config()
    
    print("\n✅ All tests passed!\n")
