import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from backend.live.signal_generator import generate_live_signal

# Mock data
@pytest.fixture
def mock_data_feed():
    with patch("backend.live.signal_generator.data_feed") as mock:
        # Mock candle data
        mock.get_chart_data.return_value = [
            {"time": 1000, "open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000}
        ] * 50
        yield mock

@pytest.fixture
def mock_predictors():
    with patch("backend.live.signal_generator.stat") as stat, \
         patch("backend.live.signal_generator.rule") as rule, \
         patch("backend.live.signal_generator.ml") as ml:
        yield stat, rule, ml

def test_all_up_consensus(mock_data_feed, mock_predictors):
    stat, rule, ml = mock_predictors
    
    # Setup: All Predictors UP
    stat.predict.return_value = {"direction": "up", "score": 0.8}
    rule.predict.return_value = {"direction": "up", "score": 0.9}
    ml.predict.return_value = {"direction": "up", "score": 0.7}
    
    result = generate_live_signal("TEST", "1d", 50)
    final = result["final_signal"]
    
    assert final["action"] == "buy"
    assert "Strong bullish consensus" in final["reason"]
    # Confidence should be average of scores (no penalty)
    expected_conf = (0.8 + 0.9 + 0.7) / 3
    assert final["confidence"] == round(expected_conf, 2)

def test_all_down_consensus(mock_data_feed, mock_predictors):
    stat, rule, ml = mock_predictors
    
    # Setup: All Predictors DOWN
    stat.predict.return_value = {"direction": "down", "score": 0.8}
    rule.predict.return_value = {"direction": "down", "score": 0.9}
    ml.predict.return_value = {"direction": "down", "score": 0.7}
    
    result = generate_live_signal("TEST", "1d", 50)
    final = result["final_signal"]
    
    assert final["action"] == "sell"
    assert "Strong bearish consensus" in final["reason"]
    # Confidence should be average of scores (no penalty)
    expected_conf = (0.8 + 0.9 + 0.7) / 3
    assert final["confidence"] == round(expected_conf, 2)

def test_mixed_signals_confusion(mock_data_feed, mock_predictors):
    stat, rule, ml = mock_predictors
    
    # Setup: 1 UP, 1 DOWN, 1 FLAT (Mixed)
    stat.predict.return_value = {"direction": "up", "score": 0.8}
    rule.predict.return_value = {"direction": "down", "score": 0.8}
    ml.predict.return_value = {"direction": "flat", "score": 0.5}
    
    result = generate_live_signal("TEST", "1d", 50)
    final = result["final_signal"]
    
    assert final["action"] == "hold"
    assert "Predictors disagree" in final["reason"]
    
    # Confidence should be penalized (0.70)
    base_conf = (0.8 + 0.8 + 0.5) / 3
    expected_conf = base_conf * 0.70
    assert final["confidence"] == round(expected_conf, 2)

def test_bullish_bias_with_disagreement(mock_data_feed, mock_predictors):
    stat, rule, ml = mock_predictors
    
    # Setup: 2 UP, 1 DOWN (Avoid Rule=DOWN to prevent specific override)
    stat.predict.return_value = {"direction": "up", "score": 0.8}
    rule.predict.return_value = {"direction": "up", "score": 0.6} 
    ml.predict.return_value = {"direction": "down", "score": 0.7} # Disagreement here
    
    result = generate_live_signal("TEST", "1d", 50)
    final = result["final_signal"]
    
    assert final["action"] == "buy"
    assert "Bullish bias" in final["reason"]
    
    # Confidence should be penalized (0.85)
    base_conf = (0.8 + 0.6 + 0.7) / 3
    expected_conf = base_conf * 0.85
    # Use approx for floating point comparison
    assert final["confidence"] == pytest.approx(expected_conf, abs=0.01)

def test_specific_reason_logic(mock_data_feed, mock_predictors):
    stat, rule, ml = mock_predictors
    
    # Setup: Stat+ML UP, Rule DOWN (Trend Reversal pattern)
    stat.predict.return_value = {"direction": "up", "score": 0.8}
    rule.predict.return_value = {"direction": "down", "score": 0.6}
    ml.predict.return_value = {"direction": "up", "score": 0.7}
    
    result = generate_live_signal("TEST", "1d", 50)
    final = result["final_signal"]
    
    assert final["action"] == "buy"
    # Should trigger the specific reason override
    assert "Stat+ML are bullish, but Rule is bearish" in final["reason"]
