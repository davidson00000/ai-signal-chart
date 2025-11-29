# Backtest Simulation Specification

## Overview
The backtest simulation system allows testing trading strategies on historical data to evaluate performance before live trading.

---

## Backtest Engine

### Core Logic

**Portfolio Simulation:**
1. Start with initial capital (default: ¥1,000,000)
2. For each trading day:
   - Check strategy signal (BUY=1, HOLD=0, SELL=-1)
   - Execute trades at close price
   - Apply commission (default: 0.05%)
   - Update portfolio (cash, position, equity)
3. Track equity curve and trade history

**Position Sizing:**
- Full capital deployment (position_size=1.0)
- Buy maximum shares with available cash
- Commission deducted from proceeds

**Commission Calculation:**
```
Buy: total_cost = shares * price + (shares * price * commission)
Sell: net_proceeds = shares * price - (shares * price * commission)
```

---

## Performance Metrics

### Total P&L
```
total_pnl = final_equity - initial_capital
```

### Return Percentage
```
return_pct = (total_pnl / initial_capital) * 100
```

### Maximum Drawdown
```
drawdown = (equity - peak_equity) / peak_equity * 100
max_drawdown = min(drawdown_series)
```

### Win Rate
```
win_rate = (winning_trades / total_trades) * 100
```

---

## Strategy Interface

### BaseStrategy

All strategies must inherit from `BaseStrategy`:

```python
from backend.strategies.base import BaseStrategy
import pandas as pd

class MyStrategy(BaseStrategy):
    def __init__(self, **params):
        super().__init__(**params)
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate position signals
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
                Index must be DatetimeIndex
        
        Returns:
            pd.Series with values:
                1: Long position (BUY)
                0: Flat (HOLD/EXIT)
               -1: Short position (not implemented for stocks)
        """
        # Strategy logic here
        position = pd.Series(0, index=df.index)
        # ... calculate signals ...
        return position
```

### Required Methods

- `generate_signals(df)`: Returns position series
- `validate_dataframe(df)`: Validates input data
- `get_params()`: Returns strategy parameters

---

## Built-in Strategies

### MA Cross Strategy

**Parameters:**
- `short_window`: Short MA period (default: 9)
- `long_window`: Long MA period (default: 21)

**Signal Logic:**
- BUY (1): short_ma > long_ma
- HOLD (0): short_ma <= long_ma

**Usage:**
```python
from backend.strategies.ma_cross import MACrossStrategy

strategy = MACrossStrategy(short_window=5, long_window=20)
```

---

## API Specification

### POST /simulate

**Request:**
```json
{
  "symbol": "AAPL",
  "timeframe": "1d",
  "start_date": "2020-01-01",
  "end_date": "2023-12-31",
  "strategy": "ma_cross",
  "initial_capital": 1000000,
  "commission": 0.0005,
  "position_size": 1.0,
  "short_window": 9,
  "long_window": 21
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "timeframe": "1d",
  "strategy": "MACrossStrategy(short_window=9, long_window=21)",
  "metrics": {
    "initial_capital": 1000000,
    "final_equity": 1150000,
    "total_pnl": 150000,
    "return_pct": 15.0,
    "max_drawdown": -5.2,
    "win_rate": 65.5,
    "trade_count": 42,
    "winning_trades": 28,
    "losing_trades": 14
  },
  "trades": [
    {
      "date": "2020-03-15T00:00:00",
      "side": "BUY",
      "price": 250.50,
      "quantity": 3990,
      "commission": 500.25,
      "cash_after": 1500.75
    },
    ...
  ],
  "equity_curve": [
    {
      "date": "2020-01-01T00:00:00",
      "equity": 1000000,
      "cash": 1000000,
      "position_value": 0
    },
    ...
  ],
  "data_points": 1000
}
```

---

## Usage Examples

### cURL
```bash
curl -X POST "http://localhost:8000/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "timeframe": "1d",
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "strategy": "ma_cross",
    "short_window": 5,
    "long_window": 20
  }'
```

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/simulate",
    json={
        "symbol": "AAPL",
        "timeframe": "1d",
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "strategy": "ma_cross",
        "short_window": 5,
        "long_window": 20
    }
)

results = response.json()
print(f"Return: {results['metrics']['return_pct']:.2f}%")
print(f"Max DD: {results['metrics']['max_drawdown']:.2f}%")
```

---

## Testing

### Run Tests
```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_backtester.py

# Run with verbose output
pytest tests/ -v
```

### Test Coverage
- Strategy interface validation
- MA cross signal generation
- Backtest engine execution
- Trade execution logic
- Metrics calculation
- Commission calculation

---

## Limitations

1. **No Short Selling**: Current implementation only supports long positions
2. **Full Position**: Always uses 100% of capital (configurable with position_size)
3. ** No Slippage**: Assumes perfect execution at close price
4. **No Market Impact**: Assumes unlimited liquidity
5. **Commission Only**: Does not include other fees (stamp duty, etc.)

---

## Future Enhancements

- Multi-strategy support
- Parameter optimization
- Walk-forward analysis
- Monte Carlo simulation
- Short selling support
- Position sizing strategies
- Risk management rules
