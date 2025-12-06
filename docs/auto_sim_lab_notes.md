# Auto Sim Lab Notes

## Overview

Auto Sim Lab is an automated paper trading simulation engine that supports multiple strategy modes and position sizing options.

## Strategy Modes

### 1. Final Signal (Default)
Uses the same signal generation logic as Live Signal, combining:
- **Stat Predictor**: Statistical analysis
- **Rule Predictor V2**: Technical rule-based signals
- **ML Predictor**: Machine learning predictions

Signals are aggregated by majority voting:
- 3/3 up → Strong BUY
- 2/3 up → BUY
- 3/3 down → Strong SELL
- 2/3 down → SELL
- Otherwise → HOLD

### 2. MA Crossover (Match Strategy Lab)
Uses Moving Average Crossover strategy, identical to Strategy Lab's `MACrossStrategy`.

**Signal Logic:**
- **Golden Cross** (Short MA crosses ABOVE Long MA) → BUY
- **Death Cross** (Short MA crosses BELOW Long MA) → SELL
- No crossover → HOLD

**Parameters:**
- `ma_short_window`: Short MA period (e.g., 10, 50)
- `ma_long_window`: Long MA period (e.g., 30, 60)

**Important:** The MA crossover detection uses crossover events (signal change from previous bar), not just the MA relationship. This matches Strategy Lab's behavior.

## Position Sizing Modes

| Mode | Description | Parameters |
|------|-------------|------------|
| `percent_of_equity` | Risk X% of equity per trade | `risk_per_trade` (decimal) |
| `full_equity` | Use 100% of equity per trade | None |
| `fixed_shares` | Fixed number of shares | `fixed_shares` (int) |
| `fixed_dollar` | Fixed dollar amount | `fixed_dollar_amount` (float) |

### Position Size Calculation

```python
if mode == "percent_of_equity":
    size = int((equity * risk_per_trade) / price)
elif mode == "full_equity":
    size = int(equity / price)
elif mode == "fixed_shares":
    size = fixed_shares
elif mode == "fixed_dollar":
    size = int(fixed_dollar_amount / price)
```

## API Usage

### Historical Simulation
```bash
POST /auto-simulate
{
  "symbol": "AAPL",
  "timeframe": "1d",
  "initial_capital": 100000,
  "strategy_mode": "ma_crossover",  # or "final_signal"
  "ma_short_window": 50,
  "ma_long_window": 60,
  "position_sizing_mode": "full_equity",
  "max_bars": 200
}
```

### Realtime Simulation
```bash
POST /realtime-sim/start
{
  "symbol": "AAPL",
  "timeframe": "1m",
  "initial_capital": 100000,
  "risk_per_trade": 0.01
}
```

## Consistency with Strategy Lab

### Verification Scenario
To verify MA Crossover mode matches Strategy Lab:

1. **Strategy Lab Settings:**
   - Strategy: Moving Average Crossover
   - Short Window: 50
   - Long Window: 60
   - Symbol: NVDA
   - Date Range: 2025-01-01 to 2025-12-06

2. **Auto Sim Lab Settings:**
   - Strategy Mode: MA Crossover
   - MA Short Window: 50
   - MA Long Window: 60
   - Position Sizing: Full Equity
   - Symbol: NVDA
   - Date Range: Same as above

3. **Expected Result:**
   - Total Return should be within 1-2% of Strategy Lab
   - Equity curves should have similar shapes
   - Trade entry/exit timing should match

### Known Differences
- **Entry timing**: Both systems enter on the signal bar's close price
- **Fees**: Auto Sim Lab does not apply commission fees (can be added if needed)
- **Warmup period**: MA mode requires `long_window + 5` bars for warmup

## Code Structure

```
backend/
├── auto_sim_lab.py           # Main engine
│   ├── AutoSimConfig         # Configuration model
│   ├── AutoSimResult         # Result model
│   ├── generate_action_for_bar()  # Strategy-agnostic signal generation
│   ├── calculate_position_size()  # Position sizing
│   └── run_auto_simulation()      # Main simulation loop
├── realtime_sim_engine.py    # Realtime engine
├── realtime_sim_manager.py   # Session manager
└── strategies/
    └── ma_cross.py           # Canonical MA Crossover (reused by Auto Sim Lab)
```

## Changelog

### 2025-12-06
- Added `strategy_mode` field: `final_signal` (default) or `ma_crossover`
- Added `position_sizing_mode` field with 4 options
- Implemented `generate_action_for_bar()` for strategy-agnostic signal generation
- Added MA Crossover logic matching Strategy Lab's implementation
- Updated Streamlit UI with strategy and position sizing selectors
