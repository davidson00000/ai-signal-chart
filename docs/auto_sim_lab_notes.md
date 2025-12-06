# Auto Sim Lab Notes

## Overview

Auto Sim Lab is an automated paper trading simulation engine that supports multiple strategy modes, position sizing options, R-based risk management, execution modes, and loss control.

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

## Position Sizing Modes

| Mode | Description | Parameters |
|------|-------------|------------|
| `percent_of_equity` | Risk X% of equity per trade | `risk_per_trade` (decimal) |
| `full_equity` | Use 100% of equity per trade | None |
| `fixed_shares` | Fixed number of shares | `fixed_shares` (int) |
| `fixed_dollar` | Fixed dollar amount | `fixed_dollar_amount` (float) |

## R-Management

R-management tracks trades in risk multiples using virtual stops.

### Configuration

```python
use_r_management: bool = False
virtual_stop_method: Literal["atr", "percent"] = "percent"
virtual_stop_atr_multiplier: float = 2.0  # For ATR method
virtual_stop_percent: float = 0.03        # For percent method (3%)
```

### Virtual Stop Calculation

**ATR Method:**
```
stop_price = entry_price - (ATR × virtual_stop_atr_multiplier)
```

**Percent Method:**
```
stop_price = entry_price × (1 - virtual_stop_percent)
```

### R-Value Calculation

```
risk_amount = (entry_price - stop_price) × position_size
r_value = pnl / risk_amount
```

### Output

When R-management is enabled, trades include:
- `r_value`: PnL in R multiples
- `stop_price`: Virtual stop price
- `risk_amount`: Dollar risk per trade

Summary includes:
- `total_r`: Total R across all trades
- `avg_r`: Average R per trade
- `best_r` / `worst_r`: Extremes

## Execution Mode

| Mode | Description |
|------|-------------|
| `same_bar_close` | Execute at signal bar's close (default) |
| `next_bar_open` | Execute at next bar's open (more realistic) |

### Commission & Slippage

```python
commission_percent: float = 0.0  # e.g., 0.001 = 0.1%
slippage_percent: float = 0.0    # e.g., 0.001 = 0.1%
```

**Price Adjustment:**
- Buy: `execution_price = raw_price × (1 + slippage_percent)`
- Sell: `execution_price = raw_price × (1 - slippage_percent)`

## Loss Control

### Max Drawdown

```python
max_drawdown_percent: Optional[float] = None  # e.g., 0.20 = 20%
```

If drawdown exceeds this limit, simulation is halted.

```
current_dd = (peak_equity - current_equity) / peak_equity
if current_dd >= max_drawdown_percent → HALT
```

### Max Daily Loss (R)

```python
max_daily_loss_r: Optional[float] = None  # e.g., 3.0 = 3R
```

If cumulative R loss for the day exceeds this limit, no new positions for that day.

```
if daily_r_loss <= -max_daily_loss_r → SKIP new entries for the day
```

## API Usage

### Historical Simulation (Full Example)

```bash
POST /auto-simulate
{
  "symbol": "AAPL",
  "timeframe": "1d",
  "initial_capital": 100000,
  "strategy_mode": "ma_crossover",
  "ma_short_window": 10,
  "ma_long_window": 30,
  "position_sizing_mode": "full_equity",
  "use_r_management": true,
  "virtual_stop_method": "percent",
  "virtual_stop_percent": 0.03,
  "execution_mode": "next_bar_open",
  "commission_percent": 0.001,
  "slippage_percent": 0.001,
  "max_drawdown_percent": 0.20,
  "max_daily_loss_r": 3.0,
  "max_bars": 200
}
```

### Response Summary

```json
{
  "summary": {
    "strategy_mode": "ma_crossover",
    "position_sizing_mode": "full_equity",
    "execution_mode": "next_bar_open",
    "total_trades": 3,
    "wins": 2,
    "losses": 1,
    "win_rate": 66.67,
    "total_pnl": 23530.00,
    "avg_pnl": 7843.33,
    "total_r": 7.8,
    "avg_r": 2.6,
    "best_r": 4.2,
    "worst_r": -1.5,
    "simulation_halted": false,
    "halt_reason": null
  }
}
```

## Backward Compatibility

All new features are optional with defaults that match previous behavior:

- `use_r_management: false` → No R tracking
- `execution_mode: "same_bar_close"` → Original behavior
- `commission_percent: 0.0` → No fees
- `slippage_percent: 0.0` → No slippage
- `max_drawdown_percent: null` → No DD limit
- `max_daily_loss_r: null` → No daily limit

Existing API calls will work exactly as before.

## Code Structure

```
backend/
├── auto_sim_lab.py           # Main engine
│   ├── AutoSimConfig         # Extended configuration
│   ├── AutoSimResult         # Result model
│   ├── calculate_atr()       # ATR calculation
│   ├── calculate_virtual_stop()  # Virtual stop
│   ├── calculate_execution_price()  # With fees
│   ├── generate_action_for_bar()  # Strategy-agnostic
│   ├── calculate_position_size()  # Position sizing
│   └── run_auto_simulation()      # Main loop
├── models/
│   └── decision_log.py       # Extended DecisionEvent
└── strategies/
    └── ma_cross.py           # MA Crossover strategy
```

## Changelog

### 2025-12-06 (v2)
- Added R-management with virtual stops (ATR / percent)
- Added execution modes (same_bar_close / next_bar_open)
- Added commission and slippage
- Added loss control (max drawdown, max daily loss in R)
- Extended DecisionEvent with new fields
- Extended UI with 3 new panels

### 2025-12-06 (v1)
- Added `strategy_mode` field: `final_signal` (default) or `ma_crossover`
- Added `position_sizing_mode` field with 4 options
- Implemented `generate_action_for_bar()` for strategy-agnostic signal generation
- Added MA Crossover logic matching Strategy Lab's implementation
- Updated Streamlit UI with strategy and position sizing selectors
