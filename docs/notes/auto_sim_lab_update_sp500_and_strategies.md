# Auto Sim Lab Update: S&P 500 Universe & Additional Strategies

**Date:** 2025-12-06  
**Feature Release:** S&P 500 Preset, Buy & Hold, RSI Mean Reversion

---

## Overview

This update adds:

1. **S&P 500 (All Constituents)** universe preset for Multi-Symbol Auto Sim
2. **Buy & Hold** strategy - baseline comparison strategy
3. **RSI Mean Reversion** strategy - buys on oversold, sells on overbought

---

## Task A: S&P 500 Universe Preset

### A-1. Symbol Universes Configuration

**File:** `backend/config/symbol_universes.py`

Single source of truth for all symbol universe definitions:

| Universe ID | Label | Symbol Count |
|-------------|-------|--------------|
| `mega_caps` | MegaCaps (MAG7) | 7 |
| `us_semiconductors` | US Semiconductors | 8 |
| `kousuke_watchlist_v1` | Kousuke Watchlist v1 | 10 |
| `sp500_top50` | S&P 500 Top 50 | 50 |
| `sp500_all` | S&P 500 (All Constituents) | 506 |

### A-2. Symbol Universes API

**Endpoints:**

```
GET /symbol-universes
GET /symbol-universes/{universe_id}
```

**Example Response:**

```json
{
  "mega_caps": {
    "label": "MegaCaps (MAG7)",
    "description": "Magnificent 7 - largest tech companies",
    "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
  },
  "sp500_all": {
    "label": "S&P 500 (All Constituents)",
    "symbols": ["A", "AAL", "AAPL", ...]
  }
}
```

### A-3. Auto Sim Lab UI Update

The Multi-Symbol mode now fetches universes from the API:

1. **Universe Preset Dropdown** - Shows all available universes
2. **Custom Mode** - Allows manual symbol entry
3. **Read-Only Mode** - When preset selected, symbols are displayed but not editable

### A-4. Verification

**Script:** `tools/verify_sp500_universe.py`

**Run:**
```bash
python tools/verify_sp500_universe.py
```

**Results:** `docs/verify_sp500_universe.md`

---

## Task B: Additional Strategies

### B-1. Buy & Hold Strategy

**File:** `backend/strategies/buy_and_hold.py`

| Property | Value |
|----------|-------|
| Strategy ID | `buy_and_hold` |
| Parameters | None |
| Logic | Buy on first bar, hold until end |
| R-Management | Disabled (N/A for buy & hold) |

**Use Case:** Baseline comparison - compare other strategies against simple buy & hold.

### B-2. RSI Mean Reversion Strategy

**File:** `backend/strategies/rsi_mean_reversion.py`

| Property | Value |
|----------|-------|
| Strategy ID | `rsi_mean_reversion` |
| Parameters | `rsi_period`, `rsi_oversold`, `rsi_overbought` |
| Logic | Buy when RSI < oversold, Sell when RSI > overbought |
| Position Type | Long only |

**Default Parameters:**
- RSI Period: 14
- Oversold Level: 30
- Overbought Level: 70

### B-3. Strategy Selection in Auto Sim Lab

Multi-Symbol mode now supports strategy selection:

| Strategy | Parameters |
|----------|------------|
| MA Crossover | MA Short Window, MA Long Window |
| Buy & Hold | None |
| RSI Mean Reversion | RSI Period, Oversold, Overbought |

**Conditional Parameters:**
- MA parameters shown only for `ma_crossover`
- RSI parameters shown only for `rsi_mean_reversion`
- R-Management disabled for `buy_and_hold`

### B-4. Verification

**Script:** `tools/verify_strategies_basic.py`

**Run:**
```bash
python tools/verify_strategies_basic.py
```

**Results:** `docs/verify_strategies_basic.md`

---

## Usage Guide

### Running Multi-Symbol with S&P 500

1. Open Auto Sim Lab → Multi-Symbol mode
2. Select "S&P 500 (All Constituents)" from Universe Preset
3. Choose strategy (MA Crossover, Buy & Hold, or RSI)
4. Configure parameters
5. Click "Run Multi-Symbol Simulation"

**Note:** Running 500+ symbols may take 5-10 minutes. Timeout is set to 600 seconds.

### Comparing Strategies

To compare Buy & Hold vs other strategies:

1. Run Multi-Symbol with "Buy & Hold" strategy
2. Note the average return and total R
3. Run again with "MA Crossover" or "RSI Mean Reversion"
4. Compare results in the ranking table

---

## Verification Results

### S&P 500 Universe

✅ All checks passed:
- 506 symbols in sp500_all
- No duplicates
- All universes accessible via API

### Strategies

✅ All checks passed:

| Strategy | NVDA | AAPL |
|----------|------|------|
| Buy & Hold | 1 trade ✅ | 1 trade ✅ |
| RSI Mean Reversion | 3 trades, all long ✅ | 2 trades, all long ✅ |

---

## Files Changed

### New Files
- `backend/config/symbol_universes.py` - Universe definitions
- `backend/strategies/buy_and_hold.py` - Buy & Hold strategy
- `tools/verify_sp500_universe.py` - Universe verification
- `tools/verify_strategies_basic.py` - Strategy verification

### Modified Files
- `backend/main.py` - Added `/symbol-universes` endpoints
- `backend/auto_sim_lab.py` - Added strategy modes, RSI params
- `backend/models/multi_sim.py` - Added RSI params
- `backend/multi_sim_engine.py` - Pass RSI params to AutoSimConfig
- `backend/strategies/rsi_mean_reversion.py` - Added STRATEGY_METADATA
- `dev_dashboard.py` - Updated Multi-Symbol UI

---

## Future Improvements

- [ ] CSV import for custom universe
- [ ] Short positions for RSI strategy
- [ ] Strategy comparison charts
- [ ] Parallel execution for faster multi-symbol runs
