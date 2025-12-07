# Verification: Strategy Sweep

## Changes Made

### 1. Backend (`backend/auto_sim_lab.py`)
- Updated `StrategyConfig` model to support individual strategy parameters (e.g. `ma_short_window`, `rsi_period`) and `label` field.
- Updated `run_strategy_sweep` to correctly map these parameters to `AutoSimConfig`.
- Added support for `rsi_mean_reversion` in `run_strategy_for_symbol` and `run_simulation_core` (warmup calculation).
- Fixed `run_auto_simulation` to use default `limit=5000` (matching Strategy Sweep) instead of `500`.

### 2. Frontend (`dev_dashboard.py`)
- Updated `_render_sweep_results` to handle the flat `StrategySweepResult` JSON structure returned by the backend.
- Fixed `Max DD %` display scaling (multiplied by 100).
- Improved Strategy column display to show "Strategy | Preset".

### 3. Verification Script (`tools/verify_strategy_sweep.py`)
- Created a script to compare Single Auto Sim results with Strategy Sweep results for consistency.

## How Strategy Sweep Works

1.  **Frontend**: User selects symbols and strategies (with presets).
2.  **Frontend**: Constructs `StrategySweepRequest` with a list of `StrategyConfig` objects. Each config contains the strategy mode, label, and specific parameters (e.g. `ma_short_window`).
3.  **Backend**: `run_strategy_sweep` iterates through symbols.
    - Fetches data once per symbol (`limit=5000`).
    - Iterates through strategies.
    - Constructs `AutoSimConfig` for each strategy using the provided parameters.
    - Calls `run_simulation_core` (reused from Single Auto Sim).
    - Aggregates results into `StrategySweepResult`.
4.  **Frontend**: Receives list of `StrategySweepResult` and renders the ranking table.

## Verification

Run the verification script:

```bash
cd /Users/kousukenakamura/dev/ai-signal-chart
python tools/verify_strategy_sweep.py
```

Expected Output:

```
Verifying Strategy Sweep for GOOGL with ma_crossover (20/50)...
Running Single Auto Sim...
Single Result: Return=433.69%, R=71.01, Trades=32, MaxDD=3811.00%
Running Strategy Sweep...
Sweep Result:  Return=433.69%, R=71.01, Trades=32, MaxDD=3811.00%
PASS: Return % match.
PASS: Total R match.
PASS: Trades match.
PASS: Max DD match.

OVERALL: PASS
```
