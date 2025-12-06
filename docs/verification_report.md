# Auto Sim Lab Verification Report

**Generated:** 2025-12-06T17:10:50.818975
**Commit:** `d66dca6e4dccd02b8daf6d1496f7a38f9913530b`

## Parameters Used

- **symbol:** NVDA
- **timeframe:** 1d
- **start_date:** 2025-01-01
- **end_date:** 2025-12-06
- **ma_short:** 50
- **ma_long:** 60
- **initial_capital:** 100000.0
- **r_management:** enabled
- **virtual_stop_method:** percent
- **stop_percent:** 3.0%

---

## Verification Results

### ✅ Task 1: Strategy Lab vs Auto Sim Lab: Pass

All 3 trades match between Strategy Lab and Auto Sim Lab (MA Crossover mode)

```
| # | SL Entry | AS Entry | SL Exit | AS Exit | Match |
|---|----------|----------|---------|---------|-------|
| 1 | 2025-05-23 | 2025-05-23 | 2025-06-05 | 2025-06-05 | ✅ |
| 2 | 2025-06-06 | 2025-06-06 | 2025-10-22 | 2025-10-22 | ✅ |
| 3 | 2025-10-30 | 2025-10-30 | 2025-12-05 | 2025-12-05 | ✅ |
```

### ✅ Task 2: R-Management Verification: Pass

All 3 trades have correct R-management calculations

Formula used:
- stop_price = entry_price * (1 - stop_percent)
- risk_amount = (entry_price - stop_price) * size
- r_value = pnl / risk_amount

```
Trade #1: ✅
  Entry: $131.27, Size: 761
  Stop: $127.33 (expected: $127.33) ✅
  Risk Amount: $2996.81 (expected: $2996.81) ✅
  R-Value: 2.21R (expected: 2.21R) ✅
  PnL: $6619.50

Trade #2: ✅
  Entry: $141.69, Size: 752
  Stop: $137.44 (expected: $137.44) ✅
  Risk Amount: $3196.62 (expected: $3196.62) ✅
  R-Value: 9.07R (expected: 9.07R) ✅
  PnL: $29008.91

Trade #3: ✅
  Entry: $202.88, Size: 668
  Stop: $196.79 (expected: $196.79) ✅
  Risk Amount: $4065.69 (expected: $4065.69) ✅
  R-Value: -3.36R (expected: -3.36R) ✅
  PnL: $-13673.09
```

### ✅ Task 3: Trade Inspector / Decision Log: Pass

All 3 trades have correct trade_id linking to Decision Log

Each trade has:
- Exactly 1 entry event with matching trade_id
- Exactly 1 exit event with matching trade_id
- Entry event price matches trade entry_price

```
Trade #1: ✅
  Entry time: 2025-05-23T00:00:00
  Exit time: 2025-06-05T00:00:00
  Linked events: 10 total
    - entry: 1 ✅
    - exit: 1 ✅
    - signal_decision: 8
  Entry price match: ✅

Trade #2: ✅
  Entry time: 2025-06-06T00:00:00
  Exit time: 2025-10-22T00:00:00
  Linked events: 97 total
    - entry: 1 ✅
    - exit: 1 ✅
    - signal_decision: 95
  Entry price match: ✅

Trade #3: ✅
  Entry time: 2025-10-30T00:00:00
  Exit time: 2025-12-05T00:00:00
  Linked events: 27 total
    - entry: 1 ✅
    - exit: 1 ✅
    - signal_decision: 25
  Entry price match: ✅
```

---

## Summary

**✅ All verifications passed.** Auto Sim Lab is ready for use.

- Task 1 (Strategy Lab vs Auto Sim Lab): ✅ Pass
- Task 2 (R-Management): ✅ Pass
- Task 3 (Trade Inspector): ✅ Pass
