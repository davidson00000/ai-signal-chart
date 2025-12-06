# Multi-Symbol Auto Sim Verification Report

**Generated:** 2025-12-06T17:25:11.683023
**Commit:** `cccff0db25e870140ac8fb1a2180a033494b86f6`

## Parameters Used

- **test_symbols:** ['NVDA', 'AAPL', 'MSFT', 'TSLA', 'AMD']
- **timeframe:** 1d
- **start_date:** 2025-01-01
- **end_date:** 2025-12-06

---

## Verification Results

### ✅ Task 1: API No Crashes: Pass

Multi-sim completed successfully for 5 symbols.
Successful: 5, Failed: 0

```
Symbols: NVDA, AAPL, MSFT, TSLA, AMD
```

### ✅ Task 2: Metrics Present: Pass

All expected metrics present for 2 symbols

```
NVDA: All fields present ✅
AAPL: All fields present ✅
```

### ✅ Task 3: Ranking Order: Pass

All 5 symbols correctly ranked by Total R (descending)

```

Ranking Table:
| Rank | Symbol | Total R | Return % |
|------|--------|---------|----------|
| 1 | AMD | 25.39R | 79.47% |
| 2 | NVDA | 7.92R | 21.96% |
| 3 | AAPL | 7.72R | 23.25% |
| 4 | TSLA | 2.93R | 6.03% |
| 5 | MSFT | 1.76R | 4.40% |
```

### ✅ Task 4: Single vs Multi Match: Pass

Single-symbol multi-sim result matches Auto Sim Lab for NVDA

```
| Metric | Multi-Sim | Auto Sim | Match |
|--------|-----------|----------|-------|
| Final Equity | $121,955.32 | $121,955.32 | ✅ |
| Total Return | 21.96% | 21.96% | ✅ |
| Trades | 3 | 3 | ✅ |
| Total R | 7.92R | 7.92R | ✅ |
```

---

## Summary

**✅ All verifications passed.** Multi-Symbol Auto Sim is ready for use.

- Task 1 (API No Crashes): ✅ Pass
- Task 2 (Metrics Present): ✅ Pass
- Task 3 (Ranking Order): ✅ Pass
- Task 4 (Single vs Multi Match): ✅ Pass
