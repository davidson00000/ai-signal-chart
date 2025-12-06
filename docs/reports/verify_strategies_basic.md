# Strategies Verification Report

**Generated:** 2025-12-06T18:24:46.397111
**Commit:** `ef2e6daf79abe2a84676f3759718e273a226012f`

## Test Parameters

- **Symbols:** NVDA, AAPL
- **Period:** 2025-01-01 to 2025-12-06

---

## Buy & Hold Strategy

| Symbol | Trade Count | Entry Time | Exit Time | Passed |
|--------|-------------|------------|-----------|--------|
| NVDA | 1 | 2025-03-18T00:00:00 | 2025-12-05T00:00:00 | ✅ |
| AAPL | 1 | 2025-03-18T00:00:00 | 2025-12-05T00:00:00 | ✅ |

**Result:** ✅ Pass

**Expected:** Exactly 1 trade, entry near start date, exit at end date.

---

## RSI Mean Reversion Strategy

| Symbol | Trade Count | All Long | Total R | Passed |
|--------|-------------|----------|---------|--------|
| NVDA | 3 | Yes | 8.55R | ✅ |
| AAPL | 2 | Yes | 7.22R | ✅ |

**Result:** ✅ Pass

**Expected:** At least 1 trade, all trades are long positions.

---

## Summary

- **Buy & Hold:** ✅ Pass
- **RSI Mean Reversion:** ✅ Pass

**✅ All strategy verifications passed.**