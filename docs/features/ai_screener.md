# AI Screener - Symbol Selection System

**Version**: 1.0  
**Status**: Production Ready  
**Last Updated**: 2025-12-08

---

## Overview

The AI Screener is EXITON's automated symbol selection system that ranks stocks based on multiple technical factors. It helps identify the most promising trading candidates from a universe of symbols using a rule-based multi-factor scoring system.

### Purpose

**Problem**: With hundreds of symbols available, manually identifying which ones to trade is time-consuming and subjective.

**Solution**: Automated technical screening that scores and ranks symbols based on:
- Trend strength
- Volatility appropriateness
- Price momentum
- Oversold/overbought conditions
- Volume activity

---

## API Specification

### Endpoint

```
GET /recommended-symbols
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `universe` | string | No | `sp500_top50` | Universe identifier |
| `limit` | integer | No | 20 | Maximum symbols to return |

### Supported Universes

- `mega_caps` - Magnificent 7 (AAPL, MSFT, GOOGL, etc.)
- `sp500_top50` - Top 50 S&P 500 by market cap
- `us_semiconductors` - Major semiconductor stocks
- `kousuke_watchlist_v1` - Curated watchlist

### Response Structure

```json
{
  "universe": "sp500_top50",
  "as_of": "2025-12-08T12:00:00Z",
  "total_screened": 50,
  "total_returned": 20,
  "symbols": [
    {
      "symbol": "AAPL",
      "score": 87.3,
      "factors": {
        "trend_score": 90.0,
        "volatility_score": 70.0,
        "momentum_score": 80.0,
        "oversold_score": 60.0,
        "volume_spike_score": 95.0
      }
    },
    ...
  ]
}
```

---

## Scoring Logic (v1)

### Overall Score Calculation

```python
overall_score = (
    trend_score * 0.35 +
    volatility_score * 0.20 +
    momentum_score * 0.20 +
    oversold_score * 0.15 +
    volume_spike_score * 0.10
)
```

All scores are normalized to 0-100 range.

### 1. Trend Score (35% weight)

**Indicators**:
- MA Short (20) vs MA Long (50)
- MACD histogram
- ADX proxy (via MA spread)

**Logic**:
- Base score: 50
- Short MA > Long MA: +20 bonus
- Strong MA spread: up to +15 bonus
- Positive MACD histogram: +10 bonus
- Negative MACD: -5 penalty

**Range**: 0-100

### 2. Volatility Score (20% weight)

**Indicator**:
- ATR (Average True Range) as % of price

**Logic**:
- Optimal range: 1.5% - 4.0% ATR
  - Within optimal: 90 points
- Too low (< 1.5%): Scaled 50-90
- Too high (> 4.0%): Decreasing, min 30

**Rationale**: Moderate volatility = tradeable but not risky

**Range**: 0-100

### 3. Momentum Score (20% weight)

**Indicators**:
- ROC (Rate of Change) over 10 periods
- RSI trend (5-period change)

**Logic**:
- Base score: 50
- Positive ROC: up to +30
- Negative ROC: up to -30
- Rising RSI: up to +20 bonus

**Range**: 0-100

### 4. Oversold Score (15% weight)

**Indicator**:
- RSI (14-period)

**Logic**:
- RSI < 30: 80-100 (oversold = buy opportunity)
- RSI 30-40: 60-80
- RSI 40-60: 50 (neutral)
- RSI 60-70: 30-50
- RSI > 70: 10-30 (overbought = avoid)

**Range**: 0-100

### 5. Volume Spike Score (10% weight)

**Indicator**:
- Current volume / 5-day average volume

**Logic**:
- Ratio >= 1.5: 90-100
- Ratio 1.2-1.5: 70-90
- Ratio 0.8-1.2: 50-70
- Ratio < 0.8: 0-50

**Rationale**: High volume = attention/liquidity

**Range**: 0-100

---

## UI (Developer Dashboard)

### Access

1. Open dev_dashboard: `streamlit run dev_dashboard.py`
2. Select "AI Screener" from mode dropdown
3. Configure universe and limit
4. Click "🚀 Run Screener"

### Features

- **Universe Selection**: Choose from predefined symbol universes
- **Top N Control**: Limit results to top N symbols
- **Scoring Explanation**: Expandable section explaining methodology
- **Results Table**: Color-coded heatmap of scores
- **Top 3 Highlights**: Quick view of best candidates with factor breakdowns

---

## Usage Examples

### cURL

```bash
# Get top 20 from S&P 500 Top 50
curl "http://localhost:8001/recommended-symbols?universe=sp500_top50&limit=20"

# Get top 10 from Mega Caps
curl "http://localhost:8001/recommended-symbols?universe=mega_caps&limit=10"
```

### Python

```python
import requests

response = requests.get(
    "http://localhost:8001/recommended-symbols",
    params={"universe": "sp500_top50", "limit": 20}
)

if response.status_code == 200:
    data = response.json()
    top_symbol = data["symbols"][0]
    print(f"Top symbol: {top_symbol['symbol']} (score: {top_symbol['score']})")
```

---

## Performance Considerations

### Current Implementation (v1)

- **Processing Time**: ~1-3 seconds for 50 symbols
- **No Caching**: Each request fetches fresh data
- **Timeout**: 120 seconds for large universes

### Future Optimizations

1. **Caching Layer**
   ```python
   # Cache mechanism (example)
   @lru_cache(maxsize=10, ttl=900)  # 15-minute cache
   def get_screener_results(universe, limit):
       ...
   ```

2. **Parallel Processing**
   ```python
   # Process symbols in parallel
   with ThreadPoolExecutor(max_workers=10) as executor:
       futures = [executor.submit(score_symbol, sym) for sym in symbols]
       results = [f.result() for f in futures]
   ```

3. **Pre-computed Scores**
   - Run screening every 15 minutes via cron
   - Store results in database
   - API returns cached results

---

## Extension Points

### Adding New Factors

```python
# 1. Add calculation method to SymbolScreener
def _calculate_my_new_score(self, df: pd.DataFrame) -> float:
    # Your logic here
    return score

# 2. Update score_symbol() method
def score_symbol(self, df, symbol):
    my_new_score = self._calculate_my_new_score(df)
    
    overall_score = (
        trend_score * 0.30 +  # Adjust weights
        ...
        my_new_score * 0.10    # Add new factor
    )
```

### Machine Learning Integration

```python
# Replace rule-based scoring with ML model
class MLSymbolScreener(SymbolScreener):
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def score_symbol(self, df, symbol):
        # Extract features
        features = self._extract_features(df)
        
        # Predict score
        score = self.model.predict(features)[0]
        
        return SymbolScoreResult(...)
```

---

## Testing

### Unit Tests

Location: `backend/tests/test_screener.py`

**Coverage**:
- Basic scoring functionality
- Score component ranges (0-100)
- Insufficient data handling
- Individual factor calculations
- Multi-symbol screening
- Error handling

**Run Tests**:
```bash
pytest backend/tests/test_screener.py -v
```

**Expected Output**:
```
12 passed in 0.75s
```

### Integration Testing

1. Start backend: `uvicorn backend.main:app --port 8001`
2. Test API: `curl "http://localhost:8001/recommended-symbols"`
3. Launch UI: `streamlit run dev_dashboard.py`
4. Select "AI Screener" mode
5. Run screening and verify results display

---

## Roadmap

### Phase 2 (Short-term)

- [ ] Add caching (15-minute TTL)
- [ ] Implement parallel processing
- [ ] Add more universes (sectors, industries)
- [ ] Add custom universe creation

### Phase 3 (Medium-term)

- [ ] ML-based scoring (replace rule-based)
- [ ] Backtesting screener performance
- [ ] API for custom factor weights
- [ ] Historical screening results storage

### Phase 4 (Long-term)

- [ ] Real-time screening (WebSocket)
- [ ] Alert system (notify when top picks change)
- [ ] Portfolio construction (optimize basket)
- [ ] Integration with Live Signal system

---

## Troubleshooting

### Issue: Timeout on Large Universes

**Solution**: Use smaller universe or increase timeout
```python
response = requests.get(..., timeout=300)  # 5 minutes
```

### Issue: All Scores Are 50

**Symptom**: Every symbol gets ~50 score  
**Cause**: Insufficient data or calculation error  
**Fix**: Check that symbols have >60 bars of OHLCV data

### Issue: No Results Returned

**Symptom**: `symbols` array is empty  
**Possible Causes**:
1. Data fetch failing for all symbols
2. All symbols filtered out (< 60 bars)
3. Backend error (check logs)

**Debug**:
```bash
# Check backend logs
tail -f backend.log

# Test single symbol API
curl "http://localhost:8001/api/chart-data?symbol=AAPL&timeframe=1d&limit=100"
```

---

## Architecture

```
User (Browser)
    ↓
Developer Dashboard (Streamlit)
    ↓
GET /recommended-symbols
    ↓
FastAPI Backend (main.py)
    ↓
SymbolScreener (screener.py)
    ↓
Data Feed (yfinance/ccxt)
    ↓
OHLCV Data
    ↓
Multi-Factor Scoring
    ↓
Ranked Results
```

---

## Code Locations

| Component | File |
|-----------|------|
| Screener Logic | `backend/signals/screener.py` |
| API Endpoint | `backend/main.py` (line ~1170) |
| UI Page | `dev_dashboard.py` (`render_ai_screener_page()`) |
| Tests | `backend/tests/test_screener.py` |
| Universe Config | `backend/config/symbol_universes.py` |

---

## Changelog

### v1.0 (2025-12-08)

- Initial release
- 5-factor scoring system
- REST API endpoint
- Streamlit UI integration
- 12 automated tests
- Documentation

---

**Maintained by**: EXITON Team  
**Contact**: See main README for support
