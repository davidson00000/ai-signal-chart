"""
Symbol Screener - AI-based stock screening system

This module implements the core logic for scoring and ranking symbols
based on multiple technical factors.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class SymbolScoreResult:
    """Result of scoring a single symbol"""
    symbol: str
    score: float
    factors: Dict[str, float]
    as_of: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "score": round(self.score, 2),
            "factors": {k: round(v, 2) for k, v in self.factors.items()}
        }


class SymbolScreener:
    """
    Symbol Screener v1 - Rule-based scoring system
    
    Scores symbols based on 5 key factors:
    1. Trend Score (35%): MACD, ADX, MA comparison
    2. Volatility Score (20%): ATR relative to price
    3. Momentum Score (20%): ROC, RSI trend
    4. Oversold Score (15%): RSI levels
    5. Volume Spike Score (10%): Volume vs average
    
    Future: This can be replaced with ML-based scoring
    """
    
    # Weights for overall score
    WEIGHTS = {
        'trend_score': 0.35,
        'volatility_score': 0.20,
        'momentum_score': 0.20,
        'oversold_score': 0.15,
        'volume_spike_score': 0.10
    }
    
    def __init__(self):
        pass
    
    def score_symbol(self, df: pd.DataFrame, symbol: str) -> Optional[SymbolScoreResult]:
        """
        Score a single symbol based on its price data
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            symbol: Symbol name
        
        Returns:
            SymbolScoreResult or None if insufficient data
        """
        try:
            # Need at least 60 bars for reliable scoring
            if len(df) < 60:
                return None
            
            # Calculate all factors
            trend_score = self._calculate_trend_score(df)
            volatility_score = self._calculate_volatility_score(df)
            momentum_score = self._calculate_momentum_score(df)
            oversold_score = self._calculate_oversold_score(df)
            volume_spike_score = self._calculate_volume_spike_score(df)
            
            # Calculate weighted overall score
            overall_score = (
                trend_score * self.WEIGHTS['trend_score'] +
                volatility_score * self.WEIGHTS['volatility_score'] +
                momentum_score * self.WEIGHTS['momentum_score'] +
                oversold_score * self.WEIGHTS['oversold_score'] +
                volume_spike_score * self.WEIGHTS['volume_spike_score']
            )
            
            # Get timestamp of last bar
            if hasattr(df.index, 'to_pydatetime'):
                as_of = df.index[-1].to_pydatetime()
            else:
                as_of = datetime.now()
            
            return SymbolScoreResult(
                symbol=symbol,
                score=overall_score,
                factors={
                    'trend_score': trend_score,
                    'volatility_score': volatility_score,
                    'momentum_score': momentum_score,
                    'oversold_score': oversold_score,
                    'volume_spike_score': volume_spike_score
                },
                as_of=as_of
            )
            
        except Exception as e:
            print(f"Error scoring {symbol}: {str(e)}")
            return None
    
    def _calculate_trend_score(self, df: pd.DataFrame) -> float:
        """
        Trend strength score (0-100)
        
        Factors:
        - MA short vs MA long
        - MACD histogram
        - ADX (Average Directional Index) proxy
        """
        score = 50.0  # Base score
        
        # MA crossover (20 vs 50)
        ma_short = df['close'].rolling(window=20).mean()
        ma_long = df['close'].rolling(window=50).mean()
        
        if not pd.isna(ma_short.iloc[-1]) and not pd.isna(ma_long.iloc[-1]):
            if ma_short.iloc[-1] > ma_long.iloc[-1]:
                score += 20  # Uptrend bonus
                
                # Additional bonus for strong uptrend
                ma_diff_pct = (ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1] * 100
                score += min(ma_diff_pct * 2, 15)  # Cap at +15
            else:
                score -= 10  # Downtrend penalty
        
        # MACD
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        if not pd.isna(histogram.iloc[-1]):
            if histogram.iloc[-1] > 0:
                score += 10  # Bullish MACD
            else:
                score -= 5  # Bearish MACD
        
        # Clamp to 0-100
        return max(0, min(100, score))
    
    def _calculate_volatility_score(self, df: pd.DataFrame) -> float:
        """
        Volatility appropriateness score (0-100)
        
        Too low volatility = hard to profit
        Too high volatility = too risky
        Optimal range = moderate volatility
        """
        # Calculate ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()
        
        if pd.isna(atr.iloc[-1]) or df['close'].iloc[-1] == 0:
            return 50.0
        
        # ATR as percentage of price
        atr_pct = (atr.iloc[-1] / df['close'].iloc[-1]) * 100
        
        # Optimal range: 1.5% - 4.0%
        if 1.5 <= atr_pct <= 4.0:
            score = 90.0
        elif atr_pct < 1.5:
            # Too low volatility
            score = 50.0 + (atr_pct / 1.5) * 40
        else:
            # Too high volatility
            score = max(30, 90 - (atr_pct - 4.0) * 10)
        
        return max(0, min(100, score))
    
    def _calculate_momentum_score(self, df: pd.DataFrame) -> float:
        """
        Momentum score (0-100)
        
        Factors:
        - ROC (Rate of Change)
        - RSI trend (rising RSI = momentum)
        """
        score = 50.0
        
        # ROC over 10 periods
        roc_10 = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        
        if not pd.isna(roc_10.iloc[-1]):
            if roc_10.iloc[-1] > 0:
                # Positive momentum
                score += min(roc_10.iloc[-1] * 2, 30)  # Cap at +30
            else:
                # Negative momentum
                score += max(roc_10.iloc[-1] * 2, -30)  # Cap at -30
        
        # RSI trend (is RSI rising?)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        if len(rsi) >= 5 and not pd.isna(rsi.iloc[-1]) and not pd.isna(rsi.iloc[-5]):
            rsi_change = rsi.iloc[-1] - rsi.iloc[-5]
            if rsi_change > 0:
                score += min(rsi_change, 20)  # Rising RSI bonus
        
        return max(0, min(100, score))
    
    def _calculate_oversold_score(self, df: pd.DataFrame) -> float:
        """
        Oversold/Overbought score (0-100)
        
        High score for oversold (buy opportunity)
        Low score for overbought (avoid)
        """
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        if pd.isna(rsi.iloc[-1]):
            return 50.0
        
        rsi_val = rsi.iloc[-1]
        
        if rsi_val < 30:
            # Oversold - good buy opportunity
            score = 80 + (30 - rsi_val)  # 80-100
        elif rsi_val < 40:
            # Slightly oversold
            score = 60 + (40 - rsi_val) * 2
        elif rsi_val <= 60:
            # Neutral range
            score = 50
        elif rsi_val <= 70:
            # Slightly overbought
            score = 50 - (rsi_val - 60) * 2
        else:
            # Overbought - avoid
            score = max(10, 30 - (rsi_val - 70))
        
        return max(0, min(100, score))
    
    def _calculate_volume_spike_score(self, df: pd.DataFrame) -> float:
        """
        Volume spike score (0-100)
        
        High volume = increased attention/liquidity
        """
        if 'volume' not in df.columns:
            return 50.0  # Neutral if no volume data
        
        # Volume vs 5-day average
        volume_sma = df['volume'].rolling(window=5).mean()
        
        if pd.isna(volume_sma.iloc[-1]) or volume_sma.iloc[-1] == 0:
            return 50.0
        
        volume_ratio = df['volume'].iloc[-1] / volume_sma.iloc[-1]
        
        if volume_ratio >= 1.5:
            score = 90 + min((volume_ratio - 1.5) * 10, 10)  # 90-100
        elif volume_ratio >= 1.2:
            score = 70 + (volume_ratio - 1.2) * 66  # 70-90
        elif volume_ratio >= 0.8:
            score = 50 + (volume_ratio - 0.8) * 50  # 50-70
        else:
            score = volume_ratio * 62.5  # 0-50
        
        return max(0, min(100, score))


def screen_symbols(
    symbols: List[str],
    data_fetcher,
    timeframe: str = '1d',
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Screen multiple symbols and return top-ranked ones
    
    Args:
        symbols: List of symbol strings
        data_fetcher: Function to fetch OHLCV data (symbol, timeframe) -> DataFrame
        timeframe: Timeframe for analysis
        limit: Number of top results to return
    
    Returns:
        List of scored symbols (sorted by score descending)
    
    Performance Notes:
        - Small universes (< 50 symbols): ~1-3 seconds
        - Medium universes (50-100 symbols): ~5-10 seconds
        - Large universes (sp500_all ~500 symbols): ~30-60 seconds
        
        For production use with large universes, consider:
        1. Implementing caching (15-minute TTL recommended)
        2. Using parallel processing (ThreadPoolExecutor)
        3. Pre-computing scores via scheduled jobs
    """
    screener = SymbolScreener()
    results = []
    
    for symbol in symbols:
        try:
            # Fetch data
            df = data_fetcher(symbol, timeframe)
            
            if df is None or len(df) < 60:
                continue
            
            # Score the symbol
            score_result = screener.score_symbol(df, symbol)
            
            if score_result is not None:
                results.append(score_result)
                
        except Exception as e:
            print(f"Error screening {symbol}: {str(e)}")
            continue
    
    # Sort by score descending
    results.sort(key=lambda x: x.score, reverse=True)
    
    # Return top N
    return [r.to_dict() for r in results[:limit]]
