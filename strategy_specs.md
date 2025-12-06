# AI Signal Chart -- Strategy Specs v0.1

こうすけ専用「戦略工場」用の仕様書。\
バックエンド / フロント / Antigravity で共通して参照する。

## 0. 共通インターフェース設計

### 0.1 strategy_type

各戦略は一意な `strategy_type` 文字列で識別する：

-   `ma_cross`
-   `ema_cross`
-   `macd_trend`
-   `rsi_mean_reversion`
-   `stoch_oscillator`
-   `bollinger_mean_reversion`
-   `bollinger_breakout`
-   `donchian_breakout`
-   `atr_trailing_ma`
-   `roc_momentum`

### 0.2 入出力仕様（Python 側）

    def generate_signals(df: pd.DataFrame, params: dict) -> pd.Series:
        '''
        df: 時系列の価格データ
        params: 戦略固有のパラメータ辞書
        return: signal Series （+1 / 0）
        '''

バックテスト本体（PnL 計算）は共通。\
戦略側は「いつ +1/0 を出すか」だけに集中する。

------------------------------------------------------------------------

## 1. MA Cross（ma_cross）

既に実装済。

### パラメータ

    short_window: int
    long_window: int

### ルール

短期SMA が長期SMAを上抜け → BUY\
下抜け → EXIT

------------------------------------------------------------------------

## 2. EMA Cross（ema_cross）

EMA を用いたトレンドフォロー。

------------------------------------------------------------------------

## 3. MACD Trend（macd_trend）

MACD のゴールデンクロス/デッドクロス。

------------------------------------------------------------------------

## 4. RSI Mean Reversion（rsi_mean_reversion）

RSI \< oversold → BUY\
RSI \> overbought → EXIT

------------------------------------------------------------------------

## 5. Stochastic Oscillator（stoch_oscillator）

ストキャスティクスのクロスで売買。

------------------------------------------------------------------------

## 6. Bollinger Mean Reversion（bollinger_mean_reversion）

下バンド割れ → BUY\
ミドルバンド到達 → EXIT

------------------------------------------------------------------------

## 7. Bollinger Breakout（bollinger_breakout）

上バンドブレイク → BUY\
ミドル割れ → EXIT

------------------------------------------------------------------------

## 8. Donchian Breakout（donchian_breakout）

高値ブレイク → BUY\
安値割れ → EXIT

------------------------------------------------------------------------

## 9. ATR Trailing MA（atr_trailing_ma）

MA Cross で入って ATR トレールで出る。

------------------------------------------------------------------------

## 10. ROC Momentum（roc_momentum）

ROC \> threshold → BUY\
以下 → EXIT
