# Sprint 18: Multi-Strategy Comparison v0.1

## 目的
Backtest Lab で Strategy Library に保存された複数ストラテジーを同時にバックテストし、
成績を並べて比較できる機能を追加する。

## 実装内容

### 1. UI コンポーネント - Strategy Comparison セクション

**配置**: Backtest Lab、Loaded Strategy Info Display の後

**主要コンポーネント**:
- **Multi-select**: 複数の保存済みストラテジーを選択
  - 表示形式: `{name} | {symbol} {timeframe} | MA({short},{long}) | Return: {return_pct}%`
- **🔬 Run Comparison ボタン**: 比較実行

### 2. バリデーションロジック

比較実行前に以下を検証:
- 最低2つのストラテジーが選択されている
- すべてのストラテジーが同一 Symbol を使用
- すべてのストラテジーが同一 Timeframe を使用

**エラーケース**:
- 選択数不足: "Please select at least 2 strategies to compare."
- Symbol 不一致: "❌ All strategies must use the same symbol. Selected symbols: ..."
- Timeframe 不一致: "❌ All strategies must use the same timeframe. Selected timeframes: ..."

### 3. 比較実行ロジック

各選択されたストラテジーに対して:
1. パラメータを抽出（symbol, timeframe, short_window, long_window）
2. Backtest Lab の現在の設定（date range, capital, commission）を使用
3. `/simulate` エンドポイントを呼び出し
4. 結果を収集

**実行時の表示**:
- ✅ 成功メッセージ: "Comparing X strategies with {symbol} / {timeframe}"
- Loading spinner: "Running comparisons..."
- エラー発生時: "Failed to run backtest for '{strategy_name}': {error}"

### 4. 結果表示

#### 4.1 情報ボックス
```
Strategy Comparison  
Comparing multiple strategies with the same symbol, timeframe, and date range.
- Return (%): (Final Equity / Initial Capital - 1) × 100
- Max Drawdown (%): Maximum peak-to-trough decline
- Sharpe Ratio: Risk-adjusted return measure
- Win Rate (%): Percentage of profitable trades
```

#### 4.2 Comparison Table

**カラム**:
- Name
- Symbol
- Timeframe
- Short (MA short window)
- Long (MA long window)
- Return (%)
- Max DD (%)
- Sharpe
- Win Rate (%)
- Trades

**ベストパフォーマー表示**:
- 🏆 Best Performer: **{best_name}** with {best_return}% return

#### 4.3 Equity Curve Overlay Chart

**使用ライブラリ**: Altair

**特徴**:
- X軸: Date
- Y軸: Equity
- 色分け: Strategy (凡例付き)
- インタラクティブ: ズーム・パン可能
- ツールチップ: Date, Equity, Strategy 表示

**実装**:
```python
chart = alt.Chart(df_equity).mark_line().encode(
    x=alt.X('date:T', title='Date'),
    y=alt.Y('equity:Q', title='Equity'),
    color=alt.Color('strategy:N', title='Strategy'),
    tooltip=['date:T', 'equity:Q', 'strategy:N']
).properties(
    height=400,
    title='Equity Curve Comparison'
).interactive()
```

## 実行方法

### 1. 準備
Strategy Lab で2つ以上のストラテジーを保存:
```bash
streamlit run dev_dashboard.py
```
- Strategy Lab → Parameter Optimization → Run Optimization
- Best Parameters を Strategy Library に保存（複数回実施）

### 2. 比較実行
Backtest Lab タブを開く:
1. **📊 Strategy Comparison** セクションへスクロール
2. Multi-select で比較したいストラテジーを選択（2つ以上）
3. "🔬 Run Comparison" ボタンをクリック
4. 結果を確認:
   - Comparison Table で各指標を比較
   - Equity Curve Comparison でビジュアル比較
   - 🏆 Best Performer を確認

## 技術的詳細

### データフロー
1. ユーザーが複数ストラテジーを選択
2. バリデーション実行
3. 各ストラテジーで `/simulate` を順次呼び出し
4. 結果を集約
5. Table と Chart を生成・表示

### パフォーマンス考慮
- **順次実行**: API呼び出しは並列化せず順次実行（MVP）
- **タイムアウト**: 各リクエストに30秒のタイムアウト
- **エラーハンドリング**: 個別のストラテジーで失敗しても他の実行は継続

### 使用ライブラリ
- **Altair**: Equity curve チャート作成
- **Pandas**: データフレーム操作
- **Requests**: バックエンド API 呼び出し

## 制約事項（v0.1）
- 同一 Symbol / Timeframe のストラテジーのみ比較可能
- API 呼び出しは順次実行（並列化なし）
- 日付範囲は Backtest Lab の設定を全ストラテジーで共通使用

## 今後の拡張案
- **異なる Symbol** の比較（正規化リターンで比較）
- **並列実行**: API呼び出しの並列化でパフォーマンス向上
- **ストラテジー固有の設定**: 保存時の date range / capital を使用
- **統計的検定**: ストラテジー間の有意差を検定
- **エクスポート**: 比較結果を CSV / PDF でエクスポート
- **リスク指標追加**: Sortino ratio, Calmar ratio など
