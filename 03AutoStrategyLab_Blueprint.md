# Auto-Strategy Lab — Blueprint v0.9
**Author:** Kousuke × ChatGPT（AI Quant Lab Mode）

高速バックテスト × 戦略生成 × 自動評価 × 自動進化  
EXITON 自動投資システムの「頭脳」レイヤーの設計図。

---

## 1. コンセプト

Auto-Strategy Lab とは：

- 投資戦略を “自動で作り”
- “自動で評価” し
- “自動で進化” させるラボ環境

目的：

- こうすけが「研究者」として戦略を1本ずつ手作業で作るのではなく、  
  **システム自身が戦略を量産・自己改善する** 状態にする。

役割：

- 完全自動運用システムの「戦略供給工場」
- LLM/AI を組み合わせた **AI Quant 研究環境**

---

## 2. レイヤー構造

Auto-Strategy Lab は、既存の Data / Brain / Execution / Dashboard の上に  
「戦略研究レイヤー」として乗るイメージ。

```
[ Layer 0 ] Data Provider        （既存：yfinance / ccxt）
[ Layer 1 ] Backtest Engine      （最優先で実装）
[ Layer 2 ] Strategy Workshop    （戦略の部品化・組み立て）
[ Layer 3 ] Auto-Optimization    （自動パラメータ最適化）
[ Layer 4 ] Evolution Engine     （戦略の自動進化）
[ Layer 5 ] LLM Advisor          （AIクオンツ研究員）
```

### 2.1 Layer 0 — Data Provider（既存）

- 日足・1時間足などのヒストリカルデータ取得
- ローカルキャッシュ
- API: `get_chart_data(symbol, timeframe, start, end)`

### 2.2 Layer 1 — Backtest Engine（最優先）

**Auto-Strategy Lab の「実験装置」。**

要件：

- ローソク足を1本ずつ時間順に流す
- 既存 `PaperTrader` を内部で利用
- 戦略やパラメータが違っても、同じフレームで公平に評価できる

代表的な出力：

- total_pnl
- win_rate
- max_drawdown
- sharpe
- equity_curve（日次 or バーごと）
- trades（全トレード一覧）

API（FastAPI）：

- `POST /backtest`

### 2.3 Layer 2 — Strategy Workshop

目的：

- 戦略を **積み木（ブロック）** として表現する。

ブロック例：

- Indicator ブロック  
  - `MA(period)`  
  - `RSI(period)`  
  - `MACD()`  
  - `ATR(period)`

- Condition ブロック  
  - `>` / `<`  
  - `cross_up(a, b)`  
  - `cross_down(a, b)`

- Logic ブロック  
  - `AND` / `OR` / `NOT`

表現イメージ：

```text
IF MA(5) cross_up MA(25) AND RSI(14) < 60 THEN BUY
IF MA(5) cross_down MA(25) OR RSI(14) > 70 THEN SELL
```

= **木構造のルール** として Python で構造化する。

### 2.4 Layer 3 — Auto-Optimization

- まずは MA クロス戦略の
  - short: [5, 10, 20]
  - long:  [50, 100, 200]
  をグリッドサーチするシンプルな実装から開始。

- その後、Optuna などを使ったベイズ最適化に拡張。

目的関数の例：

- Sharpe Ratio 最大化
- max_drawdown を一定以下にしつつ total_pnl を最大化

API：

- `POST /optimize`  
  - 戦略名 + 探索範囲 + 目的関数 を指定

### 2.5 Layer 4 — Evolution Engine

ここから **「戦略そのもの」を自動で作る** フェーズ。

手法：

- 遺伝的アルゴリズム（Genetic Algorithm, GA）

プロセス：

1. 戦略ブロックのランダム組み合わせで初期集団を生成
2. 各戦略についてバックテスト
3. 上位戦略を「親」として選択（selection）
4. 突然変異（mutation）・交叉（crossover）で子戦略を生成
5. 2〜4 を繰り返す

スコアリング：

- Sharpe
- max_drawdown
- trade_count（あまり少なすぎるものは除外）

### 2.6 Layer 5 — LLM Advisor

LLM（ChatGPT / Gemini / Claude など）の役割：

- 過去のバックテスト結果の要約
- 「どんな特徴の戦略がうまくいっているか」の分析
- 次に試すべき戦略ブロックの提案
- 「ドローダウンが小さい戦略だけを残すにはどうするか」の助言

イメージ：

- LLM を **“クオンツ研究員”** として扱う
- 人間（こうすけ）は PI / PM として、LLM に「研究テーマ」を投げる

---

## 3. モジュール構成案

```
backend/
  |- data_feed.py
  |- paper_trade.py
  |- strategy/
  |     |- blocks.py       # 戦略ブロック（Indicator / Condition / Logic）
  |     |- rules.py        # 具体的な戦略定義
  |     |- generator.py    # ランダム戦略生成
  |     |- optimizer.py    # グリッドサーチ / Optuna
  |     |- evolver.py      # GA による進化
  |
  |- backtest.py           # run_backtest()
  |- api/
        |- main.py         # FastAPI エントリ
        |- routes_backtest.py
        |- routes_optimize.py
        |- routes_evolve.py
        |- routes_llm.py
  |
  |- llm/
        |- advisor.py      # LLM 呼び出し＆結果整形
```

---

## 4. Streamlit / ダッシュボード構成案

`dev_dashboard.py` に以下のタブを追加：

- **Tab: Live / PaperTrade**  
  - 現在の AI-Signal-Chart と同等

- **Tab: Backtest**  
  - symbol / timeframe / period / strategy / params 入力  
  - equity curve / メトリクス表示

- **Tab: Optimization**  
  - パラメータ範囲を指定してグリッドサーチ / ベイズ探索  
  - ランキングテーブル + ヒートマップ

- **Tab: Evolution**  
  - 世代ごとのベスト戦略のスコア推移（グラフ）  
  - 現在のベスト戦略のルール表示（人間が読める形）

- **Tab: LLM Advisor**  
  - 直近 N 本の実験結果を LLM に要約させる  
  - 「次世代の実験プラン」を自然言語で提案させる

---

## 5. 開発フェーズ（7日間 MVP プラン）

### Day 1–2: Backtest Engine MVP

- `backtest.py` 作成
- `run_backtest()` 実装（MAクロスのみ）
- `POST /backtest` 実装
- Streamlit Backtestタブでグラフ表示

### Day 3–4: Auto-Optimization v1

- グリッドサーチ実装（MAクロス）
- `POST /optimize` 追加
- ランキングテーブル + ベストパラメータを可視化

### Day 5: Strategy Blocks v1

- `blocks.py` で Indicator / Condition / Logic をクラス化
- 既存の MAクロス戦略をブロックで書き直す

### Day 6: Evolution Engine v1

- ランダム戦略生成
- 簡易 GA（selection + mutation のみ）で進化テスト

### Day 7: LLM Advisor v1

- 実験ログ（JSON / CSV）を LLM に投げる
- 「良かった戦略の共通点」「次に試すべき案」を出させる

---

## 6. 2〜3ヶ月後のフルバージョン像

- 1分足〜日足のマルチタイムフレーム対応
- 日本株 / 米国株 / ETF など複数マーケット対応
- ポートフォリオ最適化（Markowitz など）を追加
- レジーム判定（トレンド / ボラティリティ / VIX）と連動
- 戦略クラスタリングと「戦略マップ」の可視化
- 「EXITON Quant Lab」としてブランド化

---

## 7. この Blueprint の位置づけ

- `00ROADMAP.md`：プロジェクト全体の道筋  
- `01architecture.md`：システム構造の詳細  
- `02API_SPEC.md`：外部から叩くインタフェース  
- `03AutoStrategyLab_Blueprint.md`（本書）：  
  - 戦略自動設計レイヤーの詳細設計

**Auto-Strategy Lab は、「完全自動投資システム」の前に作るべき “AI研究レイヤー”**。  
このレイヤーがあることで、運用システムに「育ちの良い戦略」だけを流し込めるようになる。
