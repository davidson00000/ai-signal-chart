# EXITON Strategy DSL v0.1

目的：
- 「テキストで戦略を書く → バックテストに流し込む」ための超シンプル DSL
- まずは **既存の Python 戦略クラス（例: MACrossStrategy）にパラメータを渡す** 設計から始める

実装ポリシー：
- 中身は **JSON** にする（標準ライブラリだけでパース可能）
- 将来 LLM で自動生成しやすい形

---

## 1. 基本フォーマット

拡張子: `.dsl`  
中身: JSON 1オブジェクト

```json
{
  "name": "MA Cross 9-21",
  "type": "ma_cross",
  "description": "Simple moving average cross strategy (9 / 21).",
  "params": {
    "short_window": 9,
    "long_window": 21
  },
  "meta": {
    "author": "kousuke",
    "tags": ["ma", "trend-follow", "example"]
  }
}
```

### 必須フィールド

- `name` (str)  
  戦略の名前（UIやログ表示用）

- `type` (str)  
  バックエンド側の戦略クラスを指す識別子  
  - `ma_cross` → `backend.strategies.ma_cross.MACrossStrategy`
  - 将来: `rsi`, `bbands`, `breakout`, `ai_generated` など追加予定

- `params` (object)  
  戦略クラスのコンストラクタに渡すパラメータ  
  - 例: `MACrossStrategy(short_window=9, long_window=21)`

### 任意フィールド

- `description` (str)  
  文章での説明

- `meta` (object)  
  - `author` (str)
  - `tags` (list[str])
  - `notes` (str) など自由

---

## 2. 例：MAクロス戦略

`strategy_lab/examples/ma_cross.dsl`

```json
{
  "name": "MA Cross 9-21",
  "type": "ma_cross",
  "description": "SMA 9 / 21 cross, full-in full-out.",
  "params": {
    "short_window": 9,
    "long_window": 21
  },
  "meta": {
    "author": "kousuke",
    "tags": ["ma", "trend-follow", "example"]
  }
}
```

---

## 3. DSL → Python の対応

現状（v0.1）の対応表：

| DSL `type` | Python クラス                                   | 備考         |
|------------|--------------------------------------------------|--------------|
| `ma_cross` | `backend.strategies.ma_cross.MACrossStrategy`   | 実装済み     |

将来：
- `rsi` → `RSIStrategy`
- `bbands` → `BollingerBandStrategy`
- `breakout` → `BreakoutStrategy`
- `ai_generated` → LLM から生成されたコードをロード

---

## 4. ワークフロー（構想）

1. 人間 or LLM が `.dsl` を作る
2. `strategy_lab.auto_generator.load_strategy_from_dsl(path)` で読み込み
3. 戦略インスタンス（`BaseStrategy` を継承）として取得
4. `strategy_lab.evaluator.run_backtest_from_dsl(...)` でバックテスト実行
5. `strategy_lab.optimizers` でパラメータ探索・ランキング

この v0.1 では：
- `ma_cross` 戦略 + パラメータ + BacktestEngine をつなぐところまで実装する
