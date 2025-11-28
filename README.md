# 🎯 EXITON - AI Signal Chart システム

自動トレーディング戦略のバックテストと紙トレードを実現するプラットフォーム

## 📁 プロジェクト構成

```
ai-signal-chart/
├── backend/              # FastAPI バックエンド
│   ├── data_feed.py      # 価格データ取得（ccxt/yfinance）
│   ├── strategy.py       # MA cross シグナル生成
│   ├── paper_trade.py    # 紙トレードエンジン
│   ├── main.py          # FastAPI アプリ（7エンドポイント）
│   ├── models/          # Pydantic データモデル
│   └── utils/           # MA計算などのヘルパー
├── dev_dashboard.py     # Streamlit 開発者ダッシュボード
├── requirements.txt     # 依存関係
└── TASK*_REPORT.md     # 各タスクの詳細ドキュメント
```

---

## 🚀 セットアップ

### 1. 依存関係のインストール

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. バックエンドの起動

```bash
uvicorn backend.main:app --reload
```

→ FastAPI: http://localhost:8000
→ API Docs: http://localhost:8000/docs

### 3. ダッシュボードの起動

```bash
streamlit run dev_dashboard.py
```

→ Streamlit: http://localhost:8501

---

## 🎮 実装済み機能

### 1️⃣ **FastAPI バックエンド** (`backend/main.py`)

**7つのエンドポイント:**

| エンドポイント | 機能 | 例 |
|---------------|------|-----|
| `GET /health` | ヘルスチェック | システム状態確認 |
| `GET /api/chart-data` | チャートデータ取得 | ローソク足 + MA |
| `GET /signal` | トレードシグナル生成 | BUY/SELL/HOLD |
| `POST /paper-order` | 紙トレード注文 | 仮想売買実行 |
| `GET /positions` | ポジション一覧 | 保有状況 + 損益 |
| `GET /trades` | トレード履歴 | 執行履歴 |
| `GET /pnl` | 損益レポート | PnL推移 |

**主な特徴:**
- ✅ 実際の市場価格で紙トレード実行
- ✅ リアルタイム損益計算（unrealized P&L）
- ✅ JSON body + クエリパラメータ両対応
- ✅ 株式（yfinance）と暗号通貨（ccxt/bybit）対応

---

### 2️⃣ **Streamlit 開発者ダッシュボード** (`dev_dashboard.py`)

**インタラクティブな監視・操作UI:**

#### 📈 チャート機能
- **Plotlyベースのインタラクティブチャート**
  - Close価格ライン（常に表示）
  - Short MA / Long MA（トグルON/OFF可能）
  - ホバーでツールチップ表示
  
#### 🎛️ Y軸コントロール
- **Y軸ズームスライダー**
  - リアルタイムでY軸範囲調整
  - 30%ヘッドルーム（データを超えて表示可能）
  - ステップ自動計算（滑らか操作）
  
- **FITボタン**
  - ワンクリックで最適表示
  - データ + 30%余白で自動調整
  - スライダーも連動してジャンプ

#### 📊 シグナル＆注文
- 現在のシグナル表示（BUY/SELL/HOLD）
- ワンクリック注文ボタン
- 注文確認メッセージ

#### 📋 3タブ構成
1. **Positions**: 現在のポジション（unrealized P&L付き）
2. **Trades**: 注文履歴（最新が上）
3. **P&L**: 損益推移グラフ + テーブル

---

## 🏗️ アーキテクチャ（モジュール分割）

### Data Layer
- `data_feed.py`: 価格データ取得
  - `get_chart_data()`: ローソク足取得
  - `get_latest_price()`: 最新価格取得

### Brain Layer
- `strategy.py`: シグナル生成
  - MA cross 戦略実装
  - TP/SL計算

### Execution Layer
- `paper_trade.py`: 紙トレード
  - `PaperTrader`: 注文執行・ポジション管理
  - 仮想的な損益計算

### Models Layer
- `models/`: Pydantic モデル
  - `Candle`, `Signal`, `Trade`, `Position`
  - リクエスト/レスポンスモデル

### Utils Layer
- `utils/indicators.py`: テクニカル指標
  - `simple_moving_average()`: MA計算

---

## 📈 開発履歴（全9タスク完了）

| タスク | 内容 | 成果 |
|--------|------|------|
| **Task 1** | PaperTrader価格アップグレード | 実際の市場価格で紙トレード |
| **Task 2** | JSON body対応 | `/paper-order`がJSON受付 |
| **Task 3** | Streamlitダッシュボード | 監視・操作UIを実装 |
| **Task 4** | MAトグル + Y軸ズーム | Plotly移行、インタラクティブ化 |
| **Task 5** | FITボタン実装 | Y軸自動調整機能 |
| **Task 6** | Close表示バグ修正 | MAトグル影響を排除 |
| **Task 7** | スライダーヘッドルーム | 30%余白でズーム拡張 |
| **Task 8** | FIT余白対応 | FITも30%余白使用 |
| **Task 9** | FIT同期修正 | スライダーと完全連動 |

各タスクの詳細は `TASK*_REPORT.md` を参照してください。

---

## 💡 技術スタック

**バックエンド:**
- FastAPI（REST API）
- Pydantic（データ検証）
- ccxt（暗号通貨データ）
- yfinance（株式データ）

**フロントエンド:**
- Streamlit（ダッシュボードUI）
- Plotly（インタラクティブチャート）
- pandas（データ処理）

**デプロイ:**
- Vercel対応（設定済み）

---

## 📊 サポートシンボル

### 株式（yfinance）
- AAPL, MSFT, TSLA, GOOGL
- 7203.T, 6758.T（日本株）
- その他 Yahoo Finance対応銘柄

### 暗号通貨（ccxt/bybit）
- BTC/USDT, ETH/USDT
- その他 Bybit対応ペア

---

## 🎯 使い方

### 基本的な流れ

1. **バックエンド起動**
   ```bash
   uvicorn backend.main:app --reload
   ```

2. **ダッシュボード起動**
   ```bash
   streamlit run dev_dashboard.py
   ```

3. **シンボル選択**
   - サイドバーでシンボル入力（例: AAPL, BTC/USDT）
   - タイムフレーム選択（1m, 5m, 1h, 1d など）
   - Data Points 調整

4. **チャート分析**
   - MAトグルでライン表示切替
   - Y軸スライダーで詳細分析
   - FITボタンで全体表示

5. **紙トレード**
   - シグナル確認（BUY/SELL/HOLD）
   - ワンクリックで注文実行
   - Positionsタブで損益確認

---

## 🔧 API使用例

### チャートデータ取得
```bash
curl "http://localhost:8000/api/chart-data?symbol=AAPL&timeframe=1d&limit=100"
```

### シグナル取得
```bash
curl "http://localhost:8000/signal?symbol=AAPL&timeframe=1d"
```

### 紙トレード注文（JSON）
```bash
curl -X POST "http://localhost:8000/paper-order" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": 10
  }'
```

### ポジション確認
```bash
curl "http://localhost:8000/positions"
```

---

## 📝 TODO / 今後の拡張候補

- [ ] LightGBM戦略の追加
- [ ] 実ブローカーAPI接続
- [ ] Discord Webhook通知
- [ ] SvelteKitフロントエンド
- [ ] バックテストレポート機能
- [ ] 複数シンボル同時監視
- [ ] アラート機能
- [ ] CSVエクスポート

---

## 📄 ライセンス

MIT License

---

## 🤝 貢献

Issue・PRを歓迎します！

---

**開発者**: Kosuke Nakamura  
**最終更新**: 2025-11-28  
**バージョン**: v1.0 (MVP完成)
