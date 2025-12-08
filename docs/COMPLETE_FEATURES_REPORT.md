# 🎯 EXITON AI Signal Chart - 完全機能レポート

**最終更新**: 2025-12-07  
**バージョン**: v2.0 (Explainability + Mobile Live View完成)

---

## 📱 昨日・今日の実装内容（2025-12-06～07）

### 🧠 **1. Explainability Layer（シグナル根拠の可視化）**

**実装日**: 2025-12-06

#### 機能概要
全てのトレーディングシグナルに「なぜそのシグナルが出たのか」を説明する情報を付与。

#### 主要機能
- **指標値の表示**: short_ma, long_ma, rsi, macd, close など
- **発火条件の説明**: "Short MA > Long MA", "RSI < 30 (oversold)" など
- **信頼度スコア**: 0.0～1.0の範囲で、シグナルの強さを数値化

#### 実装内容
1. **バックエンド**:
   - `StrategyBase.explain()` メソッド追加
   - 3戦略に実装: MA Cross, RSI Mean Reversion, MACD Trend
   - `/simulate` エンドポイントにExplain情報統合

2. **フロントエンド**:
   - `SignalDetail.tsx` コンポーネント作成（React）
   - シグナルクリックでポップアップ表示
   - 信頼度ゲージ（プログレスバー）
   - インジケーター値の整形表示

3. **テスト**:
   - **10個の自動テスト**作成（全てPASS）
   - Indicator Integrity（指標値の正確性）
   - Rule Trigger Consistency（条件の整合性）
   - Confidence Score Coherence（信頼度の一貫性）
   - `pytest` で実行、0.5秒で完了

4. **ドキュメント**:
   - `docs/tests/explainability_test_plan.md` - 詳細テスト計画
   - `docs/tests/SUMMARY.md` - クイックリファレンス
   - `docs/tests/README.md` - テストガイド

#### コード例
```json
{
  "symbol": "AAPL",
  "type": "BUY",
  "price": 189.50,
  "explain": {
    "indicators": {
      "short_ma": 189.1,
      "long_ma": 185.3,
      "rsi": 28.4,
      "macd_hist": 0.12
    },
    "conditions_triggered": [
      "Short MA (9) > Long MA (21)",
      "RSI < 30 (oversold)",
      "MACD histogram positive"
    ],
    "confidence": 0.78
  }
}
```

---

### 📱 **2. Mobile Live View（スマホ向けライブシグナル閲覧）**

**実装日**: 2025-12-07

#### 機能概要
スマートフォンから快適にLive Signalを閲覧できる専用UI。

#### 主要機能
- **モバイル最適化UI**: スマホでの閲覧に特化したレイアウト
- **自動更新**: 60秒ごとに自動的にシグナルを更新
- **複数銘柄対応**: カンマ区切りで複数銘柄を一括表示
- **3戦略対応**: MA Cross, RSI Mean Reversion, MACD Trend
- **Explainability統合**: 各シグナルの詳細情報を表示

#### 実装内容
1. **バックエンドAPI**:
   - `GET /live-signals` エンドポイント追加
   - パラメータ: `symbols`, `strategy`
   - レスポンス: Explain情報付きシグナルのリスト

2. **フロントエンドUI** (`mobile_live_view.py`):
   - **Streamlit**ベースのモバイルアプリ
   - シグナルカード表示（BUY=緑、SELL=赤）
   - 信頼度メーター（カラフルなプログレスバー）
   - 詳細情報展開（指標値・条件・信頼度）
   - サイドバー設定（銘柄・戦略選択）

3. **外部アクセス対応**:
   - Cloudflare Tunnel設定ガイド作成
   - セキュリティガイドライン（認証推奨）
   - 7ステップの設定手順書

4. **ドキュメント**:
   - `docs/mobile_live_view.md` - 使い方ガイド
   - `docs/deploy/cloudflare_tunnel_mobile_view.md` - Tunnel設定
   - `docs/deploy/mobile_live_view_summary.md` - 技術サマリ
   - `docs/deploy/IMPLEMENTATION_REPORT.md` - 実装レポート

5. **ユーティリティ**:
   - `start_exiton_mobile.sh` - ワンコマンド起動スクリプト

#### アクセス方法
```bash
# ローカル（PC）
http://localhost:8502

# 同一Wi-Fi（スマホ）
http://<Mac-miniのIP>:8502

# 外部アクセス（Cloudflare Tunnel経由）
https://signal.example.com
```

---

## 🏗️ システム全体アーキテクチャ

### バックエンド（FastAPI）

#### 主要エンドポイント（15+）

| カテゴリ | エンドポイント | 機能 |
|---------|---------------|------|
| **基本** | `GET /health` | ヘルスチェック |
| | `GET /api/chart-data` | ローソク足データ |
| **シグナル** | `GET /signal` | トレードシグナル生成 |
| | `GET /live/signal` | Live Signal（単一戦略） |
| | `GET /live-signals` | Live Signal（複数銘柄）★NEW |
| **バックテスト** | `POST /simulate` | バックテストシミュレーション |
| | `POST /optimize` | パラメータ最適化 |
| **戦略管理** | `GET /live-strategy` | Live戦略設定取得 |
| | `POST /live-strategy` | Live戦略設定保存 |
| **紙トレード** | `POST /paper-order` | 紙トレード注文 |
| | `GET /positions` | ポジション一覧 |
| | `GET /trades` | トレード履歴 |
| | `GET /pnl` | 損益レポート |
| **Strategy Lab** | `POST /run-strategy-lab` | 戦略検証バッチ |
| | `POST /auto-sim/sweep-start` | パラメータスイープ |
| | `GET /auto-sim/sweep-progress` | スイープ進捗 |
| | `GET /auto-sim/sweep-result` | スイープ結果 |

#### 戦略エンジン（25種類）

**実装済み戦略**:
1. `ma_cross` - MA Crossover ★Explain対応
2. `ema_cross` - EMA Crossover
3. `rsi_mean_reversion` - RSI Mean Reversion ★Explain対応
4. `macd_trend` - MACD Trend ★Explain対応
5. `bollinger_mean_reversion` - Bollinger Mean Reversion
6. `bollinger_breakout` - Bollinger Breakout
7. `donchian_breakout` - Donchian Channel Breakout
8. `stoch_oscillator` - Stochastic Oscillator
9. `roc_momentum` - Rate of Change Momentum
10. `atr_trailing_ma` - ATR Trailing Stop with MA
11. `ema9_dip_buy` - EMA9 Dip Buying
12. `buy_and_hold` - Buy and Hold
13. その他12戦略...

**戦略の特徴**:
- ✅ 統一されたインターフェース（`StrategyBase`）
- ✅ パラメータスキーマ自動生成
- ✅ Explainability対応（一部）
- ✅ バックテスト・リアルタイム両対応

### フロントエンド

#### 1. **開発者ダッシュボード** (`dev_dashboard.py`)

**Streamlitベースの統合管理UI**

**タブ構成**:
1. **Backtest（バックテスト）**:
   - シンボル・戦略・パラメータ設定
   - バックテスト実行
   - 結果表示（Equity Curve, Metrics, Trades）
   - Experimentとして保存

2. **Experiments（実験管理）**:
   - 過去のバックテスト履歴
   - 実験の読み込み・再実行
   - パラメータ比較

3. **Optimization（パラメータ最適化）**:
   - グリッドサーチによる最適化
   - Top N結果表示
   - パラメータのBacktestタブへの適用

4. **Strategy Lab（戦略検証ラボ）**:
   - 複数銘柄バッチ検証
   - Universe Preset（S&P500など）
   - 最適パラメータ発見

5. **Designer（JSON戦略デザイナー）**:
   - JSONで戦略を定義
   - カスタムインジケーター・ルール
   - ノーコード戦略作成

**特徴**:
- Plotlyインタラクティブチャート
- Y軸ズームスライダー
- FITボタン（自動調整）
- リアルタイムシグナル表示
- ポジション・損益管理

#### 2. **モバイルライブビュー** (`mobile_live_view.py`) ★NEW

**スマホ専用Live Signal閲覧アプリ**

**画面構成**:
- ヘッダー（タイトル・更新間隔表示）
- シグナルカード（グリッドレイアウト）
- サイドバー（設定）

**シグナルカード内容**:
- 銘柄シンボル + サイドバッジ（BUY/SELL）
- 価格 + タイムスタンプ
- 理由サマリ（短い説明）
- 信頼度メーター
- 詳細情報（展開可能）

---

## 📊 データソース

### 対応マーケット

**株式（yfinance経由）**:
- 米国株: AAPL, MSFT, GOOGL, TSLA, NVDA, META など
- 日本株: 7203.T（トヨタ）, 6758.T（ソニー）, 9984.T（ソフトバンクG）など
- その他Yahoo Finance対応の全銘柄

**暗号通貨（ccxt/Bybit経由）**:
- BTC/USDT, ETH/USDT, SOL/USDT など
- Bybit対応の全ペア

**Universe Preset**:
- `sp500_all`: S&P500全銘柄（500+）
- `sp500_top100`: 時価総額上位100
- `sp500_tech`: テクノロジーセクター
- `sp500_finance`: 金融セクター
- `sp500_healthcare`: ヘルスケアセクター

### タイムフレーム
- `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1w`

---

## 🧪 テスト・品質保証

### 自動テスト

**Explainability Tests** (`backend/tests/test_explainability.py`):
- 10テストケース（全てPASS）
- 実行時間: 0.5秒
- カバレッジ: 3戦略（MA Cross, RSI, MACD）

**テストカテゴリ**:
1. Indicator Integrity（指標値の正確性）
2. Rule Trigger Consistency（条件の整合性）
3. Confidence Score Coherence（信頼度の一貫性）
4. Integration（統合テスト）

**実行方法**:
```bash
pytest backend/tests/test_explainability.py -v
```

### ドキュメント

**包括的なドキュメント体系**:
```
docs/
├── tests/                          # テスト関連
│   ├── explainability_test_plan.md # Explainabilityテスト計画
│   ├── SUMMARY.md                  # テストサマリ
│   └── README.md                   # テストガイド
├── deploy/                         # デプロイ関連
│   ├── cloudflare_tunnel_mobile_view.md  # Tunnel設定
│   ├── mobile_live_view_summary.md       # モバイルビュー技術サマリ
│   └── IMPLEMENTATION_REPORT.md          # 実装レポート
├── mobile_live_view.md             # モバイルビュー使い方
├── reports/                        # 検証レポート
├── notes/                          # 開発ノート
└── sprints/                        # スプリント記録
```

---

## 🚀 使い方（クイックスタート）

### 最小構成

```bash
# 1. バックエンド起動
uvicorn backend.main:app --host 0.0.0.0 --port 8001

# 2. 開発者ダッシュボード起動
streamlit run dev_dashboard.py

# 3. モバイルビュー起動（オプション）
streamlit run mobile_live_view.py
```

### ワンコマンド起動

```bash
# すべてを一度に起動
./start_exiton_mobile.sh
```

### アクセス

- **FastAPI**: http://localhost:8001
- **API Docs**: http://localhost:8001/docs
- **開発者ダッシュボード**: http://localhost:8501
- **モバイルビュー**: http://localhost:8502

---

## 💡 主な用途・ユースケース

### 1. **戦略開発・検証**
- 新しいテクニカル戦略のアイデアを実装
- 過去データでバックテスト
- パラメータ最適化
- 複数銘柄で検証（Strategy Lab）

### 2. **紙トレード**
- 実際の市場価格でシミュレーション
- リアルタイム損益追跡
- トレード履歴管理
- ポートフォリオ管理

### 3. **Live Signal監視**
- リアルタイムシグナル生成
- スマホから外出先で確認（Mobile Live View）
- Explainabilityで根拠を確認
- 複数銘柄同時監視

### 4. **研究・分析**
- 複数戦略のパフォーマンス比較
- セクター別パフォーマンス分析
- Universe Presetで大規模検証
- JSON Designerでカスタム戦略作成

---

## 🔒 セキュリティ

### 現状
⚠️ **Mobile Live Viewには認証機能なし**

### 推奨対策
1. **ローカルネットワーク内のみ使用**
2. **Cloudflare Access**（無料でメール認証）
3. **Streamlit認証**（コード実装）
4. **IP制限**（Cloudflare Firewall Rules）

詳細: `docs/deploy/cloudflare_tunnel_mobile_view.md`

---

## 📈 パフォーマンス指標

### システムメトリクス
- **戦略数**: 25種類
- **対応銘柄数**: 無制限（データソース依存）
- **バックテスト速度**: 100銘柄×3年間 ≈ 数秒
- **テスト実行時間**: 0.5秒（10テスト）
- **自動更新間隔**: 60秒（モバイルビュー）

### コード規模
- **バックエンド**: 約15,000行
- **フロントエンド**: 約5,000行
- **テスト**: 約500行
- **ドキュメント**: 約10,000行

---

## 🎯 今後の拡張案

### Phase 1（短期）
- [ ] 認証機能追加（Mobile Live View）
- [ ] 残り22戦略へのExplainability実装
- [ ] E2Eテスト（Playwright）
- [ ] CI/CD統合（GitHub Actions）

### Phase 2（中期）
- [ ] Push通知（Pushbullet/LINE）
- [ ] ポートフォリオ管理機能
- [ ] パフォーマンストラッキング
- [ ] シグナル履歴・アーカイブ

### Phase 3（長期）
- [ ] 実ブローカーAPI接続
- [ ] 自動執行エンジン
- [ ] ML戦略の統合
- [ ] マルチアセットポートフォリオ

---

## 📦 技術スタック

### Backend
- **FastAPI** - REST API フレームワーク
- **Pydantic** - データバリデーション
- **pandas** - データ処理
- **NumPy** - 数値計算
- **ccxt** - 暗号通貨データ
- **yfinance** - 株式データ
- **pytest** - テスティング

### Frontend
- **Streamlit** - ダッシュボードUI
- **Plotly** - インタラクティブチャート
- **React** - SignalDetailコンポーネント（TypeScript）

### DevOps
- **Git** - バージョン管理
- **Cloudflare Tunnel** - 外部アクセス（オプション）
- **pytest** - 自動テスト

---

## 📝 まとめ

### 🎉 達成したこと（2日間）

**2025-12-06（昨日）**:
✅ **Explainability Layer完成**
- 3戦略にexplain()実装
- 10自動テスト（全PASS）
- フロントエンドUI（SignalDetail.tsx）
- 包括的ドキュメント

**2025-12-07（今日）**:
✅ **Mobile Live View完成**
- `/live-signals` API追加
- Streamlitモバイルアプリ
- Cloudflare Tunnel設定ガイド
- ワンコマンド起動スクリプト

### 💪 システムの強み

1. **包括性**: バックテスト → 紙トレード → Live Signal まで一貫
2. **拡張性**: 25戦略、簡単に追加可能
3. **透明性**: Explainabilityで根拠が明確
4. **モバイル対応**: スマホから快適に閲覧
5. **品質保証**: 自動テストでリグレッション防止
6. **ドキュメント**: 完全なガイド・手順書

### 🚀 現在の能力

このシステムを使えば：
- ✅ 新しい戦略のアイデアを数分で検証できる
- ✅ 100銘柄以上を一度にスキャンできる
- ✅ スマホから外出先でシグナルを確認できる
- ✅ シグナルの根拠を明確に理解できる
- ✅ 紙トレードで実践経験を積める

---

**開発者**: Kosuke Nakamura  
**プロジェクト**: EXITON AI Signal Chart  
**ステータス**: Production Ready（本番利用可能）  
**最終更新**: 2025-12-07 17:40 JST
