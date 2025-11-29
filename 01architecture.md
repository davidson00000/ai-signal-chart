# AI 自動投資システム アーキテクチャ設計書  
**File:** `architecture.md`  
**Author:** Kousuke × ChatGPT（AI PM / Architect Mode）  

このドキュメントは、AI-Signal-Chart を中核とした **個人向け AI 自動投資システム** の  
アーキテクチャ（構造・データフロー・将来拡張方針）をまとめたものです。

---

## 1. ゴールと前提

### ゴール
- 株価データを取得し、AIやルールベースの戦略でシグナルを生成し、  
  ペーパートレード → 将来は現物トレードまで行える **堅牢なシステム** を作る。
- 個人開発者（こうすけ）が **1人でも運用・改修し続けられるシンプルさ** を保つ。

### 前提キーワード
- **AI-Signal-Chart**：チャートとシグナル表示を行うフロントエンドUI
- **Phase 概念**：  
  - Phase 1：Streamlit + ペーパートレード  
  - Phase 2：FastAPI API 化  
  - Phase 3：SvelteKit + Webhook  
  - Phase 4：証券会社API連携  

---

## 2. 全体アーキテクチャ概要

システムは次の 4 レイヤで考える：

```text
[ Data ] → [ Brain ] → [ Execution ] → [ Dashboard / Notification ]
```

- **Data**  
  - 市場データ（株価、出来高、指標）  
  - ローカルCSV / SQLite / 将来的には PostgreSQL + TimescaleDB

- **Brain（戦略ロジック）**  
  - テクニカル指標（MA, RSI, ボリンジャーなど）  
  - 機械学習モデル（LightGBM / Transformer など）  
  - リスク管理・ポジションサイズ計算

- **Execution（実行レイヤー）**  
  - 最初はペーパートレード（仮想売買）  
  - 将来的に証券会社APIによる現物注文  
  - エラーハンドリング・リトライ・安全ストップロジック

- **Dashboard / Notification**  
  - チャート表示・履歴・シグナル一覧  
  - 成績ダッシュボード（P/L, ドローダウンなど）  
  - Webhook による Discord / Slack 通知  
  - 将来的に LLM による日次レポート生成

---

## 3. 現在のアーキテクチャ（Phase 1 〜 1.5）

### 3.1 構成イメージ

```text
[ブラウザ] ──(HTTP)── [Vercel / ローカルのフロントエンド]
                      （AI-Signal-Chart：HTML + JS）

※ バックエンド（Python）は別プロジェクトとして存在し、
   今後 FastAPI 化してフロントと連携する予定。
```

- フロントエンド
  - `frontend/index.html`（もしくは相当する React/JS ファイル）
  - 現状：AAPL のチャート表示は成功
  - 課題：MA（移動平均）の表示がエラーで落ちている

- バックエンド
  - `backend/main.py`（Python）
  - 現状：最小限の構成のみ
  - 今後：テクニカル計算やシグナル生成・APIとしての公開を担当

---

## 4. 目標アーキテクチャ（Phase 2 以降）

### 4.1 論理構成図

```text
               ┌────────────────────────────┐
               │         Dashboard          │
               │  - SvelteKit / Streamlit   │
               │  - チャート / PnL / 履歴      │
               └──────────┬─────────────────┘
                          │ HTTP (REST API)
               ┌──────────▼─────────────────┐
               │         Backend            │
               │        (FastAPI)           │
               │                            │
               │  /chart-data   /signal     │
               │  /paper-order  /pnl        │
               └───────┬────────┬──────────┘
                       │        │
           ┌───────────▼─┐   ┌─▼──────────────┐
           │   Brain      │   │  Execution     │
           │ (Strategy)   │   │ (Orders)       │
           │ MA / AI etc. │   │ PaperTrade     │
           └───────┬──────┘   └──────┬────────┘
                   │                 │
          ┌────────▼────────┐       │
          │      Data        │       │
          │  (DB / CSV)      │       │
          └────────┬────────┘       │
                   │                 │
            ┌──────▼───────┐        │
            │ Market Data  │        │
            │ (API / yfin) │        │
            └──────────────┘        │
                                     │
                            ┌────────▼────────┐
                            │  Notification   │
                            │ (Discord, etc.) │
                            └─────────────────┘
```

---

## 5. モジュール別詳細

### 5.1 Data レイヤ

#### 役割
- 市場データの取得・保存・提供

#### 主なコンポーネント
- `data_feed.py`  
  - yfinance / 証券会社API から株価を取得
- データストア
  - Phase1：CSV / SQLite  
  - Phase2以降：PostgreSQL + TimescaleDB（任意）

#### データフロー（例：日次更新）

```text
Cron / 手動実行
    ↓
data_feed.py が対象銘柄一覧を読み込む
    ↓
yfinance などから日足データ取得
    ↓
データストア（CSV/DB）へ保存
```

---

### 5.2 Brain（戦略ロジック）

#### 役割
- 「買うべきか／売るべきか／何もしないか」を決める

#### コンポーネント
- `strategy.py`
  - ルールベース戦略
    - 例：短期MAと長期MAのクロス、RSI 閾値など
  - 機械学習戦略（Phase2以降）
    - LightGBM：翌日上昇確率  
    - LSTM/GRU/Transformer：トレンド検知

- `models/`
  - 学習済みモデルの保存場所（pickle, joblib, etc.）

#### 出力
- シグナルオブジェクト
  - `date`
  - `symbol`
  - `signal`（BUY / SELL / HOLD）
  - `confidence`（AIの確信度など）
  - `reason`（MAクロス / NN予測）

---

### 5.3 Execution（実行）

#### 役割
- シグナルを実際の「注文」に変換するレイヤ

#### モード
1. **ペーパートレード（Phase1〜）**
   - 注文をDB/CSVに記録するだけ
   - 損益は実際の価格から計算

2. **セミオート（Phase4前半）**
   - シグナルと注文候補をUIに表示
   - ユーザーの「実行」ボタンクリックで本物の注文送信

3. **フルオート（最後）**
   - 条件付きで自動的にAPI注文

#### コンポーネント
- `paper_trade.py`
  - シグナルから仮想注文を生成
  - 約定価格・数量を記録
- `broker_api.py`（将来）
  - 証券会社APIとのインターフェース

---

### 5.4 Dashboard / Notification

#### 役割
- チャート・成績・シグナルの可視化
- イベント（約定・エラー・日報）通知

#### フロントエンド
- Phase1：`ui_streamlit.py`（ローカル or 一時的なクラウド）
- Phase3：SvelteKit（Vercel）

#### 通知
- Discord / Slack Webhook  
  - 注文成功・失敗  
  - 日次レポート  
  - バックテスト完了通知  

---

## 6. データフロー例

### 6.1 日次のフロー（ペーパートレード）

```text
1. データ更新
   - data_feed.py を実行
   - 前日までの株価データを更新

2. シグナル生成
   - backend (FastAPI) の /signal を呼び出す
   - Brain が最新データからシグナルリストを生成

3. ペーパートレード
   - /paper-order にシグナルを渡す
   - Execution が仮想注文を記録
   - 損益を更新

4. ダッシュボード表示
   - フロントエンドが /chart-data /pnl を叩く
   - チャート＆P/Lグラフを表示

5. 通知
   - 重要イベントを Webhook で Discord へ発報
```

---

## 7. デプロイ構成（将来像）

### 7.1 開発環境（ローカル）

```text
[local browser] → Streamlit / SvelteKit dev server
                 FastAPI (localhost:8000)
                 DB (SQLite / local Postgres)
```

### 7.2 本番候補構成

```text
[User Browser]
    │
    │ HTTPS
    ▼
[Vercel (SvelteKit)]
    │ (REST API)
    ▼
[Backend (FastAPI on VPS / Railway)]
    │
    ├─ [DB: PostgreSQL / TimescaleDB]
    │
    ├─ [Discord Webhook]
    │
    └─ [Broker API (将来)]
```

---

## 8. 現在地点と次の一手

### 現在
- AAPL チャート表示まで成功（AI-Signal-Chart）  
- MA 表示バグが残っている  
- バックエンドはまだシンプルな構成

### 次にやるべきこと（アーキテクチャ視点での優先度）

1. **フロントとバックエンドのデータ構造を固定する**
   - 例：`{ time, open, high, low, close, volume, ma_short, ma_long }` など

2. **MA計算・シグナル生成をバックエンド側に寄せる**
   - フロントはできるだけ「表示に専念」させる

3. **ペーパートレード用の最小限 Execution 実装**
   - 約定履歴の保存
   - P/L 計算

4. **この architecture.md & ROADMAP.md をベースに、コードを段階的に整理**

---

## 9. まとめ

- このアーキテクチャは、  
  **「1人で作り続けられるシンプルさ」と「将来的な拡張性」の両立** を狙った構成です。
- Data / Brain / Execution / Dashboard の分離を徹底することで、
  - バグ調査がしやすく  
  - 機能追加も局所的に行えるようになります。

今後はこの `architecture.md` と `ROADMAP.md` を「開発の憲法」として使い、  
新しい機能を追加するたびに、必要に応じてこのドキュメントもアップデートしていきます。

[[architecture]]
