# Sprint 14: Strategy Library v0.1

## 目的
Strategy Lab で見つけた有望な MA Cross 戦略パラメータを「名前付きの戦略」として保存し、後で再利用（ロード）できるようにする。これにより、パラメータのメモ書きや手動入力の手間を省き、バックテストの効率化を図る。

## 実装内容 (Option A: Local JSON)

### 1. データ保存 (`data/strategies.json`)
- ローカルの JSON ファイルを使用して戦略データを管理。
- **保存データ**:
  - `id`: UUID
  - `name`: 戦略名（ユーザー入力）
  - `description`: 説明（ユーザー入力）
  - `created_at`: 作成日時
  - `symbol`, `timeframe`: 銘柄と時間足
  - `strategy_type`: "ma_cross"
  - `params`: { "short_window", "long_window" }
  - `metrics`: { "return_pct", "sharpe_ratio", ... }

### 2. Strategy Lab UI (`dev_dashboard.py`)
- **Save Section**:
  - 最適化実行後、Best Parameters を保存するためのフォームを追加。
  - 戦略名と説明を入力して保存可能。
- **Saved Strategies List**:
  - 保存された戦略を一覧表示（テーブル形式）。
  - 各戦略の主要パラメータ（Short/Long）とパフォーマンス（Return）を確認可能。
- **Load Function**:
  - 一覧から戦略を選択して "Load" ボタンを押すと、そのパラメータがシステムにロードされる。
- **State Persistence**:
  - Grid Search 結果を `st.session_state` に保持し、リロード（Save時など）しても結果が消えないように修正。

### 3. Backtest Lab 連携
- 戦略をロードした状態で Backtest Lab に移動すると、サイドバーの入力フォーム（Symbol, Short, Long）にロードした戦略の値が自動的にセットされる。

## 実行方法
1. `streamlit run dev_dashboard.py`
2. **Strategy Lab** > **MA Cross** > **Parameter Optimization**
3. 最適化を実行し、結果が表示されたら画面下部の **Save to Strategy Library** セクションへ。
4. 名前を入力して **Save Strategy** をクリック。
5. **Saved Strategies** セクションに保存された戦略が表示されることを確認。
6. 戦略を選択して **Load Strategy** をクリック。
7. **Backtest Lab** タブへ移動し、パラメータが反映されていることを確認。

## 今後の拡張案
- データベース（SQLite/PostgreSQL）への移行。
- 複数戦略タイプ（RSI, Breakout）への対応。
- 戦略の編集・削除機能。
- REST API 化。
