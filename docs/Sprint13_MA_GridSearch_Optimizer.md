# Sprint 13: MA Grid Search Optimizer (Strategy Lab v0.3)

## 目的
Strategy Lab にて、MA Cross 戦略の最適なパラメータ（Short/Long Window）を自動的に探索する「Grid Search Optimizer」を実装する。

## 実装内容

### Backend
- **新規エンドポイント**: `POST /optimize/ma_cross`
- **リクエストモデル**: `MACrossOptimizationRequest`
  - 探索範囲 (`short_min`, `short_max`, `short_step` 等) を指定可能。
  - 組み合わせ数が 400 を超える場合はエラーを返す安全装置付き。
- **ロジック**: 既存の `GridSearchOptimizer` を利用して総当たりバックテストを実行。

### Frontend (Strategy Lab)
- **Parameter Optimization タブ**:
  - MA Cross 戦略選択時に「Single Run」と「Parameter Optimization」のタブを表示。
  - 最適化タブでは探索範囲を指定して実行可能。
- **可視化**:
  - **Best Parameters**: 最もリターンの高かったパラメータと指標を表示（ツールチップ解説付き）。
  - **Heatmap**: Short Window vs Long Window のリターンをヒートマップで可視化（Altair使用）。
  - **Result Table**: 上位の結果をテーブル表示（フォーマット整形済み）。

## UI改善 (v0.3.1)
- **数値フォーマット**:
  - Total Return, Max Drawdown, Win Rate をパーセント表示（小数第2位）に統一。
  - Sharpe Ratio を小数第2位まで表示。
  - Total PnL をカンマ区切り表示。
- **ツールチップ**:
  - 各メトリクス（Short/Long Window, Return, Sharpe, MDD, Win Rate）にマウスオーバーで解説を表示する機能を追加。
  - **Top Results テーブル** の各カラムヘッダーにも解説ツールチップを追加。
- **テーブル整形**:
  - カラム名を直感的な名称（Short, Long, Total Return (%) 等）に変更。
  - `st.column_config` を使用して数値フォーマットとソート機能を最適化。

## 実行方法
1. `streamlit run dev_dashboard.py`
2. サイドバーで **Strategy Lab** を選択。
3. Strategy Template で **MA Cross** を選択。
4. **Parameter Optimization** タブを選択。
5. 探索範囲を入力し、**🔍 Run Optimization** をクリック。

## 今後の拡張案
- RSI Reversal, Breakout 戦略への最適化機能の追加。
- 3D Surface Chart の導入（Plotly 依存）。
- 最適化指標の選択（Sharpe Ratio, Max Drawdown 等）。
