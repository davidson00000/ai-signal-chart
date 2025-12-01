# Sprint 11: Strategy Lab v0.1 (Multi-Strategy UI Base)

## 目的
Strategy Lab の UI 基盤を構築し、複数の戦略テンプレート（MA Cross, RSI Reversal, Breakout）を選択してパラメータを入力できるインターフェースを提供する。
※ v0.1 ではバックエンド実行機能は未実装。

## UI 仕様
- **Strategy Lab** ページを追加（`dev_dashboard.py`）
- **戦略テンプレート選択**:
  - MA Cross
  - RSI Reversal
  - Breakout
- **動的パラメータフォーム**:
  - 選択した戦略に応じて入力項目が切り替わる。
- **実行ボタン**:
  - クリックすると入力されたパラメータを表示し、実行機能が未実装であることを通知する。

## ファイル変更点
- `dev_dashboard.py`: `render_strategy_lab` 関数を更新。

## 今後のバージョン計画
- **v0.2**: バックエンド API と連携し、単一戦略のバックテストを実行可能にする。
- **v0.3**: パラメータ最適化（グリッドサーチ）機能の UI 統合。
