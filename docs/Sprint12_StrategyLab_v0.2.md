# Sprint 12: Strategy Lab v0.2 (MA Cross Backtest Integration)

## 目的
Strategy Lab の "MA Cross" テンプレートをバックエンドの `/simulate` エンドポイントと連携させ、実際のバックテストを実行・可視化できるようにする。

## 実装内容
- **Strategy Lab UI 更新 (`dev_dashboard.py`)**:
  - **共通入力フォーム**: Symbol, Timeframe, Start Date, End Date, Initial Capital, Commission Rate を追加。
  - **MA Cross 連携**:
    - "MA Cross" 選択時に「Run Strategy Analysis」をクリックすると、バックエンド API を呼び出すロジックを実装。
    - レスポンス（Metrics, Equity Curve, Trades）を可視化。
    - 取引履歴の CSV ダウンロード機能を追加。
  - **他戦略のプレースホルダー**:
    - "RSI Reversal", "Breakout" は v0.3 以降での実装予定であることを明記。

## Backtest Lab との関係
- **Backtest Lab**: 汎用的なバックテスト実行環境（JSONパラメータを直接扱うイメージに近い）。
- **Strategy Lab**: 特定の戦略テンプレートに基づき、パラメータを調整して分析するための環境。
- 両者はバックエンドの同じ `/simulate` エンドポイントを利用しているが、UI の目的が異なる。

## 今後の拡張案
- **v0.3**: RSI Reversal, Breakout のバックエンド連携。
- **v0.4**: パラメータ最適化（グリッドサーチ）機能の統合。
