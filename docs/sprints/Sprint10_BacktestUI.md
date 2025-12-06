# Sprint 10: Backtest UI v0.1 Implementation

## Goal
Streamlit ダッシュボードに「Backtest」タブを追加し、バックエンドの `/simulate` エンドポイントを利用してバックテストを実行・可視化できるようにする。

## Tasks Completed
- [x] `dev_dashboard.py` に "Backtest Lab" モードを追加
- [x] バックテスト実行用フォームの実装 (Symbol, Timeframe, Dates, Parameters)
- [x] バックエンド API (`POST /simulate`) の呼び出し処理実装
- [x] 結果の可視化 (Metrics, Equity Curve, Trades Table)
- [x] 取引履歴の CSV ダウンロード機能実装

## Files Changed
- `dev_dashboard.py`: Backtest Lab UI の追加

## How to Run

### 1. Start Backend
```bash
python -m backend.main
```
Backend will start at `http://localhost:8000`.

### 2. Start Streamlit Dashboard
```bash
streamlit run dev_dashboard.py
```
Dashboard will open in your browser (usually `http://localhost:8501`).

### 3. Run Backtest
1. Select "Backtest Lab" from the **Mode** sidebar.
2. Enter parameters (e.g., Symbol: AAPL, Timeframe: 1d).
3. Click "▶ Run Backtest".
4. View results and download CSV.

## Known Limitations
- Currently only supports "MA Cross" strategy parameters in the UI form.
- Error handling for API connection failures is basic.
- Chart data is fetched from the backend's `data_feed` which currently mocks data if not configured with a real provider.
