# Sprint 15: Symbol Preset Manager v0.1

## 目的
Streamlit UI 上から Symbol プリセット (AAPL, MSFT, BTC-USD など) の追加・削除を行い、
Developer Dashboard / Backtest Lab / Strategy Lab の3つのモード全てで共通して利用できるようにする。

## 実装内容

### 1. データ永続化
- **`data/symbol_presets.json`**: シンボルプリセットを JSON 形式で保存
- 構造: `{"symbols": [{"symbol": "AAPL", "label": "Apple"}, ...]}`
- 初期値として8つのシンボルを設定（AAPL, MSFT, TSLA, GOOGL, AMZN, SPY, BTC-USD, USDJPY=X）

### 2. ヘルパー関数
- `load_symbol_presets()`: JSON ファイルからプリセットを読み込み
- `save_symbol_presets(symbols)`: プリセットを JSON ファイルに保存

### 3. UI 統合
- `render_symbol_selector`: プリセット読み込みに対応（JSON から動的にロード）
- Developer Dashboard / Backtest Lab / Strategy Lab 全てで同じプリセットを使用
- "Custom..." オプションで任意シンボルの入力も可能

### 4. Symbol Preset Settings UI
Strategy Lab の下部に開発者向け設定セクションを追加:
- **Current Presets**: 現在のプリセット一覧を表示
- **Add New Preset**: 新しいシンボルとラベルを追加
- **Delete Preset**: 既存プリセットを削除

## 実行方法
1. `streamlit run dev_dashboard.py`
2. Strategy Lab を開く
3. ページ下部の「⚙️ Symbol Preset Settings」を展開
4. 新しいシンボル（例: NVDA）とラベル（例: NVIDIA）を入力して「Add Preset」
5. Developer Dashboard / Backtest Lab / Strategy Lab の Symbol 選択肢に反映されていることを確認

## 技術的詳細
- **永続化**: ローカル JSON ファイル (`data/symbol_presets.json`)
- **状態管理**: `st.session_state` で選択状態を保持
- **再描画**: 追加・削除後は `st.rerun()` で画面を更新

## 今後の拡張案
- プリセットの編集機能
- シンボルの並び替え
- カテゴリー分類（株式、暗号通貨、FX など）
- バックエンド API 化
