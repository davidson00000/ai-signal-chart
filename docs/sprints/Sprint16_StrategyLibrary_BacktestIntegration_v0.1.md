# Sprint 16: Strategy Library → Backtest Lab Integration v0.1

## 目的
Strategy Lab で保存した戦略を Backtest Lab から直接ロードし、
パラメータを自動セットして簡単に再テストできるようにする。

## 実装内容

### 1. Strategy Library 既存機能の確認
- **クラス**: `StrategyLibrary` (既存)
- **メソッド**: `load_strategies()` で保存された全戦略を取得
- **データ構造**:
  ```json
  {
    "id": "uuid",
    "name": "戦略名",
    "symbol": "AAPL",
    "timeframe": "1d",
    "strategy_type": "ma_cross",
    "params": {
      "short_window": 13,
      "long_window": 40
    },
    "metrics": {
      "return_pct": 1625.54,
      "sharpe_ratio": 0.99,
      ...
    }
  }
  ```

### 2. Backtest Lab への統合
Backtest Lab のメインエリアに「Load from Strategy Library」セクションを追加:

**UI コンポーネント**:
- **Strategy 選択**: `st.selectbox` で保存済み戦略を一覧表示
- **表示形式**: `{name} | {symbol} {timeframe} | MA({short},{long}) | Return: {return_pct}%`
- **Load Parameters ボタン**: クリックで選択戦略のパラメータを Backtest Lab にロード

**自動セット項目**:
- Symbol (`shared_symbol_preset` 経由)
- Short Window (session state: `bt_short_window`)
- Long Window (session state: `bt_long_window`)

**動作**:
1. 戦略をセレクトボックスから選択
2. "📂 Load Parameters" ボタンをクリック
3. Backtest Lab のサイドバーパラメータが自動更新
4. そのまま "▶ Run Backtest" で同条件テスト実行可能

### 3. Session State の活用
Backtest Lab のパラメータ入力を session state に対応:
- `st.sidebar.number_input` に `key="bt_short_window"` 等を追加
- デフォルト値を `st.session_state.get("bt_short_window", 9)` から取得
- ロード時に session state を更新して `st.rerun()` で画面更新

## 実行方法
1. **Strategy Lab で戦略を保存**:
   - MA Cross Grid Search を実行
   - Best Parameters を Strategy Library に保存

2. **Backtest Lab で戦略をロード**:
   - Backtest Lab タブを開く
   - "📚 Load from Strategy Library" セクションを確認
   - 保存した戦略を選択
   - "📂 Load Parameters" をクリック

3. **バックテスト実行**:
   - パラメータが自動セットされたことを確認
   - "▶ Run Backtest" で実行

## 技術的詳細
- **データソース**: `data/strategies.json` (Strategy Library)
- **状態管理**: `st.session_state` で Symbol / Short / Long を保持
- **UI 配置**: Backtest Lab メインエリア (サイドバーの後)

## 今後の拡張案
- Timeframe, Start/End Date, Initial Capital の自動ロード
- 戦略フィルタリング (Symbol や Timeframe でフィルタ)
- 複数戦略の比較実行機能
- ロード履歴の保存
