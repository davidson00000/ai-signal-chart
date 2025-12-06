# EMA9 Dip Buy 手動テストチェックリスト

## 前提条件
- [ ] Backend 起動確認: `http://localhost:8001/docs` が開ける
- [ ] Frontend 起動確認: `http://localhost:8505` が開ける

## テスト1: Quick Presetsの動作確認

### 手順
1. `http://localhost:8505` を開く
2. Mode: **Strategy Lab** を選択
3. Strategy Template: **EMA9 Dip Buy** を選択
4. Quick Presets セクションまでスクロール

### テストケース
- [ ] **Apply Conservative (慎重)** をクリック
  - [ ] Deviation Threshold % が **1.5** に変わる
  - [ ] Stop Loss Buffer % が **0.3** に変わる
  - [ ] Risk/Reward Ratio が **1.5** に変わる
  - [ ] Volume Lookback が **20** に変わる
  
- [ ] **Apply Balanced (バランス)** をクリック
  - [ ] Deviation Threshold % が **2.0** に変わる
  - [ ] Stop Loss Buffer % が **0.5** に変わる
  - [ ] Risk/Reward Ratio が **2.0** に変わる
  - [ ] Volume Lookback が **20** に変わる

- [ ] **Apply Aggressive (積極)** をクリック
  - [ ] Deviation Threshold % が **3.0** に変わる
  - [ ] Stop Loss Buffer % が **0.8** に変わる
  - [ ] Risk/Reward Ratio が **2.5** に変わる
  - [ ] Volume Lookback が **15** に変わる

- [ ] 各プリセット適用後、**🚀 Run Strategy Analysis** をクリックして成功することを確認

## テスト2: Grid Search Optimizerの動作確認

### 手順
1. Strategy Lab で EMA9 Dip Buy を選択
2. **Parameter Optimization** タブをクリック

### 範囲設定
以下の値を入力（小さい範囲でテスト）:
- **Deviation Threshold %**:
  - Min: `1.0`
  - Max: `1.5`
  - Step: `0.5`
- **Risk/Reward Ratio**:
  - Min: `1.5`
  - Max: `2.0`
  - Step: `0.5`
- **Stop Loss Buffer %**:
  - Min: `0.3`
  - Max: `0.5`
  - Step: `0.1`
- **Volume Lookback**:
  - Min: `15`
  - Max: `20`
  - Step: `5`

### 実行確認
- [ ] Total combinations が **16** と表示される (2×2×3×2 = 24... 調整必要)
  - もし 400 を超える場合は範囲を調整
- [ ] **🔍 Run Optimization** ボタンが有効（disabled でない）
- [ ] ボタンをクリック
- [ ] **"Running EMA9 Optimization..."** スピナーが表示される
- [ ] **"Optimization Completed! Tested XX combinations."** メッセージが表示される
- [ ] エラーメッセージが表示されない

## テスト3: 最適化結果の表示確認

### Parameter Optimization Results セクション（ページ下部）

- [ ] **Best Parameters (Filtered)** が表示される
  - [ ] Deviation % の値
  - [ ] Risk/Reward の値
  - [ ] Total Return % の値
  - [ ] Score の値

- [ ] **Heatmap (Deviation vs Risk/Reward)** が表示される
  - [ ] カラースケールが適切
  - [ ] ツールチップが動作

- [ ] **Top Results** テーブルが表示される
  - [ ] deviation_threshold, risk_reward, stop_buffer, lookback_volume 列
  - [ ] score, return_pct, sharpe_ratio, max_drawdown, trade_count 列
  - [ ] データが正しくソートされている（score 降順）

- [ ] **Apply to Single Run** ボタンをクリック
  - [ ] Single Run タブのパラメータが Best Parameters の値に更新される

## テスト4: Single Run の動作確認

- [ ] **Single Run** タブに戻る
- [ ] **🚀 Run Strategy Analysis** をクリック
- [ ] **"Analysis Completed!"** メッセージが表示される
- [ ] チャートが表示される:
  - [ ] Price & Trade Signals
  - [ ] Equity Curve
  - [ ] Trades テーブル
- [ ] メトリクスが表示される:
  - [ ] Total Return
  - [ ] Win Rate
  - [ ] Max Drawdown
  - [ ] Trades

## 問題が発生した場合

### バックエンドログの確認
```bash
cd /Users/kousukenakamura/dev/ai-signal-chart
tail -f backend.log
```

### Streamlit ログの確認
```bash
tail -f streamlit.log
```

### ブラウザの DevTools Console を確認
- F12 → Console タブ
- エラーメッセージを確認

## 期待される結果
- ✅ すべてのチェックボックスにチェックが入る
- ✅ エラーメッセージが一切表示されない
- ✅ MA Cross と同等の UX で動作する
