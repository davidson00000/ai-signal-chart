# Strategies - 売買戦略仕様

このフォルダには各売買戦略の詳細仕様を格納しています。

## 📄 ファイル一覧

| ファイル | 戦略名 | 説明 |
|---------|--------|------|
| `ma_crossover.md` | MA Crossover | 移動平均クロスオーバー戦略 |
| `ema9_dip_buy.md` | EMA9 Dip Buy | EMA9からの押し目買い戦略 |
| `rsi_reversal.md` | RSI Reversal | RSIミーンリバージョン戦略 |
| `template.md` | テンプレート | 新規戦略追加用テンプレート |

## 📌 新規戦略の追加方法

1. `template.md` をコピー
2. 戦略名に合わせてリネーム
3. 各セクションを記述
4. Backend の `strategies/` にも実装を追加

---

*各戦略は Strategy Lab および Auto Sim Lab でテスト可能です。*
