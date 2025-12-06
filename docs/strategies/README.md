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

---

## 🧩 戦略IDの命名規則

戦略IDは次の形式で命名する：

`STR_<カテゴリ>_<3桁連番>`

### 例

- `STR_MA_001` : Moving Average Crossover Basic  
- `STR_MA_002` : MA Crossover + Filter 条件付き  
- `STR_RSI_001`: RSI Reversal Basic  

今後、新しい戦略ドキュメントを追加する場合は、

- ファイル名：人間が読んで分かる名前（例: `ma_crossover.md`）
- 戦略ID：上記ルールに従って本文中で宣言

とする。
