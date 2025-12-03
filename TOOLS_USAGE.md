# Tools Usage Guide

## 📈 `up_move_stats`

`up_move_stats` は指定した銘柄リストに対して、過去 N 日間の株価データを取得し、**+1% / +5% / +10%** の上昇日数を集計する CLI ツールです。

### 実行コマンド
```bash
python -m tools.up_move_stats.up_move_stats \
    --symbols_file <path/to/symbols.csv> \
    --lookback_days <days> \
    --output <output.csv> \
    [--fx_rate <rate>]
```

| オプション | 型 | デフォルト | 説明 |
|------------|----|------------|------|
| `--symbols_file` | `str` | `tools/symbols_universe.csv` | 銘柄リスト（CSV）。CSV は少なくとも `symbol` 列を持つ必要があります。 |
| `--symbols` | `str` | *なし* | カンマ区切りで銘柄コードを直接指定（このオプションがあると `--symbols_file` は無視されます）。 |
| `--lookback_days` | `int` | `365` | 今日からさかのぼる日数。期間は「今日」から `lookback_days` 前までです。 |
| `--output` | `str` | `up_move_stats_result.csv` | 結果を書き出す CSV ファイル名。 |
| `--fx_rate` | `float` | `150.0` | USD→JPY の為替レート。`<= 0` にすると JPY 換算は行いません。 |

### 主な出力列（CSV）
| 列名 | 内容 |
|------|------|
| `symbol` | 銘柄コード |
| `days_total` | 集計対象日数 |
| `up_1pct_days` | +1% 以上の上昇があった日数 |
| `up_5pct_days` | +5% 以上の上昇があった日数 |
| `up_10pct_days` | +10% 以上の上昇があった日数 |
| `start_date` | 集計開始日 |
| `end_date` | 集計終了日 |
| `last_price_usd` | 期間最終日の終値（USD） |
| `min_invest_jpy` | `fx_rate` が正の場合、最終日の価格を JPY に換算した金額 |

### 使用例
#### デフォルト銘柄で 1 年分を集計
```bash
python -m tools.up_move_stats.up_move_stats \
    --output sp500_default_2025.csv
```

#### S&P500 銘柄リストで 180 日分を集計し、結果を `sp500_180d.csv` に保存
```bash
python -m tools.up_move_stats.up_move_stats \
    --symbols_file ./tools/symbols_universe_sp500.csv \
    --lookback_days 180 \
    --output sp500_180d.csv
```

#### 銘柄を直接列挙して実行（カンマ区切り）
```bash
python -m tools.up_move_stats.up_move_stats \
    --symbols AAPL,MSFT,GOOGL \
    --lookback_days 365 \
    --output my_symbols_2025.csv
```

#### JPY 換算を無効にしたい場合
```bash
python -m tools.up_move_stats.up_move_stats \
    --symbols_file ./tools/symbols_universe.csv \
    --fx_rate 0 \
    --output stats_no_jpy.csv
```

---

## 🛠️ その他のツール（参考）

現在 `tools/` ディレクトリに実装されている主要ツールは **`up_move_stats`** のみです。将来的に新しいツールが追加された場合は、このドキュメントに追記してください。

---

*このファイルはプロジェクトのルートに配置されています。*
