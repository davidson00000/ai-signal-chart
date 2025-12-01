# 上昇変動統計ツール (Up Move Stats Tool)

このツールは、指定された銘柄リストについて、過去1年間（または指定期間）に大幅なプラスリターン（+1%、+5%、+10%）を記録した日数を計算します。

## 目的
AI自動投資システムの「銘柄選定ステップ」において、ボラティリティが高い銘柄やモメンタムの強い銘柄を特定するために使用されます。

## インストール
必要な依存関係がインストールされていることを確認してください：
```bash
pip install yfinance pandas
```

## 使用方法
プロジェクトのルートディレクトリからツールを実行します：

```bash
python -m tools.up_move_stats.up_move_stats
```

### 引数
- `--symbols_file`: 銘柄が含まれるCSVファイルのパス（デフォルト: `tools/symbols_universe.csv`）
- `--symbols`: ファイルを上書きして指定するカンマ区切りの銘柄リスト（例: `AAPL,MSFT`）
- `--lookback_days`: 過去に遡る日数（デフォルト: `365`）
- `--output`: 出力CSVファイルのパス（デフォルト: `up_move_stats_result.csv`）

### 使用例

**デフォルト設定で実行:**
```bash
python -m tools.up_move_stats.up_move_stats
```

**特定の銘柄を指定して実行:**
```bash
python -m tools.up_move_stats.up_move_stats --symbols AAPL,TSLA,NVDA
```

**期間と出力ファイルを指定して実行:**
```bash
python -m tools.up_move_stats.up_move_stats --lookback_days 180 --output my_stats.csv
```

## 出力CSVフォーマット
出力されるCSVには以下の列が含まれます：
- `symbol`: 銘柄コード（ティッカー）
- `days_total`: 期間中の総取引日数
- `up_1pct_days`: リターンが +1% 以上だった日数
- `up_5pct_days`: リターンが +5% 以上だった日数
- `up_10pct_days`: リターンが +10% 以上だった日数
- `start_date`: データの開始日
- `end_date`: データの終了日
