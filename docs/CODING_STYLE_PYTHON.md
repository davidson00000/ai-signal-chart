# CODING_STYLE_PYTHON.md
EXITON Python コーディング規約 v0.1
==================================

本ドキュメントは、EXITON プロジェクト（特に自動投資システムのバックエンド）が
Pythonで実装される際の **コーディング規約・命名規則・プロジェクト構造** を定義する。

PEP8 をベースとしつつ、EXITON 独自のルールも追加する。

---

## 1. 基本スタイル

### CS-PY-001 フォーマット
- 原則として **PEP8 準拠** とする。
- 自動整形ツールの使用を推奨：
  - `black`
  - `isort`（import 並び替え）

### CS-PY-002 インデント
- インデントはスペース4つ。
- タブは禁止。

### CS-PY-003 行長
- 1行の長さは 100 文字を上限の目安とする。
- やむを得ず長くなる場合は、適切に改行して可読性を優先する。

---

## 2. 型ヒントとドキュメント

### CS-PY-010 型ヒント必須
- すべての公開関数・メソッドには、
  - 引数・戻り値に型ヒントを付与する。

例：
```python
def calculate_position_size(capital: float, risk_per_trade: float, stop_distance: float) -> int:
    ...
```

### CS-PY-011 Docstring
- 主要な関数・クラスには docstring を付ける。
- 形式は Google Style もしくは NumPy Style のいずれかに統一する。

例（Google Style）：
```python
def calculate_position_size(capital: float, risk_per_trade: float, stop_distance: float) -> int:
    """Calculate position size in shares.

    Args:
        capital: Total account capital.
        risk_per_trade: Risk per trade as a fraction (e.g., 0.01 for 1%).
        stop_distance: Distance between entry and stop in price units.

    Returns:
        Number of shares to buy (rounded down to an integer).
    """
    ...
```

---

## 3. 命名規則

### CS-PY-020 ファイル名
- Python ファイル名は `snake_case.py` とする。
  - 例：`backtest_engine.py`, `signal_generator.py`

### CS-PY-021 クラス名
- クラス名は `PascalCase`。
  - 例：`BacktestEngine`, `SignalGenerator`

### CS-PY-022 変数・関数名
- 変数名・関数名は `snake_case`。
  - 例：`entry_price`, `generate_signals()`

### CS-PY-023 定数
- 定数は `UPPER_SNAKE_CASE`。
  - 例：`DEFAULT_RISK_PER_TRADE = 0.01`

### CS-PY-024 戦略ID・ファイル名
- 戦略実装ファイル：`strategy_<name>.py`
  - 例：`strategy_ma_crossover.py`
- 戦略クラス名：`<Name>Strategy`
  - 例：`MACrossoverStrategy`

---

## 4. プロジェクト構造（推奨）

### CS-PY-030 ディレクトリ例

```text
backend/
  api/
    routes_backtest.py
    routes_signals.py
  core/
    backtest_engine.py
    signal_generator.py
  strategies/
    strategy_ma_crossover.py
    strategy_rsi.py
  models/
    bar.py
    trade.py
    position.py
  utils/
    logging_utils.py
    date_utils.py
  tests/
    test_backtest_engine.py
    test_strategy_ma_crossover.py
```

- `core/`：ドメインロジック（バックテスト・シグナル）
- `strategies/`：各種戦略の実装
- `models/`：データ構造（ローソク足・トレード等）
- `api/`：FastAPI/Flask 等のエンドポイント
- `utils/`：汎用ユーティリティ（ただし増やしすぎない）

---

## 5. 例外処理・エラーハンドリング

### CS-PY-040 例外の扱い
- 例外を握りつぶさない。
- キャッチした例外は：
  - ログに記録し
  - 上位レイヤに適切なエラーとして返す

### CS-PY-041 カスタム例外
- 領域ごとにカスタム例外クラスを定義してもよい：
  - 例：`BacktestError`, `DataSourceError`

```python
class BacktestError(Exception):
    """Raised when a backtest fails due to invalid configuration or data."""
```

---

## 6. テストに関する規約

### CS-PY-050 pytest 利用
- テストは原則 `pytest` を用いる。
- テストファイル：`tests/test_*.py`

### CS-PY-051 テスト命名
- テスト関数名は、何を検証しているかがわかる名前にする：
  - `test_ma_crossover_generates_buy_signal_when_short_above_long()`

### CS-PY-052 回帰バグ
- バグ修正時には、必ずそのバグを再現するテストを追加する。

---

## 7. インポート・依存管理

### CS-PY-060 インポート順
- 標準ライブラリ → サードパーティ → ローカルモジュール の順に並べる。

```python
import datetime
from typing import List

import numpy as np
import pandas as pd

from backend.core.backtest_engine import BacktestEngine
```

### CS-PY-061 不要インポート
- 未使用のインポートは削除する。

---

## 8. コメント・TODO

### CS-PY-070 コメント方針
- コードそのものが意図を伝えられるように書く。
- なぜその実装にしたか、という「理由」はコメントで補足してよい。

### CS-PY-071 TODO 記法
- 未実装の箇所・改善予定箇所には `# TODO:` を付ける。
  - 可能であれば Issue 番号も併記する。

```python
# TODO: support short-selling (ISSUE-23)
```

---

以上。
