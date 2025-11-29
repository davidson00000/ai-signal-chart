
---

## 2️⃣ `CHANGELOG.md` 用スニペット（日本語でも）

```markdown
## [0.2.0] - Backtest Engine & /simulate API

### Added
- 戦略抽象化レイヤー (`backend/strategies/`):
  - `BaseStrategy` 抽象クラス
  - `MACrossStrategy` 実装（短期/長期MAクロス）
- バックテストエンジン (`backend/backtester.py`):
  - ポートフォリオシミュレーション（現金 + ポジション + 資産）
  - 手数料対応（デフォルト0.05%）
  - 日次終値での売買シミュレーション
  - P&L, リターン, 最大ドローダウン, 勝率, トレード数などの指標計算
- シミュレーションAPI:
  - `POST /simulate` エンドポイント
  - Pydanticモデル (`backend/models/backtest.py`) による型安全なリクエスト/レスポンス
- テストスイート:
  - `tests/test_strategies.py`
  - `tests/test_backtester.py`
- ドキュメント:
  - `03BACKTEST_SPEC.md` （バックテスト仕様書）
- `requirements.txt` に `pytest` を追加

### Changed
- 既存の MA 戦略実装を削除し、プラグイン可能な戦略パターンに移行

### Notes
- これにより、今後は新しい戦略を `BaseStrategy` を継承して追加するだけで
  `/simulate` API からバックテスト可能。
- フロントエンド統合は今後のリリースで実装予定。
