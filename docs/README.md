# EXITON Documentation

このディレクトリは **EXITON Trading System** のドキュメントを体系的に管理するための構造です。

---

## 📁 ディレクトリ構成

```
docs/
├── specs/        ← システムの設計原則・非機能要件・憲法
├── strategies/   ← 各売買戦略の仕様と説明
├── sprints/      ← 開発履歴とスプリントの作業記録
├── reports/      ← バックテストやマルチシミュレーションの検証レポート
└── notes/        ← 作業メモ・プロトタイピング・研究ノート
```

---

## 📂 フォルダ説明

### `specs/` - 仕様の"法典"
- `EXITON_AI_DEV_SYSTEM_v1.3_noClaude.md` - AI開発システムの基本仕様
- `DOMAIN_RULES_EXITON_TRADING_v0.1.md` - トレーディングドメインルール
- `NFR_EXITON_TRADING_v0.1.md` - 非機能要件
- `APPLY_EXITON_CONSTITUTION_EXITON_TRADING.md` - 憲法の適用ガイド
- `SAAS_DECISIONS_EXITON_TRADING.md` - SaaS決定事項

### `strategies/` - 売買戦略
- `ma_crossover.md` - MA Crossover戦略
- `ema9_dip_buy.md` - EMA9 Dip Buy戦略
- `rsi_reversal.md` - RSI Reversal戦略
- `template.md` - 新規戦略のテンプレート

### `sprints/` - スプリントログ
Sprint 10 から 18 までの開発履歴を時系列で管理。

### `reports/` - 検証レポート
- `verification_report.md` - 基本検証レポート
- `multi_sim_verification_report.md` - マルチシミュレーション検証
- `verify_sp500_universe.md` - S&P500ユニバース検証
- `verify_strategies_basic.md` - ストラテジー基本検証

### `notes/` - 作業メモ
- `strategy_lab_notes.md` - Strategy Lab開発メモ
- `auto_sim_lab_notes.md` - Auto Sim Lab開発メモ
- `json_strategy_schema.md` - JSONストラテジースキーマ
- `auto_sim_lab_update_sp500_and_strategies.md` - S&P500アップデートノート

---

## 🔗 関連リンク

- **Backend**: `/backend/` - FastAPI サーバー
- **Frontend**: `/dev_dashboard.py` - Streamlit ダッシュボード
- **Tools**: `/tools/` - 検証・ユーティリティスクリプト

---

*Last Updated: 2025-12-06*
