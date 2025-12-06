# Reports - 検証レポート

このフォルダにはバックテスト結果、マルチシミュレーション検証、機能検証などのレポートを格納しています。

## 📄 ファイル一覧

| ファイル | 説明 |
|---------|------|
| `verification_report.md` | 基本機能の検証レポート |
| `multi_sim_verification_report.md` | マルチシミュレーション検証 |
| `verify_sp500_universe.md` | S&P500 ユニバース検証 |
| `verify_strategies_basic.md` | ストラテジー基本検証 |

## 📌 レポート生成

検証スクリプトは `/tools/` フォルダに格納されています:

```bash
python tools/verify_multi_sim.py
python tools/verify_sp500_universe.py
python tools/verify_strategies_basic.py
```

---

*新規機能追加時は検証スクリプトと対応するレポートを作成してください。*
