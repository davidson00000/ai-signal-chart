# EXITON /docs/specs — 標準プロジェクト仕様テンプレート
EXITON Standard Project Specification Guide  
v0.1 (Template)

---

このディレクトリは、EXITON のあらゆるプロジェクトが  
**統一された形で仕様・要件・規範を管理するための標準テンプレート** である。

新しいプロジェクトを開始するときは、  
以下のファイル構成をコピーし、必要に応じて編集して使う。

---

# 📁 0. このフォルダの目的

- プロジェクトの「意図」「品質基準」「ルール」「非機能要件」を明文化し、  
  AI（Antigravity / ChatGPT / Gemini）と人間が迷わないようにすること。
- 全プロジェクト間で共通の設計思想（EXITON Standard）を共有すること。
- 要件の抜け漏れ・仕様のブレを防ぎ、開発〜テスト〜保守を安定化させること。

---

# 📁 1. このフォルダに置く標準ファイル一覧

| ファイル名 | 役割 | 必須 |
|-----------|------|------|
| **EXITON_AI_DEV_SYSTEM_v1.3.md** | EXITON AI 開発憲法（PM・Dev・QA の三権分立） | ✅ |
| **SAAS_STANDARD_v0.1.md** | EXITON の SaaS 設計標準（Auth/DB/ログ/環境分離） | ✅ |
| **CODING_STYLE_PYTHON.md** | Python コーディング規約（Exiton標準） | （Python案件のみ必須） |
| **DOMAIN_RULES_TEMPLATE.md** | ドメインルールを作るためのテンプレ | ✅ |
| **NFR_TEMPLATE.md** | 非機能要件を作るためのテンプレ | ✅ |
| **DOMAIN_RULES_*.md** | プロジェクト固有の業務ルール | プロジェクト開始時に作成 |
| **NFR_*.md** | プロジェクト固有の非機能要件 | プロジェクト開始時に作成 |
| **SPEC_HISTORY.md** | 仕様の変更履歴（ChangeLog） | 推奨 |

---

# 📘 2. プロジェクト開始時の手順（Checklist）

以下のステップをプロジェクト開始直後に行う。

---

## ① ドメインルールを作成する  
`DOMAIN_RULES_TEMPLATE.md` をコピーして：

```
DOMAIN_RULES_<PROJECT>.md
```

にリネームして、以下を埋める：

- システムの目的
- 何を“やらない”か
- 対象ユーザー
- データの扱い（時刻/欠損/一貫性）
- 業務ルール・禁止事項
- 変更手順・バージョニング

---

## ② 非機能要件を作成する  
`NFR_TEMPLATE.md` をコピーし：

```
NFR_<PROJECT>.md
```

を作り、以下を定義する：

- 応答速度・スループット
- 壊れ方（silent failure禁止）
- ログ・再現性
- セキュリティ方針
- 運用方針（環境分離・設定管理）
- テスト方針

---

## ③ EXITON 憲法の適用範囲を決める  
`EXITON_AI_DEV_SYSTEM_v1.3.md` を読み、

- このプロジェクトのどこを Antigravity（Dev）に任せるか  
- どこを ChatGPT（破壊 QA）に任せるか  
- どこを人間が判断するか  

の境界を決める。

---

## ④ SaaS標準のどこを今回採用するか決める  
`SAAS_STANDARD_v0.1.md` から：

- 認証は外部？Supabase？Auth0？
- DB は自前？Supabase？Neon？
- ログは Sentry？Better Stack？
- 本番と開発環境の分離どうする？

をプロジェクト用に整理して、  
`SAAS_DECISIONS_<PROJECT>.md` を作ると明確になる。

---

## ⑤ 技術スタックに応じて規約を追加  
Python なら：

- `CODING_STYLE_PYTHON.md`

TypeScript/Next.js を使う場合は、  
`CODING_STYLE_TS.md`（将来追加予定）

---

# 📁 3. Antigravity / AIエージェントに与える初期プロンプト

```
あなたは EXITON の Dev Agent です。
このプロジェクトは /docs/specs 内の以下のルールに従って開発します：

- EXITON_AI_DEV_SYSTEM_v1.3.md
- DOMAIN_RULES_<PROJECT>.md
- NFR_<PROJECT>.md
- SAAS_STANDARD_v0.1.md
- CODING_STYLE_PYTHON.md（該当する場合）

これらを遵守し、Pull Request ベースで安全に開発を進めてください。
```

---

# 📁 4. SPEC_HISTORY（仕様変更履歴）

```
# SPEC_HISTORY.md
- 2025-XX-XX: DOMAIN_RULES_<PROJECT>.md を作成
- 2025-XX-XX: NFR_<PROJECT>.md を作成
- 2025-XX-XX: SaaS 設計方針を更新
- 2025-XX-XX: EXITON 憲法の適用範囲を定義
```

仕様の変更理由を必ず記録し、未来の開発者（人間・AI問わず）が  
意図を失わないようにする。

---

# ✔ この README の役割  
- EXITON のすべてのプロジェクトが **同じ構造で始まる**  
- 仕様が最初から見える化される  
- AI エージェントが迷わず実装できる  
- ヒューマンエラーが激減する  

---

以上。
