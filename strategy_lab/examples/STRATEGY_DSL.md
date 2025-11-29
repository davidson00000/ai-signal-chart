# EXITON Strategy DSL v0.1

目的：
- 「テキストで戦略を書く → バックテストに流し込む」ための超シンプル DSL
- まずは **既存の Python 戦略クラス（例: MACrossStrategy）にパラメータを渡す** 設計から始める

実装ポリシー：
- 中身は **JSON** にする（標準ライブラリだけでパース可能）
- 将来 LLM で自動生成しやすい形

---

## 1. 基本フォーマット

拡張子: `.dsl`  
中身: JSON 1オブジェクト

```json
{
  "name": "MA Cross 9-21",
  "type": "ma_cross",
  "description": "Simple moving average cross strategy (9 / 21).",
  "params": {
    "short_window": 9,
    "long_window": 21
  },
  "meta": {
    "author": "kousuke",
    "tags": ["ma", "trend-follow", "example"]
  }
}
