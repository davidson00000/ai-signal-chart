#!/usr/bin/env bash
set -e

# プロジェクトルートから実行する想定
DOCS_DIR="docs/specs"
OUT_DIR="docs/pdfs"
mkdir -p "$OUT_DIR"

# 共通オプション
PDF_ENGINE="xelatex"
MAIN_FONT="Noto Sans CJK JP"   # お好みで変えてOK（ヒラギノでも可）

convert_md () {
  local input="$1"
  local output="$2"

  echo "=> $input -> $output"

  pandoc "$DOCS_DIR/$input" \
    -o "$OUT_DIR/$output" \
    --pdf-engine="$PDF_ENGINE" \
    -V mainfont="$MAIN_FONT" \
    -V geometry:margin=20mm \
    -V linewidth=120 \
    --toc \
    --toc-depth=3
}

# ここが今回の「憲法セット」
convert_md "EXITON_AI_DEV_SYSTEM_v1.3_noClaude.md"        "EXITON_AI_DEV_SYSTEM_v1.3_noClaude.pdf"
convert_md "DOMAIN_RULES_EXITON_TRADING_v0.1.md"          "DOMAIN_RULES_EXITON_TRADING_v0.1.pdf"
convert_md "NFR_EXITON_TRADING_v0.1.md"                   "NFR_EXITON_TRADING_v0.1.pdf"
convert_md "APPLY_EXITON_CONSTITUTION_EXITON_TRADING.md"  "APPLY_EXITON_CONSTITUTION_EXITON_TRADING.pdf"
convert_md "SAAS_DECISIONS_EXITON_TRADING.md"             "SAAS_DECISIONS_EXITON_TRADING.pdf"

echo "✅ DONE. PDF files are in $OUT_DIR"
