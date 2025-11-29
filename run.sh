#!/usr/bin/env bash
set -e

# ==== 設定 ====
PROJECT_DIR="/Users/kousukenakamura/dev/ai-signal-chart"
BACKEND_PORT=8000
DASHBOARD_PORT=8501

# ==== 初期化 ====
cd "$PROJECT_DIR"

echo "[run.sh] Using project dir: $PROJECT_DIR"

# venv 有効化
if [ -f ".venv/bin/activate" ]; then
  echo "[run.sh] Activating virtualenv..."
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "[run.sh] .venv が見つかりません。先に仮想環境を作ってください。"
  exit 1
fi

# ==== Backend 起動 ====
echo "[run.sh] Starting FastAPI backend on port ${BACKEND_PORT} ..."
uvicorn backend.main:app --reload --port "${BACKEND_PORT}" &
BACKEND_PID=$!
echo "[run.sh] Backend PID = ${BACKEND_PID}"

# ==== ダッシュボード起動（Streamlit） ====
echo "[run.sh] Starting Streamlit dashboard on port ${DASHBOARD_PORT} ..."
echo "[run.sh] → ブラウザで http://localhost:${DASHBOARD_PORT} を開いてください。"

streamlit run dev_dashboard.py --server.port "${DASHBOARD_PORT}"

# ==== 終了処理 ====
echo "[run.sh] Streamlit が終了したので backend を停止します..."
kill "${BACKEND_PID}" 2>/dev/null || true
echo "[run.sh] All done. Bye!"
