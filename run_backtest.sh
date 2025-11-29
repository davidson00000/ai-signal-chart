#!/usr/bin/env bash
set -e

# プロジェクトルートに移動
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "== EXITON Backtest Simulation =="
echo "Project root: $PROJECT_ROOT"
echo

# --- Python 仮想環境の有効化(.venv前提) ---
if [ -d ".venv" ]; then
  echo "[*] Activating venv (.venv)"
  # zshの人でもbashで動くように source を使う
  source .venv/bin/activate
else
  echo "[!] .venv が見つかりません。"
  echo "    先に以下を実行してください："
  echo "    python -m venv .venv"
  echo "    source .venv/bin/activate"
  echo "    pip install -r requirements.txt"
  exit 1
fi

# --- バックエンド起動 ---
echo "[*] Starting backend (FastAPI)…"
# 好きな方を使う：
#BACKEND_CMD="python backend/main.py"
BACKEND_CMD="python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

$BACKEND_CMD &
BACKEND_PID=$!
echo "    backend PID: $BACKEND_PID"

# --- フロントエンド起動 ---
echo "[*] Starting frontend (Vite)…"
cd "$PROJECT_ROOT/frontend"

# 初回だけ npm install が必要
if [ ! -d "node_modules" ]; then
  echo "[*] node_modules がないので npm install を実行します"
  npm install
fi

npm run dev -- --host 0.0.0.0 --port 3000 &
FRONTEND_PID=$!
echo "    frontend PID: $FRONTEND_PID"

# --- ブラウザを開く（macOS 用） ---
cd "$PROJECT_ROOT"
if command -v open >/dev/null 2>&1; then
  echo "[*] Opening http://localhost:3000 in your browser…"
  open "http://localhost:3000"
fi

echo
echo ">>> EXITON Backtest Simulation is running!"
echo "    Backend : http://localhost:8000"
echo "    Frontend: http://localhost:3000"
echo
echo "終了するときは、このターミナルで Ctrl + C を押してください。"

# Ctrl+C で子プロセスをまとめて kill
trap "echo; echo '[*] Stopping services…'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true; exit 0" INT TERM

# どちらかが落ちたら終了
wait
