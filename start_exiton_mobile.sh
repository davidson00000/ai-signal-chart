#!/usr/bin/env bash
# start_exiton_mobile.sh

cd ~/dev/ai-signal-chart

# 1. backend
echo "[1/3] Starting backend..."
uvicorn backend.main:app --host 0.0.0.0 --port 8001 &

# 2. mobile view
echo "[2/3] Starting Streamlit mobile view..."
streamlit run mobile_live_view.py &

# 3. quick tunnel
echo "[3/3] Starting Cloudflare quick tunnel..."
cloudflared tunnel --url http://localhost:8501
