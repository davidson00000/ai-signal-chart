start_command.md
# ターミナル1: バックエンド
uvicorn backend.main:app --host 0.0.0.0 --port 8001

# ターミナル2: PC向けダッシュボード（ポート 8501）
streamlit run dev_dashboard.py --server.port 8501

# ターミナル3: スマホ向けLiveビュー（ポート 8502）
streamlit run mobile_live_view.py --server.port 8502

# ターミナル4: cloudflare
cloudflared tunnel --url http://localhost:8501