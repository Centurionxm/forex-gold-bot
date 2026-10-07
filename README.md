# Forex & Gold Bot

Streamlit dashboard for an EMA-crossover MT5 trading bot using MetaApi, with optional Claude risk-manager approval.

Features:
- M5, M15 and H1 timeframes
- EMA crossover entries
- ATR-based stop loss and risk/reward take profit
- Daily loss kill switch
- Optional Claude veto/approval
- SQLite configuration and decision logs
- Streamlit dashboard
- MetaApi reconnect handling

Required environment variables:
- ANTHROPIC_API_KEY
- METAAPI_TOKEN
- METAAPI_ACCOUNT_ID

Optional:
- DASH_PASSWORD
- CLAUDE_MODEL
- DB_PATH

Run locally:
1. Copy .env.example to .env and fill in credentials.
2. pip install -r requirements.txt
3. streamlit run app.py

Use a demo trading account first. This software can place real orders.
