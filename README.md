# Forex & Gold Rules Bot

Streamlit trading dashboard for MT5/Deriv through MetaApi.

## Strategies

1. **EMA Crossover** — EMA 9/21 crossover.
2. **SMA200 + MACD Histogram + ATR** — trend filter, MACD histogram zero-cross, ATR-above-median volatility filter, and a simple recent H1 supply/demand zone.
3. **RSI Mean Reversion** — RSI extreme plus EMA confirmation.

The Anthropic/Claude API has been completely removed. The bot now uses deterministic rule-based strategies only.

## Environment variables

- METAAPI_TOKEN
- METAAPI_ACCOUNT_ID
- DASH_PASSWORD
- DB_PATH (optional)

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Use a demo MT5 account first. Verify the exact symbol name, volume rules, stop distance and trading hours for your broker before enabling live orders.

The bot is not a guarantee of profit. Rule-based signals should be backtested and forward-tested before risking real money.
