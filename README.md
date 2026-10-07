# Forex & Gold Rules Bot

A Streamlit rule-based trading dashboard using the current Deriv Options API.

## Important execution change

MetaApi and Anthropic have been removed.

This version uses Deriv's authenticated REST + WebSocket Options API. Deriv's current API supports real-time market data, proposals, buying/selling contracts, portfolio monitoring, and demo accounts.

**This is NOT an MT5/CFD bridge.** The bot executes CALL/PUT Options contracts using a USD stake and contract duration. The existing rule-based signals are retained, but MT5-style lots and broker SL/TP are not used.

## Strategies

1. EMA Crossover
2. SMA200 + MACD Histogram + ATR + recent H1 supply/demand
3. RSI Mean Reversion

## Railway variables

Set these in Railway:

- `DERIV_API_TOKEN`
- `DERIV_APP_ID`
- `DERIV_ACCOUNT_ID`
- `DASH_PASSWORD`
- `DB_PATH`

Never commit your Deriv token.

## Deriv setup

1. Log in to the Deriv developer dashboard.
2. Register an application.
3. Create a Personal Access Token with trading permission.
4. Use a demo Options account.
5. Add the token, App ID and demo account ID to Railway variables.

The app defaults to `frxXAUUSD` and M15. Confirm the symbol is available to your Deriv account before enabling the bot.

## Railway

The repository includes a Procfile:

```
web: streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT
```

## Safety

Bot starts with trading disabled. Use a demo account first. Rule-based strategies are not guaranteed profitable.
