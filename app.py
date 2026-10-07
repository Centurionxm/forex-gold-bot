import os
import json
import time
import sqlite3
import threading
import asyncio
import datetime as dt
from typing import Any

import pandas as pd
import requests
import streamlit as st
import websockets

DB = os.getenv("DB_PATH", "bot.db")
DERIV_TOKEN = os.getenv("DERIV_API_TOKEN")
DERIV_APP_ID = os.getenv("DERIV_APP_ID")
DERIV_ACCOUNT_ID = os.getenv("DERIV_ACCOUNT_ID")

GRANULARITY = {"M5": 300, "M15": 900, "H1": 3600}
STRATEGIES = {
    "EMA Crossover": "ema",
    "SMA200 + MACD Histogram + ATR": "trend_macd",
    "RSI Mean Reversion": "rsi",
}
DEFAULTS = {
    "symbol": "frxXAUUSD",
    "tf": "M15",
    "strategy": "ema",
    "fast": 9,
    "slow": 21,
    "stake": 1.0,
    "duration": 15,
    "max_daily_loss": 20.0,
    "running": 0,
    "last_bar": "",
    "snap": {},
    "err": "",
}


def db():
    c = sqlite3.connect(DB, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE IF NOT EXISTS cfg(k TEXT PRIMARY KEY,v TEXT)")
    c.execute(
        "CREATE TABLE IF NOT EXISTS signal_log("
        "ts TEXT,symbol TEXT,strategy TEXT,side TEXT,signal INTEGER,reason TEXT)"
    )
    return c


def get_cfg():
    c = db()
    out = dict(DEFAULTS)
    for k, v in c.execute("SELECT k,v FROM cfg"):
        out[k] = json.loads(v)
    c.close()
    return out


def set_cfg(**kw):
    c = db()
    for k, v in kw.items():
        c.execute(
            "INSERT OR REPLACE INTO cfg(k,v) VALUES(?,?)",
            (k, json.dumps(v)),
        )
    c.commit()
    c.close()


def log_signal(symbol, strategy, side, signal, reason):
    c = db()
    c.execute(
        "INSERT INTO signal_log VALUES(?,?,?,?,?,?)",
        (
            time.strftime("%Y-%m-%d %H:%M"),
            symbol,
            strategy,
            side or "-",
            int(signal),
            reason,
        ),
    )
    c.commit()
    c.close()


class DerivClient:
    """Current Deriv Options API client: REST OTP + authenticated WebSocket."""

    def __init__(self, token: str, app_id: str, account_id: str):
        self.token = token
        self.app_id = app_id
        self.account_id = account_id
        self.ws = None
        self.req_id = 0

    async def connect(self):
        if not self.token or not self.app_id or not self.account_id:
            raise RuntimeError(
                "DERIV_API_TOKEN, DERIV_APP_ID and DERIV_ACCOUNT_ID must be configured"
            )

        r = requests.post(
            f"https://api.derivws.com/trading/v1/options/accounts/"
            f"{self.account_id}/otp",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Deriv-App-ID": self.app_id,
            },
            timeout=20,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Deriv OTP error {r.status_code}: {r.text[:500]}")

        data = r.json().get("data", {})
        url = data.get("url")
        if not url:
            raise RuntimeError(f"Deriv OTP response did not contain a WebSocket URL: {r.text[:500]}")

        self.ws = await websockets.connect(url, ping_interval=20, ping_timeout=20)

    async def close(self):
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        self.ws = None

    async def call(self, payload):
        if not self.ws:
            await self.connect()
        self.req_id += 1
        rid = self.req_id
        msg = dict(payload)
        msg["req_id"] = rid
        await self.ws.send(json.dumps(msg))

        while True:
            raw = await asyncio.wait_for(self.ws.recv(), timeout=30)
            data = json.loads(raw)
            if data.get("error"):
                err = data["error"]
                raise RuntimeError(f"{err.get('code')}: {err.get('message')}")
            if data.get("req_id") == rid:
                return data

    async def candles(self, symbol, granularity, count=260):
        res = await self.call(
            {
                "ticks_history": symbol,
                "end": "latest",
                "count": count,
                "style": "candles",
                "granularity": granularity,
                "subscribe": 0,
            }
        )
        rows = res.get("candles", [])
        if not rows:
            raise RuntimeError(f"No candle data returned for {symbol}")
        return pd.DataFrame(rows)

    async def active_symbols(self):
        res = await self.call({"active_symbols": "brief"})
        return res.get("active_symbols", [])

    async def balance(self):
        res = await self.call({"balance": 1})
        return res.get("balance", {})

    async def portfolio(self):
        res = await self.call({"portfolio": 1})
        return res.get("portfolio", {}).get("contracts", [])

    async def profit_table(self, limit=100):
        res = await self.call(
            {"profit_table": 1, "limit": limit, "sort": "DESC"}
        )
        return res.get("profit_table", {}).get("transactions", [])

    async def proposal(self, symbol, side, stake, duration):
        contract_type = "CALL" if side == "BUY" else "PUT"
        return await self.call(
            {
                "proposal": 1,
                "amount": float(stake),
                "basis": "stake",
                "contract_type": contract_type,
                "currency": "USD",
                "duration": int(duration),
                "duration_unit": "m",
                "underlying_symbol": symbol,
            }
        )

    async def buy(self, proposal_id, price):
        return await self.call(
            {"buy": proposal_id, "price": float(price)}
        )

    async def sell(self, contract_id):
        return await self.call(
            {"sell": int(contract_id), "price": 0}
        )


def indicators(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]

    ema_fast = close.ewm(span=9, adjust=False).mean()
    ema_slow = close.ewm(span=21, adjust=False).mean()
    sma200 = close.rolling(200).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))

    return ema_fast, ema_slow, sma200, macd_hist, atr, rsi


def get_signal(strategy, df, atr, supply=None, demand=None):
    if len(df) < 30:
        return None, "not enough candles"

    close = df["close"]
    ema_fast, ema_slow, sma200, macd_hist, atr_series, rsi = indicators(df)

    if pd.isna(atr) or atr <= 0:
        return None, "ATR unavailable"

    if strategy == "ema":
        if ema_fast.iloc[-2] <= ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1]:
            return "BUY", "EMA 9 crossed above EMA 21"
        if ema_fast.iloc[-2] >= ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1]:
            return "SELL", "EMA 9 crossed below EMA 21"
        return None, "no EMA crossover"

    if strategy == "trend_macd":
        if len(df) < 200 or pd.isna(sma200.iloc[-1]):
            return None, "waiting for 200 SMA"

        atr_median = atr_series.tail(100).median()
        atr_ok = not pd.isna(atr_median) and atr > atr_median
        bullish_cross = macd_hist.iloc[-2] <= 0 and macd_hist.iloc[-1] > 0
        bearish_cross = macd_hist.iloc[-2] >= 0 and macd_hist.iloc[-1] < 0

        price = float(close.iloc[-1])
        buy_zone = demand is not None and abs(price - demand) <= 0.5 * atr
        sell_zone = supply is not None and abs(price - supply) <= 0.5 * atr

        if price > sma200.iloc[-1] and bullish_cross and atr_ok and buy_zone:
            return "BUY", "SMA200 + MACD histogram + ATR + H1 demand"
        if price < sma200.iloc[-1] and bearish_cross and atr_ok and sell_zone:
            return "SELL", "SMA200 + MACD histogram + ATR + H1 supply"
        return None, "trend/MACD/ATR/zone conditions not aligned"

    if strategy == "rsi":
        if rsi.iloc[-1] < 30 and close.iloc[-1] > ema_slow.iloc[-1]:
            return "BUY", "RSI oversold recovery above EMA21"
        if rsi.iloc[-1] > 70 and close.iloc[-1] < ema_slow.iloc[-1]:
            return "SELL", "RSI overbought rejection below EMA21"
        return None, "no RSI setup"

    return None, "unknown strategy"


async def h1_zone(client, symbol):
    h1 = await client.candles(symbol, 3600, 20)
    for col in ("high", "low", "close"):
        h1[col] = pd.to_numeric(h1[col], errors="coerce")
    h1 = h1.dropna(subset=["high", "low", "close"]).sort_values("epoch")
    if len(h1) < 10:
        return None, None
    recent = h1.iloc[:-1].tail(8)
    return float(recent["high"].max()), float(recent["low"].min())


async def snapshot(client, symbol):
    bal = await client.balance()
    positions = await client.portfolio()
    transactions = await client.profit_table(100)

    today = dt.datetime.now(dt.timezone.utc).date()
    today_pnl = 0.0
    deals = []

    for t in transactions:
        ts = t.get("sell_time") or t.get("purchase_time") or t.get("transaction_time")
        try:
            when = dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).date()
        except Exception:
            when = None
        profit = float(t.get("profit", 0) or 0)
        if when == today:
            today_pnl += profit
        deals.append(
            {
                "time": str(ts or ""),
                "symbol": t.get("underlying_symbol", t.get("symbol", "")),
                "profit": round(profit, 2),
            }
        )

    snap = {
        "bal": float(bal.get("balance", 0) or 0),
        "cur": bal.get("currency", "USD"),
        "pnl24": round(today_pnl, 2),
        "open": [
            {
                "contract_id": p.get("contract_id"),
                "symbol": p.get("underlying_symbol", p.get("symbol", "")),
                "type": p.get("contract_type", ""),
                "buy_price": p.get("buy_price", 0),
                "profit": p.get("profit", 0),
            }
            for p in positions
        ],
        "deals": deals[:30],
        "total": round(sum(float(t.get("profit", 0) or 0) for t in transactions), 2),
    }
    set_cfg(snap=snap)
    return today_pnl


async def trade(client, c, day_pnl):
    if day_pnl <= -abs(float(c["max_daily_loss"])):
        return

    open_contracts = await client.portfolio()
    if any(
        p.get("underlying_symbol", p.get("symbol")) == c["symbol"]
        for p in open_contracts
    ):
        return

    candles = await client.candles(
        c["symbol"], GRANULARITY[c["tf"]], 260
    )
    for col in ("open", "high", "low", "close"):
        candles[col] = pd.to_numeric(candles[col], errors="coerce")
    df = candles.dropna(subset=["high", "low", "close"]).sort_values("epoch").reset_index(drop=True)

    if len(df) < 60:
        raise RuntimeError("Not enough candles returned for the selected symbol")

    # Ignore the currently forming candle.
    df = df.iloc[:-1].copy()
    bar = str(df["epoch"].iloc[-1])
    if bar == c["last_bar"]:
        return

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])

    supply = demand = None
    if c["strategy"] == "trend_macd":
        supply, demand = await h1_zone(client, c["symbol"])

    side, reason = get_signal(
        c["strategy"], df, atr, supply=supply, demand=demand
    )
    set_cfg(last_bar=bar)

    if not side:
        return

    log_signal(c["symbol"], c["strategy"], side, True, reason)

    proposal = await client.proposal(
        c["symbol"], side, float(c["stake"]), int(c["duration"])
    )
    p = proposal.get("proposal", {})
    proposal_id = p.get("id")
    ask = p.get("ask_price") or p.get("display_value") or p.get("price")

    if not proposal_id or ask is None:
        raise RuntimeError(f"Deriv did not return a tradable proposal: {proposal}")

    result = await client.buy(proposal_id, float(ask))
    if "buy" not in result:
        raise RuntimeError(f"Buy failed: {result}")

    buy = result["buy"]
    print(
        f"DERIV DEMO TRADE: {side} {c['symbol']} "
        f"stake={c['stake']} duration={c['duration']}m "
        f"contract={buy.get('contract_id')}"
    )


async def bot_main():
    client = DerivClient(DERIV_TOKEN, DERIV_APP_ID, DERIV_ACCOUNT_ID)

    while True:
        try:
            if client.ws is None:
                await client.connect()

            c = get_cfg()
            pnl = await snapshot(client, c["symbol"])
            set_cfg(err="")

            if c["running"]:
                try:
                    await trade(client, c, pnl)
                except Exception as e:
                    set_cfg(err=f"trade: {e}")
                    print("trade error:", e)

        except Exception as e:
            set_cfg(err=f"connection: {e}")
            print("connection error:", e)
            await client.close()

        await asyncio.sleep(30)


@st.cache_resource
def start_bot():
    threading.Thread(
        target=lambda: asyncio.run(bot_main()),
        daemon=True,
        name="deriv-trading-bot",
    ).start()
    return True


st.set_page_config(page_title="FX/Gold Bot", page_icon="🤖", layout="wide")

pw = os.getenv("DASH_PASSWORD")
if pw and not st.session_state.get("ok"):
    st.title("🔒 Login")
    entered = st.text_input("Password", type="password")
    if entered == pw:
        st.session_state["ok"] = True
        st.rerun()
    if entered:
        st.error("Incorrect password")
    st.stop()

start_bot()
cfg = get_cfg()

st.title("🤖 Forex & Gold Rules Bot")
st.caption("Deriv Options API — rule-based demo trading; Anthropic and MetaApi removed.")

with st.sidebar:
    st.header("Settings")

    symbol = st.text_input(
        "Deriv symbol",
        cfg["symbol"],
        help="Gold is commonly frxXAUUSD. Use a symbol returned by Deriv active_symbols.",
    )

    tf = st.selectbox(
        "Timeframe",
        list(GRANULARITY),
        index=list(GRANULARITY).index(cfg["tf"]),
    )

    strategy = st.selectbox(
        "Trading strategy",
        list(STRATEGIES),
        index=list(STRATEGIES.values()).index(cfg["strategy"]),
    )

    fast = st.slider("Fast EMA", 3, 30, int(cfg["fast"]))
    slow = st.slider("Slow EMA", 10, 100, int(cfg["slow"]))

    stake = st.number_input(
        "Stake per trade (USD)",
        0.35,
        1000.0,
        float(cfg["stake"]),
        step=0.25,
    )

    duration = st.number_input(
        "Contract duration (minutes)",
        1,
        1440,
        int(cfg["duration"]),
        step=1,
    )

    loss = st.number_input(
        "Max daily loss (USD)",
        1.0,
        5000.0,
        float(cfg["max_daily_loss"]),
    )

    running = st.toggle("Bot running", bool(cfg["running"]))

    if st.button("Save settings"):
        set_cfg(
            symbol=symbol,
            tf=tf,
            strategy=STRATEGIES[strategy],
            fast=fast,
            slow=slow,
            stake=stake,
            duration=duration,
            max_daily_loss=loss,
            running=int(running),
            last_bar="",
        )
        st.success("Settings saved")

    if st.button("🛑 Stop bot"):
        set_cfg(running=0)
        st.error("Bot stopped. Existing Deriv contracts continue until expiry.")

st.warning(
    "Important: Deriv API trading here uses CALL/PUT Options contracts, not MT5/CFD "
    "lot orders. Your SMA200/MACD/ATR and RSI signals are retained, but the execution "
    "model is stake + contract duration. Start on DEMO with Bot running OFF."
)


@st.fragment(run_every=10)
def view():
    c = get_cfg()
    s = c["snap"]

    if c["err"]:
        st.error(c["err"])

    if not s:
        st.info("Waiting for Deriv connection...")
        return

    a, b, d, e = st.columns(4)
    a.metric("Balance", f"{s['bal']:,.2f} {s['cur']}")
    b.metric("Today's PnL", f"{s['pnl24']:+.2f}")
    d.metric("Open contracts", len(s["open"]))
    e.metric("Closed PnL shown", f"{s['total']:+.2f}")

    if s["pnl24"] <= -abs(float(c["max_daily_loss"])):
        st.error("Kill switch active: no new entries.")

    if s["open"]:
        st.subheader("Open Deriv contracts")
        st.dataframe(pd.DataFrame(s["open"]), use_container_width=True, hide_index=True)

    if s["deals"]:
        st.subheader("Recent closed contracts")
        st.dataframe(pd.DataFrame(s["deals"]), use_container_width=True, hide_index=True)

    con = db()
    logs = pd.read_sql(
        "SELECT * FROM signal_log ORDER BY rowid DESC LIMIT 20",
        con,
    )
    con.close()

    if len(logs):
        st.subheader("Strategy signals")
        st.dataframe(logs, use_container_width=True, hide_index=True)


view()
