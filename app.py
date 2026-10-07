import os
import json
import time
import sqlite3
import threading
import asyncio
import datetime as dt
import pandas as pd
import streamlit as st
from metaapi_cloud_sdk import MetaApi

DB = os.getenv("DB_PATH", "bot.db")
TOKEN = os.getenv("METAAPI_TOKEN")
ACCOUNT_ID = os.getenv("METAAPI_ACCOUNT_ID")

TFS = {"M5": "5m", "M15": "15m", "H1": "1h"}
STRATEGIES = {
    "EMA Crossover": "ema",
    "SMA200 + MACD Histogram + ATR": "trend_macd",
    "RSI Mean Reversion": "rsi",
}

DEFAULTS = {
    "symbol": "XAUUSD",
    "tf": "M15",
    "strategy": "ema",
    "fast": 9,
    "slow": 21,
    "lots": 0.01,
    "atr_mult": 1.5,
    "rr": 2.0,
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


def utc(t):
    if isinstance(t, str):
        t = pd.Timestamp(t).to_pydatetime()
    return t.replace(tzinfo=dt.timezone.utc) if t.tzinfo is None else t


def net(d):
    return (
        float(d.get("profit", 0) or 0)
        + float(d.get("commission", 0) or 0)
        + float(d.get("swap", 0) or 0)
    )


async def get_conn(api):
    if not TOKEN or not ACCOUNT_ID:
        raise RuntimeError(
            "METAAPI_TOKEN and METAAPI_ACCOUNT_ID must be configured"
        )

    account = await api.metatrader_account_api.get_account(ACCOUNT_ID)
    if account.state != "DEPLOYED":
        await account.deploy()

    await account.wait_connected()
    conn = account.get_rpc_connection()
    await conn.connect()
    await conn.wait_synchronized()
    return account, conn


async def snapshot(conn):
    acc = await conn.get_account_information()
    pos = await conn.get_positions()
    now = dt.datetime.now(dt.timezone.utc)

    res = await conn.get_deals_by_time_range(
        now - dt.timedelta(days=30), now + dt.timedelta(days=1)
    )
    deals = [
        d for d in res.get("deals", [])
        if d.get("entryType") == "DEAL_ENTRY_OUT"
    ]

    pnl24 = sum(
        net(d)
        for d in deals
        if utc(d["time"]) >= now - dt.timedelta(hours=24)
    )
    floating = sum(float(p.get("profit", 0) or 0) for p in pos)

    snap = {
        "bal": float(acc.get("balance", 0)),
        "eq": float(acc.get("equity", 0)),
        "cur": acc.get("currency", ""),
        "pnl24": round(pnl24, 2),
        "float": round(floating, 2),
        "open": [
            {
                "symbol": p["symbol"],
                "side": "BUY" if "BUY" in p.get("type", "") else "SELL",
                "lots": p["volume"],
                "entry": p["openPrice"],
                "pnl": p.get("profit", 0),
            }
            for p in pos
        ],
        "deals": [
            {
                "time": utc(d["time"]).strftime("%m-%d %H:%M"),
                "symbol": d.get("symbol"),
                "profit": round(net(d), 2),
            }
            for d in deals[-30:]
        ][::-1],
        "total": round(sum(net(d) for d in deals), 2),
    }

    set_cfg(snap=snap)
    return pnl24 + floating


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


async def h1_zone(account, symbol):
    candles = await account.get_historical_candles(symbol, "1h", None, 20)
    if not candles or len(candles) < 10:
        return None, None

    h1 = pd.DataFrame(candles).sort_values("time").reset_index(drop=True)
    for col in ("high", "low", "close"):
        h1[col] = pd.to_numeric(h1[col], errors="coerce")
    h1 = h1.dropna(subset=["high", "low", "close"])

    if len(h1) < 9:
        return None, None

    # Simple recent supply/demand zones from the last 8 completed H1 candles.
    recent = h1.iloc[:-1].tail(8)
    return float(recent["high"].max()), float(recent["low"].min())


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
            return "BUY", "SMA200 trend + MACD histogram cross + ATR filter + H1 demand"
        if price < sma200.iloc[-1] and bearish_cross and atr_ok and sell_zone:
            return "SELL", "SMA200 trend + MACD histogram cross + ATR filter + H1 supply"

        return None, "trend/MACD/ATR/zone conditions not aligned"

    if strategy == "rsi":
        # Mean reversion: use RSI extremes with a simple EMA direction check.
        if rsi.iloc[-1] < 30 and close.iloc[-1] > ema_slow.iloc[-1]:
            return "BUY", "RSI oversold recovery above EMA21"
        if rsi.iloc[-1] > 70 and close.iloc[-1] < ema_slow.iloc[-1]:
            return "SELL", "RSI overbought rejection below EMA21"
        return None, "no RSI setup"

    return None, "unknown strategy"


async def trade(account, conn, c, day_pnl):
    sym = c["symbol"]

    if any(p["symbol"] == sym for p in await conn.get_positions()):
        return

    if day_pnl <= -abs(float(c["max_daily_loss"])):
        return

    candles = await account.get_historical_candles(
        sym, TFS[c["tf"]], None, 260
    )
    if not candles or len(candles) < 60:
        raise RuntimeError(
            f"Not enough candles for {sym}; check the exact MT5 symbol name"
        )

    df = (
        pd.DataFrame(candles)
        .sort_values("time")
        .reset_index(drop=True)
        .iloc[:-1]
        .copy()
    )

    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["high", "low", "close"])

    bar = str(df["time"].iloc[-1])
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
    atr_series = tr.rolling(14).mean()
    atr = float(atr_series.iloc[-1])

    supply = demand = None
    if c["strategy"] == "trend_macd":
        supply, demand = await h1_zone(account, sym)

    side, reason = get_signal(
        c["strategy"], df, atr, supply=supply, demand=demand
    )
    set_cfg(last_bar=bar)

    if not side:
        return

    log_signal(sym, c["strategy"], side, True, reason)

    price = await conn.get_symbol_price(sym)
    spec = await conn.get_symbol_specification(sym)

    dist = atr * float(c["atr_mult"])
    digits = int(spec.get("digits", 2))
    step = float(spec.get("volumeStep", 0.01))
    minimum = float(spec.get("minVolume", 0.01))

    requested = float(c["lots"])
    vol = max(minimum, round(requested / step) * step)
    vol = round(vol, 8)

    if side == "BUY":
        entry = float(price["ask"])
        sl = round(entry - dist, digits)
        tp = round(entry + dist * float(c["rr"]), digits)
        res = await conn.create_market_buy_order(
            sym, vol, sl, tp, {"comment": "rules-bot"}
        )
    else:
        entry = float(price["bid"])
        sl = round(entry + dist, digits)
        tp = round(entry - dist * float(c["rr"]), digits)
        res = await conn.create_market_sell_order(
            sym, vol, sl, tp, {"comment": "rules-bot"}
        )

    if res.get("stringCode") not in (
        "TRADE_RETCODE_DONE",
        "TRADE_RETCODE_PLACED",
        "ERR_NO_ERROR",
    ):
        raise RuntimeError(f"Order issue: {res}")


async def bot_main():
    api = MetaApi(TOKEN)
    account = None
    conn = None

    while True:
        try:
            if conn is None:
                account, conn = await get_conn(api)

            c = get_cfg()
            pnl = await snapshot(conn)
            set_cfg(err="")

            if c["running"]:
                try:
                    await trade(account, conn, c, pnl)
                except Exception as e:
                    set_cfg(err=f"trade: {e}")
                    print("trade error:", e)

        except Exception as e:
            set_cfg(err=f"connection: {e}")
            print("connection error:", e)
            try:
                if conn is not None:
                    await conn.close()
            except Exception:
                pass
            conn = None

        await asyncio.sleep(30)


@st.cache_resource
def start_bot():
    threading.Thread(
        target=lambda: asyncio.run(bot_main()),
        daemon=True,
        name="metaapi-trading-bot",
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
st.caption("Deriv MT5 via MetaApi — no external AI/Anthropic API required")

with st.sidebar:
    st.header("Settings")

    symbol = st.text_input(
        "Symbol (exactly as Deriv MT5 names it)",
        cfg["symbol"],
    )

    tf = st.selectbox(
        "Timeframe",
        list(TFS),
        index=list(TFS).index(cfg["tf"]),
    )

    strategy = st.selectbox(
        "Trading strategy",
        list(STRATEGIES),
        index=list(STRATEGIES.values()).index(cfg["strategy"]),
    )

    fast = st.slider("Fast EMA", 3, 30, int(cfg["fast"]))
    slow = st.slider("Slow EMA", 10, 100, int(cfg["slow"]))
    lots = st.number_input(
        "Lot size", 0.01, 10.0, float(cfg["lots"]), step=0.01
    )
    atrm = st.slider(
        "Stop distance (x ATR)",
        0.5,
        4.0,
        float(cfg["atr_mult"]),
    )
    rr = st.slider(
        "Reward : Risk",
        1.0,
        4.0,
        float(cfg["rr"]),
    )
    loss = st.number_input(
        "Max daily loss (account currency)",
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
            lots=lots,
            atr_mult=atrm,
            rr=rr,
            max_daily_loss=loss,
            running=int(running),
            last_bar="",
        )
        st.success("Settings saved")

    if st.button("🛑 Stop bot"):
        set_cfg(running=0)
        st.error("Bot stopped. Existing trades keep their SL/TP.")

st.info(
    "Strategies: EMA crossover, SMA200 + MACD histogram + ATR/H1 zones, "
    "and RSI mean reversion. Test on a demo account before live trading."
)


@st.fragment(run_every=10)
def view():
    c = get_cfg()
    s = c["snap"]

    if c["err"]:
        st.error(c["err"])

    if not s:
        st.info("Connecting to MetaApi... (the first connection can take a minute)")
        return

    a, b, d, e = st.columns(4)
    a.metric("Balance", f"{s['bal']:,.2f} {s['cur']}")
    b.metric("Equity", f"{s['eq']:,.2f}", f"{s['float']:+.2f} floating")
    d.metric("Last 24h closed", f"{s['pnl24']:+.2f}")
    e.metric("Total closed PnL", f"{s['total']:+.2f}")

    if s["pnl24"] + s["float"] <= -abs(float(c["max_daily_loss"])):
        st.error("Kill switch active: no new entries.")

    if s["open"]:
        st.subheader("Open trades")
        st.dataframe(
            pd.DataFrame(s["open"]),
            use_container_width=True,
            hide_index=True,
        )

    if s["deals"]:
        st.subheader("Closed trades")
        st.dataframe(
            pd.DataFrame(s["deals"]),
            use_container_width=True,
            hide_index=True,
        )

    con = db()
    logs = pd.read_sql(
        "SELECT * FROM signal_log ORDER BY rowid DESC LIMIT 20",
        con,
    )
    con.close()

    if len(logs):
        st.subheader("Strategy signals")
        st.dataframe(
            logs,
            use_container_width=True,
            hide_index=True,
        )


view()
