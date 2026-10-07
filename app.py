import os
import json
import time
import sqlite3
import threading
import asyncio
import datetime as dt
import pandas as pd
import streamlit as st
from anthropic import Anthropic
from metaapi_cloud_sdk import MetaApi

DB=os.getenv("DB_PATH","bot.db")
MODEL=os.getenv("CLAUDE_MODEL","claude-sonnet-4-5")
TOKEN=os.getenv("METAAPI_TOKEN")
ACCOUNT_ID=os.getenv("METAAPI_ACCOUNT_ID")
TFS={"M5":"5m","M15":"15m","H1":"1h"}
DEFAULTS={"symbol":"XAUUSD","tf":"M15","fast":9,"slow":21,"lots":0.01,"atr_mult":1.5,"rr":2.0,"max_daily_loss":20.0,"running":0,"agent_on":1,"last_bar":"","snap":{},"err":""}

def db():
    c=sqlite3.connect(DB,timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE IF NOT EXISTS cfg(k TEXT PRIMARY KEY,v TEXT)")
    c.execute("CREATE TABLE IF NOT EXISTS agent_log(ts TEXT,symbol TEXT,side TEXT,approve INTEGER,reason TEXT)")
    return c

def get_cfg():
    c=db(); out=dict(DEFAULTS)
    for k,v in c.execute("SELECT k,v FROM cfg"): out[k]=json.loads(v)
    c.close(); return out

def set_cfg(**kw):
    c=db()
    for k,v in kw.items(): c.execute("INSERT OR REPLACE INTO cfg(k,v) VALUES(?,?)",(k,json.dumps(v)))
    c.commit(); c.close()

def ask_agent(symbol,side,df,atr):
    key=os.getenv("ANTHROPIC_API_KEY")
    if not key: return False,"ANTHROPIC_API_KEY is not configured"
    try:
        closes=[round(float(x),2) for x in df["close"].tail(40).tolist()]
        prompt=(f"You are a cautious risk manager for a trading bot. An EMA crossover signals a {side} on {symbol}.\n"
                f"Last 40 closes: {closes}\nAverage candle range (ATR): {atr:.3f}\n"
                'Veto chop, spike exhaustion, or abnormal volatility. Reply ONLY as JSON: {"approve": true/false, "reason": "<max 20 words>"}')
        r=Anthropic(api_key=key).messages.create(model=MODEL,max_tokens=200,messages=[{"role":"user","content":prompt}])
        raw="".join(x.text for x in r.content if hasattr(x,"text")).strip()
        d=json.loads(raw)
        return bool(d.get("approve",False)),str(d.get("reason","No reason supplied"))
    except Exception as e: return False,f"agent error, vetoed: {e}"

def log_agent(symbol,side,ok,why):
    c=db(); c.execute("INSERT INTO agent_log VALUES(?,?,?,?,?)",(time.strftime("%Y-%m-%d %H:%M"),symbol,side,int(ok),why)); c.commit(); c.close()

def utc(t):
    if isinstance(t,str): t=pd.Timestamp(t).to_pydatetime()
    return t.replace(tzinfo=dt.timezone.utc) if t.tzinfo is None else t

def net(d): return float(d.get("profit",0) or 0)+float(d.get("commission",0) or 0)+float(d.get("swap",0) or 0)

async def get_conn(api):
    if not TOKEN or not ACCOUNT_ID: raise RuntimeError("METAAPI_TOKEN and METAAPI_ACCOUNT_ID must be configured")
    account=await api.metatrader_account_api.get_account(ACCOUNT_ID)
    if account.state!="DEPLOYED": await account.deploy()
    await account.wait_connected()
    conn=account.get_rpc_connection(); await conn.connect(); await conn.wait_synchronized()
    return account,conn

async def snapshot(conn):
    acc=await conn.get_account_information(); pos=await conn.get_positions(); now=dt.datetime.now(dt.timezone.utc)
    res=await conn.get_deals_by_time_range(now-dt.timedelta(days=30),now+dt.timedelta(days=1))
    deals=[d for d in res.get("deals",[]) if d.get("entryType")=="DEAL_ENTRY_OUT"]
    pnl24=sum(net(d) for d in deals if utc(d["time"])>=now-dt.timedelta(hours=24))
    floating=sum(float(p.get("profit",0) or 0) for p in pos)
    snap={"bal":float(acc.get("balance",0)),"eq":float(acc.get("equity",0)),"cur":acc.get("currency",""),"pnl24":round(pnl24,2),"float":round(floating,2),
          "open":[{"symbol":p["symbol"],"side":"BUY" if "BUY" in p.get("type","") else "SELL","lots":p["volume"],"entry":p["openPrice"],"pnl":p.get("profit",0)} for p in pos],
          "deals":[{"time":utc(d["time"]).strftime("%m-%d %H:%M"),"symbol":d.get("symbol"),"profit":round(net(d),2)} for d in deals[-30:]][::-1],
          "total":round(sum(net(d) for d in deals),2)}
    set_cfg(snap=snap); return pnl24+floating

async def trade(account,conn,c,day_pnl):
    sym=c["symbol"]
    if [p for p in await conn.get_positions() if p["symbol"]==sym] or day_pnl<=-abs(float(c["max_daily_loss"])): return
    candles=await account.get_historical_candles(sym,TFS[c["tf"]],None,150)
    if not candles or len(candles)<60: raise RuntimeError(f"Not enough candles for {sym}; check the exact MT5 symbol name")
    df=pd.DataFrame(candles).sort_values("time").reset_index(drop=True).iloc[:-1].copy()
    for col in ("open","high","low","close"): df[col]=pd.to_numeric(df[col],errors="coerce")
    df=df.dropna(subset=["high","low","close"])
    if len(df)<30: return
    bar=str(df["time"].iloc[-1])
    if bar==c["last_bar"]: return
    f=df["close"].ewm(span=int(c["fast"]),adjust=False).mean(); s=df["close"].ewm(span=int(c["slow"]),adjust=False).mean()
    side=None
    if f.iloc[-2]<=s.iloc[-2] and f.iloc[-1]>s.iloc[-1]: side="BUY"
    elif f.iloc[-2]>=s.iloc[-2] and f.iloc[-1]<s.iloc[-1]: side="SELL"
    if not side: return
    set_cfg(last_bar=bar)
    atr=float((df["high"]-df["low"]).rolling(14).mean().iloc[-1])
    if not atr or pd.isna(atr): return
    if c["agent_on"]:
        ok,why=await asyncio.to_thread(ask_agent,sym,side,df,atr); log_agent(sym,side,ok,why)
        if not ok: return
    price=await conn.get_symbol_price(sym); spec=await conn.get_symbol_specification(sym)
    dist=atr*float(c["atr_mult"]); digits=int(spec.get("digits",2)); step=float(spec.get("volumeStep",0.01)); minimum=float(spec.get("minVolume",0.01))
    vol=round(max(minimum,round(float(c["lots"])/step)*step),2)
    if side=="BUY":
        e=float(price["ask"]); sl=round(e-dist,digits); tp=round(e+dist*float(c["rr"]),digits)
        res=await conn.create_market_buy_order(sym,vol,sl,tp,{"comment":"claude-bot"})
    else:
        e=float(price["bid"]); sl=round(e+dist,digits); tp=round(e-dist*float(c["rr"]),digits)
        res=await conn.create_market_sell_order(sym,vol,sl,tp,{"comment":"claude-bot"})
    if res.get("stringCode") not in ("TRADE_RETCODE_DONE","TRADE_RETCODE_PLACED","ERR_NO_ERROR"): raise RuntimeError(f"Order issue: {res}")

async def bot_main():
    api=MetaApi(TOKEN); account=None; conn=None
    while True:
        try:
            if conn is None: account,conn=await get_conn(api)
            c=get_cfg(); pnl=await snapshot(conn); set_cfg(err="")
            if c["running"]:
                try: await trade(account,conn,c,pnl)
                except Exception as e: set_cfg(err=f"trade: {e}"); print("trade error:",e)
        except Exception as e:
            set_cfg(err=f"connection: {e}"); print("connection error:",e)
            try:
                if conn is not None: await conn.close()
            except Exception: pass
            conn=None
        await asyncio.sleep(30)

@st.cache_resource
def start_bot():
    threading.Thread(target=lambda:asyncio.run(bot_main()),daemon=True,name="metaapi-trading-bot").start()
    return True

st.set_page_config(page_title="FX/Gold Bot",page_icon="🤖",layout="wide")
pw=os.getenv("DASH_PASSWORD")
if pw and not st.session_state.get("ok"):
    st.title("🔒 Login"); entered=st.text_input("Password",type="password")
    if entered==pw: st.session_state["ok"]=True; st.rerun()
    if entered: st.error("Incorrect password")
    st.stop()

start_bot(); cfg=get_cfg()
st.title("🤖 Forex & Gold Bot"); st.caption("Deriv MT5 via MetaApi")

with st.sidebar:
    st.header("Settings")
    symbol=st.text_input("Symbol (exactly as Deriv MT5 names it)",cfg["symbol"])
    tf=st.selectbox("Timeframe",list(TFS),index=list(TFS).index(cfg["tf"]))
    fast=st.slider("Fast EMA",3,30,int(cfg["fast"])); slow=st.slider("Slow EMA",10,100,int(cfg["slow"]))
    lots=st.number_input("Lot size",0.01,10.0,float(cfg["lots"]),step=0.01)
    atrm=st.slider("Stop distance (x ATR)",0.5,4.0,float(cfg["atr_mult"])); rr=st.slider("Reward : Risk",1.0,4.0,float(cfg["rr"]))
    loss=st.number_input("Max daily loss (account currency)",1.0,5000.0,float(cfg["max_daily_loss"]))
    agent=st.toggle("Claude agent veto",bool(cfg["agent_on"])); running=st.toggle("Bot running",bool(cfg["running"]))
    if st.button("Save settings"):
        set_cfg(symbol=symbol,tf=tf,fast=fast,slow=slow,lots=lots,atr_mult=atrm,rr=rr,max_daily_loss=loss,agent_on=int(agent),running=int(running),last_bar="")
        st.success("Settings saved")
    if st.button("🛑 Stop bot"): set_cfg(running=0); st.error("Bot stopped. Existing trades keep their SL/TP.")

@st.fragment(run_every=10)
def view():
    c=get_cfg(); s=c["snap"]
    if c["err"]: st.error(c["err"])
    if not s: st.info("Connecting to MetaApi... (the first connection can take a minute)"); return
    a,b,d,e=st.columns(4); a.metric("Balance",f"{s['bal']:,.2f} {s['cur']}"); b.metric("Equity",f"{s['eq']:,.2f}",f"{s['float']:+.2f} floating"); d.metric("Last 24h closed",f"{s['pnl24']:+.2f}"); e.metric("Total closed PnL",f"{s['total']:+.2f}")
    if s["pnl24"]+s["float"]<=-abs(float(c["max_daily_loss"])): st.error("Kill switch active: no new entries.")
    if s["open"]: st.subheader("Open trades"); st.dataframe(pd.DataFrame(s["open"]),use_container_width=True,hide_index=True)
    if s["deals"]: st.subheader("Closed trades"); st.dataframe(pd.DataFrame(s["deals"]),use_container_width=True,hide_index=True)
    con=db(); logs=pd.read_sql("SELECT * FROM agent_log ORDER BY rowid DESC LIMIT 20",con); con.close()
    if len(logs):
        logs["approve"]=logs["approve"].map({1:"approved",0:"vetoed"}); st.subheader("Claude agent decisions"); st.dataframe(logs,use_container_width=True,hide_index=True)
view()
