import os
import io
import json
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ---------------------------
# Page / theme
# ---------------------------
st.set_page_config(page_title="Panamax Freight Game", layout="wide")

# ---------------------------
# Config & data
# ---------------------------
with open("config.json", "r") as f:
    config = json.load(f)

selected_date = str(config.get("current_day"))  # e.g., "9/1/2025"

CURVE_FILE = "data/forward_curves.csv"
NEWS_FILE = "data/news_stories.csv"
LOG_FILE = "data/trader_log.csv"

# Contract universe (labels in forward_curves.csv)
CONTRACTS = ["Sep", "Oct", "Nov", "Dec"]

# Position limits
MAX_PER_BUCKET = 200   # max net per contract month
MAX_SLATE = 100        # max net across all four months

# ---------------------------
# Safe datetime parsing
# ---------------------------
def parse_dt(x):
    """Safely parse dates; return pandas Timestamp or NaT."""
    return pd.to_datetime(str(x), errors="coerce", infer_datetime_format=True)

# ---------------------------
# Load datasets
# ---------------------------
curve_df = pd.read_csv(CURVE_FILE)
news_df = pd.read_csv(NEWS_FILE)

# Normalize date columns to string for exact match
curve_df["date"] = curve_df["date"].astype(str)
news_df["date"] = news_df["date"].astype(str)

curve_today = curve_df[curve_df["date"] == str(selected_date)].copy()
news_today = news_df[news_df["date"] == str(selected_date)].copy()

# ---------------------------
# Helpers
# ---------------------------
def _append_rows(rows: list[dict], path: str):
    df_new = pd.DataFrame(rows)
    if os.path.exists(path):
        df_existing = pd.read_csv(path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(path, index=False)

def _ensure_log_schema(df: pd.DataFrame) -> pd.DataFrame:
    req = [
        "timestamp","date","trader","type","contract","side","price","lots",
        "buy_month","sell_month","spread_px","mtm_next","pnl_day"
    ]
    for c in req:
        if c not in df.columns:
            df[c] = ""
    # types
    for c in ["price","lots","spread_px"]:
        df[c] = df[c].replace("", "0")
    try:
        df["lots"] = df["lots"].astype(int)
    except Exception:
        df["lots"] = pd.to_numeric(df["lots"], errors="coerce").fillna(0).astype(int)
    for c in ["price","spread_px","mtm_next","pnl_day"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[req]

def _mid(df_row) -> float:
    return float((df_row["bid"] + df_row["ask"]) / 2)

def get_today_bid_ask(contract: str):
    row = curve_today[curve_today["contract"] == contract]
    if row.empty:
        return None, None
    r = row.iloc[0]
    return float(r["bid"]), float(r["ask"])

def next_mtm_price(contract: str, trade_date_str: str, curves_all: pd.DataFrame, fallback_same_day=False):
    cdf = curves_all.copy()
    cdf["date_dt"] = parse_dt(cdf["date"])
    trade_dt = parse_dt(trade_date_str)
    later = cdf[(cdf["contract"] == contract) & (cdf["date_dt"] > trade_dt)].sort_values("date_dt")
    if not later.empty:
        head = later.iloc[0]
        return float((head["bid"] + head["ask"]) / 2)
    if fallback_same_day:
        same = cdf[(cdf["contract"] == contract) & (cdf["date_dt"] == trade_dt)]
        if not same.empty:
            head = same.iloc[0]
            return float((head["bid"] + head["ask"]) / 2)
    return None

def compute_positions_asof(trader: str, asof_date: str) -> dict:
    """Net lots per contract & slate through asof_date (inclusive). Robust to bad dates."""
    if not os.path.exists(LOG_FILE):
        return {m: 0 for m in CONTRACTS} | {"slate": 0}
    tl = pd.read_csv(LOG_FILE)
    if tl.empty:
        return {m: 0 for m in CONTRACTS} | {"slate": 0}
    tl["date"] = tl["date"].astype(str)
    tl = _ensure_log_schema(tl)

    tl["date_dt"] = parse_dt(tl["date"])
    asof_dt = parse_dt(asof_date)
    tl = tl[(tl["trader"] == trader) & (tl["date_dt"].notna()) & (tl["date_dt"] <= asof_dt)]
    if tl.empty:
        return {m: 0 for m in CONTRACTS} | {"slate": 0}

    net = {m: 0 for m in CONTRACTS}

    # outrights
    outs = tl[tl["type"] == "outright"]
    for _, r in outs.iterrows():
        if r["contract"] in CONTRACTS:
            sgn = 1 if str(r["side"]).lower() == "buy" else -1
            net[r["contract"]] += sgn * int(r["lots"])

    # spreads (buy_month long, sell_month short)
    spr = tl[tl["type"] == "spread"]
    for _, r in spr.iterrows():
        bm = str(r["buy_month"])
        sm = str(r["sell_month"])
        if bm in CONTRACTS:
            net[bm] += int(r["lots"])
        if sm in CONTRACTS:
            net[sm] -= int(r["lots"])

    slate = sum(net.values())
    net["slate"] = slate
    return net

def would_violate_limits(trader: str, asof_date: str, deltas: dict) -> tuple[bool, str]:
    """deltas = {'Oct': +10, 'Nov': -10, ...} => check MAX_PER_BUCKET and MAX_SLATE."""
    current = compute_positions_asof(trader, asof_date)
    trial = {m: current.get(m, 0) + deltas.get(m, 0) for m in CONTRACTS}
    slate = sum(trial.values())
    offenders = [m for m in CONTRACTS if abs(trial[m]) > MAX_PER_BUCKET]
    if offenders:
        return True, f"Per-month limit exceeded ({', '.join(offenders)} would be > {MAX_PER_BUCKET})."
    if abs(slate) > MAX_SLATE:
        return True, f"Slate limit exceeded (|{slate}| > {MAX_SLATE})."
    return False, ""

# ---------------------------
# UI — header, news, curve
# ---------------------------
st.title("🚢 Panamax Freight Paper Trading Game")

if curve_today.empty or news_today.empty:
    st.error(f"No market data found for {selected_date}. Check your config or data files.")
    st.stop()

st.subheader(f"📅 Market Day: {selected_date}")

st.markdown("### 📰 Tradewinds News")
st.markdown(news_today["headline"].values[0])

st.markdown("### 📈 Forward Curve")
fig, ax = plt.subplots(figsize=(7, 2.8))  # compact
contracts = curve_today["contract"]
mids = (curve_today["bid"] + curve_today["ask"]) / 2
ax.plot(contracts, mids, marker='o', label="Mid Price")
ax.fill_between(contracts, curve_today["bid"], curve_today["ask"], alpha=0.2, label="Bid/Ask Spread")
for _, row in curve_today.iterrows():
    ax.text(row["contract"], row["bid"] - 40, f"B: {int(row['bid'])}", ha='center', fontsize=8)
    ax.text(row["contract"], row["ask"] + 40, f"O: {int(row['ask'])}", ha='center', fontsize=8)
ax.set_ylabel("USD/Day")
ax.legend(loc="upper right")
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ---------------------------
# Trading forms
# ---------------------------
st.markdown("## 🧾 Submit Trades")

# ----- Outright trade -----
with st.form("ou_form", clear_on_submit=True):
    c1, c2, c3, c4, c5 = st.columns([1.3, 1, 1, 1, 1])
    trader_ou = c1.text_input("Trader Name", key="ou_trader")
    contract_ou = c2.selectbox("Contract", options=list(curve_today["contract"]), key="ou_contract")
    side_ou = c3.selectbox("Side", ["Buy", "Sell"], key="ou_side")

    # auto-prefill price from today's bid/offer
    def _default_ou_price():
        b, a = get_today_bid_ask(contract_ou)
        return int(a if side_ou == "Buy" else b) if (b is not None and a is not None) else 0

    price_ou = c4.number_input("Price", step=1, value=_default_ou_price(), key="ou_price")
    lots_ou = c5.number_input("Lots (days)", min_value=1, step=1, value=1, key="ou_lots")

    submit_ou = st.form_submit_button("Submit Outright Trade")

if submit_ou:
    if not trader_ou:
        st.error("Enter trader name.")
    else:
        delta = {m: 0 for m in CONTRACTS}
        sgn = 1 if side_ou == "Buy" else -1
        if contract_ou in delta:
            delta[contract_ou] = sgn * int(lots_ou)
        viol, msg = would_violate_limits(trader_ou, selected_date, delta)
        if viol:
            st.error(f"❌ {msg}")
        else:
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date": selected_date,
                "trader": trader_ou,
                "type": "outright",
                "contract": contract_ou,
                "side": side_ou,
                "price": float(price_ou),
                "lots": int(lots_ou),
                "buy_month": "",
                "sell_month": "",
                "spread_px": "",
                "mtm_next": "",
                "pnl_day": ""
            }
            _append_rows([row], LOG_FILE)
            st.success(f"✅ Outright: {trader_ou} {side_ou} {lots_ou}d {contract_ou} @ {int(price_ou)}")

st.markdown("### 🔁 Calendar Spread")

# ----- Spread trade -----
with st.form("sp_form", clear_on_submit=True):
    s1, s2, s3, s4, s5 = st.columns([1.3, 1, 1, 1, 1])
    trader_sp = s1.text_input("Trader Name", key="sp_trader")
    buy_m = s2.selectbox("Buy month", options=list(curve_today["contract"]), key="sp_buy")
    sell_m = s3.selectbox("Sell month", options=list(curve_today["contract"]), key="sp_sell")
    lots_sp = s4.number_input("Lots (days)", min_value=1, step=1, value=1, key="sp_lots")

    # auto-prefill spread = (Buy ask – Sell bid)
    b_bid, b_ask = get_today_bid_ask(buy_m)
    s_bid, s_ask = get_today_bid_ask(sell_m)
    default_spread = int(b_ask - s_bid) if (b_ask is not None and s_bid is not None) else 0
    spread_px = s5.number_input("Spread Price (Buy–Sell)", step=1, value=default_spread, key="sp_px")

    submit_sp = st.form_submit_button("Submit Spread Trade")

if submit_sp:
    if not trader_sp:
        st.error("Enter trader name.")
    elif buy_m == sell_m:
        st.error("Buy and Sell month must differ.")
    else:
        delta = {m: 0 for m in CONTRACTS}
        if buy_m in delta:  delta[buy_m]  += int(lots_sp)
        if sell_m in delta: delta[sell_m] -= int(lots_sp)
        viol, msg = would_violate_limits(trader_sp, selected_date, delta)
        if viol:
            st.error(f"❌ {msg}")
        else:
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date": selected_date,
                "trader": trader_sp,
                "type": "spread",
                "contract": "",
                "side": "",
                "price": "",
                "lots": int(lots_sp),
                "buy_month": buy_m,
                "sell_month": sell_m,
                "spread_px": float(spread_px),
                "mtm_next": "",
                "pnl_day": ""
            }
            _append_rows([row], LOG_FILE)
            st.success(f"✅ Spread: {trader_sp} BUY {buy_m} / SELL {sell_m} {lots_sp}d @ {int(spread_px)}")

st.markdown("---")

# ---------------------------
# Repair tool
# ---------------------------
st.markdown("### 🧹 Repair Trade Log (backfill missing dates/schema)")
if os.path.exists(LOG_FILE):
    if st.button("Repair now"):
        tl = pd.read_csv(LOG_FILE, dtype=str).fillna("")
        if "date" not in tl.columns:
            tl["date"] = ""
        fixed = int((tl["date"].astype(str).str.strip() == "").sum())
        tl.loc[tl["date"].astype(str).str.strip() == "", "date"] = str(selected_date)
        tl = _ensure_log_schema(tl)
        tl.to_csv(LOG_FILE, index=False)
        st.success(f"Repaired {fixed} row(s) with missing dates and normalized columns.")

# ---------------------------
# Admin downloads (log + P&L)
# ---------------------------
st.markdown("### 🔐 Admin Access")
pwd = st.text_input("Enter admin password", type="password")
if pwd == "freightadmintrader":
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("📥 Download Trade Log", f, file_name="trader_log.csv")

        # ---- Compute T+1 P&L and offer download ----
        tl = pd.read_csv(LOG_FILE)
        if not tl.empty:
            tl = _ensure_log_schema(tl)
            curves_all = pd.read_csv(CURVE_FILE)
            curves_all["date"] = curves_all["date"].astype(str)

            pnl_rows = []
            pending = 0

            for _, r in tl.iterrows():
                tdate = str(r["date"])
                lots = int(r["lots"]) if pd.notna(r["lots"]) else 0

                if str(r["type"]) == "outright":
                    con = str(r["contract"])
                    side = str(r["side"]).lower()
                    trade_px = float(r["price"]) if pd.notna(r["price"]) else None
                    mtm_next = next_mtm_price(con, tdate, curves_all, fallback_same_day=False)
                    if mtm_next is None or trade_px is None:
                        pending += 1
                        pnl = None
                    else:
                        sgn = 1 if side == "buy" else -1
                        pnl = (mtm_next - trade_px) * sgn * lots
                    pnl_rows.append({
                        "timestamp": r["timestamp"],
                        "date": tdate,
                        "trader": r["trader"],
                        "type": "outright",
                        "contract": con,
                        "side": r["side"],
                        "price": trade_px,
                        "lots": lots,
                        "buy_month": "",
                        "sell_month": "",
                        "spread_px": "",
                        "mtm_next": mtm_next,
                        "pnl_day": pnl
                    })

                else:  # spread
                    bm = str(r["buy_month"])
                    sm = str(r["sell_month"])
                    spx = float(r["spread_px"]) if pd.notna(r["spread_px"]) else None
                    mtm_b = next_mtm_price(bm, tdate, curves_all, fallback_same_day=False)
                    mtm_s = next_mtm_price(sm, tdate, curves_all, fallback_same_day=False)
                    if (mtm_b is None) or (mtm_s is None) or (spx is None):
                        pending += 1
                        pnl = None
                        mtm_next = None
                    else:
                        mtm_spread_next = mtm_b - mtm_s
                        pnl = (mtm_spread_next - spx) * lots
                        mtm_next = mtm_spread_next

                    pnl_rows.append({
                        "timestamp": r["timestamp"],
                        "date": tdate,
                        "trader": r["trader"],
                        "type": "spread",
                        "contract": "",
                        "side": "",
                        "price": "",
                        "lots": lots,
                        "buy_month": bm,
                        "sell_month": sm,
                        "spread_px": spx,
                        "mtm_next": mtm_next,
                        "pnl_day": pnl
                    })

            pnl_df = pd.DataFrame(pnl_rows)
            if not pnl_df.empty:
                daily = pnl_df.groupby(["date", "trader"], dropna=False)["pnl_day"].sum(min_count=1).reset_index()
                daily = daily.sort_values(["trader", "date"])
                daily["cum_pnl"] = daily.groupby("trader")["pnl_day"].cumsum(min_count=1)

                out = io.StringIO()
                out.write("# Trades with T+1 P&L\n")
                pnl_df.to_csv(out, index=False)
                out.write("\n# Daily & Cumulative per Trader\n")
                daily.to_csv(out, index=False)
                st.download_button("📊 Download P&L (CSV)", data=out.getvalue().encode("utf-8"),
                                   file_name="pnl_results.csv")

            if pending > 0:
                st.warning("Some trades have no next-day curve yet (P&L pending). "
                           "Upload the next day’s forward_curves to finalize those rows.")
    else:
        st.info("No trades yet.")
else:
    st.caption("Enter admin password to access downloads.")
