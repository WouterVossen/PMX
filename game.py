# game.py — Panamax Freight Paper Trading Game
# Outrights + Spreads (Buy month / Sell month), with position limits

import streamlit as st
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import json

# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Panamax Freight Game", layout="wide")

# Position rules
POS_LIMIT_PER_MONTH = 200   # absolute position limit per month (long or short)
SLATE_LIMIT = 100           # absolute net across all months

# Trade log setup (single file for outrights & spreads)
LOG_FILE = "data/trader_log.csv"
LOG_COLUMNS = [
    "timestamp", "date", "trader", "type",
    "contract", "side", "price", "lots",
    # for spreads:
    "buy_month", "sell_month", "spread_px"
]

def ensure_log_schema(path: str):
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Add any missing columns then reorder
        for c in LOG_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        df = df[LOG_COLUMNS]
        df.to_csv(path, index=False)
    else:
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(path, index=False)

ensure_log_schema(LOG_FILE)

# ─────────────────────────────────────────────────────────────────────────────
# Load config & data
# ─────────────────────────────────────────────────────────────────────────────
with open("config.json", "r") as f:
    config = json.load(f)
selected_date = config.get("current_day")

curve_df = pd.read_csv("data/forward_curves.csv")
curve_today = curve_df[curve_df["date"] == selected_date]

st.title("🚢 Panamax Freight Paper Trading Game")

if curve_today.empty:
    st.error(f"No market data found for {selected_date}. Please check your config or data files.")
    st.stop()

# contracts visible today (expecting Sep, Oct, Nov, Dec in your file)
contracts_today = curve_today["contract"].tolist()

st.subheader(f"📅 Market Day: {selected_date}")

# ─────────────────────────────────────────────────────────────────────────────
# Forward curve chart
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📈 Forward Curve")
fig, ax = plt.subplots(figsize=(8, 2.4))
mids = (curve_today["bid"] + curve_today["ask"]) / 2
ax.plot(curve_today["contract"], mids, marker='o', label="Mid Price")
ax.fill_between(curve_today["contract"], curve_today["bid"], curve_today["ask"],
                alpha=0.2, label="Bid/Ask Spread")
for _, row in curve_today.iterrows():
    ax.text(row["contract"], row["bid"] - 50, f"B: {int(row['bid'])}", ha='center', fontsize=8)
    ax.text(row["contract"], row["ask"] + 50, f"O: {int(row['ask'])}", ha='center', fontsize=8)
ax.set_ylabel("USD/Day")
ax.legend()
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers: current positions & rule checking
# ─────────────────────────────────────────────────────────────────────────────
def load_log() -> pd.DataFrame:
    df = pd.read_csv(LOG_FILE)
    if not df.empty:
        # types
        df["lots"] = pd.to_numeric(df["lots"], errors="coerce").fillna(0).astype(int)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # normalize missing columns
        for c in LOG_COLUMNS:
            if c not in df.columns:
                df[c] = ""
    return df

def positions_for_trader(trader: str, up_to_date: str) -> dict:
    """
    Returns net positions per contract month up to and including the given date.
    Long = +lots, Short = -lots.
    """
    pos = {m: 0 for m in contracts_today}  # initialize only with today's months
    df = load_log()
    if df.empty:
        return pos
    cutoff = pd.to_datetime(up_to_date, errors="coerce")
    if pd.isna(cutoff):
        return pos
    tdf = df[(df["trader"] == trader) & (df["date"] <= cutoff)]
    for _, r in tdf.iterrows():
        typ = str(r.get("type", "")).lower()
        if typ == "outright":
            m = r.get("contract", "")
            if m in pos:
                if str(r.get("side","")).lower() == "buy":
                    pos[m] += int(r["lots"])
                else:
                    pos[m] -= int(r["lots"])
        elif typ == "spread":
            buy_m = r.get("buy_month", "")
            sell_m = r.get("sell_month", "")
            lots = int(r["lots"])
            if buy_m in pos:
                pos[buy_m] += lots
            if sell_m in pos:
                pos[sell_m] -= lots
    return pos

def would_violate_rules(current_pos: dict, delta: dict):
    """
    current_pos: dict month->net lots now
    delta: dict month->change if trade is executed
    Returns (violates: bool, msg: str)
    """
    # new per-month
    new_pos = current_pos.copy()
    for m, v in delta.items():
        new_pos[m] = new_pos.get(m, 0) + v

    # per-month limit
    viol_months = {m: new_pos[m] for m in new_pos if abs(new_pos[m]) > POS_LIMIT_PER_MONTH}
    if viol_months:
        detail = ", ".join([f"{m}:{v:+d}" for m, v in viol_months.items()])
        return True, f"❌ Per-month limit exceeded (> {POS_LIMIT_PER_MONTH}). Would be: {detail}"

    # slate limit
    slate = sum(new_pos.values())
    if abs(slate) > SLATE_LIMIT:
        return True, f"❌ Slate limit exceeded (> {SLATE_LIMIT}). Would be slate {slate:+d}"

    return False, ""

# ─────────────────────────────────────────────────────────────────────────────
# Trade entry (Outright + Spread in ONE section)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🧾 Submit Your Trade")

# Show live positions for a trader (updates when name entered)
name_col, _ = st.columns([1, 2])
with name_col:
    trader_name = st.text_input("Trader Name", key="trader_name")

if trader_name:
    pos_now = positions_for_trader(trader_name, selected_date)
    slate_now = sum(pos_now.values())
    pos_str = "  •  ".join([f"{m}:{v:+d}" for m, v in pos_now.items()])
    st.info(f"**Current positions** — {pos_str}  |  **Slate:** {slate_now:+d}")

# --- Outright form -----------------------------------------------------------
with st.form("outright_form"):
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    ou_contract = c1.selectbox("Contract", options=contracts_today)
    ou_side = c2.selectbox("Side", ["Buy", "Sell"])
    ou_price = c3.number_input("Price", step=1)
    ou_lots = c4.number_input("Lots (days)", min_value=1, step=1, value=1)
    submit_outright = st.form_submit_button("Submit Outright Trade")

if submit_outright:
    if not trader_name:
        st.error("Please enter trader name above.")
    else:
        current = positions_for_trader(trader_name, selected_date)
        # Delta for this outright
        delta = {m: 0 for m in contracts_today}
        delta[ou_contract] = ou_lots if ou_side == "Buy" else -ou_lots
        violates, msg = would_violate_rules(current, delta)
        if violates:
            st.error(msg)
        else:
            st.success(f"✅ {trader_name} {ou_side} {ou_lots}d of {ou_contract} @ ${ou_price}")
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date": selected_date,
                "trader": trader_name,
                "type": "outright",
                "contract": ou_contract,
                "side": ou_side,
                "price": ou_price,
                "lots": int(ou_lots),
                "buy_month": "",
                "sell_month": "",
                "spread_px": ""
            }
            pd.DataFrame([row])[LOG_COLUMNS].to_csv(LOG_FILE, mode='a', header=False, index=False)

# --- Spread form (Buy month / Sell month) -----------------------------------
with st.form("spread_form"):
    sc1, sc2, sc3, sc4 = st.columns([1,1,1,1.2])
    sp_buy = sc1.selectbox("Buy month", options=contracts_today, key="sp_buy")
    sp_sell = sc2.selectbox("Sell month", options=[m for m in contracts_today if m != sp_buy], key="sp_sell")
    sp_lots = sc3.number_input("Lots (days)", min_value=1, step=1, value=1, key="sp_lots")
    sp_px = sc4.number_input("Spread Price (Buy−Sell)", step=1, key="sp_px")
    submit_spread = st.form_submit_button("Submit Spread Trade")

if submit_spread:
    if not trader_name:
        st.error("Please enter trader name above.")
    else:
        if sp_buy == sp_sell:
            st.error("Buy month and Sell month must be different.")
        else:
            current = positions_for_trader(trader_name, selected_date)
            # Delta for spread: +lots on buy, -lots on sell
            delta = {m: 0 for m in contracts_today}
            delta[sp_buy] += sp_lots
            delta[sp_sell] -= sp_lots
            violates, msg = would_violate_rules(current, delta)
            if violates:
                st.error(msg)
            else:
                st.success(
                    f"✅ {trader_name} Buy {sp_buy} / Sell {sp_sell}  {sp_lots}d  @ {sp_px}"
                )
                row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "date": selected_date,
                    "trader": trader_name,
                    "type": "spread",
                    "contract": f"{sp_buy}-{sp_sell}",
                    "side": "Buy/Sell",
                    "price": sp_px,          # duplicate into 'price' for simple exports
                    "lots": int(sp_lots),
                    "buy_month": sp_buy,
                    "sell_month": sp_sell,
                    "spread_px": sp_px
                }
                pd.DataFrame([row])[LOG_COLUMNS].to_csv(LOG_FILE, mode='a', header=False, index=False)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Admin download
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("🔐 ### Admin Access")
password = st.text_input("Enter admin password to download trade log", type="password")
if password == "freightadmintrader":
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("📥 Download Trade Log", f, file_name="trader_log.csv")
