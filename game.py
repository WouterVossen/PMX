# game.py — Panamax Freight Paper Trading Game (outrights + calendar spreads)

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

# Global monthly limit (adjustable)
MAX_MONTHLY_LOTS = 1000

# Trade log schema (single file for outrights & spreads)
LOG_FILE = "data/trader_log.csv"
LOG_COLUMNS = [
    "timestamp", "date", "trader", "type",
    "contract", "side", "price", "lots",
    "near", "far", "spread_px", "near_leg_px", "far_leg_px"
]

def ensure_log_schema(path: str):
    """Make sure the CSV exists with the correct columns/order."""
    if os.path.exists(path):
        old = pd.read_csv(path)
        # Add any missing columns
        for c in LOG_COLUMNS:
            if c not in old.columns:
                old[c] = ""
        old = old[LOG_COLUMNS]
        old.to_csv(path, index=False)
    else:
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(path, index=False)

ensure_log_schema(LOG_FILE)

# ─────────────────────────────────────────────────────────────────────────────
# Load config/current day
# ─────────────────────────────────────────────────────────────────────────────
with open("config.json", "r") as f:
    config = json.load(f)
selected_date = config.get("current_day")

# ─────────────────────────────────────────────────────────────────────────────
# Load curve
# ─────────────────────────────────────────────────────────────────────────────
curve_df = pd.read_csv("data/forward_curves.csv")
curve_today = curve_df[curve_df["date"] == selected_date]

st.title("🚢 Panamax Freight Paper Trading Game")

if curve_today.empty:
    st.error(f"No market data found for {selected_date}. Please check your config or data files.")
    st.stop()

st.subheader(f"📅 Market Day: {selected_date}")

# ─────────────────────────────────────────────────────────────────────────────
# Curve chart
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📈 Forward Curve")
fig, ax = plt.subplots(figsize=(8, 2))  # compact height
contracts = curve_today["contract"]
mids = (curve_today["bid"] + curve_today["ask"]) / 2
ax.plot(contracts, mids, marker='o', label="Mid Price")
ax.fill_between(contracts, curve_today["bid"], curve_today["ask"], alpha=0.2, label="Bid/Ask Spread")
for _, row in curve_today.iterrows():
    ax.text(row["contract"], row["bid"] - 50, f"B: {int(row['bid'])}", ha='center', fontsize=8)
    ax.text(row["contract"], row["ask"] + 50, f"O: {int(row['ask'])}", ha='center', fontsize=8)
ax.set_ylabel("USD/Day")
ax.legend()
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Outright trade entry
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("🧾 ### Submit Your Trade (Outright)")

with st.form("trade_form"):
    trader = st.text_input("Trader Name")
    contract = st.selectbox("Contract", options=contracts)
    side = st.selectbox("Side", ["Buy", "Sell"])
    price = st.number_input("Price", step=1)
    lots = st.number_input("Lots (days)", min_value=1, step=1)
    submitted = st.form_submit_button("Submit Trade")

if submitted:
    # Monthly lot cap check (counts spread lots once and outright lots once)
    log_df = pd.read_csv(LOG_FILE)
    # coerce types
    if "lots" in log_df.columns:
        log_df["lots"] = pd.to_numeric(log_df["lots"], errors="coerce").fillna(0)
    if "date" in log_df.columns:
        log_df["date"] = pd.to_datetime(log_df["date"], errors="coerce")

    sel_dt = pd.to_datetime(selected_date, errors="coerce")
    if pd.isna(sel_dt):
        st.error("Config date format not recognized; please use e.g. 9/1/2025.")
    else:
        same_month = (log_df["date"].dt.month == sel_dt.month) & (log_df["date"].dt.year == sel_dt.year)
        trader_mask = (log_df["trader"] == trader)
        total_lots_this_month = log_df.loc[same_month & trader_mask, "lots"].sum()

        if total_lots_this_month + lots > MAX_MONTHLY_LOTS:
            st.error(f"❌ Trade exceeds monthly limit of {MAX_MONTHLY_LOTS} lots. "
                     f"You have already traded {int(total_lots_this_month)} lots.")
        else:
            st.success(f"✅ Trade submitted: {trader} {side} {lots}d of {contract} @ ${price}")
            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date": selected_date,
                "trader": trader,
                "type": "outright",
                "contract": contract,
                "side": side,
                "price": price,
                "lots": int(lots),
                "near": "", "far": "",
                "spread_px": "", "near_leg_px": "", "far_leg_px": ""
            }
            pd.DataFrame([row])[LOG_COLUMNS].to_csv(
                LOG_FILE, mode='a', header=False, index=False
            )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Calendar Spreads (Near – Far)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("🔀 ### Calendar Spread (Near–Far)")

def _row_for(df, ct: str):
    r = df[df["contract"] == ct]
    return None if r.empty else r.iloc[0]

def calc_spread_quotes(curve_today: pd.DataFrame, near_ct: str, far_ct: str):
    rn = _row_for(curve_today, near_ct)
    rf = _row_for(curve_today, far_ct)
    if rn is None or rf is None:
        return None
    near_bid, near_ask = float(rn["bid"]), float(rn["ask"])
    far_bid, far_ask = float(rf["bid"]), float(rf["ask"])
    near_mid = (near_bid + near_ask) / 2
    far_mid = (far_bid + far_ask) / 2
    return {
        "near_bid": near_bid, "near_ask": near_ask,
        "far_bid": far_bid, "far_ask": far_ask,
        "bid": near_bid - far_ask,
        "ask": near_ask - far_bid,
        "mid": near_mid - far_mid
    }

def exec_spread(curve_today: pd.DataFrame, side: str, near_ct: str, far_ct: str):
    rn = _row_for(curve_today, near_ct)
    rf = _row_for(curve_today, far_ct)
    if rn is None or rf is None:
        return None
    if side == "Buy":
        near_px = float(rn["ask"])  # long near at ask
        far_px = float(rf["bid"])   # short far at bid
    else:
        near_px = float(rn["bid"])  # short near at bid
        far_px = float(rf["ask"])   # long far at ask
    return {"near_px": near_px, "far_px": far_px, "spread_px": near_px - far_px}

contracts_today = curve_today["contract"].tolist()

if len(contracts_today) < 2:
    st.info("Need at least two contracts listed today to quote a spread.")
else:
    c1, c2 = st.columns(2)
    with c1:
        near_ct = st.selectbox("Near", options=contracts_today, key="spr_near")
    with c2:
        far_ct = st.selectbox("Far", options=[c for c in contracts_today if c != near_ct], key="spr_far")

    q = calc_spread_quotes(curve_today, near_ct, far_ct)
    if q is None:
        st.error("Could not compute spread quotes.")
    else:
        st.write(
            f"**Quoted {near_ct}–{far_ct}**  |  "
            f"**Bid:** {q['bid']:,.0f}  •  **Mid:** {q['mid']:,.0f}  •  **Ask:** {q['ask']:,.0f}"
        )
        with st.expander("Leg markets", expanded=False):
            st.write(
                f"- {near_ct}: Bid {q['near_bid']:,.0f} / Ask {q['near_ask']:,.0f}\n"
                f"- {far_ct}:  Bid {q['far_bid']:,.0f} / Ask {q['far_ask']:,.0f}"
            )

        with st.form("spread_form"):
            sc1, sc2, sc3, sc4 = st.columns([1, 1, 1, 1.2])
            spr_trader = sc1.text_input("Trader")
            spr_side = sc2.selectbox("Side", ["Buy", "Sell"], help="Buy = long Near/short Far; Sell = short Near/long Far")
            spr_lots = sc3.number_input("Lots (days)", min_value=1, step=1, value=1)
            spr_submit = sc4.form_submit_button("Execute Spread")

        if spr_submit:
            # monthly cap check (count spread lots once)
            log_df = pd.read_csv(LOG_FILE)
            if "lots" in log_df.columns:
                log_df["lots"] = pd.to_numeric(log_df["lots"], errors="coerce").fillna(0)
            if "date" in log_df.columns:
                log_df["date"] = pd.to_datetime(log_df["date"], errors="coerce")
            sel_dt = pd.to_datetime(selected_date, errors="coerce")
            same_month = (log_df["date"].dt.month == sel_dt.month) & (log_df["date"].dt.year == sel_dt.year)
            total_lots_this_month = log_df.loc[(log_df["trader"] == spr_trader) & same_month, "lots"].sum()

            if total_lots_this_month + spr_lots > MAX_MONTHLY_LOTS:
                st.error(f"❌ Trade exceeds monthly limit of {MAX_MONTHLY_LOTS} lots. "
                         f"You have already traded {int(total_lots_this_month)} lots.")
            else:
                fills = exec_spread(curve_today, spr_side, near_ct, far_ct)
                if fills is None:
                    st.error("Pricing error; missing leg.")
                else:
                    st.success(
                        f"✅ {spr_trader} {spr_side} {spr_lots}d of {near_ct}-{far_ct} @ {fills['spread_px']:,.0f}  "
                        f"(legs: {near_ct} {'Buy' if spr_side=='Buy' else 'Sell'} {fills['near_px']:,.0f} / "
                        f"{far_ct} {'Sell' if spr_side=='Buy' else 'Buy'} {fills['far_px']:,.0f})"
                    )
                    row = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "date": selected_date,
                        "trader": spr_trader,
                        "type": "spread",
                        "contract": f"{near_ct}-{far_ct}",
                        "side": spr_side,
                        "price": round(fills["spread_px"], 0),  # store spread price in 'price' too
                        "lots": int(spr_lots),
                        "near": near_ct,
                        "far": far_ct,
                        "spread_px": round(fills["spread_px"], 0),
                        "near_leg_px": round(fills["near_px"], 0),
                        "far_leg_px": round(fills["far_px"], 0)
                    }
                    pd.DataFrame([row])[LOG_COLUMNS].to_csv(
                        LOG_FILE, mode='a', header=False, index=False
                    )

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
