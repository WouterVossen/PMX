# === game.py (Panamax Freight Game) ===
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

# -----------------------------
# Page & constants
# -----------------------------
st.set_page_config(page_title="Panamax Freight Game", layout="wide")
ALLOWED_CONTRACTS = ["Sep", "Oct", "Nov", "Dec"]  # trading buckets
LOG_FILE = "data/trader_log.csv"

# Risk limits
PER_MONTH_LIMIT = 200      # abs(net) per month
SLATE_LIMIT = 100          # abs(sum of nets across months)

# -----------------------------
# Load config & data
# -----------------------------
with open("config.json", "r") as f:
    config = json.load(f)
selected_date = config.get("current_day")

curve_df = pd.read_csv("data/forward_curves.csv")
# optional: strip spaces in header/values
curve_df.columns = [c.strip() for c in curve_df.columns]
for col in ["date", "contract"]:
    curve_df[col] = curve_df[col].astype(str).str.strip()

# Filter to today + allowed contracts
curve_today = curve_df[(curve_df["date"] == str(selected_date)) &
                       (curve_df["contract"].isin(ALLOWED_CONTRACTS))].copy()

st.title("🚢 Panamax Freight Paper Trading Game")

if curve_today.empty:
    st.error(f"No market data found for {selected_date}. Check `config.json` or `data/forward_curves.csv`.")
    st.stop()

# Build quick maps for auto-pricing
bid_map = dict(zip(curve_today["contract"], curve_today["bid"]))
ask_map = dict(zip(curve_today["contract"], curve_today["ask"]))
contracts_today = [m for m in ALLOWED_CONTRACTS if m in curve_today["contract"].tolist()]

# -----------------------------
# Plot the forward curve
# -----------------------------
st.subheader(f"📅 Market Day: {selected_date}")
st.markdown("### 📈 Forward Curve")

fig, ax = plt.subplots(figsize=(8, 3))
contracts = curve_today["contract"]
mids = (curve_today["bid"] + curve_today["ask"]) / 2
ax.plot(contracts, mids, marker='o', label="Mid Price")
ax.fill_between(contracts, curve_today["bid"], curve_today["ask"], alpha=0.2, label="Bid/Ask Spread")
for _, row in curve_today.iterrows():
    ax.text(row["contract"], row["bid"] - 50, f"B: {int(row['bid'])}", ha="center", fontsize=8)
    ax.text(row["contract"], row["ask"] + 50, f"O: {int(row['ask'])}", ha="center", fontsize=8)
ax.set_ylabel("USD/Day")
ax.legend(loc="upper right")
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# -----------------------------
# Helpers: positions & checks
# -----------------------------
def load_log():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        # normalize dtypes
        if "date" in df.columns:
            df["date"] = df["date"].astype(str)
        return df
    return pd.DataFrame()

def month_year(s):
    """return (year, month number) from 'M/D/YYYY' or 'YYYY-MM-DD' string"""
    try:
        dt = pd.to_datetime(s)
        return dt.year, dt.month
    except Exception:
        return None, None

def current_month_slice(df: pd.DataFrame, date_str: str):
    y, m = month_year(date_str)
    if y is None:
        return df.iloc[0:0]
    dft = df.copy()
    dft["__dt"] = pd.to_datetime(dft["date"], errors="coerce")
    return dft[(dft["__dt"].dt.year == y) & (dft["__dt"].dt.month == m)].drop(columns="__dt")

def net_positions_for_trader(trader: str, date_str: str):
    """
    Return dict { 'Sep': net_lots, 'Oct': .. } for the month of `date_str`,
    based on existing LOG_FILE.
    Outright: Buy +lots / Sell -lots
    Spread: +lots on buy_month, -lots on sell_month
    """
    log = load_log()
    if log.empty:
        return {m: 0 for m in ALLOWED_CONTRACTS}

    log = current_month_slice(log, date_str)
    log = log[log["trader"].astype(str).str.strip().str.lower() == str(trader).strip().lower()]

    nets = {m: 0 for m in ALLOWED_CONTRACTS}

    for _, r in log.iterrows():
        typ = str(r.get("type", "")).upper()
        if typ == "OUTRIGHT":
            m = str(r.get("contract", "")).strip()
            side = str(r.get("side", "")).capitalize()
            lots = int(r.get("lots", 0))
            if m in nets:
                nets[m] += lots if side == "Buy" else -lots
        elif typ == "SPREAD":
            buy_m = str(r.get("buy_month", "")).strip()
            sell_m = str(r.get("sell_month", "")).strip()
            lots = int(r.get("lots", 0))
            if buy_m in nets:
                nets[buy_m] += lots
            if sell_m in nets:
                nets[sell_m] -= lots

    return nets

def would_break_limits(nets_before: dict, delta: dict):
    """
    Given existing nets and a proposed delta dict (e.g., {'Oct': +10, 'Nov': -10}),
    return (ok, message, nets_after).
    """
    nets_after = nets_before.copy()
    for m, v in delta.items():
        nets_after[m] = nets_after.get(m, 0) + int(v)

    # Per-month abs limit
    for m in ALLOWED_CONTRACTS:
        if abs(nets_after.get(m, 0)) > PER_MONTH_LIMIT:
            return (False, f"Per-month limit exceeded in {m}: "
                           f"{nets_after.get(m, 0)} (limit {PER_MONTH_LIMIT})", nets_after)

    # Slate abs limit
    slate = sum(nets_after.get(m, 0) for m in ALLOWED_CONTRACTS)
    if abs(slate) > SLATE_LIMIT:
        return (False, f"Slate limit exceeded: {slate} (limit {SLATE_LIMIT})", nets_after)

    return True, "OK", nets_after

def append_log_row(row: dict):
    df = pd.DataFrame([row])
    df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)

# -----------------------------
# Trading forms (Outright + Spread)
# -----------------------------
st.markdown("### 🧾 Submit Trades")

trader = st.text_input("Trader Name", key="trader_name", placeholder="e.g., Alice")

if not trader:
    st.info("Enter your name to enable trade tickets.")
    st.stop()

# Compute current nets for display
current_nets = net_positions_for_trader(trader, selected_date)
slate_now = sum(current_nets[m] for m in ALLOWED_CONTRACTS)

with st.expander("📊 Your Current Net Positions (month-to-date)", expanded=True):
    cols = st.columns(len(ALLOWED_CONTRACTS) + 1)
    for i, m in enumerate(ALLOWED_CONTRACTS):
        cols[i].metric(m, f"{current_nets.get(m, 0)}")
    cols[-1].metric("Slate", f"{slate_now}")

st.divider()

# ----- Outright form -----
st.markdown("#### Outright")
c1, c2, c3, c4 = st.columns([1.2, 0.8, 0.8, 0.8])

# Session defaults so the number_input can be prefilled
ou_default_contract = st.session_state.get("ou_contract", contracts_today[0])
ou_default_side = st.session_state.get("ou_side", "Buy")

# Default price from the book
default_ou_price = int(ask_map.get(ou_default_contract, 0)) if ou_default_side == "Buy" else int(bid_map.get(ou_default_contract, 0))

with st.form("outright_form"):
    ou_contract = c1.selectbox("Contract", options=contracts_today, key="ou_contract")
    ou_side = c2.selectbox("Side", ["Buy", "Sell"], key="ou_side")
    # recompute default based on current selections
    default_ou_price = int(ask_map[ou_contract]) if ou_side == "Buy" else int(bid_map[ou_contract])
    ou_price = c3.number_input("Price", step=1, value=default_ou_price, key="ou_price")
    ou_lots = c4.number_input("Lots (days)", min_value=1, step=1, value=1, key="ou_lots")
    submit_outright = st.form_submit_button("Submit Outright Trade")

if submit_outright:
    # Build the delta for limits
    delta = {m: 0 for m in ALLOWED_CONTRACTS}
    delta[ou_contract] = ou_lots if ou_side == "Buy" else -ou_lots

    ok, msg, nets_after = would_break_limits(current_nets, delta)
    if not ok:
        st.error(f"❌ {msg}")
    else:
        append_log_row({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date": selected_date,
            "trader": trader,
            "type": "OUTRIGHT",
            "contract": ou_contract,
            "side": ou_side,
            "price": int(ou_price),
            "lots": int(ou_lots)
        })
        st.success(f"✅ {trader} {ou_side} {int(ou_lots)}d {ou_contract} @ ${int(ou_price)}")
        st.rerun()

st.divider()

# ----- Spread form -----
st.markdown("#### Calendar Spread (Buy one month, Sell another)")
sc1, sc2, sc3, sc4 = st.columns([1.2, 1.2, 0.8, 1.0])

# sensible defaults
sp_buy_default = st.session_state.get("sp_buy", contracts_today[0])
sp_sell_default = st.session_state.get("sp_sell", contracts_today[1] if len(contracts_today) > 1 else contracts_today[0])

with st.form("spread_form"):
    sp_buy = sc1.selectbox("Buy month", options=contracts_today, key="sp_buy")
    # ensure sell != buy
    sell_opts = [m for m in contracts_today if m != sp_buy] or contracts_today
    sp_sell = sc2.selectbox("Sell month", options=sell_opts, index=0, key="sp_sell")
    sp_lots = sc3.number_input("Lots (days)", min_value=1, step=1, value=1, key="sp_lots")
    # default spread price = Ask(buy) - Bid(sell)
    default_sp_px = int(ask_map[sp_buy]) - int(bid_map[sp_sell]) if sp_buy != sp_sell else 0
    sp_px = sc4.number_input("Spread Price (Buy−Sell)", step=1, value=default_sp_px, key="sp_px")
    submit_spread = st.form_submit_button("Submit Spread Trade")

if submit_spread:
    if sp_buy == sp_sell:
        st.error("❌ Buy and Sell months must be different.")
    else:
        delta = {m: 0 for m in ALLOWED_CONTRACTS}
        # +lots on buy leg, -lots on sell leg
        delta[sp_buy] += int(sp_lots)
        delta[sp_sell] -= int(sp_lots)

        ok, msg, nets_after = would_break_limits(current_nets, delta)
        if not ok:
            st.error(f"❌ {msg}")
        else:
            append_log_row({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date": selected_date,
                "trader": trader,
                "type": "SPREAD",
                "buy_month": sp_buy,
                "sell_month": sp_sell,
                "spread_price": int(sp_px),
                "lots": int(sp_lots)
            })
            st.success(f"✅ {trader} BUY {int(sp_lots)}d {sp_buy} / SELL {int(sp_lots)}d {sp_sell} @ {int(sp_px)}")
            st.rerun()

st.markdown("---")

# -----------------------------
# Admin download (optional)
# -----------------------------
st.markdown("🔐 ### Admin Access")
password = st.text_input("Enter admin password to download trade log", type="password")
if password == "freightadmintrader":
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("📥 Download Trade Log", f, file_name="trader_log.csv")
    else:
        st.info("No trades logged yet.")
