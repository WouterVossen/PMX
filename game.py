import streamlit as st
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import json

# -----------------------------
# Page & basic configuration
# -----------------------------
st.set_page_config(page_title="Panamax Freight Game", layout="wide")

DATA_FOLDER = "data"
CURVE_FILE = os.path.join(DATA_FOLDER, "forward_curves.csv")
LOG_FILE = os.path.join(DATA_FOLDER, "trader_log.csv")
CONFIG_FILE = "config.json"

# Trading limits
ALLOWED_CONTRACTS = ["Sep", "Oct", "Nov", "Dec"]   # game months
MAX_MONTH_NET = 200      # absolute net limit per month
MAX_SLATE_NET = 100      # absolute net across all months

# -----------------------------
# Helpers
# -----------------------------
def read_config_current_day() -> str:
    with open(CONFIG_FILE, "r") as f:
        cfg = json.load(f)
    return cfg.get("current_day")

def load_curve_for_day(day: str) -> pd.DataFrame:
    df = pd.read_csv(CURVE_FILE)
    return df[df["date"] == day].copy()

def load_log() -> pd.DataFrame:
    if not os.path.exists(LOG_FILE):
        cols = [
            "timestamp","date","trader","type",
            "contract","side","price","lots",
            "buy_month","sell_month","spread_price",
            "buy_leg_price","sell_leg_price"
        ]
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(LOG_FILE)
    # Ensure types
    if "lots" in df.columns:
        df["lots"] = pd.to_numeric(df["lots"], errors="coerce").fillna(0).astype(int)
    return df

def signed_effects_for_trade(row: dict) -> dict:
    """
    Return dict of {month: signed_lots} produced by a single row-like trade.
    Outright: Buy +lots, Sell -lots
    Spread:  Buy month +lots, Sell month -lots
    """
    effects = {m: 0 for m in ALLOWED_CONTRACTS}
    if row["type"] == "outright":
        sign = 1 if str(row["side"]).lower() == "buy" else -1
        m = str(row["contract"])
        if m in effects:
            effects[m] += sign * int(row["lots"])
    elif row["type"] == "spread":
        bm = str(row["buy_month"])
        sm = str(row["sell_month"])
        if bm in effects:
            effects[bm] += int(row["lots"])
        if sm in effects:
            effects[sm] -= int(row["lots"])
    return effects

def net_positions_for_trader(trader: str, up_to_date: str) -> dict:
    """Month-to-date nets per month for a trader up to and including selected date."""
    logs = load_log()
    if logs.empty:
        return {m: 0 for m in ALLOWED_CONTRACTS}
    # Only include this trader and trades dated <= selected date
    logs = logs[(logs["trader"] == trader) & (logs["date"] <= up_to_date)]
    nets = {m: 0 for m in ALLOWED_CONTRACTS}
    for _, r in logs.iterrows():
        eff = signed_effects_for_trade(r)
        for m, v in eff.items():
            nets[m] += v
    return nets

def check_limits_after(trader: str, selected_date: str, hypothetical_effects: dict) -> tuple[bool, str]:
    """
    Check if applying hypothetical_effects on top of current nets violates limits.
    Returns (ok, message).
    """
    nets_now = net_positions_for_trader(trader, selected_date)
    nets_new = {m: nets_now.get(m, 0) + hypothetical_effects.get(m, 0) for m in ALLOWED_CONTRACTS}
    # Month net limit
    for m, v in nets_new.items():
        if abs(v) > MAX_MONTH_NET:
            return (False, f"Month limit exceeded in {m}: would be {v} (limit {MAX_MONTH_NET}).")
    # Slate net limit
    slate = sum(nets_new.values())
    if abs(slate) > MAX_SLATE_NET:
        return (False, f"Slate limit exceeded: would be {slate} (limit {MAX_SLATE_NET}).")
    return (True, "OK")

def bid_ask_dicts(curve_today: pd.DataFrame):
    bids = dict(zip(curve_today["contract"], curve_today["bid"]))
    asks = dict(zip(curve_today["contract"], curve_today["ask"]))
    return bids, asks

# -----------------------------
# Load data for today
# -----------------------------
selected_date = read_config_current_day()
curve_today = load_curve_for_day(selected_date)

st.title("🚢 Panamax Freight Paper Trading Game")

if curve_today.empty:
    st.error(f"No market data found for {selected_date}. Please check your config or data files.")
    st.stop()

# Keep only game contracts & ensure ordering
curve_today = curve_today[curve_today["contract"].isin(ALLOWED_CONTRACTS)]
curve_today["contract"] = pd.Categorical(curve_today["contract"], ALLOWED_CONTRACTS, ordered=True)
curve_today = curve_today.sort_values("contract")
contracts_today = list(curve_today["contract"])

bids, asks = bid_ask_dicts(curve_today)

# -----------------------------
# Chart
# -----------------------------
st.subheader(f"📅 Market Day: {selected_date}")
st.markdown("### 📈 Forward Curve")

fig, ax = plt.subplots(figsize=(6.5, 2.8))
contracts = curve_today["contract"]
mids = (curve_today["bid"] + curve_today["ask"]) / 2.0
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

# -----------------------------
# Trading section (outright + spread)
# -----------------------------
st.markdown("### 🧾 Submit Your Trade")

# Trader name (used for limits & logging)
trader = st.text_input("Trader Name", key="trader_name").strip()

# ---- Outright trade inputs (outside the form so we can auto-fill price) ----
st.markdown("**Outright**")
c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.2])

# State defaults
if "ou_contract" not in st.session_state:
    st.session_state["ou_contract"] = contracts_today[0]
if "ou_side" not in st.session_state:
    st.session_state["ou_side"] = "Buy"

# Selectors
st.session_state["ou_contract"] = c1.selectbox("Contract", options=contracts_today, key="ou_contract_select")
st.session_state["ou_side"] = c2.selectbox("Side", ["Buy", "Sell"], key="ou_side_select")
ou_lots = c3.number_input("Lots (days)", min_value=1, step=1, key="ou_lots", value=1)

# Auto-suggest price
sel_contract = st.session_state["ou_contract"]
sel_side = st.session_state["ou_side"]
suggest_ou_price = asks[sel_contract] if sel_side == "Buy" else bids[sel_contract]
ou_price = c4.number_input("Price (auto-filled)", value=float(suggest_ou_price), step=1.0, key="ou_price")

# ---- Spread trade inputs (outside the form to auto-fill spread price) ----
st.markdown("**Calendar Spread (Buy one month / Sell another)**")
s1, s2, s3, s4 = st.columns([1.2, 1.2, 1, 1.2])

if "sp_buy" not in st.session_state:
    st.session_state["sp_buy"] = contracts_today[0]
if "sp_sell" not in st.session_state:
    st.session_state["sp_sell"] = contracts_today[1] if len(contracts_today) > 1 else contracts_today[0]

st.session_state["sp_buy"] = s1.selectbox("Buy month", options=contracts_today, key="sp_buy_select")
st.session_state["sp_sell"] = s2.selectbox("Sell month", options=contracts_today, key="sp_sell_select")
sp_lots = s3.number_input("Lots (days)", min_value=1, step=1, key="sp_lots", value=1)

buy_m = st.session_state["sp_buy"]
sell_m = st.session_state["sp_sell"]
# Conservative auto-filled spread price = (buy leg ask) - (sell leg bid)
suggest_spread_price = float(asks[buy_m]) - float(bids[sell_m])
sp_price = s4.number_input("Spread Price (Buy–Sell)", value=suggest_spread_price, step=1.0, key="sp_price")

# -----------------------------
# Forms (just submit buttons)
# -----------------------------
col_left, col_right = st.columns(2)

with col_left.form("ou_form"):
    ou_submit = st.form_submit_button("Submit Outright Trade")

with col_right.form("sp_form"):
    sp_submit = st.form_submit_button("Submit Spread Trade")

# -----------------------------
# Submission handling
# -----------------------------
def append_rows(rows: list[dict]):
    df = pd.DataFrame(rows)
    df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)

# Outright
if ou_submit:
    if not trader:
        st.error("Please enter your Trader Name before submitting.")
    else:
        hypot = {m: 0 for m in ALLOWED_CONTRACTS}
        sign = 1 if sel_side == "Buy" else -1
        hypot[sel_contract] = sign * int(ou_lots)

        ok, msg = check_limits_after(trader, selected_date, hypot)
        if not ok:
            st.error(f"❌ {msg}")
        else:
            append_rows([{
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date": selected_date,
                "trader": trader,
                "type": "outright",
                "contract": sel_contract,
                "side": sel_side,
                "price": ou_price,
                "lots": int(ou_lots),
                "buy_month": "",
                "sell_month": "",
                "spread_price": "",
                "buy_leg_price": "",
                "sell_leg_price": ""
            }])
            st.success(f"✅ Recorded: {trader} {sel_side} {int(ou_lots)}d {sel_contract} @ ${ou_price:,.0f}")

# Spread
if sp_submit:
    if not trader:
        st.error("Please enter your Trader Name before submitting.")
    elif buy_m == sell_m:
        st.error("Buy month and Sell month must be different for a spread.")
    else:
        # Effects: +lots on buy_m, -lots on sell_m
        hypot = {m: 0 for m in ALLOWED_CONTRACTS}
        hypot[buy_m] += int(sp_lots)
        hypot[sell_m] -= int(sp_lots)

        ok, msg = check_limits_after(trader, selected_date, hypot)
        if not ok:
            st.error(f"❌ {msg}")
        else:
            append_rows([{
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date": selected_date,
                "trader": trader,
                "type": "spread",
                "contract": "",
                "side": "",
                "price": "",
                "lots": int(sp_lots),
                "buy_month": buy_m,
                "sell_month": sell_m,
                "spread_price": sp_price,
                # Store the legs we implied for transparency
                "buy_leg_price": asks[buy_m],
                "sell_leg_price": bids[sell_m]
            }])
            st.success(
                f"✅ Recorded: {trader} BUY {int(sp_lots)}d {buy_m} / SELL {sell_m} @ spread ${sp_price:,.0f} "
                f"(legs used: Buy {buy_m} O={asks[buy_m]:,.0f}, Sell {sell_m} B={bids[sell_m]:,.0f})"
            )

st.markdown("---")

# Admin download (unchanged)
st.markdown("🔐 ### Admin Access")
password = st.text_input("Enter admin password to download trade log", type="password")
if password == "freightadmintrader":
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("📥 Download Trade Log", f, file_name="trader_log.csv")
