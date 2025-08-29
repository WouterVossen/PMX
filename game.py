import streamlit as st
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import json

# --------------------------
# Page setup
# --------------------------
st.set_page_config(page_title="Panamax Freight Game", layout="wide")

# --------------------------
# Load config (current trading day)
# --------------------------
with open("config.json", "r") as f:
    config = json.load(f)
selected_date = config.get("current_day")  # e.g., "9/1/2025" to match CSV format

# --------------------------
# Load forward curve (for the selected day only)
# --------------------------
curve_df = pd.read_csv("data/forward_curves.csv")
curve_today = curve_df[curve_df["date"] == selected_date].copy()

st.title("🚢 Panamax Freight Paper Trading Game")

if curve_today.empty:
    st.error(f"No market data found for {selected_date}. Please check your config or data files.")
    st.stop()

# Ensure contract order and prepare bid/ask dicts
contracts_today = list(curve_today["contract"])
bids = dict(zip(curve_today["contract"], curve_today["bid"]))
asks = dict(zip(curve_today["contract"], curve_today["ask"]))

# --------------------------
# Plot: compact forward curve with bid/offer labels
# --------------------------
st.subheader(f"📅 Market Day: {selected_date}")
st.markdown("### 📈 Forward Curve")

fig, ax = plt.subplots(figsize=(7, 3))  # smaller visual
mids = (curve_today["bid"] + curve_today["ask"]) / 2
ax.plot(curve_today["contract"], mids, marker="o", label="Mid Price")
ax.fill_between(curve_today["contract"], curve_today["bid"], curve_today["ask"], alpha=0.20, label="Bid/Ask Spread")

for _, row in curve_today.iterrows():
    ax.text(row["contract"], row["bid"] - 40, f"B: {int(row['bid'])}", ha="center", fontsize=8)
    ax.text(row["contract"], row["ask"] + 40, f"O: {int(row['ask'])}", ha="center", fontsize=8)

ax.set_ylabel("USD/Day")
ax.legend(loc="upper right")
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

# ==========================
# Position & risk utilities
# ==========================

LOG_FILE = "data/trader_log.csv"
PER_MONTH_CAP = 200   # |net| per single contract month
SLATE_CAP = 100       # |sum of net across months|

MONTH_ORDER = ["Sep", "Oct", "Nov", "Dec"]  # target game months

def _ensure_log_exists():
    cols = [
        "timestamp","date","trader","type",
        "contract","side","price","lots",
        "spread_buy","spread_sell","spread_price"
    ]
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=cols).to_csv(LOG_FILE, index=False)

def _load_log_df():
    _ensure_log_exists()
    df = pd.read_csv(LOG_FILE)
    # keep date parsing tolerant (the log’s date is game-day string)
    return df

def _month_from_contract(contract_str: str) -> str:
    # Contracts are month labels like "Sep","Oct","Nov","Dec"
    return contract_str

def _positions_for_trader_mtd(trader: str, upto_date_str: str):
    """
    Return dict of net lots per month (Sep/Oct/Nov/Dec) and slate net,
    including only trades with log 'date' in the same month & year as selected_date.
    """
    df = _load_log_df()
    if df.empty:
        return {m: 0 for m in MONTH_ORDER}, 0

    # Convert log 'date' and selected to datetime for same-month filtering
    try:
        df["date_dt"] = pd.to_datetime(df["date"])
        selected_dt = pd.to_datetime(upto_date_str)
        same_month = (df["date_dt"].dt.month == selected_dt.month) & (df["date_dt"].dt.year == selected_dt.year)
        df = df[same_month]
    except Exception:
        # if parsing fails, fall back to using the exact string match
        df = df[df["date"] == upto_date_str]

    df = df[df["trader"].astype(str).str.strip().str.lower() == str(trader).strip().lower()]

    per_month = {m: 0 for m in MONTH_ORDER}

    for _, r in df.iterrows():
        ttype = str(r.get("type","")).lower()
        lots = int(r.get("lots", 0))
        if ttype == "outright":
            m = _month_from_contract(str(r.get("contract","")))
            side = str(r.get("side","")).lower()
            if m in per_month:
                per_month[m] += lots if side == "buy" else -lots
        elif ttype == "spread":
            buy_m = _month_from_contract(str(r.get("spread_buy","")))
            sell_m = _month_from_contract(str(r.get("spread_sell","")))
            if buy_m in per_month:
                per_month[buy_m] += lots
            if sell_m in per_month:
                per_month[sell_m] -= lots

    slate = sum(per_month.values())
    return per_month, slate

def _check_limits_after(trader: str, date_str: str, changes: dict):
    """
    changes: dict of {month: delta_lots}, positive for long, negative for short.
    Returns (ok: bool, message: str)
    """
    per_month, slate = _positions_for_trader_mtd(trader, date_str)
    # apply changes
    new_per_month = per_month.copy()
    for m, dlt in changes.items():
        if m in new_per_month:
            new_per_month[m] += dlt
    new_slate = sum(new_per_month.values())

    # Per-month cap
    for m, net in new_per_month.items():
        if abs(net) > PER_MONTH_CAP:
            return False, f"Per-month limit exceeded in {m}: |{net}| > {PER_MONTH_CAP}."

    # Slate cap
    if abs(new_slate) > SLATE_CAP:
        return False, f"Slate limit exceeded: |{new_slate}| > {SLATE_CAP}."

    return True, "OK"

def _append_log_row(row: dict):
    df = pd.DataFrame([row])
    df.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0, index=False)

# ==========================
# Trade entry
# ==========================

st.markdown("### 🧾 Submit Your Trades")

# ---------- OUTRIGHT ----------
st.markdown("**Outright**")
with st.form("ou_form"):
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.2])

    ou_contract = c1.selectbox("Contract", options=contracts_today, key="ou_contract")
    ou_side = c2.selectbox("Side", ["Buy", "Sell"], key="ou_side")
    ou_lots = c3.number_input("Lots (days)", min_value=1, step=1, value=1, key="ou_lots")

    # Auto-fill price: Buy at ask, Sell at bid
    default_ou_price = float(asks[ou_contract]) if ou_side == "Buy" else float(bids[ou_contract])
    ou_price = c4.number_input("Price (auto-filled)", value=default_ou_price, step=1.0, key="ou_price")

    ou_trader = st.text_input("Trader Name", key="ou_trader")
    ou_submit = st.form_submit_button("Submit Outright Trade")

if ou_submit:
    if not ou_trader.strip():
        st.error("Please enter your Trader Name for the outright trade.")
    else:
        # Position impact
        delta = { _month_from_contract(ou_contract): (ou_lots if ou_side == "Buy" else -ou_lots) }
        ok, msg = _check_limits_after(ou_trader, selected_date, delta)
        if not ok:
            st.error(f"❌ {msg}")
        else:
            _append_log_row({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date": selected_date,
                "trader": ou_trader,
                "type": "outright",
                "contract": ou_contract,
                "side": ou_side,
                "price": ou_price,
                "lots": int(ou_lots),
                "spread_buy": "",
                "spread_sell": "",
                "spread_price": ""
            })
            st.success(f"✅ Outright submitted: {ou_trader} {ou_side} {int(ou_lots)}d {ou_contract} @ {int(ou_price)}")

st.markdown("---")

# ---------- SPREAD ----------
st.markdown("**Calendar Spread (Buy one month / Sell another)**")
with st.form("sp_form"):
    s1, s2, s3, s4 = st.columns([1.2, 1.2, 1, 1.2])

    sp_buy = s1.selectbox("Buy month", options=contracts_today, key="sp_buy")
    sp_sell = s2.selectbox("Sell month", options=[m for m in contracts_today if m != sp_buy], key="sp_sell")
    sp_lots = s3.number_input("Lots (days)", min_value=1, step=1, value=1, key="sp_lots")

    # Auto-fill spread price: buy at ask, sell at bid => spread = buy_ask - sell_bid
    default_spread = float(asks[sp_buy]) - float(bids[sp_sell])
    sp_price = s4.number_input("Spread Price (Buy–Sell)", value=default_spread, step=1.0, key="sp_price")

    sp_trader = st.text_input("Trader Name", key="sp_trader")
    sp_submit = st.form_submit_button("Submit Spread Trade")

if sp_submit:
    if sp_buy == sp_sell:
        st.error("Choose two different months for a spread.")
    elif not sp_trader.strip():
        st.error("Please enter your Trader Name for the spread.")
    else:
        # Position impact: +lots on buy month, -lots on sell month
        delta = {
            _month_from_contract(sp_buy): int(sp_lots),
            _month_from_contract(sp_sell): -int(sp_lots)
        }
        ok, msg = _check_limits_after(sp_trader, selected_date, delta)
        if not ok:
            st.error(f"❌ {msg}")
        else:
            # Log spread summary row
            _append_log_row({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date": selected_date,
                "trader": sp_trader,
                "type": "spread",
                "contract": "",
                "side": "",
                "price": "",
                "lots": int(sp_lots),
                "spread_buy": sp_buy,
                "spread_sell": sp_sell,
                "spread_price": sp_price
            })
            # And log both legs as outright for auditability of positions
            _append_log_row({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date": selected_date,
                "trader": sp_trader,
                "type": "outright",
                "contract": sp_buy,
                "side": "Buy",
                "price": float(asks[sp_buy]),
                "lots": int(sp_lots),
                "spread_buy": "",
                "spread_sell": "",
                "spread_price": ""
            })
            _append_log_row({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date": selected_date,
                "trader": sp_trader,
                "type": "outright",
                "contract": sp_sell,
                "side": "Sell",
                "price": float(bids[sp_sell]),
                "lots": int(sp_lots),
                "spread_buy": "",
                "spread_sell": "",
                "spread_price": ""
            })
            st.success(
                f"✅ Spread submitted: {sp_trader} BUY {int(sp_lots)}d {sp_buy} / SELL {int(sp_lots)}d {sp_sell} @ {int(sp_price)}"
            )

st.markdown("---")

# ---------- Admin download ----------
st.markdown("🔐 **Admin Access**")
password = st.text_input("Enter admin password to download trade log", type="password")
if password == "freightadmintrader":
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("📥 Download Trade Log", f, file_name="trader_log.csv")
    else:
        st.info("No trades logged yet.")
