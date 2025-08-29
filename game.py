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
# Absolute data paths (prevents day-to-day resets from CWD changes)
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------
# Load config (current trading day)
# --------------------------
with open(os.path.join(BASE_DIR, "config.json"), "r") as f:
    config = json.load(f)
selected_date = config.get("current_day")  # e.g., "9/1/2025" to match CSV format

# --------------------------
# Load forward curve (for the selected day only)
# --------------------------
curve_df = pd.read_csv(os.path.join(DATA_DIR, "forward_curves.csv"))
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

LOG_FILE = os.path.join(DATA_DIR, "trader_log.csv")
PER_MONTH_CAP = 200   # |net| per single contract month
SLATE_CAP = 100       # |sum of net across months|

# Canonical month set (UI + CSV must map here)
MONTH_ORDER = ["Sep", "Oct", "Nov", "Dec"]

# Canonicalizer to ensure spreads and outrights land in the SAME bucket
_CANON_MAP = {"sep":"Sep","oct":"Oct","nov":"Nov","dec":"Dec"}
def _canon_month(s: str) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    if not t:
        return ""
    if t in MONTH_ORDER:
        return t
    key = t[:3].lower()
    return _CANON_MAP.get(key, t)

# ==========================
# Positions sheet utilities (persistent per-trader buckets)
# ==========================
POS_FILE = os.path.join(DATA_DIR, "positions.csv")

def _norm_trader_key(trader: str) -> str:
    return str(trader).strip().lower()

def _ensure_positions_exists():
    cols = ["trader", "trader_key"] + MONTH_ORDER + ["slate"]
    if not os.path.exists(POS_FILE):
        pd.DataFrame(columns=cols).to_csv(POS_FILE, index=False)

def _load_positions():
    _ensure_positions_exists()
    df = pd.read_csv(POS_FILE)
    if "trader_key" not in df.columns:
        df["trader_key"] = df["trader"].astype(str).map(_norm_trader_key)
    for m in MONTH_ORDER:
        if m not in df.columns:
            df[m] = 0
        df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0).astype(int)
    if "slate" not in df.columns:
        df["slate"] = df[MONTH_ORDER].sum(axis=1).astype(int)
    df["trader"] = df["trader"].astype(str)
    df["trader_key"] = df["trader_key"].astype(str)
    return df

def _save_positions(df: pd.DataFrame):
    if not df.empty:
        if "trader_key" not in df.columns:
            df["trader_key"] = df["trader"].astype(str).map(_norm_trader_key)
        for m in MONTH_ORDER:
            df[m] = pd.to_numeric(df[m], errors="coerce").fillna(0).astype(int)
        df["slate"] = df[MONTH_ORDER].sum(axis=1).astype(int)
    df.to_csv(POS_FILE, index=False)

def _get_trader_positions(trader: str):
    df = _load_positions()
    key = _norm_trader_key(trader)
    mask = df["trader_key"] == key
    if not mask.any():
        new = {"trader": trader, "trader_key": key}
        new.update({m: 0 for m in MONTH_ORDER})
        new["slate"] = 0
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        _save_positions(df)
        return {m: 0 for m in MONTH_ORDER}, 0
    row = df[mask].iloc[0]
    per_month = {m: int(row[m]) for m in MONTH_ORDER}
    slate = int(row.get("slate", sum(per_month.values())))
    return per_month, slate

def _apply_position_changes(trader: str, changes: dict):
    df = _load_positions()
    key = _norm_trader_key(trader)
    mask = df["trader_key"] == key
    if not mask.any():
        new = {"trader": trader, "trader_key": key}
        new.update({m: 0 for m in MONTH_ORDER})
        new["slate"] = 0
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
        mask = df["trader_key"] == key
    idx = df[mask].index[0]
    for m, d in changes.items():
        cm = _canon_month(m)
        if cm in MONTH_ORDER:
            df.at[idx, cm] = int(df.at[idx, cm]) + int(d)
    _save_positions(df)

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
    return df

def _month_from_contract(contract_str: str) -> str:
    return _canon_month(contract_str)

def _positions_for_trader_mtd(trader: str, upto_date_str: str):
    # legacy helper (not used for limits)
    df = _load_log_df()
    if df.empty:
        return {m: 0 for m in MONTH_ORDER}, 0
    try:
        df["date_dt"] = pd.to_datetime(df["date"])
        selected_dt = pd.to_datetime(upto_date_str)
        same_month = (df["date_dt"].dt.month == selected_dt.month) & (df["date_dt"].dt.year == selected_dt.year)
        df = df[same_month]
    except Exception:
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
    Strong enforcement (Option A): block any order if any single leg would push a month over PER_MONTH_CAP.
    Applies to both outrights and spreads. 'changes' is {month: delta_lots}.
    """
    # Current live positions
    per_month, _ = _get_trader_positions(trader)

    # 1) Legwise pre-flight: immediate block if any proposed leg breaches cap
    for m, dlt in changes.items():
        cm = _canon_month(m)
        if cm not in per_month:
            return False, f"Unknown/disabled month '{m}'."
        proposed = int(per_month[cm]) + int(dlt)
        if abs(proposed) > PER_MONTH_CAP:
            return False, f"Per-month limit exceeded in {cm}: |{proposed}| > {PER_MONTH_CAP}."

    # 2) Apply changes and re-check defensively (covers multi-leg collisions)
    new_per_month = per_month.copy()
    for m, dlt in changes.items():
        cm = _canon_month(m)
        new_per_month[cm] += int(dlt)

    for m, net in new_per_month.items():
        if abs(net) > PER_MONTH_CAP:
            return False, f"Per-month limit exceeded in {m}: |{net}| > {PER_MONTH_CAP}."

    # 3) Slate cap unchanged (your slate restriction already works)
    new_slate = sum(new_per_month.values())
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
        cm = _month_from_contract(ou_contract)
        if cm not in MONTH_ORDER:
            st.error(f"Unknown month/contract '{ou_contract}'.")
        else:
            delta = { cm: (ou_lots if ou_side == "Buy" else -ou_lots) }
            ok, msg = _check_limits_after(ou_trader, selected_date, delta)
            if not ok:
                st.error(f"❌ {msg}")
            else:
                _append_log_row({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "date": selected_date,
                    "trader": ou_trader,
                    "type": "outright",
                    "contract": cm,
                    "side": ou_side,
                    "price": ou_price,
                    "lots": int(ou_lots),
                    "spread_buy": "",
                    "spread_sell": "",
                    "spread_price": ""
                })
                st.success(f"✅ Outright submitted: {ou_trader} {ou_side} {int(ou_lots)}d {cm} @ {int(ou_price)}")
                _apply_position_changes(ou_trader, { cm: (int(ou_lots) if ou_side == "Buy" else -int(ou_lots)) })

st.markdown("---")

# ---------- SPREAD ----------
st.markdown("**Calendar Spread (Buy one month / Sell another)**")
with st.form("sp_form"):
    s1, s2, s3, s4 = st.columns([1.2, 1.2, 1, 1.2])

    sp_buy_raw = s1.selectbox("Buy month", options=contracts_today, key="sp_buy")
    sp_sell_raw = s2.selectbox("Sell month", options=[m for m in contracts_today if m != sp_buy_raw], key="sp_sell")
    sp_lots = s3.number_input("Lots (days)", min_value=1, step=1, value=1, key="sp_lots")

    # Auto-fill spread price: buy at ask, sell at bid => spread = buy_ask - sell_bid
    default_spread = float(asks[sp_buy_raw]) - float(bids[sp_sell_raw])
    sp_price = s4.number_input("Spread Price (Buy–Sell)", value=default_spread, step=1.0, key="sp_price")

    sp_trader = st.text_input("Trader Name", key="sp_trader")
    sp_submit = st.form_submit_button("Submit Spread Trade")

if sp_submit:
    sp_buy = _month_from_contract(sp_buy_raw)
    sp_sell = _month_from_contract(sp_sell_raw)

    if sp_buy == sp_sell:
        st.error("Choose two different months for a spread.")
    elif sp_buy not in MONTH_ORDER or sp_sell not in MONTH_ORDER:
        st.error(f"Unknown months in spread: BUY '{sp_buy_raw}' / SELL '{sp_sell_raw}'.")
    elif not sp_trader.strip():
        st.error("Please enter your Trader Name for the spread.")
    else:
        # HARD LIMIT ENFORCEMENT: same buckets as outrights
        delta = { sp_buy: int(sp_lots), sp_sell: -int(sp_lots) }
        ok, msg = _check_limits_after(sp_trader, selected_date, delta)
        if not ok:
            st.error(f"❌ {msg}")
        else:
            # Log spread summary row (canonical months)
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
            # Log outright legs (audit only)
            _append_log_row({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date": selected_date,
                "trader": sp_trader,
                "type": "outright",
                "contract": sp_buy,
                "side": "Buy",
                "price": float(asks[sp_buy_raw]),
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
                "price": float(bids[sp_sell_raw]),
                "lots": int(sp_lots),
                "spread_buy": "",
                "spread_sell": "",
                "spread_price": ""
            })
            st.success(
                f"✅ Spread submitted: {sp_trader} BUY {int(sp_lots)}d {sp_buy} / SELL {int(sp_lots)}d {sp_sell} @ {int(sp_price)}"
            )
            # Apply net effect to the same live buckets
            _apply_position_changes(sp_trader, { sp_buy:  int(sp_lots), sp_sell: -int(sp_lots) })

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
