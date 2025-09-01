import streamlit as st
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import json
from io import BytesIO, StringIO

# --------------------------
# Page setup
# --------------------------
st.set_page_config(page_title="Panamax Freight Game", layout="wide")

# --------------------------
# Absolute data paths
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CURVE_FILE = os.path.join(DATA_DIR, "forward_curves.csv")
LOG_FILE   = os.path.join(DATA_DIR, "trader_log.csv")
POS_FILE   = os.path.join(DATA_DIR, "positions.csv")

# --------------------------
# Load config (current trading day)
# --------------------------
with open(os.path.join(BASE_DIR, "config.json"), "r") as f:
    config = json.load(f)
selected_date = config.get("current_day")  # e.g. "9/1/2025"

# --------------------------
# Load forward curve (for the selected day only)
# --------------------------
curve_df = pd.read_csv(CURVE_FILE)
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

fig, ax = plt.subplots(figsize=(7, 2))  # compact
mids = (curve_today["bid"] + curve_today["ask"]) / 2
ax.plot(curve_today["contract"], mids, marker="o")
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

PER_MONTH_CAP = 200   # |net| per single contract month
SLATE_CAP     = 100   # |sum of net across months|

# Canonical month set (UI + CSV must map here)
MONTH_ORDER = ["Sep", "Oct", "Nov", "Dec"]
_CANON_MAP = {"sep": "Sep", "oct": "Oct", "nov": "Nov", "dec": "Dec"}

def _canon_month(s: str) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    if not t:
        return ""
    if t in MONTH_ORDER:
        return t
    return _CANON_MAP.get(t[:3].lower(), t)

def _norm_trader_key(trader: str) -> str:
    return str(trader).strip().lower()

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
    return pd.read_csv(LOG_FILE)

# ---- positions.csv maintained only for your audit; not used for checks
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

# ===== LIVE positions from LOG (truth) =====
def _live_positions_from_log(trader: str, upto_date_str: str):
    _ensure_log_exists()
    df = pd.read_csv(LOG_FILE)
    if df.empty:
        return {m: 0 for m in MONTH_ORDER}
    key = _norm_trader_key(trader)
    df["trader_key"] = df["trader"].astype(str).str.strip().str.lower()
    df = df[df["trader_key"] == key]
    df = df[df["date"].astype(str) <= str(upto_date_str)]  # string compare to match CSV format
    pos = {m: 0 for m in MONTH_ORDER}
    for _, r in df.iterrows():
        ttype = str(r.get("type", "")).lower()
        lots  = int(pd.to_numeric(r.get("lots", 0), errors="coerce"))
        if ttype == "outright":
            m = _canon_month(r.get("contract", ""))
            side = str(r.get("side", "")).lower()
            if m in pos:
                pos[m] += lots if side == "buy" else -lots
        elif ttype == "spread":
            # Ignore spread summary row; we also log outright legs and those carry exposure.
            continue
    return pos

# ===== Risk checks using LOG state =====
def _check_limits_after(trader: str, date_str: str, changes: dict):
    current = _live_positions_from_log(trader, date_str)

    # legwise
    for m, dlt in changes.items():
        cm = _canon_month(m)
        if cm not in current:
            return False, f"Unknown/disabled month '{m}'."
        proposed = int(current[cm]) + int(dlt)
        if abs(proposed) > PER_MONTH_CAP:
            return False, f"Per-month limit exceeded in {cm}: |{proposed}| > {PER_MONTH_CAP}."

    # both legs together
    new_pos = current.copy()
    for m, dlt in changes.items():
        cm = _canon_month(m)
        new_pos[cm] = int(new_pos[cm]) + int(dlt)

    for m, net in new_pos.items():
        if abs(net) > PER_MONTH_CAP:
            return False, f"Per-month limit exceeded in {m}: |{net}| > {PER_MONTH_CAP}."

    slate = sum(new_pos.values())
    if abs(slate) > SLATE_CAP:
        return False, f"Slate limit exceeded: |{slate}| > {SLATE_CAP}."
    return True, "OK"

def _append_log_row(row: dict):
    df = pd.DataFrame([row])
    header_needed = not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0
    df.to_csv(LOG_FILE, mode="a", header=header_needed, index=False)

# ==========================
# Trade entry
# ==========================

st.markdown("### 🧾 Submit Your Trades")

# ---------- OUTRIGHT ----------
st.markdown("**Outright**")
with st.form("ou_form"):
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.2])

    ou_contract = c1.selectbox("Contract", options=contracts_today, key="ou_contract")
    ou_side     = c2.selectbox("Side", ["Buy", "Sell"], key="ou_side")
    ou_lots     = c3.number_input("Lots (days)", min_value=1, step=1, value=1, key="ou_lots")

    # Auto-fill: Buy at ask, Sell at bid
    default_ou_price = float(asks[ou_contract]) if ou_side == "Buy" else float(bids[ou_contract])
    ou_price = c4.number_input("Price (auto-filled when you press submit)", value=default_ou_price, step=1.0, key="ou_price")

    ou_trader = st.text_input("Trader Name", key="ou_trader")
    ou_submit = st.form_submit_button("Submit Outright Trade")

if ou_submit:
    if not ou_trader.strip():
        st.error("Please enter your Trader Name for the outright trade.")
    else:
        cm = _canon_month(ou_contract)
        if cm not in MONTH_ORDER:
            st.error(f"Unknown month/contract '{ou_contract}'.")
        else:
            delta = {cm: (ou_lots if ou_side == "Buy" else -ou_lots)}
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
                    "price": float(ou_price),
                    "lots": int(ou_lots),
                    "spread_buy": "",
                    "spread_sell": "",
                    "spread_price": ""
                })
                st.success(f"✅ Outright submitted: {ou_trader} {ou_side} {int(ou_lots)}d {cm} @ {int(ou_price)}")
                _apply_position_changes(ou_trader, {cm: (int(ou_lots) if ou_side == "Buy" else -int(ou_lots))})

st.markdown("---")

# ---------- SPREAD ----------
st.markdown("**Calendar Spread (Buy one month / Sell another)**")
with st.form("sp_form"):
    s1, s2, s3, s4 = st.columns([1.2, 1.2, 1, 1.2])

    sp_buy_raw  = s1.selectbox("Buy month", options=contracts_today, key="sp_buy")
    sp_sell_raw = s2.selectbox("Sell month", options=[m for m in contracts_today if m != sp_buy_raw], key="sp_sell")
    sp_lots     = s3.number_input("Lots (days)", min_value=1, step=1, value=1, key="sp_lots")

    # Auto-fill spread: buy at ask, sell at bid => spread = ask(buy) - bid(sell)
    default_spread = float(asks[sp_buy_raw]) - float(bids[sp_sell_raw])
    sp_price = s4.number_input("Spread Price (auto-filled when you press submit)", value=default_spread, step=1.0, key="sp_price")

    sp_trader = st.text_input("Trader Name", key="sp_trader")
    sp_submit = st.form_submit_button("Submit Spread Trade")

if sp_submit:
    sp_buy  = _canon_month(sp_buy_raw)
    sp_sell = _canon_month(sp_sell_raw)

    if sp_buy == sp_sell:
        st.error("Choose two different months for a spread.")
    elif sp_buy not in MONTH_ORDER or sp_sell not in MONTH_ORDER:
        st.error(f"Unknown months in spread: BUY '{sp_buy_raw}' / SELL '{sp_sell_raw}'.")
    elif not sp_trader.strip():
        st.error("Please enter your Trader Name for the spread.")
    else:
        delta = {sp_buy: int(sp_lots), sp_sell: -int(sp_lots)}
        ok, msg = _check_limits_after(sp_trader, selected_date, delta)
        if not ok:
            st.error(f"❌ {msg}")
        else:
            # Summary spread row
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
                "spread_price": float(sp_price)
            })
            # Outright audit legs (executable marks used)
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
            st.success(f"✅ Spread submitted: {sp_trader} BUY {int(sp_lots)}d {sp_buy} / SELL {int(sp_lots)}d {sp_sell} @ {int(sp_price)}")
            _apply_position_changes(sp_trader, {sp_buy: int(sp_lots), sp_sell: -int(sp_lots)})

st.markdown("---")

# ==========================
# P&L computation (on demand in Admin section)
# ==========================
def _pnl_compute_and_package():
    """Compute T+1 P&L using executable marks (Bid/Ask), not mids.
       - Outright long -> next-day Bid; short -> next-day Ask
       - Spread (Buy A / Sell B) -> next-day Bid(A) - Ask(B)
       Produces Excel if possible, else CSV bundle.
    """
    _ensure_log_exists()
    tl = pd.read_csv(LOG_FILE)
    if tl.empty:
        return None, "No trades logged yet.", 0

    # numeric types
    for c in ["price", "lots", "spread_price"]:
        if c not in tl.columns:
            tl[c] = 0
        tl[c] = pd.to_numeric(tl[c], errors="coerce").fillna(0.0)

    # load curves + prep next-day lookup
    curves = pd.read_csv(CURVE_FILE).copy()
    curves["date"] = curves["date"].astype(str)
    curves["bid"]  = pd.to_numeric(curves["bid"], errors="coerce")
    curves["ask"]  = pd.to_numeric(curves["ask"], errors="coerce")
    curves["_dt"]  = pd.to_datetime(curves["date"], errors="coerce", infer_datetime_format=True)
    unique_dates = sorted(curves["_dt"].dropna().unique())

    def _next_day_row(contract, date_str):
        try:
            d0 = pd.to_datetime(date_str, errors="coerce", infer_datetime_format=True)
        except Exception:
            return None
        later = [d for d in unique_dates if d > d0]
        if not later:
            return None
        d1 = later[0]
        sub = curves[(curves["_dt"] == d1) & (curves["contract"] == contract)]
        if sub.empty:
            # Fallback: try formatted string matching
            ds = (pd.Timestamp(d1).strftime("%-m/%-d/%Y")
                  if os.name != "nt" else pd.Timestamp(d1).strftime("%#m/%#d/%Y"))
            sub = curves[(curves["date"] == ds) & (curves["contract"] == contract)]
            if sub.empty:
                return None
        return sub.iloc[0]

    def _ou_close_px(contract, trade_side, date_str):
        row = _next_day_row(contract, date_str)
        if row is None or pd.isna(row["bid"]) or pd.isna(row["ask"]):
            return None, None
        if str(trade_side).lower() == "buy":
            return float(row["bid"]), "Bid"
        else:
            return float(row["ask"]), "Ask"

    def _sp_close_spread(buy_m, sell_m, date_str):
        a = _next_day_row(buy_m, date_str)
        b = _next_day_row(sell_m, date_str)
        if a is None or b is None or pd.isna(a["bid"]) or pd.isna(b["ask"]):
            return None
        return float(a["bid"]) - float(b["ask"])

    # mark spread audit legs so they don't double count
    tl["timestamp"]  = tl["timestamp"].astype(str)
    tl["date"]       = tl["date"].astype(str)
    tl["trader_key"] = tl["trader"].astype(str).str.strip().str.lower()
    df = tl.copy()
    df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce", infer_datetime_format=True)
    df["is_spread_leg"] = False

    spreads = df[df["type"].str.lower() == "spread"].copy()
    for _, r in spreads.iterrows():
        tkey = r["trader_key"]
        lots = int(r["lots"])
        buy_m = _canon_month(r.get("spread_buy", ""))
        sell_m = _canon_month(r.get("spread_sell", ""))
        ts = r["_ts"]

        cand = df[
            (df["trader_key"] == tkey) &
            (df["date"] == r["date"]) &
            (df["type"].str.lower() == "outright") &
            (pd.to_numeric(df["lots"], errors="coerce") == lots) &
            (df["_ts"].notna())
        ]
        window = cand
        if pd.notna(ts):
            window = cand[(cand["_ts"] >= ts - pd.Timedelta(seconds=3)) &
                          (cand["_ts"] <= ts + pd.Timedelta(seconds=3))]

        leg_buy = window[
            (window["contract"].astype(str).map(_canon_month) == buy_m) &
            (window["side"].astype(str).str.lower() == "buy")
        ]
        leg_sell = window[
            (window["contract"].astype(str).map(_canon_month) == sell_m) &
            (window["side"].astype(str).str.lower() == "sell")
        ]
        if len(leg_buy) >= 1 and len(leg_sell) >= 1:
            df.loc[leg_buy.index[:1], "is_spread_leg"] = True
            df.loc[leg_sell.index[:1], "is_spread_leg"] = True

    outs = df[(df["type"].str.lower() == "outright") & (~df["is_spread_leg"])].copy()
    sprs = df[df["type"].str.lower() == "spread"].copy()

    rows = []
    pending = 0

    # ----- Outrights MTM -----
    for _, r in outs.iterrows():
        con = _canon_month(r.get("contract", ""))
        trade_px = float(pd.to_numeric(r.get("price", 0), errors="coerce"))
        lots = int(pd.to_numeric(r.get("lots", 0), errors="coerce"))
        side = str(r.get("side", "")).lower()

        close_px, used_side = _ou_close_px(con, side, r["date"])
        if close_px is None:
            pending += 1
            pnl = None
        else:
            if side == "buy":
                pnl = (close_px - trade_px) * lots  # long marked to next-day Bid
            else:
                pnl = (trade_px - close_px) * lots  # short marked to next-day Ask

        rows.append({
            "timestamp": r["timestamp"],
            "date": r["date"],
            "trader": r["trader"],
            "type": "outright",
            "contract": con,
            "side": r.get("side", ""),
            "price": trade_px,
            "lots": lots,
            "mtm_next_used": used_side if close_px is not None else None,
            "mtm_next_px": close_px,
            "pnl_day": pnl
        })

    # ----- Spreads MTM (Buy A / Sell B) -----
    for _, r in sprs.iterrows():
        buy_m = _canon_month(r.get("spread_buy", ""))
        sell_m = _canon_month(r.get("spread_sell", ""))
        lots = int(pd.to_numeric(r.get("lots", 0), errors="coerce"))
        spx  = float(pd.to_numeric(r.get("spread_price", 0), errors="coerce"))

        close_spread = _sp_close_spread(buy_m, sell_m, r["date"])
        if close_spread is None:
            pending += 1
            pnl = None
        else:
            pnl = (close_spread - spx) * lots  # long spread MTM

        rows.append({
            "timestamp": r["timestamp"],
            "date": r["date"],
            "trader": r["trader"],
            "type": "spread",
            "buy_month": buy_m,
            "sell_month": sell_m,
            "spread_px": spx,
            "lots": lots,
            "mtm_next_spread": close_spread,
            "pnl_day": pnl
        })

    trades_pnl = pd.DataFrame(rows)

    # ----- Daily per trader and cumulative (robust for older pandas) -----
    daily = trades_pnl.copy()
    daily["pnl_day"] = pd.to_numeric(daily["pnl_day"], errors="coerce")

    daily = daily.groupby(["date", "trader"], as_index=False)["pnl_day"].sum()
    daily["date_dt"] = pd.to_datetime(daily["date"], errors="coerce", infer_datetime_format=True)
    daily = daily.sort_values(["trader", "date_dt"]).drop(columns=["date_dt"])

    daily["pnl_day"] = daily["pnl_day"].fillna(0.0)
    daily["cum_pnl"] = daily.groupby("trader")["pnl_day"].cumsum()

    # Try Excel first, fallback to CSV
    xlsx_bytes = None
    csv_text = None
    try:
        import xlsxwriter  # noqa: F401
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
            trades_pnl.to_excel(xw, index=False, sheet_name="Trades_T+1_PnL")
            daily.to_excel(xw, index=False, sheet_name="Daily_and_Cumulative")
            summary = daily.groupby("trader", dropna=False).agg(
                days_played=("pnl_day", "count"),
                total_pnl=("pnl_day", "sum")
            ).sort_values("total_pnl", ascending=False).reset_index()
            summary.to_excel(xw, index=False, sheet_name="Summary")
        xlsx_bytes = bio.getvalue()
    except Exception:
        sio = StringIO()
        sio.write("# Trades_T+1_PnL\n")
        trades_pnl.to_csv(sio, index=False)
        sio.write("\n# Daily_and_Cumulative\n")
        daily.to_csv(sio, index=False)
        csv_text = sio.getvalue()

    return xlsx_bytes, csv_text, pending

# ---------- Admin download ----------
st.markdown("🔐 **Admin Access**")
password = st.text_input("Enter admin password to download trade log", type="password")
if password == "freightadmintrader":
    # Trade log
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("📥 Download Trade Log (CSV)", f, file_name="trader_log.csv")

    # P&L pack (computed on demand)
    xlsx_bytes, csv_text, pending = _pnl_compute_and_package()
    if xlsx_bytes:
        st.download_button(
            "📊 Download P&L Pack (Excel)",
            data=xlsx_bytes,
            file_name="pnl_pack.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    elif csv_text:
        st.download_button(
            "📊 Download P&L Pack (CSV bundle)",
            data=csv_text.encode("utf-8"),
            file_name="pnl_pack.csv",
            mime="text/csv",
        )
    if pending:
        st.warning(f"Some trades have no next-day curve yet (P&L pending): {pending} row(s). Upload the next day’s forward_curves to finalize.")
else:
    st.caption("Enter the admin password to enable downloads.")
