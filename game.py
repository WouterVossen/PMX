import streamlit as st
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import json
from io import BytesIO, StringIO
import shutil  # for copying morning template into live log
from collections import defaultdict, deque  # <<< ADDED

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
TEMPLATE_FILE = os.path.join(DATA_DIR, "trader_log_template.csv")  # upload this each morning

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

# ---- Boot-time risk guard: require a non-empty trade log (or template) before trading ----
def _ensure_log_exists():
    """
    Ensure the live LOG_FILE exists and has data.
    - If LOG_FILE is missing/empty but TEMPLATE_FILE exists, copy template into place.
    - Otherwise do nothing; a boot guard will stop the app if still missing.
    """
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        return
    if os.path.exists(TEMPLATE_FILE) and os.path.getsize(TEMPLATE_FILE) > 0:
        shutil.copy(TEMPLATE_FILE, LOG_FILE)
        return
    # do not create an empty log here; trading will be blocked until restored

_ensure_log_exists()
if (not os.path.exists(LOG_FILE)) or (os.path.getsize(LOG_FILE) == 0):
    st.error(
        "❌ No trade log found. Upload your morning master 'trader_log_template.csv' into /data "
        "or restore it via Admin before trading. Limits cannot be enforced without the log."
    )
    st.stop()

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

def _load_log_df():
    # Log should exist due to boot guard; just read it.
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
    # purely audit; LOG is source of truth for risk checks
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
    df = _load_log_df()
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
            # Ignore spread summary row; outright legs carry exposure.
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

 #--------------------------
 #Live positions table (sanity)
 #--------------------------
#st.markdown("### 📊 Current Live Positions (as of selected day)")
#try:
#    live_df = _load_log_df()
#    if live_df.empty:
#        st.info("No trades found in the live log.")
#    else:
#        traders = sorted(set(str(t) for t in live_df["trader"].dropna().unique()))
#        pos_rows = []
#        for t in traders:
#            pos = _live_positions_from_log(t, selected_date)
#            pos_rows.append({"trader": t, **pos, "slate": sum(pos.values())})
#        pos_table = pd.DataFrame(pos_rows).sort_values("trader")
#        st.dataframe(pos_table, use_container_width=True)
#except Exception as e:
#    st.warning(f"Could not render live positions: {e}")

st.markdown("---")

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
    """
    Daily P&L path via inventory replay:
      - Replay trades across days in time order.
      - FIFO realized P&L on closes (price vs price).
      - Unrealized MTM each day to executable marks (longs->Bid, shorts->Ask).
      - Daily P&L = realized_today + [ MTM_today - MTM_yesterday ].
    Produces Excel (or CSV) with 'Daily_and_Cumulative' and 'Summary'.
    """
    tl = _load_log_df()
    if tl.empty:
        return None, "No trades logged yet.", 0

    # numeric types
    for c in ["price", "lots", "spread_price"]:
        if c not in tl.columns:
            tl[c] = 0
        tl[c] = pd.to_numeric(tl[c], errors="coerce").fillna(0.0)

    # ----- Load curves up to selected_date -----
    curves = pd.read_csv(CURVE_FILE).copy()
    curves["date"] = curves["date"].astype(str)
    curves["contract"] = curves["contract"].astype(str)
    curves["bid"]  = pd.to_numeric(curves["bid"], errors="coerce")
    curves["ask"]  = pd.to_numeric(curves["ask"], errors="coerce")
    curves["_dt"]  = pd.to_datetime(curves["date"], errors="coerce", infer_datetime_format=True)
    if curves["_dt"].isna().all():
        return None, "Curve dates could not be parsed.", 0

    try:
        sel_dt = pd.to_datetime(selected_date, errors="coerce", infer_datetime_format=True)
    except Exception:
        sel_dt = curves["_dt"].max()

    curves = curves[curves["_dt"] <= sel_dt].copy()
    if curves.empty:
        return None, "No curve data up to the selected day.", 0

    # quick mark lookup + ordered curve days
    bid_map = {(r["date"], r["contract"]): float(r["bid"])  for _, r in curves.iterrows()}
    ask_map = {(r["date"], r["contract"]): float(r["ask"])  for _, r in curves.iterrows()}
    curve_days = sorted(curves["date"].unique())
    curve_dt   = {d: pd.to_datetime(d, errors="coerce") for d in curve_days}

    # ----- Tag spread legs (reuse your heuristic) -----
    tl["timestamp"]  = tl["timestamp"].astype(str)
    tl["date"]       = tl["date"].astype(str)
    tl["trader_key"] = tl["trader"].astype(str).str.strip().str.lower()
    df = tl.copy()
    df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce", infer_datetime_format=True)
    df["is_spread_leg"] = False

    spreads = df[df["type"].str.lower() == "spread"].copy()
    for _, r in spreads.iterrows():
        tkey = r["trader_key"]
        lots = int(pd.to_numeric(r.get("lots", 0), errors="coerce"))
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

    # only use trades up to selected_date
    outs["date_dt"] = pd.to_datetime(outs["date"], errors="coerce", infer_datetime_format=True)
    outs = outs[outs["date_dt"] <= sel_dt].copy()

    # canonicalize and keep enabled contracts
    outs["contract"] = outs["contract"].astype(str).map(_canon_month)
    outs = outs[outs["contract"].isin(MONTH_ORDER)].copy()

    # stable intraday order
    outs["_ts"] = pd.to_datetime(outs["timestamp"], errors="coerce", infer_datetime_format=True)
    outs = outs.sort_values(["date_dt", "_ts"]).reset_index(drop=True)

    # group trades by trade day
    trades_by_day = defaultdict(list)
    for _, r in outs.iterrows():
        trades_by_day[str(r["date_dt"].date())].append({
            "trader": str(r["trader"]),
            "contract": str(r["contract"]),
            "side": str(r["side"]).strip().lower(),
            "lots": int(pd.to_numeric(r.get("lots", 0), errors="coerce")),
            "price": float(pd.to_numeric(r.get("price", 0), errors="coerce")),
            "_ts": r["_ts"]
        })
    # chronological within each day
    for d in trades_by_day:
        trades_by_day[d].sort(key=lambda x: (pd.to_datetime(x["_ts"]) if pd.notna(x["_ts"]) else pd.Timestamp.min))

    # combined timeline = all curve days ∪ all trade days (sorted)
    trade_days = list(trades_by_day.keys())
    combined_days = sorted(
        set(curve_days) | set(trade_days),
        key=lambda s: pd.to_datetime(s, errors="coerce")
    )

    # helper: for any day, use the last available curve day ≤ that day for marks
    def _effective_curve_day(day_str: str):
        dts = pd.to_datetime(day_str, errors="coerce")
        eligible = [d for d in curve_days if curve_dt[d] <= dts]
        return max(eligible, key=lambda d: curve_dt[d]) if eligible else None

    # inventory state (FIFO)
    open_long  = defaultdict(lambda: defaultdict(lambda: deque()))  # trader->con->deque[(px, qty)]
    open_short = defaultdict(lambda: defaultdict(lambda: deque()))
    realized_by_day = defaultdict(lambda: defaultdict(float))       # trader->day->realized
    mtm_by_day      = defaultdict(lambda: defaultdict(float))       # trader->day->mtm

    traders_all = sorted(set(outs["trader"].astype(str)))

    for d in combined_days:
        # 1) apply trades of day d
        for t in trades_by_day.get(d, []):
            trader, con, side, lots, px = t["trader"], t["contract"], t["side"], t["lots"], t["price"]
            if lots <= 0 or px == 0:
                continue
            if side == "buy":
                qty = lots
                while qty > 0 and open_short[trader][con]:
                    s_px, s_qty = open_short[trader][con][0]
                    close_qty = min(qty, s_qty)
                    realized = (s_px - px) * close_qty  # short close
                    realized_by_day[trader][d] += realized
                    s_qty -= close_qty
                    qty   -= close_qty
                    if s_qty == 0:
                        open_short[trader][con].popleft()
                    else:
                        open_short[trader][con][0] = (s_px, s_qty)
                if qty > 0:
                    open_long[trader][con].append((px, qty))
            elif side == "sell":
                qty = lots
                while qty > 0 and open_long[trader][con]:
                    l_px, l_qty = open_long[trader][con][0]
                    close_qty = min(qty, l_qty)
                    realized = (px - l_px) * close_qty  # long close
                    realized_by_day[trader][d] += realized
                    l_qty -= close_qty
                    qty   -= close_qty
                    if l_qty == 0:
                        open_long[trader][con].popleft()
                    else:
                        open_long[trader][con][0] = (l_px, l_qty)
                if qty > 0:
                    open_short[trader][con].append((px, qty))

        # 2) end-of-day MTM using last available curve day ≤ d
        eff = _effective_curve_day(d)
        for trader in traders_all:
            if eff is None:
                mtm_by_day[trader][d] = 0.0
                continue
            mtm = 0.0
            for con in MONTH_ORDER:
                # longs -> Bid
                bid = bid_map.get((eff, con), None)
                if bid is not None:
                    for l_px, l_qty in list(open_long[trader][con]):
                        mtm += (bid - l_px) * l_qty
                # shorts -> Ask
                ask = ask_map.get((eff, con), None)
                if ask is not None:
                    for s_px, s_qty in list(open_short[trader][con]):
                        mtm += (s_px - ask) * s_qty
            mtm_by_day[trader][d] = mtm

    # 3) Daily P&L rows per trader
    daily_rows = []
    for trader in traders_all:
        prev_mtm = 0.0
        for i, d in enumerate(combined_days):
            mtm_today = float(mtm_by_day[trader].get(d, 0.0))
            delta_unreal = (mtm_today - prev_mtm) if i > 0 else mtm_today
            realized_d = float(realized_by_day[trader].get(d, 0.0))
            total_day = realized_d + delta_unreal
            daily_rows.append({
                "date": d,
                "trader": trader,
                "realized": realized_d,
                "delta_unrealized": delta_unreal,
                "pnl_day": total_day
            })
            prev_mtm = mtm_today

    daily = pd.DataFrame(daily_rows)
    if not daily.empty:
        daily["date_dt"] = pd.to_datetime(daily["date"], errors="coerce", infer_datetime_format=True)
        daily = daily.sort_values(["trader", "date_dt"]).drop(columns=["date_dt"])
        daily["cum_pnl"] = daily.groupby("trader")["pnl_day"].cumsum()
    else:
        daily = pd.DataFrame(columns=["date","trader","realized","delta_unrealized","pnl_day","cum_pnl"])

    # Summary (keep your original layout)
    summary = daily.groupby("trader", dropna=False).agg(
        days_played=("pnl_day", "count"),
        total_pnl=("pnl_day", "sum")
    ).sort_values("total_pnl", ascending=False).reset_index()

    # Export
    xlsx_bytes = None
    csv_text = None
    try:
        import xlsxwriter  # noqa: F401
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
            daily.to_excel(xw, index=False, sheet_name="Daily_and_Cumulative")
            summary.to_excel(xw, index=False, sheet_name="Summary")
        xlsx_bytes = bio.getvalue()
    except Exception:
        sio = StringIO()
        sio.write("# Daily_and_Cumulative\n")
        daily.to_csv(sio, index=False)
        sio.write("\n# Summary\n")
        summary.to_csv(sio, index=False)
        csv_text = sio.getvalue()

    # No 'pending' under daily MTM method
    return xlsx_bytes, csv_text, 0

# ---------- Admin download / restore ----------
st.markdown("🔐 **Admin Access**")
password = st.text_input("Enter admin password to download trade log", type="password")
if password == "freightadmintrader":
    # Trade log
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "rb") as f:
            st.download_button("📥 Download Trade Log (CSV)", f, file_name="trader_log.csv")

    # Forward curves download (so you can recompute MTM offline if needed)
    if os.path.exists(CURVE_FILE):
        with open(CURVE_FILE, "rb") as f:
            st.download_button("📥 Download Forward Curves (CSV)", f, file_name="forward_curves.csv")

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

    st.markdown("### 🔄 Restore / Replace Live Trade Log")
    up = st.file_uploader("Upload master trader_log.csv to replace the live log", type=["csv"], key="restore_log")
    if up is not None:
        df_up = pd.read_csv(up)
        required = {"timestamp","date","trader","type","contract","side","price","lots","spread_buy","spread_sell","spread_price"}
        cols_lower = set(c.lower() for c in df_up.columns)
        missing = [c for c in required if c not in cols_lower]
        if missing:
            st.error(f"Uploaded CSV missing columns: {missing}")
        else:
            # Keep incoming column order as-is
            df_up.to_csv(LOG_FILE, index=False)
            st.success("✅ Live trade log replaced from uploaded file. Limits now reflect this state.")

    if os.path.exists(TEMPLATE_FILE) and os.path.getsize(TEMPLATE_FILE) > 0:
        if st.button("Use trader_log_template.csv as Live Log"):
            shutil.copy(TEMPLATE_FILE, LOG_FILE)
            st.success("✅ Live trade log refreshed from trader_log_template.csv")
    else:
        st.caption("No trader_log_template.csv found in /data (or it's empty).")
else:
    st.caption("Enter the admin password to enable downloads.")
