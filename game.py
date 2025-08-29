import streamlit as st
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import json

# ==========================
# Page setup
# ==========================
st.set_page_config(page_title="Panamax Freight Game", layout="wide")

DATA_FOLDER = "data"
CURVE_FILE = os.path.join(DATA_FOLDER, "forward_curves.csv")
LOG_FILE = os.path.join(DATA_FOLDER, "trader_log.csv")
CONFIG_FILE = "config.json"

# Contracts traded in the game
MONTH_ORDER = ["Sep", "Oct", "Nov", "Dec"]

# Risk limits
PER_MONTH_CAP = 200   # |net| per single month
SLATE_CAP = 100       # |sum of nets across months|

# ==========================
# Utility: load config & data
# ==========================
def read_config_current_day() -> str:
    with open(CONFIG_FILE, "r") as f:
        cfg = json.load(f)
    return cfg.get("current_day")

def load_curves_all() -> pd.DataFrame:
    df = pd.read_csv(CURVE_FILE)
    # Normalize columns
    df.columns = [c.strip() for c in df.columns]
    # Coerce numeric
    for col in ["bid", "ask"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["contract"] = df["contract"].astype(str).str.strip()
    df["date"] = df["date"].astype(str).str.strip()
    return df

def curves_for_day(df: pd.DataFrame, day: str) -> pd.DataFrame:
    out = df[df["date"] == day].copy()
    # keep only game months, sorted
    out = out[out["contract"].isin(MONTH_ORDER)]
    out["contract"] = pd.Categorical(out["contract"], MONTH_ORDER, ordered=True)
    out = out.sort_values("contract")
    return out

# ==========================
# Logging / positions helpers
# ==========================
def ensure_log_exists():
    if not os.path.exists(LOG_FILE):
        cols = [
            "timestamp","date","trader","type",
            "contract","side","price","lots",
            "spread_buy","spread_sell","spread_price"
        ]
        pd.DataFrame(columns=cols).to_csv(LOG_FILE, index=False)

def load_log() -> pd.DataFrame:
    ensure_log_exists()
    df = pd.read_csv(LOG_FILE)
    if not df.empty:
        # Normalize
        if "lots" in df.columns:
            df["lots"] = pd.to_numeric(df["lots"], errors="coerce").fillna(0).astype(int)
        if "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
        for c in ["trader","contract","side","type","spread_buy","spread_sell","date","timestamp"]:
            if c in df.columns:
                df[c] = df[c].astype(str)
    return df

def month_label_from_contract(contract_str: str) -> str:
    # Contracts are month strings (e.g., "Sep","Oct","Nov","Dec")
    return str(contract_str)

def positions_for_trader_mtd(trader: str, upto_date_str: str) -> tuple[dict, int]:
    """
    Return (per_month_nets, slate) for a trader for the calendar month of upto_date_str.
    Outright: Buy +lots / Sell -lots
    Spread: +lots on buy month / -lots on sell month
    """
    logs = load_log()
    if logs.empty:
        return {m: 0 for m in MONTH_ORDER}, 0

    # Month filter aligned to upto_date_str month/year
    try:
        logs["date_dt"] = pd.to_datetime(logs["date"], errors="coerce")
        ref = pd.to_datetime(upto_date_str, errors="coerce")
        logs = logs[(logs["date_dt"].dt.month == ref.month) & (logs["date_dt"].dt.year == ref.year)]
    except Exception:
        # Fallback if parsing fails: only exact date match
        logs = logs[logs["date"] == upto_date_str]

    logs = logs[logs["trader"].str.strip().str.lower() == str(trader).strip().lower()]

    per_month = {m: 0 for m in MONTH_ORDER}
    for _, r in logs.iterrows():
        ttype = str(r.get("type","")).lower()
        lots = int(r.get("lots", 0))
        if ttype == "outright":
            m = month_label_from_contract(r.get("contract",""))
            side = str(r.get("side","")).lower()
            if m in per_month:
                per_month[m] += lots if side == "buy" else -lots
        elif ttype == "spread":
            bm = month_label_from_contract(r.get("spread_buy",""))
            sm = month_label_from_contract(r.get("spread_sell",""))
            if bm in per_month:
                per_month[bm] += lots
            if sm in per_month:
                per_month[sm] -= lots

    slate = sum(per_month.values())
    return per_month, slate

def check_limits_after(trader: str, date_str: str, changes: dict) -> tuple[bool, str]:
    """
    changes: {month: delta_lots}
    Returns (ok, message)
    """
    per_month, slate = positions_for_trader_mtd(trader, date_str)
    new_per_month = per_month.copy()
    for m, dlt in changes.items():
        if m in new_per_month:
            new_per_month[m] += int(dlt)
    new_slate = sum(new_per_month.values())

    for m, net in new_per_month.items():
        if abs(net) > PER_MONTH_CAP:
            return False, f"Per-month limit exceeded in {m}: |{net}| > {PER_MONTH_CAP}."
    if abs(new_slate) > SLATE_CAP:
        return False, f"Slate limit exceeded: |{new_slate}| > {SLATE_CAP}."
    return True, "OK"

def append_log_rows(rows: list[dict]):
    ensure_log_exists()
    df = pd.DataFrame(rows)
    df.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0, index=False)

# ==========================
# P&L (T+1) helpers
# ==========================
def next_mtm_price(contract: str, trade_date_str: str, curves_all: pd.DataFrame) -> float | None:
    """
    T+1 mark: earliest available MID price for contract with curve date strictly > trade_date.
    Returns None if next-day curve not available yet.
    """
    try:
        cdf = curves_all[curves_all["contract"] == contract].copy()
        cdf["date_dt"] = pd.to_datetime(cdf["date"], errors="coerce")
        trade_dt = pd.to_datetime(trade_date_str, errors="coerce")
        cdf = cdf[cdf["date_dt"] > trade_dt].sort_values("date_dt")
        if cdf.empty:
            return None
        head = cdf.iloc[0]
        return float((head["bid"] + head["ask"]) / 2.0)
    except Exception:
        return None

def compute_pnl_Tplus1_tables(curves_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      trades_with_pnl: full log with 'mtm_next' + 'pnl_day' on outright legs (spreads informational only)
      pnl_by_trader_daily: per (date, trader) P&L
      pnl_by_trader_cum: per trader cumulative P&L
    P&L logic:
      pnl_day = (mtm_next - exec_price) * lots * sign, sign=+1 Buy, -1 Sell
      next-day MTM uses earliest curve strictly after trade date
    """
    logs = load_log()
    if logs.empty:
        return logs, pd.DataFrame(columns=["date","trader","pnl_day"]), pd.DataFrame(columns=["trader","pnl_cumulative"])

    ou = logs[logs["type"].str.lower() == "outright"].copy()
    if ou.empty:
        # No outright legs yet, build trivial outputs
        trades_with_pnl = logs.copy()
        trades_with_pnl["mtm_next"] = pd.NA
        trades_with_pnl["pnl_day"] = pd.NA
        return trades_with_pnl, pd.DataFrame(columns=["date","trader","pnl_day"]), pd.DataFrame(columns=["trader","pnl_cumulative"])

    ou["mtm_next"] = ou.apply(lambda r: next_mtm_price(str(r["contract"]), str(r["date"]), curves_all), axis=1)
    sign = ou["side"].str.lower().map({"buy": 1, "sell": -1}).fillna(0)
    ou["pnl_day"] = (ou["mtm_next"] - ou["price"]) * ou["lots"] * sign

    pnl_by_trader_daily = (
        ou.groupby(["date","trader"], dropna=False)["pnl_day"]
          .sum(min_count=1)
          .reset_index()
          .sort_values(["date","trader"])
    )

    pnl_by_trader_cum = (
        ou.groupby("trader", dropna=False)["pnl_day"]
          .sum(min_count=1)
          .reset_index()
          .rename(columns={"pnl_day":"pnl_cumulative"})
          .sort_values("pnl_cumulative", ascending=False)
    )

    merge_cols = ["timestamp","trader","contract","side","price","lots","date"]
    trades_with_pnl = logs.copy()
    trades_with_pnl = trades_with_pnl.merge(
        ou[merge_cols + ["mtm_next","pnl_day"]],
        how="left",
        on=merge_cols
    )
    return trades_with_pnl, pnl_by_trader_daily, pnl_by_trader_cum

# ==========================
# Load data for today
# ==========================
selected_date = read_config_current_day()
curve_df_all = load_curves_all()
curve_today = curves_for_day(curve_df_all, selected_date)

st.title("🚢 Panamax Freight Paper Trading Game")

if curve_today.empty:
    st.error(f"No market data found for {selected_date}. Please check your config or data files.")
    st.stop()

contracts_today = list(curve_today["contract"])
bids = dict(zip(curve_today["contract"], curve_today["bid"]))
asks = dict(zip(curve_today["contract"], curve_today["ask"]))

# ==========================
# Chart (compact)
# ==========================
st.subheader(f"📅 Market Day: {selected_date}")
st.markdown("### 📈 Forward Curve")

fig, ax = plt.subplots(figsize=(7, 3))
mids = (curve_today["bid"] + curve_today["ask"]) / 2.0
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
# Trade entry (Outright + Spread)
# ==========================
st.markdown("### 🧾 Submit Your Trades")

# ---------- OUTRIGHT ----------
st.markdown("**Outright**")
with st.form("ou_form"):
    c1, c2, c3, c4 = st.columns([1.2, 1.0, 1.0, 1.2])

    ou_contract = c1.selectbox("Contract", options=contracts_today, key="ou_contract")
    ou_side = c2.selectbox("Side", ["Buy", "Sell"], key="ou_side")
    ou_lots = c3.number_input("Lots (days)", min_value=1, step=1, value=1, key="ou_lots")

    default_ou_price = float(asks[ou_contract]) if ou_side == "Buy" else float(bids[ou_contract])
    ou_price = c4.number_input("Price (auto-filled)", value=default_ou_price, step=1.0, key="ou_price")

    ou_trader = st.text_input("Trader Name", key="ou_trader")
    ou_submit = st.form_submit_button("Submit Outright Trade")

if ou_submit:
    if not ou_trader.strip():
        st.error("Please enter your Trader Name for the outright trade.")
    else:
        # Position impact: Buy +lots / Sell -lots
        m = month_label_from_contract(ou_contract)
        delta = {m: (int(ou_lots) if ou_side == "Buy" else -int(ou_lots))}
        ok, msg = check_limits_after(ou_trader, selected_date, delta)
        if not ok:
            st.error(f"❌ {msg}")
        else:
            append_log_rows([{
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "date": selected_date,
                "trader": ou_trader,
                "type": "outright",
                "contract": ou_contract,
                "side": ou_side,
                "price": float(ou_price),
                "lots": int(ou_lots),
                "spread_buy": "",
                "spread_sell": "",
                "spread_price": ""
            }])
            st.success(f"✅ Outright submitted: {ou_trader} {ou_side} {int(ou_lots)}d {ou_contract} @ {int(ou_price)}")

st.markdown("---")

# ---------- SPREAD ----------
st.markdown("**Calendar Spread (Buy one month / Sell another)**")
with st.form("sp_form"):
    s1, s2, s3, s4 = st.columns([1.2, 1.2, 1.0, 1.2])

    sp_buy = s1.selectbox("Buy month", options=contracts_today, key="sp_buy")
    sp_sell = s2.selectbox("Sell month", options=[m for m in contracts_today if m != sp_buy], key="sp_sell")
    sp_lots = s3.number_input("Lots (days)", min_value=1, step=1, value=1, key="sp_lots")

    # Conservative auto-fill: spread = Ask(buy) - Bid(sell)
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
        # Effects: +lots on buy month, -lots on sell month
        bm = month_label_from_contract(sp_buy)
        sm = month_label_from_contract(sp_sell)
        delta = {bm: int(sp_lots), sm: -int(sp_lots)}
        ok, msg = check_limits_after(sp_trader, selected_date, delta)
        if not ok:
            st.error(f"❌ {msg}")
        else:
            # Log spread summary row
            append_log_rows([{
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
            }])
            # Log both legs as outrights (auditability and P&L on legs)
            append_log_rows([
                {
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
                },
                {
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
                }
            ])
            st.success(f"✅ Spread submitted: {sp_trader} BUY {int(sp_lots)}d {sp_buy} / SELL {int(sp_lots)}d {sp_sell} @ {int(sp_price)}")

st.markdown("---")

# ==========================
# Admin download (with T+1 P&L)
# ==========================
st.markdown("🔐 **Admin Access**")
password = st.text_input("Enter admin password to download trade log & PnL", type="password")

if password == "freightadmintrader":
    trades_with_pnl, pnl_by_trader_daily, pnl_by_trader_cum = compute_pnl_Tplus1_tables(curve_df_all)

    if trades_with_pnl.empty:
        st.info("No trades logged yet.")
    else:
        csv_trades = trades_with_pnl.to_csv(index=False).encode("utf-8")
        csv_daily  = pnl_by_trader_daily.to_csv(index=False).encode("utf-8")
        csv_cum    = pnl_by_trader_cum.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Trades_With_DailyPnL.csv",
            data=csv_trades,
            file_name="Trades_With_DailyPnL.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.download_button(
            "📥 Download PnL_By_Trader_Daily.csv",
            data=csv_daily,
            file_name="PnL_By_Trader_Daily.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.download_button(
            "📥 Download PnL_By_Trader_Cumulative.csv",
            data=csv_cum,
            file_name="PnL_By_Trader_Cumulative.csv",
            mime="text/csv",
            use_container_width=True
        )

        if trades_with_pnl["mtm_next"].isna().any():
            st.warning(
                "Some trades have no next-day curve yet (P&L pending). "
                "Upload the next day’s forward_curves to finalize those rows."
            )
