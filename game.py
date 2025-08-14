import streamlit as st
import pandas as pd
from datetime import datetime
import os
import json
import altair as alt

# ──────────────────────────────────────────────────────────────────────────────
# Page setup & styling (Suggestions 1–3)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Panamax Freight Game", layout="wide")

CARGILL_GREEN = "#007A33"  # primary accent

# Use a token so we don't fight with Python .format() vs CSS braces
_BG_CSS = """
<style>
/* Background gradient */
[data-testid="stAppViewContainer"] {{
  background: radial-gradient(1200px 800px at 20% -10%, #e8f5f0 0%, #f7faf9 40%, #ffffff 100%);
}

/* Hide Streamlit default menu/footer */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* Top header bar */
.topbar {{
  position: sticky;
  top: 0;
  z-index: 999;
  background: white;
  border-bottom: 1px solid #e6e6e6;
  padding: .9rem 1rem;
  display: flex;
  gap: 1rem;
  align-items: center;
}}
.topbar .title {{
  font-weight: 700; 
  font-size: 1.05rem; 
  color: #0f172a;
}}
.badge {{
  background: <<GREEN>>;
  color: white;
  padding: .18rem .5rem;
  border-radius: .5rem;
  font-size: .75rem;
}}
.card {{
  background: #ffffff;
  border: 1px solid #edf2f7;
  border-radius: 14px;
  padding: 1.0rem 1.2rem;
  box-shadow: 0 2px 8px rgba(16,24,40,.06);
}}
.section-title {{
  font-weight: 700;
  font-size: 1.0rem;
  color: #0f172a;
  margin-bottom: .35rem;
}}
.news-line {{
  display: flex; gap: .5rem; align-items: flex-start;
  padding: .25rem 0;
  border-left: 3px solid transparent;
}}
.news-line .dot {{
  width: 8px; height: 8px; border-radius: 999px; background: <<GREEN>>; margin-top: .4rem;
}}
.small-muted {{ color: #6b7280; font-size: .82rem; }}
.curve-legend {{
  display:flex; gap: 10px; align-items:center; margin-top: .4rem;
}}
.legend-dot {{ width:10px; height:10px; border-radius:999px; display:inline-block; }}
.legend-today {{ background: <<GREEN>>; }}
.legend-yday {{ background:#94a3b8; }}
.legend-band {{ background:linear-gradient(90deg, rgba(0,122,51,.08), rgba(0,122,51,.08)); border:1px dashed <<GREEN>>; }}
</style>
"""
BG_GRADIENT = _BG_CSS.replace("<<GREEN>>", CARGILL_GREEN)
st.markdown(BG_GRADIENT, unsafe_allow_html=True)

with st.container():
    st.markdown(
        """
        <div class="topbar">
            <span class="title">📦 Panamax Freight Paper Trading Game</span>
            <span class="badge">Cargill Internal</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────────────────────────────────────
# Load config & data
# ──────────────────────────────────────────────────────────────────────────────
with open("config.json", "r") as f:
    config = json.load(f)
selected_date = config.get("current_day")

curve_df = pd.read_csv("data/forward_curves.csv")
news_df = pd.read_csv("data/news_stories.csv")

# Ensure numeric bid/ask
for col in ["bid", "ask"]:
    if col in curve_df.columns:
        curve_df[col] = pd.to_numeric(curve_df[col], errors="coerce")

# Determine yesterday (previous available business day in file)
unique_days = curve_df["date"].dropna().unique().tolist()
def _to_dt(s):
    try:
        return datetime.strptime(s, "%m/%d/%Y")
    except Exception:
        return None
pairs = [(d, _to_dt(d)) for d in unique_days]
pairs = [p for p in pairs if p[1] is not None]
pairs.sort(key=lambda x: x[1])
unique_days_sorted = [p[0] for p in pairs]

if selected_date not in unique_days_sorted:
    st.error(f"No market data found for {selected_date}. Please check your config or data files.")
    st.stop()

idx_today = unique_days_sorted.index(selected_date)
date_yesterday = unique_days_sorted[idx_today-1] if idx_today > 0 else None

curve_today = curve_df[curve_df["date"] == selected_date].copy()
curve_yday  = curve_df[curve_df["date"] == date_yesterday].copy() if date_yesterday else pd.DataFrame()
news_today = news_df[news_df["date"] == selected_date]

# ──────────────────────────────────────────────────────────────────────────────
# Layout: two columns in a clean card grid
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.2, 1.0], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">📅 Market Day: {selected_date}</div>', unsafe_allow_html=True)

    # News feed — headline + update lines (Suggestion 3)
    st.markdown('<div class="section-title">📰 Tradewinds News</div>', unsafe_allow_html=True)
    if news_today.empty:
        st.markdown('<div class="small-muted">No news found for today.</div>', unsafe_allow_html=True)
    else:
        full_text = str(news_today["headline"].values[0]).strip()
        parts = [p.strip() for p in full_text.split(".") if p.strip()]
        if parts:
            headline = parts[0]
            updates = parts[1:]
            st.markdown(f"**{headline}.**")
            for u in updates:
                st.markdown(
                    f'<div class="news-line"><span class="dot"></span>'
                    f'<div>{u}.</div></div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(full_text)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Forward Curve</div>', unsafe_allow_html=True)

    if curve_today.empty:
        st.markdown('<div class="small-muted">No curve available for today.</div>', unsafe_allow_html=True)
    else:
        # Build interactive dataset (Suggestion 2)
        def prep(df, label):
            out = df[["contract", "bid", "ask"]].copy()
            out["mid"] = (out["bid"] + out["ask"]) / 2.0
            out["series"] = label
            return out

        data_today = prep(curve_today, "Today")
        data_plot = data_today.copy()

        if not curve_yday.empty:
            data_yday = prep(curve_yday, "Yesterday")
            data_plot = pd.concat([data_plot, data_yday], ignore_index=True)

        # Preserve today's contract order
        order = data_today["contract"].tolist()
        if order:
            data_plot["contract"] = pd.Categorical(data_plot["contract"], categories=order, ordered=True)

        # Bid–ask band (today)
        band = alt.Chart(data_today).mark_area(opacity=0.18, color=CARGILL_GREEN).encode(
            x=alt.X("contract:O", title="Contract"),
            y=alt.Y("bid:Q", title="USD / day"),
            y2="ask:Q",
            tooltip=[
                alt.Tooltip("contract:O", title="Contract"),
                alt.Tooltip("bid:Q", title="Bid", format=",.0f"),
                alt.Tooltip("ask:Q", title="Ask", format=",.0f")
            ]
        )

        # Mid lines (today vs yesterday)
        line = alt.Chart(data_plot).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("contract:O", title="Contract"),
            y=alt.Y("mid:Q", title="USD / day"),
            color=alt.Color("series:N", title="Series",
                            scale=alt.Scale(domain=["Today", "Yesterday"],
                                            range=[CARGILL_GREEN, "#94a3b8"])),
            tooltip=[
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("contract:O", title="Contract"),
                alt.Tooltip("mid:Q", title="Mid", format=",.0f"),
                alt.Tooltip("bid:Q", title="Bid", format=",.0f"),
                alt.Tooltip("ask:Q", title="Ask", format=",.0f"),
            ]
        )

        chart = (band + line).properties(height=300).interactive()
        st.altair_chart(chart, use_container_width=True)

        # Legend badges
        st.markdown("""
        <div class="curve-legend">
          <span class="legend-dot legend-today"></span><span class="small-muted">Today (Mid)</span>
          <span class="legend-dot legend-yday"></span><span class="small-muted">Yesterday (Mid)</span>
          <span class="legend-dot legend-band"></span><span class="small-muted">Bid–Ask band (today)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Trade form (same logic, nicer layout)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧾 Submit Your Trade</div>', unsafe_allow_html=True)

contracts = curve_today["contract"].tolist() if not curve_today.empty else []
with st.form("trade_form"):
    col1, col2, col3, col4, col5 = st.columns([1.1, 1, 1, 1, 1.2])
    trader = col1.text_input("Trader Name")
    contract = col2.selectbox("Contract", options=contracts if contracts else ["—"])
    side = col3.selectbox("Side", ["Buy", "Sell"])
    price = col4.number_input("Price", step=1)
    lots = col5.number_input("Lots (days)", min_value=1, step=1)
    submitted = st.form_submit_button("Submit Trade")

log_file = "data/trader_log.csv"
if submitted:
    st.success(f"✅ Trade submitted: {trader} {side} {lots}d of {contract} @ ${price}")
    trade = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date": selected_date,
        "trader": trader,
        "contract": contract,
        "side": side,
        "price": price,
        "lots": lots
    }
    df = pd.DataFrame([trade])
    if os.path.exists(log_file):
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, index=False)

st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Admin download (password-gated)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🔐 Admin Access</div>', unsafe_allow_html=True)
password = st.text_input("Enter admin password to download trade log", type="password")
if password == "freightadmintrader":
    if os.path.exists(log_file):
        with open(log_file, "rb") as f:
            st.download_button("📥 Download Trade Log", f, file_name="trader_log.csv")
st.markdown('</div>', unsafe_allow_html=True)
