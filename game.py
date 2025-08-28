import streamlit as st
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import json

# Page setup
st.set_page_config(page_title="Panamax Freight Game", layout="wide")

# Load config to determine current trading day
with open("config.json", "r") as f:
    config = json.load(f)
selected_date = config.get("current_day")

# Load data
curve_df = pd.read_csv("data/forward_curves.csv")

# Filter data for current date
curve_today = curve_df[curve_df["date"] == selected_date]

# Global monthly limit (adjustable)
MAX_MONTHLY_LOTS = 1000

st.title("📦 Panamax Freight Paper Trading Game")

if curve_today.empty:
    st.error(f"No market data found for {selected_date}. Please check your config or data files.")
else:
    st.subheader(f"📅 Market Day: {selected_date}")

    st.markdown("### 📈 Forward Curve")
    fig, ax = plt.subplots(figsize=(8, 3))
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
    st.markdown("🧾 ### Submit Your Trade")
    with st.form("trade_form"):
        trader = st.text_input("Trader Name")
        contract = st.selectbox("Contract", options=contracts)
        side = st.selectbox("Side", ["Buy", "Sell"])
        price = st.number_input("Price", step=1)
        lots = st.number_input("Lots (days)", min_value=1, step=1)
        submitted = st.form_submit_button("Submit Trade")

    log_file = "data/trader_log.csv"
    if submitted:
        # Load existing trades
        if os.path.exists(log_file):
            log_df = pd.read_csv(log_file)
            log_df["date"] = pd.to_datetime(log_df["date"])
            selected_month = pd.to_datetime(selected_date).month
            selected_year = pd.to_datetime(selected_date).year

            # Filter for current trader and month
            trader_month_df = log_df[
                (log_df["trader"] == trader) &
                (log_df["date"].dt.month == selected_month) &
                (log_df["date"].dt.year == selected_year)
            ]

            total_lots_this_month = trader_month_df["lots"].sum()
        else:
            total_lots_this_month = 0

        if total_lots_this_month + lots > MAX_MONTHLY_LOTS:
            st.error(f"❌ Trade exceeds monthly limit of {MAX_MONTHLY_LOTS} lots. You have already traded {total_lots_this_month} lots.")
        else:
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
            df.to_csv(log_file, mode='a', header=not os.path.exists(log_file), index=False)

    st.markdown("---")
    st.markdown("🔐 ### Admin Access")
    password = st.text_input("Enter admin password to download trade log", type="password")
    if password == "freightadmintrader":
        if os.path.exists(log_file):
            with open(log_file, "rb") as f:
                st.download_button("📥 Download Trade Log", f, file_name="trader_log.csv")
