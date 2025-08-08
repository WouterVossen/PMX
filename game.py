
import streamlit as st
import pandas as pd
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
news_df = pd.read_csv("data/news_stories.csv")

# Filter data for current date
curve_today = curve_df[curve_df["date"] == selected_date]
news_today = news_df[news_df["date"] == selected_date]

st.title("📦 Panamax Freight Paper Trading Game")

if curve_today.empty or news_today.empty:
    st.error(f"No market data found for {selected_date}. Please check your config or data files.")
else:
    st.subheader(f"📅 Market Day: {selected_date}")
    st.markdown("### 📰 News")
    st.markdown(news_today["headline"].values[0])

    st.markdown("### 📈 Forward Curve")
    fig, ax = plt.subplots(figsize=(8, 4))
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
    st.markdown("🛠 *To advance the game, update `config.json` with the next date.*")
