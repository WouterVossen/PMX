
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json

# Page setup
st.set_page_config(page_title="Panamax Freight Game", layout="wide")

# Load current day from config
with open("config.json", "r") as f:
    config = json.load(f)
selected_date = config["current_day"]

# Load data
curve_df = pd.read_csv("data/forward_curves.csv")
news_df = pd.read_csv("data/news_stories.csv")

# Filter today's data
curve_today = curve_df[curve_df["date"] == selected_date]
news_today = news_df[news_df["date"] == selected_date]["headline"].values[0]

# App display
st.title("📦 Panamax Freight Paper Trading Game")
st.subheader(f"📅 Market Day: {selected_date}")
st.markdown("### 📰 News")
st.markdown(news_today)

# Forward curve plot
st.markdown("### 📈 Forward Curve")
fig, ax = plt.subplots(figsize=(8, 4))
contracts = curve_today["contract"]
mids = (curve_today["bid"] + curve_today["ask"]) / 2
ax.plot(contracts, mids, marker='o', label="Mid Price")
ax.fill_between(contracts, curve_today["bid"], curve_today["ask"], alpha=0.2, label="Bid/Offer Range")
for _, row in curve_today.iterrows():
    ax.text(row["contract"], row["bid"] - 50, f"B: {int(row['bid'])}", ha='center', fontsize=8)
    ax.text(row["contract"], row["ask"] + 50, f"O: {int(row['ask'])}", ha='center', fontsize=8)
ax.set_ylabel("USD/Day")
ax.legend()
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.markdown("🛠 *To advance the game, update `config.json` with the next date.*")
