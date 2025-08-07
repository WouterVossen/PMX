
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="Panamax Trading Game", layout="wide")

# Load data
curve_df = pd.read_csv("data/forward_curves.csv")
news_df = pd.read_csv("data/news_stories.csv")
sentiment_df = pd.read_csv("data/sentiment_scores.csv")
log_df = pd.read_csv("data/trader_log_template.csv")

# Get list of unique dates and sort them
trading_days = sorted(curve_df['date'].unique())

# Sidebar - select date
selected_date = st.sidebar.selectbox("📅 Select Trading Day", trading_days)

# Filter data for selected day
curve_today = curve_df[curve_df['date'] == selected_date]
news_today = news_df[news_df['date'] == selected_date]['headline'].values[0]
sentiment_today = sentiment_df[sentiment_df['date'] == selected_date]

# Display news story
st.markdown("### 📰 Market News")
st.markdown(f"**{selected_date}** – {news_today}")

# Display sentiment scores
st.markdown("### 🌊 Market Sentiment")
atl = sentiment_today['atlantic_sentiment'].values[0]
pac = sentiment_today['pacific_sentiment'].values[0]
st.write(f"**Atlantic:** {atl}/5")
st.write(f"**Pacific:** {pac}/5")

# Plot forward curve
st.markdown("### 📈 Forward Curve")
fig, ax = plt.subplots()
contracts = curve_today['contract']
mids = (curve_today['bid'] + curve_today['ask']) / 2
ax.plot(contracts, mids, label="Mid Price", marker='o')
ax.fill_between(contracts, curve_today['bid'], curve_today['ask'], color='lightblue', alpha=0.5, label="Bid-Offer Range")
for i, row in curve_today.iterrows():
    ax.text(row['contract'], row['bid']-50, f"B: {int(row['bid'])}", ha='center', fontsize=8)
    ax.text(row['contract'], row['ask']+50, f"O: {int(row['ask'])}", ha='center', fontsize=8)
ax.set_ylabel("USD/Day")
ax.legend()
st.pyplot(fig)

# Trade form
st.markdown("### 🧾 Submit Trade")
with st.form("trade_form"):
    trader = st.text_input("Trader Name")
    contract = st.selectbox("Contract", options=contracts)
    side = st.selectbox("Buy/Sell", options=["Buy", "Sell"])
    price = st.number_input("Price (USD/day)", step=1)
    lots = st.number_input("Lots (days)", step=1, min_value=1)
    submitted = st.form_submit_button("Submit Trade")

if submitted:
    st.success(f"Trade submitted: {trader} - {side} {lots}d of {contract} at ${price}")
    # Note: This version does not save the trade

# Leaderboard (mock preview)
st.markdown("### 🏆 Leaderboard")
leaderboard_data = {
    "Trader": ["Alice", "Bob", "Charlie"],
    "Total PnL": [4200, 3900, 3200],
    "Rank": ["🥇", "🥈", "🥉"]
}
leaderboard_df = pd.DataFrame(leaderboard_data)
st.dataframe(leaderboard_df)
