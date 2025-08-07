
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Panamax Game", layout="wide")

curve_df = pd.read_csv("data/forward_curves.csv")
news_df = pd.read_csv("data/news_stories.csv")

trading_days = sorted(curve_df['date'].unique())
selected_date = st.sidebar.selectbox("📅 Select Trading Day", trading_days)

curve_today = curve_df[curve_df['date'] == selected_date]
news_today = news_df[news_df['date'] == selected_date]['headline'].values[0]

st.markdown("### 📰 Market News")
st.markdown(f"**{selected_date}** – {news_today}")

st.markdown("### 📈 Forward Curve")
fig, ax = plt.subplots()
contracts = curve_today['contract']
mids = (curve_today['bid'] + curve_today['ask']) / 2
ax.plot(contracts, mids, label="Mid Price", marker='o')
ax.fill_between(contracts, curve_today['bid'], curve_today['ask'], color='lightblue', alpha=0.5, label="Bid/Offer Spread")
for i, row in curve_today.iterrows():
    ax.text(row['contract'], row['bid']-50, f"B: {int(row['bid'])}", ha='center', fontsize=8)
    ax.text(row['contract'], row['ask']+50, f"O: {int(row['ask'])}", ha='center', fontsize=8)
ax.set_ylabel("USD/Day")
ax.legend()
st.pyplot(fig)

st.markdown("### 🧾 Submit Trade")
with st.form("trade_form"):
    trader = st.text_input("Trader Name")
    contract = st.selectbox("Contract", options=contracts)
    side = st.selectbox("Side", options=["Buy", "Sell"])
    price = st.number_input("Price", step=1)
    lots = st.number_input("Lots (days)", min_value=1, step=1)
    submit = st.form_submit_button("Submit Trade")
if submit:
    st.success(f"Trade submitted: {trader} {side} {lots}d of {contract} @ ${price}")

st.markdown("### 🏆 Leaderboard")
leaderboard = pd.DataFrame({
    "Trader": ["Alice", "Bob", "Charlie"],
    "Total PnL": [4400, 3900, 2850],
    "Rank": ["🥇", "🥈", "🥉"]
})
st.dataframe(leaderboard)
