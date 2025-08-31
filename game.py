# --- Outright P&L check ---
oct_curve = {
    "D1": {"bid":12900, "ask":13100},
    "D2": {"bid":13200, "ask":13400},
    "D3": {"bid":12750, "ask":12950},
    "D4": {"bid":13050, "ask":13250},
    "D5": {"bid":13100, "ask":13300},
    "D6": {"bid":12800, "ask":13000},
}
trades_ou = [
    # (day, side, lots, trade_px)
    ("D1", "buy", 100, 13100),
    ("D2", "sell", 40, 13200),
    ("D3", "sell", 60, 12750),
    ("D4", "buy",  50, 13250),
    ("D5", "sell", 20, 13100),
]

order_days = ["D1","D2","D3","D4","D5","D6"]
day_to_next = {order_days[i]: order_days[i+1] for i in range(len(order_days)-1)}

pnl_total = 0
for d, side, lots, px in trades_ou:
    nd = day_to_next[d]
    mark = oct_curve[nd]["bid"] if side=="buy" else oct_curve[nd]["ask"]
    pnl = (mark - px)*lots if side=="buy" else (px - mark)*lots
    pnl_total += pnl
    print(d, side, lots, px, "-> next:", nd, "mark:", mark, "PnL:", pnl)
print("Total PnL:", pnl_total)

# --- Spread P&L check (Buy A/Sell B) ---
oct_c = {
    "D1": (12900,13100), "D2": (13200,13400), "D3": (12750,12950),
    "D4": (13050,13250), "D5": (13100,13300), "D6": (12800,13000),
}
nov_c = {
    "D1": (12500,12700), "D2": (12650,12850), "D3": (12450,12650),
    "D4": (12600,12800), "D5": (12750,12950), "D6": (12600,12800),
}
def traded_spread(day, buy="Oct", sell="Nov"):
    buy_ask = (oct_c if buy=="Oct" else nov_c)[day][1]
    sell_bid = (oct_c if sell=="Oct" else nov_c)[day][0]
    return buy_ask - sell_bid

def next_mtm(day, buy="Oct", sell="Nov"):
    nd = day_to_next[day]
    buy_bid = (oct_c if buy=="Oct" else nov_c)[nd][0]
    sell_ask = (oct_c if sell=="Oct" else nov_c)[nd][1]
    return buy_bid - sell_ask

# Trade 1: Buy Oct/Sell Nov on D1, 50 lots
t1 = traded_spread("D1", "Oct", "Nov")            # 600
m1 = next_mtm("D1", "Oct", "Nov")                  # 350
pnl1 = (m1 - t1)*50                                # -12500

# Trade 2: Sell Oct/Buy Nov on D3 -> enter as Buy Nov/Sell Oct
t2 = traded_spread("D3", "Nov", "Oct")            # -100
m2 = next_mtm("D3", "Nov", "Oct")                  # -650
pnl2 = (m2 - t2)*30                                # -16500

print("Spread1 D1 PnL:", pnl1, "Spread2 D3 PnL:", pnl2, "Total:", pnl1+pnl2)
