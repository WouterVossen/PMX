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

    # Ensure numeric types
    for c in ["price", "lots", "spread_price"]:
        if c not in tl.columns:
            tl[c] = 0
        tl[c] = pd.to_numeric(tl[c], errors="coerce").fillna(0.0)

    # Load curves and prep next-day lookup
    curves = pd.read_csv(CURVE_FILE).copy()
    curves["date"] = curves["date"].astype(str)
    curves["bid"]  = pd.to_numeric(curves["bid"], errors="coerce")
    curves["ask"]  = pd.to_numeric(curves["ask"], errors="coerce")
    curves["_dt"]  = pd.to_datetime(curves["date"], errors="coerce", infer_datetime_format=True)
    unique_dates = sorted(curves["_dt"].dropna().unique())

    def _next_day_row(contract, date_str):
        """Return the curve row (with bid/ask) for the day AFTER date_str for this contract."""
        try:
            d0 = pd.to_datetime(date_str, errors="coerce", infer_datetime_format=True)
        except Exception:
            return None
        later = [d for d in unique_dates if d > d0]
        if not later:
            return None
        d1 = later[0]
        # Primary: match by datetime
        sub = curves[(curves["_dt"] == d1) & (curves["contract"] == contract)]
        if sub.empty:
            # Fallback: match by the original string (handles odd formats)
            ds = (pd.Timestamp(d1).strftime("%-m/%-d/%Y")
                  if os.name != "nt" else pd.Timestamp(d1).strftime("%#m/%#d/%Y"))
            sub = curves[(curves["date"] == ds) & (curves["contract"] == contract)]
            if sub.empty:
                return None
        return sub.iloc[0]

    def _ou_close_px(contract, trade_side, date_str):
        """Executable close for outrights:
           - long (buy): next-day Bid
           - short (sell): next-day Ask
        """
        row = _next_day_row(contract, date_str)
        if row is None or pd.isna(row["bid"]) or pd.isna(row["ask"]):
            return None, None
        if str(trade_side).lower() == "buy":
            return float(row["bid"]), "Bid"
        else:
            return float(row["ask"]), "Ask"

    def _sp_close_spread(buy_m, sell_m, date_str):
        """Executable close for Buy A / Sell B spread => next-day Bid(A) - Ask(B)."""
        a = _next_day_row(buy_m, date_str)
        b = _next_day_row(sell_m, date_str)
        if a is None or b is None or pd.isna(a["bid"]) or pd.isna(b["ask"]):
            return None
        return float(a["bid"]) - float(b["ask"])

    # Identify and drop spread audit legs (so we don’t double-count)
    tl["timestamp"] = tl["timestamp"].astype(str)
    tl["date"] = tl["date"].astype(str)
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
                pnl = (close_px - trade_px) * lots     # long marked to next-day Bid
            else:
                pnl = (trade_px - close_px) * lots     # short marked to next-day Ask

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
        spx = float(pd.to_numeric(r.get("spread_price", 0), errors="coerce"))

        close_spread = _sp_close_spread(buy_m, sell_m, r["date"])
        if close_spread is None:
            pending += 1
            pnl = None
        else:
            pnl = (close_spread - spx) * lots  # long spread MTM: [Bid(A) - Ask(B)] - traded_spread

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

    # Daily per trader and cumulative
    daily = trades_pnl.groupby(["date", "trader"], dropna=False)["pnl_day"].sum(min_count=1).reset_index()
    daily = daily.sort_values(["trader", "date"])
    daily["cum_pnl"] = daily.groupby("trader")["pnl_day"].cumsum(min_count=1)

    # Try Excel first, fallback to CSV
    xlsx_bytes = None
    try:
        import xlsxwriter
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
        csv_text = None
    except Exception:
        sio = StringIO()
        sio.write("# Trades_T+1_PnL\n")
        trades_pnl.to_csv(sio, index=False)
        sio.write("\n# Daily_and_Cumulative\n")
        daily.to_csv(sio, index=False)
        csv_text = sio.getvalue()

    return xlsx_bytes, (csv_text if xlsx_bytes is None else None), pending
