"""Functions for downloading data from WRDS (OptionMetrics and CBOE).

Every function takes `username`, your WRDS username (e.g. "kairogers"),
so anyone with a WRDS account can run them.
"""
import wrds
import numpy as np
import pandas as pd


def multi_option_single_day(username, date, secid=108105, save=True):
    """Return a cleaned option chain for one trading day.

    username : WRDS username
    date     : "YYYY-MM-DD"
    secid    : OptionMetrics security id (108105 = SPX)
    save     : if True, also writes source/data/multi_option_single_day{date}.csv
    """
    year = date[:4]
    db = wrds.Connection(wrds_username=username)

    opts = db.raw_sql(f"""
        SELECT date, optionid, exdate, cp_flag, strike_price, best_bid, best_offer,
               volume, open_interest, impl_volatility, delta,
               am_settlement, expiry_indicator, forward_price
        FROM optionm.opprcd{year}
        WHERE secid = {secid} AND date = '{date}'
    """, date_cols=["date", "exdate"])

    spot = db.raw_sql(f"""
        SELECT close FROM optionm.secprd{year}
        WHERE secid = {secid} AND date = '{date}'
    """)

    zc = db.raw_sql(f"""
        SELECT days, rate FROM optionm.zerocd
        WHERE date = '{date}' ORDER BY days
    """)

    div = db.raw_sql(f"""
        SELECT rate FROM optionm.idxdvd
        WHERE secid = {secid} AND date = '{date}'
    """)

    db.close()

    if opts.empty or spot.empty:
        raise ValueError(f"No data for secid {secid} on {date} (not a trading day?)")

    # ---- Tidy up ----
    opts["strike"] = opts["strike_price"] / 1000                     # stored x1000
    opts["mid"] = (opts["best_bid"] + opts["best_offer"]) / 2
    opts["days"] = (opts["exdate"] - opts["date"]).dt.days
    opts["T"] = opts["days"] / 365
    opts["S"] = spot["close"].iloc[0]
    opts["r"] = np.interp(opts["days"], zc["days"], zc["rate"]) / 100  # % -> decimal
    opts["q"] = div["rate"].iloc[0] / 100

    opts = opts[(opts["best_bid"] > 0) & opts["impl_volatility"].notna()].reset_index(drop=True)

    if save:
        opts.to_csv(f"source/data/multi_option_single_day_{date}.csv", index=False)

    return opts


def single_option_multi_day(username, optionid, start, end, secid=108105, save=True):
    """Return daily quotes for one option between start and end, with S, r, q, T and mid.

    username   : WRDS username
    optionid   : OptionMetrics option id
    start, end : "YYYY-MM-DD" (can span several years)
    secid      : OptionMetrics security id (108105 = SPX)
    save       : if True, also writes source/data/single_option_multi_day{start}-{end}.csv
    """
    years = range(int(start[:4]), int(end[:4]) + 1)

    def year_range(y):
        # clip the date window to year y, since opprcd/secprd are split by year
        return max(start, f"{y}-01-01"), min(end, f"{y}-12-31")

    db = wrds.Connection(wrds_username=username)

    # 1. The option each day
    opt = pd.concat([
        db.raw_sql(f"""
            SELECT date, optionid, exdate, cp_flag, strike_price, best_bid, best_offer,
                   volume, open_interest, impl_volatility, delta,
                   am_settlement, expiry_indicator, forward_price
            FROM optionm.opprcd{y}
            WHERE secid = {secid}
              AND optionid = {optionid}
              AND date BETWEEN '{year_range(y)[0]}' AND '{year_range(y)[1]}'
        """, date_cols=["date", "exdate"])
        for y in years
    ])

    # 2. Index level each day
    spot = pd.concat([
        db.raw_sql(f"""
            SELECT date, close
            FROM optionm.secprd{y}
            WHERE secid = {secid}
              AND date BETWEEN '{year_range(y)[0]}' AND '{year_range(y)[1]}'
        """, date_cols=["date"])
        for y in years
    ])

    # 3. Zero curve each day (not split by year)
    zc = db.raw_sql(f"""
        SELECT date, days, rate
        FROM optionm.zerocd
        WHERE date BETWEEN '{start}' AND '{end}'
        ORDER BY date, days
    """, date_cols=["date"])

    # 4. Dividend yield each day (not split by year)
    div = db.raw_sql(f"""
        SELECT date, rate
        FROM optionm.idxdvd
        WHERE secid = {secid}
          AND date BETWEEN '{start}' AND '{end}'
    """, date_cols=["date"])

    db.close()

    if opt.empty:
        raise ValueError(f"No data for option {optionid} between {start} and {end}")

    # ---- Attach everything by date ----
    opt = opt.sort_values("date").reset_index(drop=True)
    opt["strike"] = opt["strike_price"] / 1000                  # stored x1000
    opt["days"] = (opt["exdate"] - opt["date"]).dt.days
    opt["T"] = opt["days"] / 365
    opt["mid"] = (opt["best_bid"] + opt["best_offer"]) / 2

    # S: that day's index close
    opt = opt.merge(spot.rename(columns={"close": "S"}), on="date", how="left")

    # q: that day's dividend yield (carries the last value forward if a day is missing)
    div["q"] = div["rate"] / 100
    opt = pd.merge_asof(opt, div[["date", "q"]].sort_values("date"), on="date")

    # r: that day's zero curve, interpolated to the option's remaining maturity
    curves = {d: g for d, g in zc.groupby("date")}
    opt["r"] = [np.interp(n, curves[d]["days"], curves[d]["rate"]) / 100
                for d, n in zip(opt["date"], opt["days"])]

    if save:
        opt.to_csv(f"source/data/single_option_multi_day_{start}-{end}.csv", index=False)

    return opt


def return_multi_day(username, start, end, lookback=60, secid=108105, save=True):
    """Return daily close, simple return and log return from `start` to `end`,
    plus `lookback` extra trading days before `start` (flagged in the 'lookback' column).

    username   : WRDS username
    start, end : "YYYY-MM-DD"
    lookback   : trading days of extra history before start (0 = none)
    secid      : OptionMetrics security id (108105 = SPX)
    save       : if True, also writes source/data/return_multi_day_{start}_{end}_lb{lookback}.csv
    """
    # Trading days aren't calendar days, so fetch a generous calendar window and trim afterwards
    fetch_start = (pd.Timestamp(start)
                   - pd.Timedelta(days=int(lookback * 365 / 252) + 10)).strftime("%Y-%m-%d")

    db = wrds.Connection(wrds_username=username)

    # secprd is split into one table per year, so query each year and stack them
    spot = pd.concat([
        db.raw_sql(f"""
            SELECT date, close, return
            FROM optionm.secprd{y}
            WHERE secid = {secid}
              AND date BETWEEN '{max(fetch_start, f"{y}-01-01")}' AND '{min(end, f"{y}-12-31")}'
        """, date_cols=["date"])
        for y in range(int(fetch_start[:4]), int(end[:4]) + 1)
    ]).set_index("date").sort_index()

    db.close()

    assert spot.index.is_unique, "duplicate dates"

    # Keep exactly `lookback` trading days before start, then everything from start on
    start_ts = pd.Timestamp(start)
    before = spot[spot.index < start_ts].tail(lookback) if lookback else spot.iloc[:0]
    spot = pd.concat([before, spot[spot.index >= start_ts]])

    if len(before) < lookback:
        print(f"Warning: only {len(before)} of {lookback} lookback days available")

    spot["lookback"] = spot.index < start_ts           # True for the warm-up rows
    spot["log_return"] = np.log1p(spot["return"])      # simple return -> log return

    if save:
        spot.to_csv(f"source/data/return_multi_day_{start}_{end}_lb{lookback}.csv")

    return spot


def get_vix(username, start, end, save=True):
    """Return daily VIX open, high, low and close from `start` to `end`.

    username   : WRDS username
    start, end : "YYYY-MM-DD"
    save       : if True, also writes source/data/vix_{start}_{end}.csv

    VIX is quoted in vol points (20 = 20% annualised); 'vix_dec' is the
    close as a decimal (0.20), ready to use alongside implied vols.
    """
    db = wrds.Connection(wrds_username=username)

    # cboe.cboe holds all the Cboe vol indices (VIX, VXO, VXN, VXD) side by side;
    # select everything, then keep only the VIX columns, so a renamed column
    # can't silently break the query
    raw = db.raw_sql(f"""
        SELECT *
        FROM cboe.cboe
        WHERE date BETWEEN '{start}' AND '{end}'
    """, date_cols=["date"])

    db.close()

    if raw.empty:
        raise ValueError(f"No VIX data between {start} and {end}")

    vix = raw[["date"] + [c for c in raw.columns if c.startswith("vix")]]
    vix = vix.rename(columns={"vixo": "open", "vixh": "high",
                              "vixl": "low", "vix": "close"})
    vix = vix.set_index("date").sort_index()

    vix = vix.dropna(subset=["close"])          # drop any days with no closing level
    assert vix.index.is_unique, "duplicate dates"

    vix["vix_dec"] = vix["close"] / 100         # 20 -> 0.20

    if save:
        vix.to_csv(f"source/data/vix_{start}_{end}.csv")

    return vix


def exercise_style(username, optionid, year=None):
    """Return 'E' (European) or 'A' (American) for an OptionMetrics option.
 
    username : WRDS username
    optionid : OptionMetrics option id
    year     : a year the option traded in (e.g. 2023). Optional, but much faster:
               opprcd is split into one table per year, so without it every year
               is searched, newest first, until the option is found.
 
    Exercise style is stored per underlying security (optionm.opinfd), so this
    finds the option's secid in opprcd, then looks that secid up in opinfd.
    """
    db = wrds.Connection(wrds_username=username)
 
    if year is not None:
        years = [int(year)]
    else:
        # only the opprcd years that actually exist on WRDS, newest first
        tables = db.list_tables(library="optionm")
        years = sorted((int(t[6:]) for t in tables
                        if t.startswith("opprcd") and t[6:].isdigit()), reverse=True)
 
    style = None
    for y in years:
        res = db.raw_sql(f"""
            SELECT i.exercise_style
            FROM optionm.opprcd{y} AS p
            JOIN optionm.opinfd AS i ON i.secid = p.secid
            WHERE p.optionid = {optionid}
            LIMIT 1
        """)
        if not res.empty:
            style = res["exercise_style"].iloc[0]
            break
 
    db.close()
 
    if style is None:
        where = f"in {year}" if year is not None else "in any year"
        raise ValueError(f"Option {optionid} not found {where}")
 
    return style


if __name__ == "__main__":
    username = "kairogers"
    vix = get_vix(username, "2023-06-12", "2023-12-15", save=False)
    print(f"{len(vix)} trading days: {vix.index[0].date()} to {vix.index[-1].date()}")
    print(vix.head())
