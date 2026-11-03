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

    # idxdvd only covers indices, so for anything else use the trailing 12m distribution yield
    if not div.empty:
        opts["q"] = div["rate"].iloc[0] / 100
        opts["q_source"] = "idxdvd"
    else:
        opts["q"] = trailing_dividend_yield(username, secid, date, date, save=False)["q"].iloc[0]
        opts["q_source"] = "trailing_distrd"

    opts = opts[(opts["best_bid"] > 0) & opts["impl_volatility"].notna()].reset_index(drop=True)

    if save:
        opts.to_csv(f"source/data/multi_option_single_day_{date}.csv", index=False)

    return opts


def multi_option_multi_day(username, dates, secid=108105, save=True):
    """Return cleaned option chains for several trading days, stacked into one frame.

    Same columns and filters as multi_option_single_day, but uses a single WRDS
    connection and one query per year instead of one connection per date, so it
    is much faster for many dates.

    username : WRDS username
    dates    : iterable of dates ("YYYY-MM-DD" strings or Timestamps)
    secid    : OptionMetrics security id (108105 = SPX)
    save     : if True, also writes source/data/multi_option_multi_day_{first}_{last}.csv
    """
    dates = sorted(pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates)
    by_year = {}
    for d in dates:
        by_year.setdefault(d[:4], []).append(d)

    def in_list(ds):
        return ", ".join(f"'{d}'" for d in ds)

    db = wrds.Connection(wrds_username=username)

    # opprcd/secprd are split by year; filter quotes in SQL so fewer rows come over the network
    opts = pd.concat([
        db.raw_sql(f"""
            SELECT date, optionid, exdate, cp_flag, strike_price, best_bid, best_offer,
                   volume, open_interest, impl_volatility, delta,
                   am_settlement, expiry_indicator, forward_price
            FROM optionm.opprcd{y}
            WHERE secid = {secid} AND date IN ({in_list(ds)})
              AND best_bid > 0 AND impl_volatility IS NOT NULL
        """, date_cols=["date", "exdate"])
        for y, ds in by_year.items()
    ], ignore_index=True)

    spot = pd.concat([
        db.raw_sql(f"""
            SELECT date, close FROM optionm.secprd{y}
            WHERE secid = {secid} AND date IN ({in_list(ds)})
        """, date_cols=["date"])
        for y, ds in by_year.items()
    ], ignore_index=True)

    zc = db.raw_sql(f"""
        SELECT date, days, rate FROM optionm.zerocd
        WHERE date IN ({in_list(dates)}) ORDER BY date, days
    """, date_cols=["date"])

    div = db.raw_sql(f"""
        SELECT date, rate FROM optionm.idxdvd
        WHERE secid = {secid} AND date IN ({in_list(dates)})
    """, date_cols=["date"])

    db.close()

    if opts.empty or spot.empty:
        raise ValueError(f"No data for secid {secid} on any of the requested dates")

    missing = sorted(set(pd.to_datetime(dates)) - set(opts["date"]))
    if missing:
        print(f"Warning: no option data on {len(missing)} date(s): {[d.date() for d in missing]}")

    # ---- Tidy up ----
    opts["strike"] = opts["strike_price"] / 1000                     # stored x1000
    opts["mid"] = (opts["best_bid"] + opts["best_offer"]) / 2
    opts["days"] = (opts["exdate"] - opts["date"]).dt.days
    opts["T"] = opts["days"] / 365
    opts["S"] = opts["date"].map(spot.set_index("date")["close"])

    # r: each day's zero curve, interpolated to each option's maturity
    opts["r"] = np.nan
    for d, curve in zc.groupby("date"):
        rows = opts["date"] == d
        opts.loc[rows, "r"] = np.interp(opts.loc[rows, "days"], curve["days"], curve["rate"]) / 100

    # idxdvd only covers indices, so for anything else use the trailing 12m distribution yield
    if not div.empty:
        opts["q"] = opts["date"].map(div.set_index("date")["rate"]) / 100
        opts["q_source"] = "idxdvd"
    else:
        q = trailing_dividend_yield(username, secid, dates[0], dates[-1], save=False)["q"]
        opts["q"] = opts["date"].map(q)
        opts["q_source"] = "trailing_distrd"

    opts = opts.sort_values(["date", "optionid"]).reset_index(drop=True)

    if save:
        opts.to_csv(f"source/data/multi_option_multi_day_{dates[0]}_{dates[-1]}.csv", index=False)

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

    # q: that day's dividend yield (carries the last value forward if a day is missing);
    # idxdvd only covers indices, so for anything else use the trailing 12m distribution yield
    if not div.empty:
        div["q"] = div["rate"] / 100
        opt["q_source"] = "idxdvd"
    else:
        div = trailing_dividend_yield(username, secid, start, end, save=False).reset_index()
        opt["q_source"] = "trailing_distrd"
    opt = pd.merge_asof(opt, div[["date", "q"]].sort_values("date"), on="date")

    # r: that day's zero curve, interpolated to the option's remaining maturity
    curves = {d: g for d, g in zc.groupby("date")}
    opt["r"] = [np.interp(n, curves[d]["days"], curves[d]["rate"]) / 100
                for d, n in zip(opt["date"], opt["days"])]

    if save:
        opt.to_csv(f"source/data/single_option_multi_day_{start}-{end}.csv", index=False)

    return opt


def trailing_dividend_yield(username, secid, start, end, save=True):
    """Return daily S, trailing 12-month cash dividends and continuous dividend yield q.

    username   : WRDS username
    secid      : OptionMetrics security id (e.g. 110007 = XLB)
    start, end : "YYYY-MM-DD"
    save       : if True, also writes source/data/trailing_dividend_yield_{secid}_{start}_{end}.csv

    For each trading day t, div_12m is the sum of regular cash dividends with
    ex-date in (t - 365 days, t], and q = ln(1 + div_12m / S) as a decimal.
    Only past ex-dates are used, so there is no look-ahead.
    """
    # Need a full year of distributions before the first date
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")

    db = wrds.Connection(wrds_username=username)

    # secprd is split into one table per year, so query each year and stack them
    spot = pd.concat([
        db.raw_sql(f"""
            SELECT date, close
            FROM optionm.secprd{y}
            WHERE secid = {secid}
              AND date BETWEEN '{max(start, f"{y}-01-01")}' AND '{min(end, f"{y}-12-31")}'
        """, date_cols=["date"])
        for y in range(int(start[:4]), int(end[:4]) + 1)
    ]).set_index("date").sort_index()

    # distr_type: 1 regular dividend, 2 split, 3 stock dividend (4 capital gain, 5 special excluded)
    dist = db.raw_sql(f"""
        SELECT ex_date, amount, distr_type, adj_factor
        FROM optionm.distrd
        WHERE secid = {secid}
          AND distr_type IN ('1', '2', '3')
          AND cancel_flag = '0'
          AND ex_date > '{fetch_start}' AND ex_date <= '{end}'
        ORDER BY ex_date
    """, date_cols=["ex_date"])

    db.close()

    if spot.empty:
        raise ValueError(f"No prices for secid {secid} between {start} and {end}")
    assert spot.index.is_unique, "duplicate dates"

    # Splits change the share basis, so amounts and prices across one aren't comparable
    splits = dist[dist["distr_type"].isin(["2", "3"]) & (dist["adj_factor"] != 1)]
    if not splits.empty:
        print(f"Warning: split(s) for secid {secid} within the 12m windows, "
              f"div_12m mixes share bases:\n{splits[['ex_date', 'adj_factor']]}")

    divs = dist[dist["distr_type"] == "1"]
    ex = divs["ex_date"].values
    cum = np.concatenate([[0.0], divs["amount"].cumsum().values])

    # Sum over ex-dates in (t - 365, t]: cumulative sum up to t minus up to t - 365
    t = spot.index.values
    hi = np.searchsorted(ex, t, side="right")
    lo = np.searchsorted(ex, (spot.index - pd.Timedelta(days=365)).values, side="right")

    spot["S"] = spot["close"].abs()                    # negative close = bid-ask average
    spot["div_12m"] = cum[hi] - cum[lo]
    spot["q"] = np.log1p(spot["div_12m"] / spot["S"])
    spot = spot[["S", "div_12m", "q"]]

    n_zero = (spot["div_12m"] == 0).sum()
    if n_zero:
        print(f"Warning: no distributions in the past year on {n_zero} of {len(spot)} days "
              f"for secid {secid}, q = 0 there")

    if save:
        spot.to_csv(f"source/data/trailing_dividend_yield_{secid}_{start}_{end}.csv")

    return spot


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

    To avoid look-ahead leakage, each row holds the *previous* trading day's
    VIX: the value on date t is the VIX from t-1, so it is known before any
    option on date t is priced.
    """
    # Need the trading day before start too; 10 calendar days covers weekends/holidays
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    db = wrds.Connection(wrds_username=username)

    # cboe.cboe holds all the Cboe vol indices (VIX, VXO, VXN, VXD) side by side;
    # select everything, then keep only the VIX columns, so a renamed column
    # can't silently break the query
    raw = db.raw_sql(f"""
        SELECT *
        FROM cboe.cboe
        WHERE date BETWEEN '{fetch_start}' AND '{end}'
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

    # Lag by one trading day: date t now holds t-1's values, then trim to [start, end]
    vix = vix.shift(1)
    vix = vix[vix.index >= pd.Timestamp(start)].dropna(subset=["close"])

    if vix.empty:
        raise ValueError(f"No lagged VIX data between {start} and {end}")

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
    date = "2024-06-12"
    multi_option_single_day(username, date, secid=110007, save=True)
