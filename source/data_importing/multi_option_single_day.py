"""Download a single-day SPX option chain from OptionMetrics (WRDS)."""
import wrds
import numpy as np
import pandas as pd


def multi_option_single_day(date, secid=108105, save=True):
    """Return a cleaned option chain for one trading day.

    date  : "YYYY-MM-DD"
    secid : OptionMetrics security id (108105 = SPX)
    save  : if True, also writes source/data/multi_option_single_day{date}.csv
    """
    year = date[:4]
    db = wrds.Connection(wrds_username="kairogers")

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


if __name__ == "__main__":
    chain = get_option_chain("2023-06-12")
    print(f"{len(chain)} options, {chain['exdate'].nunique()} expiries")
    print(chain.head())