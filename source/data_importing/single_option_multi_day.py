"""Download one option's daily history from OptionMetrics (WRDS), with S, r and q attached."""
import wrds
import numpy as np
import pandas as pd


def single_option_multi_day(optionid, start, end, secid=108105, save=True):
    """Return daily quotes for one option between start and end, with S, r, q, T and mid.

    optionid   : OptionMetrics option id
    start, end : "YYYY-MM-DD" (can span several years)
    secid      : OptionMetrics security id (108105 = SPX)
    save       : if True, also writes source/data/single_option_multi_day{start}-{end}.csv
    """
    years = range(int(start[:4]), int(end[:4]) + 1)

    def year_range(y):
        # clip the date window to year y, since opprcd/secprd are split by year
        return max(start, f"{y}-01-01"), min(end, f"{y}-12-31")

    db = wrds.Connection(wrds_username="kairogers")

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


if __name__ == "__main__":
    opt = get_option_history(155036250, "2023-05-01", "2023-12-15")
    print(f"{len(opt)} days: {opt['date'].min().date()} to {opt['date'].max().date()}")
    print(opt.head())