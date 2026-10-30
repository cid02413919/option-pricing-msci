"""Download daily SPX index levels and returns from OptionMetrics (WRDS) for realised-vol work."""
import wrds
import numpy as np
import pandas as pd


def return_multi_day(start, end, lookback=60, secid=108105, save=True):
    """Return daily close, simple return and log return from `start` to `end`,
    plus `lookback` extra trading days before `start` (flagged in the 'lookback' column).

    start, end : "YYYY-MM-DD"
    lookback   : trading days of extra history before start (0 = none)
    secid      : OptionMetrics security id (108105 = SPX)
    save       : if True, also writes source/data/return_multi_day_{start}_{end}_lb{lookback}.csv
    """
    # Trading days aren't calendar days, so fetch a generous calendar window and trim afterwards
    fetch_start = (pd.Timestamp(start)
                   - pd.Timedelta(days=int(lookback * 365 / 252) + 10)).strftime("%Y-%m-%d")

    db = wrds.Connection(wrds_username="kairogers")

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


if __name__ == "__main__":
    spot = get_returns("2023-06-12", "2023-12-15")
    print(f"{len(spot)} trading days ({spot['lookback'].sum()} lookback): "
          f"{spot.index[0].date()} to {spot.index[-1].date()}")
    print(spot.head())