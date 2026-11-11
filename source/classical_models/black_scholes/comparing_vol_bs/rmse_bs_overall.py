#%%
import sys
import os
import numpy as np
import pandas as pd

sys.path.append("source")  # tells python to look in source

from classical_models.black_scholes.bs_funcs import black_scholes_call, black_scholes_put, historical_volatility_series
from data_importing.data_import import multi_option_multi_day, return_multi_day, get_vix, wednesdays

# WRDS username and period (SPX only, since VIX is SPX vol)
username = "kairogers"
start, end = "2019-01-01", "2023-12-31"

# ---- Settings: switch these on/off ----
SHOW_ALL_ERRORS = False   # True also does median absolute error and MAPE (RMSE is always done)

methods = {"hist_vol_zm": "zero-mean historical vol",
           "hist_vol_dm": "demeaned historical vol",
           "vix_dec": "VIX"}

#%%
# ---- Vols for every day ----
# rolling 60-day historical vol, shifted one day so day t only uses returns up to t-1
rtn = return_multi_day(username, start, end, lookback=60, secid=108105, save=False)
rtn["hist_vol_zm"] = historical_volatility_series(rtn["log_return"], window=60, method="zero_mean").shift(1)
rtn["hist_vol_dm"] = historical_volatility_series(rtn["log_return"], window=60, method="demeaned").shift(1)
rtn = rtn[rtn["lookback"] == False]

# get_vix already gives the previous day's value
vix = get_vix(username, start, end, save=False)


def cached(cache_file, download):
    """Load a saved download if there is one, otherwise download and save it. Delete the file to refresh."""
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file)
    df = download()
    df.to_pickle(cache_file)
    return df


def attach_vols(df):
    """Drop options expiring that day, and attach each day's vol for every method."""
    df = df[df["T"] > 0].copy()
    df["hist_vol_zm"] = df["date"].map(rtn["hist_vol_zm"])
    df["hist_vol_dm"] = df["date"].map(rtn["hist_vol_dm"])
    df["vix_dec"] = df["date"].map(vix["vix_dec"])
    return df


def error_table(df):
    """Black–Scholes price vs market mid for each vol method: RMSE (and the others if SHOW_ALL_ERRORS)."""
    is_call = df["cp_flag"] == "C"
    results = {}
    for col, name in methods.items():
        call_price = black_scholes_call(df["S"], df["strike"], df[col], df["r"], df["T"], df["q"])
        put_price = black_scholes_put(df["S"], df["strike"], df[col], df["r"], df["T"], df["q"])
        error = np.where(is_call, call_price, put_price) - df["mid"]

        results[name] = {"RMSE ($)": np.sqrt(np.mean(error ** 2))}
        if SHOW_ALL_ERRORS:
            results[name]["Median abs error ($)"] = np.median(np.abs(error))
            results[name]["MAPE (%)"] = np.mean(np.abs(error) / df["mid"]) * 100   # mid > 0 since best_bid > 0
    return pd.DataFrame(results).T

#%%
# ---- The dates we price on: every Wednesday, or the Tuesday before if Wednesday is a holiday ----
# (every trading day would be ~19M quotes / ~7 GB, too slow and too big)
dates = wednesdays(rtn.index, start, end)

# ---- Option chains on those dates (same cache file as the other SPX scripts) ----
df = cached(f"source/data/cache_weekly_chains_{start}_{end}.pkl",
            lambda: multi_option_multi_day(username, dates, secid=108105, save=False, calc_q=True))
df = attach_vols(df)

#%%
# ---- Results ----
results = error_table(df)
print(f"Overall Black–Scholes pricing error, SPX, {start} to {end} (weekly, Wednesdays)\n")
print(results.to_string(float_format="%.3f"))
print(f"\n{len(df)} option quotes on {df['date'].nunique()} days")

# %%

