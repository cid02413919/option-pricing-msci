#%%
import sys
import os
import numpy as np
import pandas as pd

sys.path.append("source")  # tells python to look in source

from classical_models.black_scholes.bs_funcs import black_scholes_call, black_scholes_put
from classical_models.black_scholes.vol_funcs import historical_volatility_series
from data_importing.data_import import multi_option_multi_day, return_multi_day, get_vix

# WRDS username and period (SPX only, since VIX is SPX vol)
username = "kairogers"
start, end = "2019-01-01", "2023-12-31"

#%%
# ---- Historical vol for every day ----
# returns from start to end, plus 60 extra trading days before start to fill the first window
rtn = return_multi_day(username, start, end, lookback=60, secid=108105, save=False)

# rolling 60-day vol, two ways
rtn["hist_vol_zm"] = historical_volatility_series(rtn["log_return"], window=60, method="zero_mean")
rtn["hist_vol_dm"] = historical_volatility_series(rtn["log_return"], window=60, method="demeaned")

# shift by one day, so the vol used on day t only uses returns up to t-1 (no leakage)
rtn["hist_vol_zm"] = rtn["hist_vol_zm"].shift(1)
rtn["hist_vol_dm"] = rtn["hist_vol_dm"].shift(1)

# drop the 60 warm-up days
rtn = rtn[rtn["lookback"] == False]

# ---- VIX for every day (get_vix already gives the previous day's value) ----
vix = get_vix(username, start, end, save=False)

# ---- The dates we price on: the first trading day of each month ----
dates = rtn.groupby(rtn.index.to_period("M")).head(1).index

#%%
# ---- Option chains on all those dates ----
# Same cache file as mse_bs_wrt_maturity.py, so it only gets downloaded once.
# Delete the file to download fresh data.
cache_file = f"source/data/cache_mse_chains_{start}_{end}.pkl"

if os.path.exists(cache_file):
    df = pd.read_pickle(cache_file)
else:
    df = multi_option_multi_day(username, dates, secid=108105, save=False)
    df.to_pickle(cache_file)

# remove options that expire on the day itself
df = df[df["T"] > 0].copy()

# attach each day's vols to every option on that day
df["hist_vol_zm"] = df["date"].map(rtn["hist_vol_zm"])
df["hist_vol_dm"] = df["date"].map(rtn["hist_vol_dm"])
df["vix_dec"] = df["date"].map(vix["vix_dec"])

#%%
# ---- Black–Scholes price for each vol method, and the overall RMSE vs the market mid price ----
# prices are in $ per unit of the index
methods = {"hist_vol_zm": "zero-mean historical vol",
           "hist_vol_dm": "demeaned historical vol",
           "vix_dec": "VIX"}

is_call = df["cp_flag"] == "C"

# one number per method over every option, all dates/maturities/strikes
results = {}
for col, name in methods.items():
    call_price = black_scholes_call(df["S"], df["strike"], df[col], df["r"], df["T"], df["q"])
    put_price = black_scholes_put(df["S"], df["strike"], df[col], df["r"], df["T"], df["q"])
    model_price = np.where(is_call, call_price, put_price)

    error = model_price - df["mid"]
    results[name] = {
        "RMSE ($)": np.sqrt(np.mean(error ** 2)),                       # root mean squared error
        "Median abs error ($)": np.median(np.abs(error)),               # median absolute error
        "MAPE (%)": np.mean(np.abs(error) / df["mid"]) * 100,           # mean absolute percentage error (mid > 0 since best_bid > 0)
    }

results = pd.DataFrame(results).T
print(f"Overall Black–Scholes pricing error, SPX, {start} to {end} (monthly)\n")
print(results.to_string(float_format="%.3f"))
print("\nn options:", len(df))

# %%
