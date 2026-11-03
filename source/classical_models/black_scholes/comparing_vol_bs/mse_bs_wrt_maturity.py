#%%
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# Downloading takes a while, so save it to a file the first time and load the file after that.
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
# ---- Black–Scholes price for each vol method, and the error vs the market mid price ----
# prices are in $ per unit of the index
methods = {"hist_vol_zm": "zero-mean historical vol",
           "hist_vol_dm": "demeaned historical vol",
           "vix_dec": "VIX"}

is_call = df["cp_flag"] == "C"

for col in methods:
    call_price = black_scholes_call(df["S"], df["strike"], df[col], df["r"], df["T"], df["q"])
    put_price = black_scholes_put(df["S"], df["strike"], df[col], df["r"], df["T"], df["q"])
    model_price = np.where(is_call, call_price, put_price)

    error = model_price - df["mid"]
    df[f"sq_err_{col}"] = error ** 2                        # for RMSE
    df[f"abs_err_{col}"] = np.abs(error)                    # for median absolute error
    df[f"abs_pct_err_{col}"] = np.abs(error) / df["mid"]    # for MAPE (mid > 0 since best_bid > 0)

#%%
# ---- Group the options into maturity buckets (days to expiry) ----
bins = [0, 30, 60, 90, 180, 365, 730]
df["maturity"] = pd.cut(df["days"], bins)

rmse = {}   # root mean squared error, $
mdae = {}   # median absolute error, $
mape = {}   # mean absolute percentage error, %
for col in methods:
    rmse[col] = np.sqrt(df.groupby("maturity", observed=True)[f"sq_err_{col}"].mean())
    mdae[col] = df.groupby("maturity", observed=True)[f"abs_err_{col}"].median()
    mape[col] = df.groupby("maturity", observed=True)[f"abs_pct_err_{col}"].mean() * 100

print("RMSE ($)\n", pd.DataFrame(rmse).to_string())
print("Median absolute error ($)\n", pd.DataFrame(mdae).to_string())
print("MAPE (%)\n", pd.DataFrame(mape).to_string())
print(df.groupby("maturity", observed=True).size().rename("n options"))

#%%
# ---- Plot ----
fig, (ax_rmse, ax_mdae, ax_mape) = plt.subplots(1, 3, figsize=(19, 5))

colours = {"hist_vol_zm": "orange", "hist_vol_dm": "red", "vix_dec": "green"}
labels = [f"{b.left}–{b.right}" for b in rmse["vix_dec"].index]

for col, name in methods.items():
    ax_rmse.plot(labels, rmse[col], "o-", lw=2, color=colours[col], label=f"Black–Scholes {name}")
    ax_mdae.plot(labels, mdae[col], "o-", lw=2, color=colours[col], label=f"Black–Scholes {name}")
    ax_mape.plot(labels, mape[col], "o-", lw=2, color=colours[col], label=f"Black–Scholes {name}")

ax_rmse.set_ylabel("RMSE ($ per unit)")
ax_mdae.set_ylabel("Median absolute error ($ per unit)")
ax_mape.set_ylabel("Mean absolute percentage error (%)")

for ax in (ax_rmse, ax_mdae, ax_mape):
    ax.set_xlabel("Days to expiry")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
ax_rmse.legend(frameon=False)

fig.suptitle(f"SPX options, pricing error by maturity, {start} to {end} (monthly)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
plt.show()

# %%
