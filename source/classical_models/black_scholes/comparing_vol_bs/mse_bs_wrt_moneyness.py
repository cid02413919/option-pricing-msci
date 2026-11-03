#%%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("source")  # tells python to look in source

from classical_models.black_scholes.bs_funcs import black_scholes_call, black_scholes_put
from classical_models.black_scholes.vol_funcs import historical_volatility
from data_importing.data_import import multi_option_single_day, return_multi_day, get_vix

# WRDS username and the day to price (SPX only, since VIX is SPX vol)
username = "kairogers"
date = "2023-06-13"

# ---- Option chain on that day ----
df = multi_option_single_day(username, date, secid=108105, save=False)
df = df[df["T"] > 0]

# ---- One vol per method, using only data known before the day ----
# 60 trading days of returns before `date` (the lookback rows), so no leakage
rtn = return_multi_day(username, date, date, lookback=60, secid=108105, save=False)
past = rtn.loc[rtn["lookback"], "log_return"]
df["hist_vol_zm"] = historical_volatility(past, window=60, method="zero_mean")
df["hist_vol_dm"] = historical_volatility(past, window=60, method="demeaned")

# get_vix already gives the previous day's VIX on each date
df["vix_dec"] = get_vix(username, date, date, save=False)["vix_dec"].iloc[0]

#%%
# ---- Black–Scholes price for each vol method, and squared error vs market mid ----
methods = {"hist_vol_zm": "zero-mean historical vol",
           "hist_vol_dm": "demeaned historical vol",
           "vix_dec": "VIX"}

is_call = df["cp_flag"] == "C"
for col in methods:
    call = black_scholes_call(df["S"], df["strike"], df[col], df["r"], df["T"], df["q"])
    put = black_scholes_put(df["S"], df["strike"], df[col], df["r"], df["T"], df["q"])
    df[f"sq_err_{col}"] = (np.where(is_call, call, put) - df["mid"]) ** 2

# Moneyness = strike / spot (1 = at the money)
df["moneyness"] = df["strike"] / df["S"]

# MSE in moneyness buckets
bins = [0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5]
df["moneyness_bin"] = pd.cut(df["moneyness"], bins)
mse = df.groupby("moneyness_bin", observed=True)[[f"sq_err_{c}" for c in methods]].mean()
print(mse.to_string())
print(df.groupby("moneyness_bin", observed=True).size().rename("n options"))

#%%
# ---- Plot ----
fig, ax = plt.subplots(figsize=(9, 5))

colours = {"hist_vol_zm": "orange", "hist_vol_dm": "red", "vix_dec": "green"}
labels = [f"{b.left}–{b.right}" for b in mse.index]
for col, name in methods.items():
    ax.plot(labels, mse[f"sq_err_{col}"], "o-", lw=2, color=colours[col], label=f"Black–Scholes {name}")

ax.set_xlabel("Moneyness (K / S)")
ax.set_ylabel("MSE (index points²)")
ax.set_title(f"SPX options, pricing error by moneyness, {date}", loc="left", fontweight="bold")

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False)

fig.tight_layout()
plt.show()

# %%
