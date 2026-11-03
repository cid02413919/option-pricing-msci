#%%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("source")  # tells python to look in source

from classical_models.black_scholes.bs_funcs import black_scholes_call, black_scholes_put
from classical_models.black_scholes.vol_funcs import historical_volatility_series
from data_importing.data_import import multi_option_single_day, return_multi_day, get_vix

# WRDS username and period (SPX only, since VIX is SPX vol)
username = "kairogers"
start, end = "2019-01-01", "2023-12-31"

# ---- Daily vols over the whole period ----
rtn = return_multi_day(username, start, end, secid=108105, save=False)
rtn["hist_vol_zm"] = historical_volatility_series(rtn["log_return"], window=60, annualization=252, method="zero_mean")
rtn["hist_vol_dm"] = historical_volatility_series(rtn["log_return"], window=60, annualization=252, method="demeaned")
rtn = rtn[~rtn["lookback"]]
vix = get_vix(username, start, end, save=False)

# one trading day per month (the first), to average over the period
dates = rtn.groupby(rtn.index.to_period("M")).head(1).index

#%%
# ---- Option chains on each date (slow: one WRDS call per date) ----
chains = []
for d in dates:
    opts = multi_option_single_day(username, d.strftime("%Y-%m-%d"), secid=108105, save=False)
    opts["hist_vol_zm"] = rtn.loc[d, "hist_vol_zm"]
    opts["hist_vol_dm"] = rtn.loc[d, "hist_vol_dm"]
    opts["vix_dec"] = vix["vix_dec"].get(d, np.nan)
    chains.append(opts)
    print(d.date(), len(opts), "options")

df = pd.concat(chains, ignore_index=True)
df = df[df["T"] > 0]

#%%
# ---- Black–Scholes price for each vol method, and squared error vs market mid ----
methods = {"impl_volatility": "implied vol",
           "hist_vol_zm": "zero-mean historical vol",
           "hist_vol_dm": "demeaned historical vol",
           "vix_dec": "VIX"}

is_call = df["cp_flag"] == "C"
for col in methods:
    call = black_scholes_call(df["S"], df["strike"], df[col], df["r"], df["T"], df["q"])
    put = black_scholes_put(df["S"], df["strike"], df[col], df["r"], df["T"], df["q"])
    df[f"sq_err_{col}"] = (np.where(is_call, call, put) - df["mid"]) ** 2

# MSE in maturity buckets (days to expiry)
bins = [0, 30, 60, 90, 180, 365, 730]
df["maturity"] = pd.cut(df["days"], bins)
mse = df.groupby("maturity", observed=True)[[f"sq_err_{c}" for c in methods]].mean()
print(mse.to_string())
print(df.groupby("maturity", observed=True).size().rename("n options"))

#%%
# ---- Plot ----
fig, ax = plt.subplots(figsize=(9, 5))

colours = {"impl_volatility": "pink", "hist_vol_zm": "orange", "hist_vol_dm": "red", "vix_dec": "green"}
labels = [f"{b.left}–{b.right}" for b in mse.index]
for col, name in methods.items():
    ax.plot(labels, mse[f"sq_err_{col}"], "o-", lw=2, color=colours[col], label=f"Black–Scholes {name}")

ax.set_yscale("log")                                # implied vol error is far smaller than the rest
ax.set_xlabel("Days to expiry")
ax.set_ylabel("MSE (index points²)")
ax.set_title(f"SPX options, pricing error by maturity, {start} to {end} (monthly)", loc="left", fontweight="bold")

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False)

fig.tight_layout()
plt.show()

# %%
