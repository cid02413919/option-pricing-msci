#%%
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("source")  # tells python to look in source

from classical_models.black_scholes.bs_funcs import black_scholes_call
from classical_models.black_scholes.vol_funcs import historical_volatility_series
from data_importing.data_import import multi_option_multi_day, return_multi_day, get_vix

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
# ---- Option chains on each date (one WRDS connection; cached locally after the first run) ----
cache = Path(f"source/data/cache_mse_chains_{start}_{end}.pkl")
if cache.exists():
    df = pd.read_pickle(cache)
else:
    df = multi_option_multi_day(username, dates, secid=108105, save=False)
    df.to_pickle(cache)
print(df.groupby("date").size().rename("n options").to_string())

df = df[df["T"] > 0].copy()
for col in ("hist_vol_zm", "hist_vol_dm"):
    df[col] = df["date"].map(rtn[col])
df["vix_dec"] = df["date"].map(vix["vix_dec"])

#%%
# ---- Black–Scholes price for each vol method, and squared error vs market mid ----
methods = {"impl_volatility": "implied vol",
           "hist_vol_zm": "zero-mean historical vol",
           "hist_vol_dm": "demeaned historical vol",
           "vix_dec": "VIX"}

S, K, r, T, q = (df[c].to_numpy() for c in ("S", "strike", "r", "T", "q"))
is_put = (df["cp_flag"] == "P").to_numpy()
parity = K * np.exp(-r * T) - S * np.exp(-q * T)        # put = call + parity
mid = df["mid"].to_numpy()
for col in methods:
    # price calls once and get puts from put-call parity, instead of pricing both for every row
    call = black_scholes_call(S, K, df[col].to_numpy(), r, T, q)
    err = call + is_put * parity - mid
    df[f"sq_err_{col}"] = err ** 2
    df[f"sq_pct_err_{col}"] = (err / mid) ** 2          # mid > 0 since best_bid > 0
    df[f"abs_pct_err_{col}"] = np.abs(err / mid)

# MSE, RMSPE and MdAPE in maturity buckets (days to expiry)
bins = [0, 30, 60, 90, 180, 365, 730]
df["maturity"] = pd.cut(df["days"], bins)
by_mat = df.groupby("maturity", observed=True)
mse = by_mat[[f"sq_err_{c}" for c in methods]].mean()
rmspe = np.sqrt(by_mat[[f"sq_pct_err_{c}" for c in methods]].mean()) * 100   # in %
mdape = by_mat[[f"abs_pct_err_{c}" for c in methods]].median() * 100        # in %
print(mse.to_string())
print(rmspe.to_string())
print(mdape.to_string())
print(df.groupby("maturity", observed=True).size().rename("n options"))

#%%
# ---- Plot ----
fig, (ax_mse, ax_pct, ax_med) = plt.subplots(1, 3, figsize=(19, 5))

colours = {"impl_volatility": "pink", "hist_vol_zm": "orange", "hist_vol_dm": "red", "vix_dec": "green"}
labels = [f"{b.left}–{b.right}" for b in mse.index]
for col, name in methods.items():
    ax_mse.plot(labels, mse[f"sq_err_{col}"], "o-", lw=2, color=colours[col], label=f"Black–Scholes {name}")
    ax_pct.plot(labels, rmspe[f"sq_pct_err_{col}"], "o-", lw=2, color=colours[col], label=f"Black–Scholes {name}")
    ax_med.plot(labels, mdape[f"abs_pct_err_{col}"], "o-", lw=2, color=colours[col], label=f"Black–Scholes {name}")

ax_mse.set_ylabel("MSE (index points²)")
ax_pct.set_ylabel("RMSPE (%)")
ax_med.set_ylabel("Median absolute percentage error (%)")

for ax in (ax_mse, ax_pct, ax_med):
    ax.set_yscale("log")                            # implied vol error is far smaller than the rest
    ax.set_xlabel("Days to expiry")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
ax_mse.legend(frameon=False)

fig.suptitle(f"SPX options, pricing error by maturity, {start} to {end} (monthly)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout()
plt.show()

# %%
