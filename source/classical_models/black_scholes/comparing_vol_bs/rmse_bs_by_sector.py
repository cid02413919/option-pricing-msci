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

# secid -> (ticker, name). The XL... are sector ETFs (American options), SPX is an index (European)
SECURITIES = {
    110007: ("XLB", "Materials"),
    213084: ("XLC", "Communication Services"),
    110011: ("XLE", "Energy"),
    110012: ("XLF", "Financials"),
    110013: ("XLI", "Industrials"),
    110014: ("XLK", "Technology"),
    110009: ("XLP", "Consumer Staples"),
    208181: ("XLRE", "Real Estate"),
    110015: ("XLU", "Utilities"),
    110008: ("XLV", "Health Care"),
    110010: ("XLY", "Consumer Discretionary"),
    108105: ("SPX", "S&P 500 Index"),
}

# WRDS username and period (same as mse_bs_wrt_maturity&moneyness.py, so the saved data files are shared)
username = "kairogers"
start, end = "2019-01-01", "2023-12-31"

# vol methods to compare, and their plot colours
methods = {"hist_vol_zm": "zero-mean historical vol",
           "hist_vol_dm": "demeaned historical vol",
           "vix_dec": "VIX"}
colours = {"hist_vol_zm": "orange", "hist_vol_dm": "red", "vix_dec": "green"}

# VIX is the same for every security, so download it once
# (it is SPX vol, so for the ETFs it is only a rough market-wide vol)
vix = get_vix(username, start, end, save=False)

#%%
# ---- Functions ----
def load_options(secid):
    """Option chains on the first trading day of each month, with each vol method attached."""
    # historical vol for every day: rolling 60-day vol, shifted one day so day t only uses returns up to t-1
    rtn = return_multi_day(username, start, end, lookback=60, secid=secid, save=False)
    rtn["hist_vol_zm"] = historical_volatility_series(rtn["log_return"], window=60, method="zero_mean").shift(1)
    rtn["hist_vol_dm"] = historical_volatility_series(rtn["log_return"], window=60, method="demeaned").shift(1)
    rtn = rtn[rtn["lookback"] == False]

    # the dates we price on: the first trading day of each month
    dates = rtn.groupby(rtn.index.to_period("M")).head(1).index

    # downloading takes a while, so save it to a file the first time and load the file after that
    cache_file = f"source/data/cache_mse_chains_{secid}_{start}_{end}.pkl"
    if os.path.exists(cache_file):
        df = pd.read_pickle(cache_file)
    else:
        df = multi_option_multi_day(username, dates, secid=secid, save=False)
        df.to_pickle(cache_file)

    # remove options that expire on the day itself
    df = df[df["T"] > 0].copy()

    # attach each day's vols to every option on that day
    df["hist_vol_zm"] = df["date"].map(rtn["hist_vol_zm"])
    df["hist_vol_dm"] = df["date"].map(rtn["hist_vol_dm"])
    df["vix_dec"] = df["date"].map(vix["vix_dec"])
    return df


def rmse_per_method(df):
    """RMSE of the Black–Scholes price vs the market mid, for each vol method ($ per unit)."""
    is_call = df["cp_flag"] == "C"
    rmse = {}
    for col in methods:
        call_price = black_scholes_call(df["S"], df["strike"], df[col], df["r"], df["T"], df["q"])
        put_price = black_scholes_put(df["S"], df["strike"], df[col], df["r"], df["T"], df["q"])
        model_price = np.where(is_call, call_price, put_price)

        error = model_price - df["mid"]
        rmse[col] = np.sqrt(np.mean(error ** 2))
    return rmse

#%%
# ---- RMSE for each security ----
results = {}    # ticker -> {method: rmse}
for secid, (ticker, name) in SECURITIES.items():
    try:
        df = load_options(secid)
    except Exception as e:
        print(f"Warning: {ticker}: no data ({e})")
        continue
    results[ticker] = rmse_per_method(df)

table = pd.DataFrame(results).T     # rows = tickers, columns = vol methods
print("RMSE ($ per unit)\n", table.to_string())

#%%
# ---- Plot: one group of bars per security, one bar per vol method ----
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(table))           # one position per security
width = 0.8 / len(methods)          # bars in a group share 0.8 of the space

for i, (col, name) in enumerate(methods.items()):
    ax.bar(x + i * width - 0.4 + width / 2, table[col], width, color=colours[col], label=f"Black–Scholes {name}")

ax.set_xticks(x)
ax.set_xticklabels(table.index)
ax.set_xlabel("Security")
ax.set_ylabel("RMSE ($ per unit)")
ax.set_title(f"Black–Scholes pricing error by sector ETF and S&P 500, {start} to {end} (monthly)",
             loc="left", fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False)

fig.tight_layout()
plt.show()

# %%
