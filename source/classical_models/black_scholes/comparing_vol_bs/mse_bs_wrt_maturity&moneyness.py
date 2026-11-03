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

# WRDS username and period (same as mse_bs_wrt_maturity.py and mse_bs_wrt_moneyness.py)
username = "kairogers"
start, end = "2019-01-01", "2023-12-31"

# vol methods to compare, and their plot colours
methods = {"hist_vol_zm": "zero-mean historical vol",
           "hist_vol_dm": "demeaned historical vol",
           "vix_dec": "VIX"}
colours = {"hist_vol_zm": "orange", "hist_vol_dm": "red", "vix_dec": "green"}

# buckets for the two sets of plots
maturity_bins = [0, 30, 60, 90, 180, 365, 730]                      # days to expiry
moneyness_bins = [0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5]    # strike / spot

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


def add_errors(df):
    """Black–Scholes price for each vol method, and the error vs the market mid price ($ per unit)."""
    is_call = df["cp_flag"] == "C"
    for col in methods:
        call_price = black_scholes_call(df["S"], df["strike"], df[col], df["r"], df["T"], df["q"])
        put_price = black_scholes_put(df["S"], df["strike"], df[col], df["r"], df["T"], df["q"])
        model_price = np.where(is_call, call_price, put_price)

        error = model_price - df["mid"]
        df[f"sq_err_{col}"] = error ** 2                        # for RMSE
        df[f"abs_err_{col}"] = np.abs(error)                    # for median absolute error
        df[f"abs_pct_err_{col}"] = np.abs(error) / df["mid"]    # for MAPE (mid > 0 since best_bid > 0)
    return df


def error_tables(df, bucket_col):
    """RMSE, median absolute error and MAPE for each vol method, in each bucket of `bucket_col`."""
    groups = df.groupby(bucket_col, observed=True)
    rmse = {}   # root mean squared error, $
    mdae = {}   # median absolute error, $
    mape = {}   # mean absolute percentage error, %
    for col in methods:
        rmse[col] = np.sqrt(groups[f"sq_err_{col}"].mean())
        mdae[col] = groups[f"abs_err_{col}"].median()
        mape[col] = groups[f"abs_pct_err_{col}"].mean() * 100
    return pd.DataFrame(rmse), pd.DataFrame(mdae), pd.DataFrame(mape)


def plot_errors(rmse, mdae, mape, xlabel, title):
    """One figure with 3 panels: RMSE, median absolute error, MAPE."""
    fig, (ax_rmse, ax_mdae, ax_mape) = plt.subplots(1, 3, figsize=(19, 5))

    labels = [f"{b.left}–{b.right}" for b in rmse.index]
    for col, name in methods.items():
        ax_rmse.plot(labels, rmse[col], "o-", lw=2, color=colours[col], label=f"Black–Scholes {name}")
        ax_mdae.plot(labels, mdae[col], "o-", lw=2, color=colours[col], label=f"Black–Scholes {name}")
        ax_mape.plot(labels, mape[col], "o-", lw=2, color=colours[col], label=f"Black–Scholes {name}")

    ax_rmse.set_ylabel("RMSE ($ per unit)")
    ax_mdae.set_ylabel("Median absolute error ($ per unit)")
    ax_mape.set_ylabel("Mean absolute percentage error (%)")

    for ax in (ax_rmse, ax_mdae, ax_mape):
        ax.set_xlabel(xlabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_rmse.legend(frameon=False)

    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    plt.show()

#%%
# ---- For each security: 3 plots by maturity, then 3 plots by moneyness ----
for secid, (ticker, name) in SECURITIES.items():
    try:
        df = load_options(secid)
    except Exception as e:
        print(f"Warning: {ticker}: no data ({e})")
        continue

    df = add_errors(df)
    df["maturity"] = pd.cut(df["days"], maturity_bins)
    df["moneyness"] = df["strike"] / df["S"]
    df["moneyness_bin"] = pd.cut(df["moneyness"], moneyness_bins)

    print(f"\n==== {ticker} ({name}): {len(df)} options on {df['date'].nunique()} days ====")

    rmse, mdae, mape = error_tables(df, "maturity")
    plot_errors(rmse, mdae, mape, "Days to expiry",
                f"{ticker} ({name}) options, pricing error by maturity, {start} to {end} (monthly)")

    rmse, mdae, mape = error_tables(df, "moneyness_bin")
    plot_errors(rmse, mdae, mape, "Moneyness (K / S)",
                f"{ticker} ({name}) options, pricing error by moneyness, {start} to {end} (monthly)")

# %%
