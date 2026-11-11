"""Shared code for the scripts that compare pricing methods against the market:
Black–Scholes with different volatilities, and Heston.

Each script has a METHODS dict of True/False switches at the top; pass chosen(METHODS)
to the functions here so only the switched-on methods are calculated and plotted.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classical_models.black_scholes.bs_funcs import black_scholes_call, black_scholes_put, historical_volatility_series, garch_vol_series
from classical_models.Heston.heston import calibrate_heston_by_date, heston_price_chain
from data_importing.data_import import multi_option_multi_day, return_multi_day, get_vix, wednesdays

# legend label and plot colour for every method
LABELS = {
    "implied":     "Black–Scholes implied vol",
    "hist_vol_zm": "Black–Scholes zero-mean historical vol",
    "hist_vol_dm": "Black–Scholes demeaned historical vol",
    "garch_vol":   "Black–Scholes GJR-GARCH vol",
    "vix_dec":     "Black–Scholes VIX",
    "heston":      "Heston (calibrated on previous day)",
}
COLOURS = {"implied": "pink", "hist_vol_zm": "orange", "hist_vol_dm": "red",
           "garch_vol": "purple", "vix_dec": "green", "heston": "brown"}


def chosen(methods):
    """The methods switched on (True) in a script's METHODS dict."""
    return [m for m, on in methods.items() if on]


def cached(cache_file, download):
    """Load a saved download if there is one, otherwise download and save it. Delete the file to refresh."""
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file)
    data = download()
    pd.to_pickle(data, cache_file)
    return data


def load_weekly_prices(username, secid, start, end, methods):
    """Option chains on every Wednesday (Tuesday if Wednesday is a holiday) from start to end,
    with a model price column 'price_{method}' for each method in `methods`.

    Every method only uses information from before the pricing day:
      historical vols : rolling 60-day vol, shifted one day
      GARCH           : re-fit on the 252 returns before the day, forecast over the option's life
      VIX             : get_vix already gives the previous day's value
      Heston          : calibrated to the previous trading day's whole option chain
    Only options that every method could price are kept, so all methods are compared on the same options.
    Downloads (and the Heston calibrations) are saved in source/data; delete the files to redo them.
    """
    # returns, with a year before start: GARCH's fitting window, and the day before the first Wednesday for Heston
    rtn = return_multi_day(username, start, end, lookback=252, secid=secid, save=False)
    log_returns = rtn["log_return"]
    prev_day = dict(zip(rtn.index[1:], rtn.index[:-1]))       # trading day -> trading day before it
    rtn["hist_vol_zm"] = historical_volatility_series(log_returns, window=60, method="zero_mean").shift(1)
    rtn["hist_vol_dm"] = historical_volatility_series(log_returns, window=60, method="demeaned").shift(1)
    rtn = rtn[rtn["lookback"] == False]

    # option chains on the pricing dates (every trading day would be ~19M quotes / ~7 GB for SPX)
    dates = wednesdays(rtn.index, start, end)
    df = cached(f"source/data/cache_weekly_chains_{secid}_{start}_{end}.pkl",
                lambda: multi_option_multi_day(username, dates, secid=secid, save=False, calc_q=True))
    df = df[df["T"] > 0].copy()                     # remove options that expire on the day itself

    # each Black–Scholes method's vol for every option
    for m in ("hist_vol_zm", "hist_vol_dm"):
        if m in methods:
            df[m] = df["date"].map(rtn[m])
    if "vix_dec" in methods:
        df["vix_dec"] = df["date"].map(get_vix(username, start, end, save=False)["vix_dec"])
    if "garch_vol" in methods:
        df["garch_vol"] = garch_vol_series(df["date"], df["days"], log_returns)
    if "implied" in methods:
        df["implied"] = df["impl_volatility"]

    is_call = df["cp_flag"] == "C"
    for m in methods:
        if m != "heston":
            call_price = black_scholes_call(df["S"], df["strike"], df[m], df["r"], df["T"], df["q"])
            put_price = black_scholes_put(df["S"], df["strike"], df[m], df["r"], df["T"], df["q"])
            df[f"price_{m}"] = np.where(is_call, call_price, put_price)

    # Heston: calibrate to the trading day before each pricing date, then price the whole chain
    if "heston" in methods:
        prev_dates = sorted({prev_day[d] for d in df["date"].unique()})
        params = cached(f"source/data/cache_heston_params_{secid}_{start}_{end}.pkl",
                        lambda: calibrate_heston_by_date(
                            multi_option_multi_day(username, prev_dates, secid=secid, save=False, calc_q=True)))
        params_by_date = {d: params[prev_day[d]] for d in df["date"].unique() if prev_day[d] in params}
        df["price_heston"] = heston_price_chain(df, params_by_date)

    n = len(df)
    df = df.dropna(subset=[f"price_{m}" for m in methods])
    if len(df) < n:
        print(f"{n - len(df)} of {n} options dropped (a method had no price for them)")
    return df


def add_errors(df, methods, show_all=False):
    """Error columns vs the market mid for each method: squared (for RMSE), and if show_all,
    absolute (for median absolute error) and absolute % (for MAPE; mid > 0 since best_bid > 0)."""
    for m in methods:
        error = df[f"price_{m}"] - df["mid"]
        df[f"sq_err_{m}"] = error ** 2
        if show_all:
            df[f"abs_err_{m}"] = error.abs()
            df[f"abs_pct_err_{m}"] = error.abs() / df["mid"]
    return df


def error_tables(df, bucket_col, methods, show_all=False):
    """RMSE ($), and if show_all median absolute error ($) and MAPE (%), for each method in each
    bucket of `bucket_col`. Needs add_errors first. Returns (rmse, mdae, mape); the last two are None if not show_all."""
    groups = df.groupby(bucket_col, observed=True)
    rmse = pd.DataFrame({m: np.sqrt(groups[f"sq_err_{m}"].mean()) for m in methods})
    if not show_all:
        return rmse, None, None
    mdae = pd.DataFrame({m: groups[f"abs_err_{m}"].median() for m in methods})
    mape = pd.DataFrame({m: groups[f"abs_pct_err_{m}"].mean() * 100 for m in methods})
    return rmse, mdae, mape


def overall_errors(df, methods, show_all=False):
    """One number per method over all options: {"rmse": {method: value}, and if show_all "mdae", "mape"}."""
    out = {"rmse": {}}
    if show_all:
        out["mdae"], out["mape"] = {}, {}
    for m in methods:
        error = df[f"price_{m}"] - df["mid"]
        out["rmse"][m] = np.sqrt(np.mean(error ** 2))                        # root mean squared error, $
        if show_all:
            out["mdae"][m] = np.median(np.abs(error))                         # median absolute error, $
            out["mape"][m] = np.mean(np.abs(error) / df["mid"]) * 100         # MAPE, %
    return out


def plot_errors(rmse, mdae, mape, methods, xlabel, title, save_as=None):
    """One figure of error vs bucket, one line per method: an RMSE panel, plus median absolute
    error and MAPE panels if they were calculated. save_as: file name in figures/, or None to not save."""
    panels = [(rmse, "RMSE ($ per unit)")]
    if mdae is not None:
        panels += [(mdae, "Median absolute error ($ per unit)"),
                   (mape, "Mean absolute percentage error (%)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(19, 5) if len(panels) > 1 else (8, 5), squeeze=False)
    axes = axes[0]

    labels = [f"{b.left:g}+" if b.right == np.inf else f"{b.left:g}–{b.right:g}" for b in rmse.index]
    for ax, (values, ylabel) in zip(axes, panels):
        for m in methods:
            ax.plot(labels, values[m], "o-", lw=2, color=COLOURS[m], label=LABELS[m])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False)

    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    if save_as:
        fig.savefig(f"figures/{save_as}.png", dpi=300, bbox_inches="tight")
    plt.show()
