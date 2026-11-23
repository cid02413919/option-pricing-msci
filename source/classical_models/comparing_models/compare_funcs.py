"""Shared code for the scripts that compare pricing methods against the market:
Black–Scholes with different volatilities, and Heston.

How the comparison scripts use this file:
  1. Each script has a METHODS dict of True/False switches; chosen(METHODS) gives the methods switched on.
  2. load_weekly_prices(...)  downloads the weekly option chains and prices every option with every method.
     How dividends are handled depends on the exercise style (see exercise_style):
       european : the dividend yield q is used and every option is kept
       american : q = 0, and only options with no ex-dividend date during their life are kept
                  (the models are European, so they are only right for American options when
                   there is no dividend, since then early exercise of a call is never worth it)
  3. add_errors / error_tables / overall_errors  measure each method's error against the market mid.
  4. plot_errors(...)  plots the error against some bucket (maturity, moneyness, ...).

Sections:
  1. Settings shared by every script (securities, labels, colours)
  2. Small helpers
  3. Pricing every option with every method
  4. Errors against the market
  5. Plotting
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classical_models.black_scholes.bs_funcs import black_scholes_call, black_scholes_put, historical_volatility_series, garch_vol_series
from classical_models.Heston.heston import calibrate_heston_by_date, heston_price_chain
from data_importing.data_import import multi_option_multi_day, return_multi_day, get_vix, wednesdays, ex_dividend_dates


# =============================================================================
# 1. Settings shared by every script
# =============================================================================

# secid -> (ticker, name, exercise style of its options).
# The XL... are sector ETFs (American options), SPX is an index (European options)
SECURITIES = {
    108105: ("SPX", "S&P 500 Index", "european"),
    110007: ("XLB", "Materials", "american"),
    213084: ("XLC", "Communication Services", "american"),
    110011: ("XLE", "Energy", "american"),
    110012: ("XLF", "Financials", "american"),
    110013: ("XLI", "Industrials", "american"),
    110014: ("XLK", "Technology", "american"),
    110009: ("XLP", "Consumer Staples", "american"),
    208181: ("XLRE", "Real Estate", "american"),
    110015: ("XLU", "Utilities", "american"),
    110008: ("XLV", "Health Care", "american"),
    110010: ("XLY", "Consumer Discretionary", "american"),
}
SPX = 108105   # the VIX is S&P 500 vol, so it is only a fair test of the VIX method on SPX

# legend label and plot colour for every method
LABELS = {
    "implied":     "Black–Scholes implied vol",
    "hist_vol_zm": "Black–Scholes zero-mean historical vol",
    "hist_vol_dm": "Black–Scholes demeaned historical vol",
    "garch_vol":   "Black–Scholes GJR-GARCH vol",
    "vix_dec":     "Black–Scholes VIX",
    "lagged_iv":   "Black–Scholes lagged IV (previous day's implied vol)",
    "heston":      "Heston (calibrated on previous day)",
}
COLOURS = {"implied": "pink", "hist_vol_zm": "orange", "hist_vol_dm": "red",
           "garch_vol": "purple", "vix_dec": "green", "lagged_iv": "teal", "heston": "brown"}

BLACK_SCHOLES_METHODS = ["implied", "hist_vol_zm", "hist_vol_dm", "garch_vol", "vix_dec", "lagged_iv"]


# =============================================================================
# 2. Small helpers
# =============================================================================

def chosen(methods):
    """The methods switched on (True) in a script's METHODS dict."""
    return [m for m, on in methods.items() if on]


def ask_secid():
    """Print the securities in SECURITIES and ask for a secid (any other OptionMetrics secid works too)."""
    print("\n  secid   ticker  exercise  name")
    for secid, (ticker, name, exercise) in SECURITIES.items():
        print(f"  {secid:<7d} {ticker:<7s} {exercise:<9s} {name}")
    while True:
        choice = input("SECID: ").strip()
        if choice.isdigit():
            return int(choice)
        print("Enter a number, e.g. 108105 for SPX")


def exercise_style(secid):
    """'american' or 'european' for a secid: from SECURITIES, or asked for if it is not in there."""
    if secid in SECURITIES:
        return SECURITIES[secid][2]
    while True:
        choice = input(f"Are secid {secid}'s options American or European? (a/e): ").strip().lower()
        if choice in ("a", "e"):
            return "american" if choice == "a" else "european"
        print("Enter a or e")


def ticker(secid):
    """Ticker for a secid, or 'secid ...' if it is not in SECURITIES."""
    return SECURITIES[secid][0] if secid in SECURITIES else f"secid {secid}"


def cached(cache_file, download):
    """Load a saved download if there is one, otherwise download and save it. Delete the file to refresh."""
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file)
    data = download()
    pd.to_pickle(data, cache_file)
    return data


# =============================================================================
# 3. Pricing every option with every method
# =============================================================================
#
# Every method only uses information from before the pricing day:
#   historical vols : rolling 60-day vol, shifted one day
#   GARCH           : re-fit on the 252 returns before the day, forecast over the option's life
#   VIX             : get_vix already gives the previous day's value
#   lagged IV          : the same option's implied vol on the previous trading day
#   Heston          : calibrated to the previous trading day's whole option chain
#   implied         : the same day's implied vol (so it is not a forecast; it sits on the market price)

def load_weekly_prices(username, secid, start, end, methods, exercise="european", daily=False):
    """Option chains on every Wednesday (Tuesday if Wednesday is a holiday) from start to end,
    or on every trading day if daily=True, with a model price column 'price_{method}' for each method in `methods`.

    exercise : 'european' -> use the dividend yield q, keep every option
               'american' -> q = 0, and only keep options with no ex-dividend date during their
                             life, both for pricing and for the Heston calibration

    Only options that every method could price are kept, so all methods are compared on the same options.
    Downloads (and the Heston calibrations) are saved in source/data; delete the files to redo them.
    """
    if exercise not in ("european", "american"):
        raise ValueError(f"exercise must be 'european' or 'american', not {exercise!r}")
    american = exercise == "american"
    tag = "_daily" if daily else ""     # daily downloads are saved separately from weekly ones

    # Step 1: daily returns (with a year before start for GARCH) and the past-only historical vols
    rtn = _load_returns(username, secid, start, end)

    # Step 2: the option chains on the pricing dates (American: q = 0 and dividend-free options only)
    ex_dates = load_ex_dividend_dates(username, secid, start, end) if american else None
    df = _load_weekly_chains(username, secid, start, end, rtn.index, use_dividends=not american, daily=daily)
    if american:
        df = drop_options_over_dividends(df, ex_dates)

    # Step 3: each Black–Scholes method's vol, then its price
    df = _add_black_scholes_vols(df, rtn, username, secid, start, end, methods, tag)
    df = _add_black_scholes_prices(df, methods)

    # Step 4: Heston, calibrated to the previous trading day
    if "heston" in methods:
        df["price_heston"] = _heston_prices(df, rtn, username, secid, start, end, ex_dates, tag)

    # Step 5: keep only options every method could price
    n = len(df)
    df = df.dropna(subset=[f"price_{m}" for m in methods])
    if len(df) < n:
        print(f"{n - len(df)} of {n} options dropped (a method had no price for them)")
    return df


def _load_returns(username, secid, start, end):
    """Daily log returns from a year before start to end, with the 60-day historical vols
    (shifted one day, so each day only uses earlier returns). Lookback rows are kept: GARCH
    fits on them and Heston needs the trading day before the first pricing date."""
    rtn = return_multi_day(username, start, end, lookback=252, secid=secid, save=False)
    rtn["hist_vol_zm"] = historical_volatility_series(rtn["log_return"], window=60, method="zero_mean").shift(1)
    rtn["hist_vol_dm"] = historical_volatility_series(rtn["log_return"], window=60, method="demeaned").shift(1)
    return rtn


def _load_weekly_chains(username, secid, start, end, trading_days, use_dividends, daily=False):
    """Every option quoted on each Wednesday between start and end, or on every trading day if daily.
    (Every trading day is ~19M quotes / ~7 GB a year for SPX, so keep daily periods short.)"""
    if daily:
        dates = trading_days[(trading_days >= start) & (trading_days <= end)]
    else:
        dates = wednesdays(trading_days, start, end)
    name = "daily" if daily else "weekly"
    df = cached(f"source/data/cache_{name}_chains_{secid}_{start}_{end}.pkl",
                lambda: multi_option_multi_day(username, dates, secid=secid, save=False, calc_q=True))
    df = df[df["T"] > 0].copy()          # remove options that expire on the day itself
    if not use_dividends:
        df["q"] = 0.0
    return df


def load_ex_dividend_dates(username, secid, start, end):
    """Ex-dividend dates from start until 3 years after end (options can live that long)."""
    until = (pd.Timestamp(end) + pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    ex_dates = cached(f"source/data/cache_ex_div_dates_{secid}_{start}_{end}.pkl",
                      lambda: ex_dividend_dates(username, secid, start, until))
    if len(ex_dates) == 0:
        print(f"Warning: no ex-dividend dates found for secid {secid}, so no options are dropped")
    return ex_dates


def drop_options_over_dividends(df, ex_dates):
    """Keep only options with no ex-dividend date in (date, exdate], i.e. no dividend is paid
    between the pricing day and expiry, so the dividend-free models (q = 0) are correct for them.
    (An ex-date on the pricing day itself is fine: the price is already ex-dividend.)
    df needs 'date' and 'exdate' columns."""
    ex = np.asarray(ex_dates, dtype="datetime64[ns]")
    # number of ex-dates <= date vs <= exdate: equal means none fall in between
    before_start = np.searchsorted(ex, df["date"].to_numpy("datetime64[ns]"), side="right")
    before_end = np.searchsorted(ex, df["exdate"].to_numpy("datetime64[ns]"), side="right")
    keep = before_start == before_end
    print(f"Dividend-free filter: kept {keep.sum()} of {len(df)} options")
    return df[keep].copy()


def _add_black_scholes_vols(df, rtn, username, secid, start, end, methods, tag=""):
    """A volatility column for every Black–Scholes method in `methods`."""
    if "implied" in methods:
        df["implied"] = df["impl_volatility"]
    for m in ("hist_vol_zm", "hist_vol_dm"):
        if m in methods:
            df[m] = df["date"].map(rtn[m])
    if "vix_dec" in methods:
        df["vix_dec"] = df["date"].map(get_vix(username, start, end, save=False)["vix_dec"])
    if "garch_vol" in methods:
        df["garch_vol"] = garch_vol_series(df["date"], df["days"], rtn["log_return"])
    if "lagged_iv" in methods:
        df["lagged_iv"] = _previous_day_implied_vols(df, rtn, username, secid, start, end, tag)
    return df


def _previous_trading_day(rtn):
    """trading day -> trading day before it"""
    return dict(zip(rtn.index[1:], rtn.index[:-1]))


def _load_previous_day_chains(df, rtn, username, secid, start, end, tag=""):
    """The whole option chain on the trading day before each pricing date in df (used by lagged IV and Heston)."""
    prev_day = _previous_trading_day(rtn)
    prev_dates = sorted({prev_day[d] for d in df["date"].unique()})
    return cached(f"source/data/cache_prev_day_chains_{secid}_{start}_{end}{tag}.pkl",
                  lambda: multi_option_multi_day(username, prev_dates, secid=secid, save=False, calc_q=True))


def _previous_day_implied_vols(df, rtn, username, secid, start, end, tag=""):
    """Each option's implied vol on the previous trading day (NaN if it was not quoted then, e.g. just listed)."""
    prev = _load_previous_day_chains(df, rtn, username, secid, start, end, tag)
    prev_vol = prev.set_index(["date", "optionid"])["impl_volatility"]
    prev_day = _previous_trading_day(rtn)
    wanted = pd.MultiIndex.from_arrays([df["date"].map(prev_day), df["optionid"]])
    return prev_vol.reindex(wanted).to_numpy()


def _add_black_scholes_prices(df, methods):
    """A 'price_{method}' column for every Black–Scholes method: the call or put price with that vol."""
    is_call = df["cp_flag"] == "C"
    for m in methods:
        if m in BLACK_SCHOLES_METHODS:
            call = black_scholes_call(df["S"], df["strike"], df[m], df["r"], df["T"], df["q"])
            put = black_scholes_put(df["S"], df["strike"], df[m], df["r"], df["T"], df["q"])
            df[f"price_{m}"] = np.where(is_call, call, put)
    return df


def _heston_prices(df, rtn, username, secid, start, end, ex_dates=None, tag=""):
    """Heston price for every option in df: calibrate to the whole option chain on the trading
    day before each pricing date, then price that date's chain with those parameters.
    ex_dates None  -> European: calibrate with the dividend yield q on every option
    ex_dates given -> American: calibrate with q = 0 on the options with no dividend during their life"""
    prev_day = _previous_trading_day(rtn)
    pricing_dates = df["date"].unique()

    def calibrate():
        chains = _load_previous_day_chains(df, rtn, username, secid, start, end, tag)
        if ex_dates is not None:
            chains = chains.assign(q=0.0)
            chains = drop_options_over_dividends(chains, ex_dates)
        return calibrate_heston_by_date(chains)

    # the parameters depend on q and on which options are fitted, so the two styles are saved separately
    suffix = tag + ("" if ex_dates is None else "_american")
    params = cached(f"source/data/cache_heston_params_{secid}_{start}_{end}{suffix}.pkl", calibrate)

    params_by_date = {d: params[prev_day[d]] for d in pricing_dates if prev_day[d] in params}
    return heston_price_chain(df, params_by_date)


# =============================================================================
# 4. Errors against the market mid
# =============================================================================

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


# =============================================================================
# 5. Plotting
# =============================================================================

def plot_errors(rmse, mdae, mape, methods, xlabel, title, save_as=None):
    """One figure of error vs bucket, one line per method: an RMSE panel, plus median absolute
    error and MAPE panels if they were calculated. save_as: file name in figures/, or None to not save."""
    panels = [(rmse, "RMSE ($ per unit)")]
    if mdae is not None:
        panels += [(mdae, "Median absolute error ($ per unit)"),
                   (mape, "Mean absolute percentage error (%)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(19, 5) if len(panels) > 1 else (8, 5), squeeze=False)
    axes = axes[0]

    # bucket labels like "30–60", and "730+" for an open-ended last bucket
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
