#%%
import sys
import pandas as pd

sys.path.append("source")  # tells python to look in source (run from the repo root)

from classical_models.comparing_models.compare_funcs import chosen, load_weekly_prices, add_errors, error_tables, plot_errors

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

# WRDS username and period (same as the other comparison scripts, so the saved data files are shared)
username = "kairogers"
start, end = "2019-01-01", "2023-12-31"

# ---- Settings: switch these on/off ----
SHOW_ALL_ERRORS = False   # True also does median absolute error and MAPE (RMSE is always done)
SAVE_FIG = False          # True saves the figure(s) to figures/

# ---- Methods to compare: True = calculate and plot, False = skip ----
# (switching off GARCH or Heston also skips their slow fitting)
METHODS = {
    "hist_vol_zm": True,   # Black–Scholes, zero-mean 60-day historical vol
    "hist_vol_dm": True,   # Black–Scholes, demeaned 60-day historical vol
    "garch_vol":   True,   # Black–Scholes, GJR-GARCH vol forecast over the option's life
    "vix_dec":     True,   # Black–Scholes, VIX (S&P 500 vol, so only a rough market-wide vol for the ETFs)
    "heston":      True,   # Heston, calibrated to the previous trading day's option chain
}
methods = chosen(METHODS)

# buckets for the two sets of plots
maturity_bins = [0, 30, 60, 90, 180, 365, 730]                      # days to expiry
moneyness_bins = [0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.5]    # strike / spot

#%%
# ---- For each security: error plots by maturity, then by moneyness ----
# Option chains on every Wednesday (Tuesday if Wednesday is a holiday), priced by each method.
# Saved to source/data the first time; delete the files to redo.
for secid, (ticker, name) in SECURITIES.items():
    print(f"\n==== {ticker} ({name}) ====")
    try:
        df = load_weekly_prices(username, secid, start, end, methods)
    except Exception as e:
        print(f"Warning: {ticker}: no data ({e})")
        continue

    df = add_errors(df, methods, SHOW_ALL_ERRORS)
    df["maturity"] = pd.cut(df["days"], maturity_bins)
    df["moneyness_bin"] = pd.cut(df["strike"] / df["S"], moneyness_bins)
    print(f"{len(df)} options on {df['date'].nunique()} days")

    rmse, mdae, mape = error_tables(df, "maturity", methods, SHOW_ALL_ERRORS)
    plot_errors(rmse, mdae, mape, methods, "Days to expiry",
                f"{ticker} ({name}) options, pricing error by maturity, {start} to {end} (weekly)",
                save_as=f"error_{ticker}_wrt_maturity" if SAVE_FIG else None)

    rmse, mdae, mape = error_tables(df, "moneyness_bin", methods, SHOW_ALL_ERRORS)
    plot_errors(rmse, mdae, mape, methods, "Moneyness (K / S)",
                f"{ticker} ({name}) options, pricing error by moneyness, {start} to {end} (weekly)",
                save_as=f"error_{ticker}_wrt_moneyness" if SAVE_FIG else None)

# %%
