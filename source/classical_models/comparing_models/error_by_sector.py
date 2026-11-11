#%%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("source")  # tells python to look in source (run from the repo root)

from classical_models.comparing_models.compare_funcs import LABELS, COLOURS, chosen, load_weekly_prices, overall_errors

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
start, end = "2023-01-01", "2023-12-31"

# ---- Settings: switch these on/off ----
SHOW_ALL_ERRORS = True   # True also does median absolute error and MAPE (RMSE is always done)
SAVE_FIG = False         # True saves the figure(s) to figures/

# ---- Methods to compare: True = calculate and plot, False = skip ----
# (switching off GARCH or Heston also skips their slow fitting)
METHODS = {
    "hist_vol_zm": True,   # Black–Scholes, zero-mean 60-day historical vol
    "hist_vol_dm": False,   # Black–Scholes, demeaned 60-day historical vol
    "garch_vol":   True,   # Black–Scholes, GJR-GARCH vol forecast over the option's life
    "vix_dec":     True,   # Black–Scholes, VIX (S&P 500 vol, so only a rough market-wide vol for the ETFs)
    "heston":      True,   # Heston, calibrated to the previous trading day's option chain
}
methods = chosen(METHODS)

# error metric -> (y-axis label, file name for saving)
metrics = {"rmse": ("RMSE ($ per unit)", "rmse_by_sector")}
if SHOW_ALL_ERRORS:
    metrics["mdae"] = ("Median absolute error ($ per unit)", "mdae_by_sector")
    metrics["mape"] = ("Mean absolute percentage error (%)", "mape_by_sector")

#%%
# ---- Errors for each security ----
# Option chains on every Wednesday (Tuesday if Wednesday is a holiday), priced by each method.
# Saved to source/data the first time; delete the files to redo.
results = {m: {} for m in metrics}    # metric -> ticker -> {method: value}
for secid, (ticker, name) in SECURITIES.items():
    print(f"\n==== {ticker} ({name}) ====")
    try:
        df = load_weekly_prices(username, secid, start, end, methods)
    except Exception as e:
        print(f"Warning: {ticker}: no data ({e})")
        continue
    for m, values in overall_errors(df, methods, SHOW_ALL_ERRORS).items():
        results[m][ticker] = values

tables = {m: pd.DataFrame(results[m]).T for m in metrics}    # rows = tickers, columns = methods
for m, (ylabel, _) in metrics.items():
    print(f"{ylabel}\n", tables[m].rename(columns=LABELS).to_string(), "\n")

#%%
# ---- Plot: one figure per error metric, one group of bars per security, one bar per method ----
for m, (ylabel, fname) in metrics.items():
    table = tables[m]
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(table))           # one position per security
    width = 0.8 / len(methods)          # bars in a group share 0.8 of the space

    for i, col in enumerate(methods):
        ax.bar(x + i * width - 0.4 + width / 2, table[col], width, color=COLOURS[col], label=LABELS[col])

    ax.set_xticks(x)
    ax.set_xticklabels(table.index)
    ax.set_xlabel("Security")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Pricing error by sector ETF and S&P 500, {start} to {end} (weekly)",
                 loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    if SAVE_FIG:
        fig.savefig(f"figures/{fname}.png", dpi=300, bbox_inches="tight")
    plt.show()

# %%
