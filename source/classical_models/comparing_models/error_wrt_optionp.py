#%%
import sys
import numpy as np
import pandas as pd

sys.path.append("source")  # tells python to look in source (run from the repo root)

from classical_models.comparing_models.compare_funcs import LABELS, chosen, load_weekly_prices, add_errors, error_tables, plot_errors

# WRDS username, security and period
username = "kairogers"
secid = 108105   # SPX (VIX is SPX vol, so SPX is the fair test for the VIX method)
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
    "vix_dec":     True,   # Black–Scholes, VIX
    "heston":      True,   # Heston, calibrated to the previous trading day's option chain
}
methods = chosen(METHODS)

#%%
# ---- Option chains on every Wednesday (Tuesday if Wednesday is a holiday), priced by each method ----
# Saved to source/data the first time (shared with the other comparison scripts); delete the files to redo.
df = load_weekly_prices(username, secid, start, end, methods)
df = add_errors(df, methods, SHOW_ALL_ERRORS)

#%%
# ---- Group the options into market price buckets (mid price, $ per unit) ----
bins = [0, 1, 5, 10, 25, 50, 100, 250, 500, 1000, np.inf]
df["price_bin"] = pd.cut(df["mid"], bins)
rmse, mdae, mape = error_tables(df, "price_bin", methods, SHOW_ALL_ERRORS)
print("RMSE ($)\n", rmse.rename(columns=LABELS).to_string())
if SHOW_ALL_ERRORS:
    print("Median absolute error ($)\n", mdae.rename(columns=LABELS).to_string())
    print("MAPE (%)\n", mape.rename(columns=LABELS).to_string())
print(df.groupby("price_bin", observed=True).size().rename("n options"))

#%%
# ---- Plot ----
plot_errors(rmse, mdae, mape, methods, "Market option price ($ per unit)",
            f"SPX options, pricing error by market option price, {start} to {end} (weekly)",
            save_as="error_wrt_optionp" if SAVE_FIG else None)

# %%
