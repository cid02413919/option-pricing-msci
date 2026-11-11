#%%
import sys
import pandas as pd

sys.path.append("source")  # tells python to look in source (run from the repo root)

from classical_models.comparing_models.compare_funcs import LABELS, chosen, load_weekly_prices, overall_errors

# WRDS username, security and period
username = "kairogers"
secid = 108105   # SPX (VIX is SPX vol, so SPX is the fair test for the VIX method)
start, end = "2019-01-01", "2023-12-31"

# ---- Settings: switch these on/off ----
SHOW_ALL_ERRORS = False   # True also does median absolute error and MAPE (RMSE is always done)

# ---- Methods to compare: True = calculate and print, False = skip ----
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

#%%
# ---- Results ----
errors = overall_errors(df, methods, SHOW_ALL_ERRORS)
names = {"rmse": "RMSE ($)", "mdae": "Median abs error ($)", "mape": "MAPE (%)"}
results = pd.DataFrame({names[k]: v for k, v in errors.items()}).rename(index=LABELS)
print(f"Overall pricing error, SPX, {start} to {end} (weekly, Wednesdays)\n")
print(results.to_string(float_format="%.3f"))
print(f"\n{len(df)} option quotes on {df['date'].nunique()} days")

# %%
