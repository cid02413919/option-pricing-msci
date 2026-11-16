#%%
import sys
import pandas as pd

sys.path.append("source")  # tells python to look in source (run from the repo root)

from classical_models.comparing_models.compare_funcs import (
    LABELS, SPX, chosen, ask_secid, exercise_style, ticker, load_weekly_prices, overall_errors)

# WRDS username and period
username = "kairogers"
start, end = "2023-01-01", "2023-12-31"

# ---- Settings: switch these on/off ----
SHOW_ALL_ERRORS = False   # True also does median absolute error and MAPE (RMSE is always done)

# ---- Methods to compare: True = calculate and print, False = skip ----
# (switching off GARCH or Heston also skips their slow fitting)
METHODS = {
    "hist_vol_zm": True,   # Black–Scholes, zero-mean 60-day historical vol
    "hist_vol_dm": False,   # Black–Scholes, demeaned 60-day historical vol
    "garch_vol":   True,   # Black–Scholes, GJR-GARCH vol forecast over the option's life
    "vix_dec":     True,   # Black–Scholes, VIX (S&P 500 vol, only a rough guide for other securities)
    "heston":      True,   # Heston, calibrated to the previous trading day's option chain
}
methods = chosen(METHODS)

# ---- Choose the underlying ----
secid = ask_secid()
name = ticker(secid)
exercise = exercise_style(secid)   # american: q = 0 and only options with no dividend before expiry
print(f"\n{name}, {start} to {end}, {exercise} options")
if "vix_dec" in methods and secid != SPX:
    print("Note: VIX is S&P 500 vol, so the VIX method is only a rough guide here")

#%%
# ---- Option chains on every Wednesday (Tuesday if Wednesday is a holiday), priced by each method ----
# Saved to source/data the first time (shared with the other comparison scripts); delete the files to redo.
df = load_weekly_prices(username, secid, start, end, methods, exercise=exercise)

#%%
# ---- Results ----
errors = overall_errors(df, methods, SHOW_ALL_ERRORS)
names = {"rmse": "RMSE ($)", "mdae": "Median abs error ($)", "mape": "MAPE (%)"}
results = pd.DataFrame({names[k]: v for k, v in errors.items()}).rename(index=LABELS)
div_note = ", no dividend before expiry, q = 0" if exercise == "american" else ""
print(f"Overall pricing error, {name}, {start} to {end} (weekly, Wednesdays{div_note})\n")
print(results.to_string(float_format="%.3f"))
print(f"\n{len(df)} option quotes on {df['date'].nunique()} days")

# %%
