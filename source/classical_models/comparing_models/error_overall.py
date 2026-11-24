#%%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("source")  # run from the repo root

from classical_models.comparing_models.compare_funcs import (
    LABELS, SPX, chosen, ask_secid, exercise_style, ticker, load_weekly_prices, overall_errors)

username = "kairogers"
start, end = "2024-12-31", "2024-12-31"

SHOW_ALL_ERRORS = False   # also show median absolute error and MAPE
WEDNESDAYS_ONLY = True    # True: only Wednesdays (Tuesday if Wednesday is a holiday). False: every trading day

METHODS = {
    "hist_vol_zm": False,
    "hist_vol_dm": False,
    "garch_vol":   False,
    "vix_dec":     True,
    "lagged_iv":   False,
    "heston":      False,
}

methods = chosen(METHODS)

secid = ask_secid()
name = ticker(secid)
exercise = exercise_style(secid)
print(f"\n{name}, {start} to {end}, {exercise} options")
if "vix_dec" in methods and secid != SPX:
    print("Note: VIX is S&P 500 vol, so the VIX method is only a rough guide here")

#%%
df = load_weekly_prices(username, secid, start, end, methods, exercise=exercise, daily=not WEDNESDAYS_ONLY)

#%%
errors = overall_errors(df, methods, SHOW_ALL_ERRORS)
names = {"rmse": "RMSE ($)", "mdae": "Median abs error ($)", "mape": "MAPE (%)"}
results = pd.DataFrame({names[k]: v for k, v in errors.items()}).rename(index=LABELS)
div_note = ", no dividend before expiry, q = 0" if exercise == "american" else ""
days_note = "weekly, Wednesdays" if WEDNESDAYS_ONLY else "every trading day"
print(f"Overall pricing error, {name}, {start} to {end} ({days_note}{div_note})\n")
print(results.to_string(float_format="%.3f"))
print(f"\n{len(df)} option quotes on {df['date'].nunique()} days")

#%%
# Histogram of residuals (model price - market price) for each method
for m in methods:
    residuals = (df[f"price_{m}"] - df["mid"]).dropna()
    limit = residuals.abs().max()
    bins = np.linspace(-limit, limit, 51)   # 50 bins, symmetric about 0

    plt.figure()
    plt.hist(residuals, bins=bins)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Residual: model price - market price ($)")
    plt.ylabel("Frequency")
    plt.title(f"{LABELS[m]}, {name}")
    plt.show()

# %%
