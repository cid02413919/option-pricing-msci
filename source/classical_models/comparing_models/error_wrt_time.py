#%%
import sys
import matplotlib.pyplot as plt

sys.path.append("source")  # tells python to look in source (run from the repo root)

from classical_models.comparing_models.compare_funcs import (
    LABELS, COLOURS, SPX, chosen, ask_secid, exercise_style, ticker, load_weekly_prices, add_errors, error_tables)

# WRDS username and period
username = "kairogers"
start, end = "2020-01-01", "2020-04-30"

# ---- Settings: switch these on/off ----
SAVE_FIG = False          # True saves the figure to figures/

# ---- Methods to compare: True = calculate and plot, False = skip ----
# (switching off GARCH or Heston also skips their slow fitting)
METHODS = {
    "hist_vol_zm": False,   # Black–Scholes, zero-mean 60-day historical vol
    "hist_vol_dm": False,   # Black–Scholes, demeaned 60-day historical vol
    "garch_vol":   False,   # Black–Scholes, GJR-GARCH vol forecast over the option's life
    "vix_dec":     False,    # Black–Scholes, VIX (S&P 500 vol, only a rough guide for other securities)
    "lagged_iv":   True,    # Black–Scholes, the same option's implied vol on the previous trading day
    "heston":      False,   # Heston, calibrated to the previous trading day's option chain
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
# ---- Option chains on every trading day, priced by each method (keep the period short: daily is a big download) ----
df = load_weekly_prices(username, secid, start, end, methods, exercise=exercise, daily=True)
df = add_errors(df, methods)

#%%
# ---- RMSE on each pricing date ----
rmse, _, _ = error_tables(df, "date", methods)
print("RMSE ($)\n", rmse.rename(columns=LABELS).to_string())

#%%
# ---- Plot ----
fig, ax = plt.subplots(figsize=(10, 5))
for m in methods:
    ax.plot(rmse.index, rmse[m], "o-", ms=3, lw=2, color=COLOURS[m], label=LABELS[m])
ax.set_xlabel("Date")
ax.set_ylabel("RMSE ($ per unit)")
div_note = ", no dividend before expiry, q = 0" if exercise == "american" else ""
ax.set_title(f"{name} options, pricing error over time, {start} to {end} (daily{div_note})",
             loc="left", fontweight="bold")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False)

fig.tight_layout()
if SAVE_FIG:
    fig.savefig(f"figures/error_wrt_time_{secid}_{exercise}.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
