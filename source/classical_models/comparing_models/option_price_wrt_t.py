#%%

import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("source")  # tells python to look in source (run from the repo root)

from classical_models.black_scholes.bs_funcs import black_scholes_call, black_scholes_put, historical_volatility_series, garch_vol_series
from classical_models.Heston.heston import calibrate_heston_by_date, heston_price_chain
from classical_models.comparing_models.compare_funcs import (
    LABELS, COLOURS, SPX, chosen, ask_secid, exercise_style, ticker, load_ex_dividend_dates, drop_options_over_dividends)
from data_importing.data_import import multi_option_single_day, multi_option_multi_day, single_option_multi_day, return_multi_day, get_vix

# WRDS username, and the day the option is chosen on (it is followed from here to expiry)
username = "kairogers"
start = "2024-06-21"

# ---- Settings: switch these on/off ----
SAVE_FIG = True        # True saves the figure(s) to figures/

# Dividends depend on the exercise style of the chosen security (from compare_funcs.SECURITIES):
#   american : q = 0, and only options with no ex-dividend date before expiry can be chosen
#   european : the dividend yield q is used, and any option can be chosen

# ---- Methods to plot: True = calculate and plot, False = skip ----
# (switching off GARCH or Heston also skips their slow fitting / extra download)
METHODS = {
    "implied":     True,   # Black–Scholes, the market's implied vol (same-day, so it sits on the mid)
    "hist_vol_zm": False,   # Black–Scholes, zero-mean 60-day historical vol
    "hist_vol_dm": False,   # Black–Scholes, demeaned 60-day historical vol
    "garch_vol":   False,   # Black–Scholes, GJR-GARCH vol forecast over the option's remaining life
    "vix_dec":     False,   # Black–Scholes, VIX (S&P 500 vol, only a rough guide for other securities)
    "ad_hoc":      True,   # Black–Scholes, the same option's implied vol on the previous trading day
    "heston":      True,   # Heston, calibrated to the previous trading day's option chain
}
methods = chosen(METHODS)


def pick(prompt, n):
    """Ask for a row number between 0 and n-1 until a valid one is given."""
    while True:
        choice = input(prompt).strip()
        if choice.isdigit() and int(choice) < n:
            return int(choice)
        print(f"Enter a number from 0 to {n - 1}")


# ---- Choose the option ----
secid = ask_secid()
name = ticker(secid)
exercise = exercise_style(secid)
american = exercise == "american"
chain = multi_option_single_day(username, start, secid=secid, save=False)
S0 = chain["S"].iloc[0]
print(f"\n{len(chain)} {exercise} options quoted on {start}, spot = {S0:g}")
if american:
    # only options with no dividend between start and expiry (so none on any later day either)
    ex_dates = load_ex_dividend_dates(username, secid, start, start)
    chain = drop_options_over_dividends(chain, ex_dates)
    if chain.empty:
        raise ValueError(f"Every option on {start} has an ex-dividend date before expiry; try another start")

# step 1: expiry
expiries = [pd.Timestamp(e) for e in sorted(chain["exdate"].unique())]
counts = chain["exdate"].value_counts()
print("\n  #  expiry       options")
for i, e in enumerate(expiries):
    print(f"{i:3d}  {e.date()}   {counts[e]}")
exdate = expiries[pick("Choose an expiry # (option data is cut off after 2025-08-29): ", len(expiries))]

# step 2: strike and call/put (strikes within 20% of spot, unless there are none)
options = chain[chain["exdate"] == exdate]
near = options[options["strike"].between(0.8 * S0, 1.2 * S0)]
options = (near if not near.empty else options).sort_values(["cp_flag", "strike"]).reset_index(drop=True)
print("\n" + options[["optionid", "cp_flag", "strike", "mid", "impl_volatility", "volume"]].to_string())
choice = options.loc[pick("Choose an option #: ", len(options))]

optionid = int(choice["optionid"])
end = exdate.strftime("%Y-%m-%d")
print(f"\nFollowing option {optionid} from {start} to expiry {end}")

#%%
# ---- Load ----
week_before = (pd.Timestamp(start) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")   # so ad hoc has a previous day
opt = single_option_multi_day(username, optionid, week_before, end, secid=secid, save=False, calc_q=not american)
opt = opt[(opt["T"] > 0) & opt["impl_volatility"].notna()].sort_values("days", ascending=False)
opt["ad_hoc"] = opt["impl_volatility"].shift(1)    # previous trading day's implied vol on this option
opt = opt[opt["date"] >= start].reset_index(drop=True)
if american:
    opt["q"] = 0.0

# returns indexed by date, with a year before start: GARCH's fitting window, and Heston's first previous day
rtn = return_multi_day(username, start, end, lookback=252, secid=secid, save=False)

# ---- Each method's vol for every day (Black–Scholes) ----
opt["implied"] = opt["impl_volatility"]
for m, how in (("hist_vol_zm", "zero_mean"), ("hist_vol_dm", "demeaned")):
    if m in methods:
        opt[m] = opt["date"].map(historical_volatility_series(rtn["log_return"], window=60, method=how))
if "vix_dec" in methods:
    opt["vix_dec"] = opt["date"].map(get_vix(username, start, end, save=False)["vix_dec"])
if "garch_vol" in methods:
    # re-fit each day on the returns before it, forecast over the option's remaining life
    opt["garch_vol"] = garch_vol_series(opt["date"], opt["days"], rtn["log_return"])

K = opt["strike"].iloc[0]
cp = opt["cp_flag"].iloc[0]
price_fn = black_scholes_call if cp == "C" else black_scholes_put
for m in methods:
    if m != "heston":
        opt[f"price_{m}"] = price_fn(opt["S"], K, opt[m], opt["r"], opt["T"], opt["q"])

# ---- Heston: calibrate to the PREVIOUS trading day's whole option chain, then price this option today ----
# with those parameters and today's S, T, r (no look-ahead, like the historical and GARCH vols)
if "heston" in methods:
    prev_day = dict(zip(rtn.index[1:], rtn.index[:-1]))          # trading day -> trading day before it
    chains = multi_option_multi_day(username, sorted({prev_day[d] for d in opt["date"]}), secid=secid,
                                    save=False, calc_q=not american)
    if american:
        # same rule as the option itself: q = 0, fitted only to options with no dividend before expiry
        chains["q"] = 0.0
        chains = drop_options_over_dividends(chains, ex_dates)
    params = calibrate_heston_by_date(chains)
    opt["price_heston"] = heston_price_chain(opt, {d: params[prev_day[d]] for d in opt["date"] if prev_day[d] in params})

#%%
# ---- Plot ----
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(opt["days"], opt["mid"], "o", ms=4, color="blue", alpha=0.8, label="Market mid")
for m in methods:
    label = LABELS[m]
    if m == "vix_dec" and secid != SPX:
        label += " (S&P 500 vol, only a rough guide here)"
    ax.plot(opt["days"], opt[f"price_{m}"], "-", lw=2, color=COLOURS[m], label=label)
ax.invert_xaxis()                                   # time runs left to right towards expiry
ax.set_xlabel("Days to expiry")
ax.set_ylabel("Option price ($)")
div_note = " (no dividend before expiry, q = 0)" if american else ""
ax.set_title(f"{name} {exercise} {'call' if cp == 'C' else 'put'}, K = {K:g}, expiry {end}{div_note}",
             loc="left", fontweight="bold")

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False)

fig.tight_layout()
if SAVE_FIG:
    fig.savefig(f"figures/option_price_wrt_t_{secid}_{optionid}.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
