import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("source")  # tells python to look in source

from classical_models.black_scholes.bs_funcs import black_scholes_call, black_scholes_put, historical_volatility_series
from data_importing.data_import import multi_option_single_day, single_option_multi_day, return_multi_day, get_vix

# WRDS username, and the day the option is chosen on (it is followed from here to expiry)
username = "kairogers"
start = "2025-06-23"

# ---- Settings: switch these on/off ----
USE_DIVIDENDS = False   # False sets the dividend yield q to 0 in the Black–Scholes prices
SAVE_FIG = False       # True saves the figure(s) to figures/


def pick(prompt, n):
    """Ask for a row number between 0 and n-1 until a valid one is given."""
    while True:
        choice = input(prompt).strip()
        if choice.isdigit() and int(choice) < n:
            return int(choice)
        print(f"Enter a number from 0 to {n - 1}")


# ---- Choose the option ----
secid = int(input("SECID (e.g. 108105 = SPX, 110007 = XLB, 110014 = XLK): "))
chain = multi_option_single_day(username, start, secid=secid, save=False)
S0 = chain["S"].iloc[0]
print(f"\n{len(chain)} options quoted on {start}, spot = {S0:g}")

# step 1: expiry
expiries = [pd.Timestamp(e) for e in sorted(chain["exdate"].unique())]
counts = chain["exdate"].value_counts()
print("\n  #  expiry       options")
for i, e in enumerate(expiries):
    print(f"{i:3d}  {e.date()}   {counts[e]}")
exdate = expiries[pick("Choose an expiry #: ", len(expiries))]

# step 2: strike and call/put (strikes within 20% of spot, unless there are none)
options = chain[chain["exdate"] == exdate]
near = options[options["strike"].between(0.8 * S0, 1.2 * S0)]
options = (near if not near.empty else options).sort_values(["cp_flag", "strike"]).reset_index(drop=True)
print("\n" + options[["optionid", "cp_flag", "strike", "mid", "impl_volatility", "volume"]].to_string())
choice = options.loc[pick("Choose an option #: ", len(options))]

optionid = int(choice["optionid"])
end = exdate.strftime("%Y-%m-%d")
print(f"\nFollowing option {optionid} from {start} to expiry {end}")

# ---- Load ----
opt = single_option_multi_day(username, optionid, start, end, secid=secid, save=False, calc_q=USE_DIVIDENDS)
rtn = return_multi_day(username, start, end, secid=secid, save=False).reset_index()   # date: index -> column

# historical vol (lookback rows give the 60-day warm-up)
rtn["hist_vol_zm"] = historical_volatility_series(rtn["log_return"], window=60, annualization=252, method="zero_mean")
rtn["hist_vol_dm"] = historical_volatility_series(rtn["log_return"], window=60, annualization=252, method="demeaned")
vix = get_vix(username, start, end, save=False).reset_index()
opt = opt.merge(rtn[["date", "hist_vol_zm", "hist_vol_dm"]], on="date", how="left")
opt = opt.merge(vix[["date", "vix_dec"]], on="date", how="left")

opt = opt[(opt["T"] > 0) & opt["impl_volatility"].notna()].sort_values("days", ascending=False)

# calc price for implied and historical vol, black-scholes
K = opt["strike"].iloc[0]
q = opt["q"] if USE_DIVIDENDS else 0
cp = opt["cp_flag"].iloc[0]
price_fn = black_scholes_call if cp == "C" else black_scholes_put
opt["bs_price_implied"] = price_fn(opt["S"], K, opt["impl_volatility"], opt["r"], opt["T"], q)
opt["bs_price_historical_zm"] = price_fn(opt["S"], K, opt["hist_vol_zm"], opt["r"], opt["T"], q)
opt["bs_price_historical_dm"] = price_fn(opt["S"], K, opt["hist_vol_dm"], opt["r"], opt["T"], q)
opt["bs_price_VIX"] = price_fn(opt["S"], K, opt["vix_dec"], opt["r"], opt["T"], q)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(9, 5))

vix_label = "Black–Scholes VIX" if secid == 108105 else "Black–Scholes VIX (S&P 500 vol, only a rough guide here)"
ax.plot(opt["days"], opt["mid"], "o", ms=4, color="blue", alpha=0.8, label="Market mid")
ax.plot(opt["days"], opt["bs_price_implied"], "-", lw=2, color="pink", label="Black–Scholes implied vol")
ax.plot(opt["days"], opt["bs_price_historical_zm"], "-", lw=2, color="orange", label="Black–Scholes zero-meaned historical vol")
ax.plot(opt["days"], opt["bs_price_historical_dm"], "-", lw=2, color="red", label="Black–Scholes demeaned historical vol")
ax.plot(opt["days"], opt["bs_price_VIX"], "-", lw=2, color="green", label=vix_label)
ax.invert_xaxis()                                   # time runs left to right towards expiry
ax.set_xlabel("Days to expiry")
ax.set_ylabel("Option price ($)")
ax.set_title(f"secid {secid} {'call' if cp == 'C' else 'put'}, K = {K:g}, expiry {end}", loc="left", fontweight="bold")

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False)

fig.tight_layout()
if SAVE_FIG:
    fig.savefig(f"figures/optionp_wrt_t_{secid}_{optionid}.png", dpi=300, bbox_inches="tight")
plt.show()

