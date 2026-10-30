#%%
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("source")  # tells python to look in source

from pricing import black_scholes_call, black_scholes_put
from vol_methods import historical_volatility_series
from data_importing.single_option_multi_day import single_option_multi_day
from data_importing.return_multi_day import return_multi_day

# choose option and dates
optionid = 155036252
start, end = "2023-06-12", "2023-12-15"

# ---- Load ----
opt = single_option_multi_day(optionid, start, end, save=False)
rtn = return_multi_day(start, end, save=False).reset_index()   # date: index -> column

# historical vol (lookback rows give the 60-day warm-up)
rtn["hist_vol_zm"] = historical_volatility_series(rtn["log_return"], window=60, annualization=252, method="zero_mean")
rtn["hist_vol_dm"] = historical_volatility_series(rtn["log_return"], window=60, annualization=252, method="demeaned")
opt = opt.merge(rtn[["date", "hist_vol_zm"]], on="date", how="left")
opt = opt.merge(rtn[["date", "hist_vol_dm"]], on="date", how="left")

opt = opt[(opt["T"] > 0) & opt["impl_volatility"].notna()].sort_values("days", ascending=False)

# calc price for implied and historical vol, black-scholes
K = opt["strike"].iloc[0]
cp = opt["cp_flag"].iloc[0]
price_fn = black_scholes_call if cp == "C" else black_scholes_put
opt["bs_price_implied"] = price_fn(opt["S"], K, opt["impl_volatility"], opt["r"], opt["T"], opt["q"])
opt["bs_price_historical_zm"] = price_fn(opt["S"], K, opt["hist_vol_zm"], opt["r"], opt["T"], opt["q"])
opt["bs_price_historical_dm"] = price_fn(opt["S"], K, opt["hist_vol_dm"], opt["r"], opt["T"], opt["q"])

# ---- Plot ----
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(opt["days"], opt["mid"], "o", ms=4, color="blue", alpha=0.8, label="Market mid")
ax.plot(opt["days"], opt["bs_price_implied"], "-", lw=2, color="green", label="Black–Scholes implied vol")
ax.plot(opt["days"], opt["bs_price_historical_zm"], "-", lw=2, color="orange", label="Black–Scholes zero-meaned historical vol")
ax.plot(opt["days"], opt["bs_price_historical_dm"], "-", lw=2, color="red", label="Black–Scholes demeaned historical vol")
ax.invert_xaxis()                                   # time runs left to right towards expiry
ax.set_xlabel("Days to expiry")
ax.set_ylabel("Option price ($)")
ax.set_title(f"SPX {'call' if cp == 'C' else 'put'}, K = {K:g}", loc="left", fontweight="bold")

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False)

fig.tight_layout()
plt.show()

# %%
