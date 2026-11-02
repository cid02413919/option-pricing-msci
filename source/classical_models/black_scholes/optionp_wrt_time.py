# #%%
# import sys
# import pandas as pd
# import matplotlib.pyplot as plt

# sys.path.append("source")  # tells python to look in source
# sys.path.append("../..")

# from pricing import black_scholes_call, black_scholes_put
# from vol_methods import historical_volatility_series
# from data_importing.data_import import single_option_multi_day, return_multi_day, get_vix

# # WRDS username, option and dates
# username = "alexmatijasevic"
# optionid = 155036252
# start, end = "2023-06-12", "2023-12-15"

# # ---- Load ----
# opt = single_option_multi_day(username, optionid, start, end, save=False)
# rtn = return_multi_day(username, start, end, save=False).reset_index()   # date: index -> column

# # historical vol (lookback rows give the 60-day warm-up)
# rtn["hist_vol_zm"] = historical_volatility_series(rtn["log_return"], window=60, annualization=252, method="zero_mean")
# rtn["hist_vol_dm"] = historical_volatility_series(rtn["log_return"], window=60, annualization=252, method="demeaned")
# vix = get_vix(username, start, end, save=False).reset_index()   # date: index -> column
# opt = opt.merge(rtn[["date", "hist_vol_zm"]], on="date", how="left")
# opt = opt.merge(rtn[["date", "hist_vol_dm"]], on="date", how="left")
# opt = opt.merge(vix[["date", "vix_dec"]], on="date", how="left")

# opt = opt[(opt["T"] > 0) & opt["impl_volatility"].notna()].sort_values("days", ascending=False)

# # calc price for implied and historical vol, black-scholes
# K = opt["strike"].iloc[0]
# cp = opt["cp_flag"].iloc[0]
# price_fn = black_scholes_call if cp == "C" else black_scholes_put
# opt["bs_price_implied"] = price_fn(opt["S"], K, opt["impl_volatility"], opt["r"], opt["T"], opt["q"])
# opt["bs_price_historical_zm"] = price_fn(opt["S"], K, opt["hist_vol_zm"], opt["r"], opt["T"], opt["q"])
# opt["bs_price_historical_dm"] = price_fn(opt["S"], K, opt["hist_vol_dm"], opt["r"], opt["T"], opt["q"])
# opt["bs_price_VIX"] = price_fn(opt["S"], K, opt["vix_dec"], opt["r"], opt["T"], opt["q"])

# # ---- Plot ----
# fig, ax = plt.subplots(figsize=(9, 5))

# ax.plot(opt["days"], opt["mid"], "o", ms=4, color="blue", alpha=0.8, label="Market mid")
# #ax.plot(opt["days"], opt["bs_price_implied"], "-", lw=2, color="green", label="Black–Scholes implied vol")
# ax.plot(opt["days"], opt["bs_price_historical_zm"], "-", lw=2, color="orange", label="Black–Scholes zero-meaned historical vol")
# ax.plot(opt["days"], opt["bs_price_historical_dm"], "-", lw=2, color="red", label="Black–Scholes demeaned historical vol")
# ax.plot(opt["days"], opt["bs_price_VIX"], "-", lw=2, color="green", label="Black–Scholes VIX")
# ax.invert_xaxis()                                   # time runs left to right towards expiry
# ax.set_xlabel("Days to expiry")
# ax.set_ylabel("Option price ($)")
# ax.set_title(f"SPX {'call' if cp == 'C' else 'put'}, K = {K:g}", loc="left", fontweight="bold")

# for side in ("top", "right"):
#     ax.spines[side].set_visible(False)
# ax.legend(frameon=False)

# fig.tight_layout()
# plt.show()

#%%

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("source")  # tells python to look in source

from pricing import black_scholes_call, black_scholes_put
from vol_methods import historical_volatility_series, forecast_realized_vol_garch
from data_importing.data_import import single_option_multi_day, return_multi_day, get_vix

# WRDS username, option and dates
username = "alexmatijasevic"
optionid = 155036252
start, end = "2023-06-12", "2023-12-15"

# ---- Load ----
opt = single_option_multi_day(username, optionid, start, end, save=False).reset_index()
rtn = return_multi_day(username, start, end, save=False).reset_index()   # date: index -> column

# historical vol (lookback rows give the 60-day warm-up)
rtn["hist_vol_zm"] = historical_volatility_series(rtn["log_return"], window=60, annualization=252, method="zero_mean")
rtn["hist_vol_dm"] = historical_volatility_series(rtn["log_return"], window=60, annualization=252, method="demeaned")

# ---- GARCH forecast, rolling per day ----
# Unlike historical vol (a plain rolling std), GARCH needs to be RE-FIT on
# each day using only data up to and including that day (no look-ahead),
# then rolled FORWARD to forecast vol over that day's own remaining option
# life. Since "remaining life" (days to expiry) changes every day, the
# GARCH forecast horizon must be recomputed per row too -- it isn't a
# single rolling-window statistic like historical vol, so we loop.
GARCH_WINDOW = 60          # lookback days used to fit GARCH each time (~1 trading yr)
GARCH_MODEL = "gjr-garch"   # captures the leverage effect; use "garch" for plain GARCH(1,1)

def rolling_garch_forecast(returns, dates, horizon_lookup, window=GARCH_WINDOW, model=GARCH_MODEL):
    returns = np.asarray(returns, dtype=float)
    out = np.full(len(dates), np.nan)   # <-- plain array, no index to misalign
    for i, d in enumerate(dates):
        if i < window:
            continue
        h = horizon_lookup.get(d, np.nan)
        if pd.isna(h) or h < 1:
            continue
        window_rets = returns[i - window:i]
        vol, _ = forecast_realized_vol_garch(window_rets, horizon_days=int(h), model=model)
        out[i] = vol
    return out

# build a date -> days-to-expiry lookup from the option chain itself
horizon_lookup = dict(zip(opt["date"], opt["days"]))

rtn["garch_vol"] = rolling_garch_forecast(rtn["log_return"].values, rtn["date"].values, horizon_lookup)

vix = get_vix(username, start, end, save=False).reset_index()   # date: index -> column

opt = opt.merge(rtn[["date", "hist_vol_zm"]], on="date", how="left")
opt = opt.merge(rtn[["date", "hist_vol_dm"]], on="date", how="left")
opt = opt.merge(rtn[["date", "garch_vol"]], on="date", how="left")
opt = opt.merge(vix[["date", "vix_dec"]], on="date", how="left")

opt = opt[(opt["T"] > 0) & opt["impl_volatility"].notna()].sort_values("days", ascending=False)

# calc price for implied, historical, GARCH and VIX vol, black-scholes
K = opt["strike"].iloc[0]
cp = opt["cp_flag"].iloc[0]
price_fn = black_scholes_call if cp == "C" else black_scholes_put
opt["bs_price_implied"]        = price_fn(opt["S"], K, opt["impl_volatility"], opt["r"], opt["T"], opt["q"])
opt["bs_price_historical_zm"]  = price_fn(opt["S"], K, opt["hist_vol_zm"],     opt["r"], opt["T"], opt["q"])
opt["bs_price_historical_dm"]  = price_fn(opt["S"], K, opt["hist_vol_dm"],     opt["r"], opt["T"], opt["q"])
opt["bs_price_garch"]          = price_fn(opt["S"], K, opt["garch_vol"],      opt["r"], opt["T"], opt["q"])
opt["bs_price_VIX"]            = price_fn(opt["S"], K, opt["vix_dec"],        opt["r"], opt["T"], opt["q"])

# ---- Plot ----
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(opt["days"], opt["mid"], "o", ms=4, color="blue", alpha=0.8, label="Market mid")
#ax.plot(opt["days"], opt["bs_price_implied"], "-", lw=2, color="green", label="Black–Scholes implied vol")
ax.plot(opt["days"], opt["bs_price_historical_zm"], "-", lw=2, color="orange", label="Black–Scholes zero-meaned historical vol")
ax.plot(opt["days"], opt["bs_price_historical_dm"], "-", lw=2, color="red", label="Black–Scholes demeaned historical vol")
ax.plot(opt["days"], opt["bs_price_garch"], "-", lw=2, color="purple", label=f"Black–Scholes GARCH ({GARCH_MODEL})")
ax.plot(opt["days"], opt["bs_price_VIX"], "-", lw=2, color="green", label="Black–Scholes VIX")
ax.invert_xaxis()                                   # time runs left to right towards expiry
ax.set_xlabel("Days to expiry")
ax.set_ylabel("Option price ($)")
ax.set_title(f"SPX {'call' if cp == 'C' else 'put'}, K = {K:g}", loc="left", fontweight="bold")

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False)

fig.tight_layout()
plt.show()
