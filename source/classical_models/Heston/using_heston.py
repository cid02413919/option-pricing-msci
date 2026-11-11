#%%
import sys
import numpy as np

sys.path.append("source")  # tells python to look in source

from classical_models.Heston.heston import calibrate_heston_day, heston_price_fft
from classical_models.black_scholes.bs_funcs import black_scholes_call, black_scholes_put
from data_importing.data_import import multi_option_single_day

# WRDS username, the day to price on, and the security
username = "kairogers"
date = "2025-06-23"
secid = 108105   # 108105 = SPX, 110007 = XLB, 110014 = XLK, ...

# ---- Settings: which option to price ----
OPTIONID = None    # an OptionMetrics optionid, or None to pick one with the settings below
CP = "C"           # "C" call or "P" put
TARGET_DAYS = 30   # days to expiry wanted (the closest expiry is used)
MONEYNESS = 1.0    # strike / spot wanted (the closest strike is used)

#%%
# ---- Load that day's option chain (same loader as the Black–Scholes scripts) ----
chain = multi_option_single_day(username, date, secid=secid, save=False, calc_q=True)
print(f"{len(chain)} options quoted on {date}, spot = {chain['S'].iloc[0]:g}")

# ---- Calibrate Heston to the whole chain (one set of 5 parameters for the day) ----
params, S0, r, q = calibrate_heston_day(chain, n_starts=3)
print("\nCalibrated Heston parameters:")
for name, value in params.items():
    print(f"  {name:8s} {value:.4f}")
print(f"  Feller 2*kappa*theta - sigma_v^2 = {2 * params['kappa'] * params['theta'] - params['sigma_v']**2:.4f} (>= 0 is nice)")

#%%
# ---- Pick the option ----
if OPTIONID is not None:
    option = chain[chain["optionid"] == OPTIONID].iloc[0]
else:
    side = chain[chain["cp_flag"] == CP]
    exdate = side.loc[(side["days"] - TARGET_DAYS).abs().idxmin(), "exdate"]
    side = side[side["exdate"] == exdate]
    option = side.loc[(side["strike"] / S0 - MONEYNESS).abs().idxmin()]

K, T, cp = option["strike"], option["T"], option["cp_flag"]
q_opt = option["q"] if "q" in option and np.isfinite(option["q"]) else 0.0

# ---- Price it: Heston vs Black–Scholes (implied vol) vs the market ----
heston = heston_price_fft(option["S"], K, T, option["r"], q=q_opt, cp_flag=cp, **params)
bs_fn = black_scholes_call if cp == "C" else black_scholes_put
bs = bs_fn(option["S"], K, option["impl_volatility"], option["r"], T, q_opt)

print(f"\nOption {int(option['optionid'])}: {'call' if cp == 'C' else 'put'}, K = {K:g}, "
      f"expiry {option['exdate'].date()} ({option['days']} days)")
print(f"  Market mid                  {option['mid']:10.3f}")
print(f"  Heston (calibrated)         {heston:10.3f}")
print(f"  Black–Scholes implied vol   {bs:10.3f}")

# %%
