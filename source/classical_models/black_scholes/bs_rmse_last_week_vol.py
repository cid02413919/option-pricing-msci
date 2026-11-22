"""RMSE of Black-Scholes when each option is priced with LAST WEEK's implied volatility.

Steps:
  1. Find every Wednesday between START and END (the Tuesday before if Wednesday is a holiday)
  2. Load all the options quoted on those days
  3. For each option, look up its implied vol from the previous pricing day (a week earlier)
  4. Price it with Black-Scholes using that vol
  5. RMSE = sqrt(mean((model price - market mid price)^2))

Run from the repo root:  python source/classical_models/black_scholes/bs_rmse_last_week_vol.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("source")  # so python can find our own files (run from repo root)
from classical_models.black_scholes.bs_funcs import black_scholes_call, black_scholes_put
from data_importing.data_import import multi_option_multi_day, return_multi_day, wednesdays

# ---- Settings: change these ----
USERNAME = "kairogers"
SECID    = 108105         # 108105 = SPX, 110007 = XLB, 110009 = XLP, ...
START    = "2023-01-01"
END      = "2023-12-31"


# =============================================================================
# 1. The pricing days: Wednesdays (or Tuesday if Wednesday is a holiday)
# =============================================================================
trading_days = return_multi_day(USERNAME, START, END, lookback=0, secid=SECID, save=False).index
dates = wednesdays(trading_days, START, END)
print(f"{len(dates)} pricing days from {dates[0].date()} to {dates[-1].date()}")


# =============================================================================
# 2. All the options on those days (re-uses the saved file if it is there)
# =============================================================================
cache_file = f"source/data/cache_weekly_chains_{SECID}_{START}_{END}.pkl"
if os.path.exists(cache_file):
    df = pd.read_pickle(cache_file)
else:
    df = multi_option_multi_day(USERNAME, dates, secid=SECID, save=False, calc_q=True)
    df.to_pickle(cache_file)

df = df[df["T"] > 0]                       # drop options expiring on the day itself
if "q" not in df.columns:
    df["q"] = 0.0                          # no dividend data -> assume no dividends
df = df.dropna(subset=["S", "strike", "T", "r", "q", "mid"])


# =============================================================================
# 3. Each option's implied vol from last week
# =============================================================================
# Map each pricing day to the one before it
pricing_days = sorted(df["date"].unique())
previous_day = dict(zip(pricing_days[1:], pricing_days[:-1]))
df["prev_date"] = df["date"].map(previous_day)

# Last week's implied vol of the same option (same optionid)
last_week = df[["date", "optionid", "impl_volatility"]].rename(
    columns={"date": "prev_date", "impl_volatility": "vol_last_week"})
df = df.merge(last_week, on=["prev_date", "optionid"], how="inner")
# (inner merge: options that did not exist last week, and the first week, are dropped)


# =============================================================================
# 4. Black-Scholes price with last week's vol
# =============================================================================
calls = df["cp_flag"] == "C"
puts  = df["cp_flag"] == "P"

df["bs_price"] = np.nan
df.loc[calls, "bs_price"] = black_scholes_call(df.loc[calls, "S"], df.loc[calls, "strike"],
                                               df.loc[calls, "vol_last_week"], df.loc[calls, "r"],
                                               df.loc[calls, "T"], df.loc[calls, "q"])
df.loc[puts, "bs_price"] = black_scholes_put(df.loc[puts, "S"], df.loc[puts, "strike"],
                                             df.loc[puts, "vol_last_week"], df.loc[puts, "r"],
                                             df.loc[puts, "T"], df.loc[puts, "q"])


# =============================================================================
# 5. RMSE
# =============================================================================
df["error"] = df["bs_price"] - df["mid"]

def rmse(errors):
    return np.sqrt(np.mean(errors**2))

print(f"Options priced: {len(df)}  ({calls.sum()} calls, {puts.sum()} puts)")
print(f"RMSE (all):   {rmse(df['error']):.4f}")
print(f"RMSE (calls): {rmse(df.loc[calls, 'error']):.4f}")
print(f"RMSE (puts):  {rmse(df.loc[puts, 'error']):.4f}")
