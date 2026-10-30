#%%
import wrds
import numpy as np
import pandas as pd

DATE = "2023-06-12"     # pick any trading day
YEAR = DATE[:4]         # tables are split by year
SECID = 108105          # SPX

db = wrds.Connection(wrds_username="kairogers")

cols_options = db.describe_table(library="optionm", table="opprcd2024")
print(len(cols_options))
print(cols_options[["name", "type"]].to_string())

cols_sec = db.describe_table(library="optionm", table="secprd2024")
print(len(cols_sec))
print(cols_sec[["name", "type"]].to_string())

cols_zerocurve = db.describe_table(library="optionm", table="zerocd")
print(len(cols_zerocurve))
print(cols_zerocurve[["name", "type"]].to_string())

cols_div = db.describe_table(library="optionm", table="idxdvd")
print(len(cols_div))
print(cols_div[["name", "type"]].to_string())

#%%

# 1. SPX options on that day
opts = db.raw_sql(f"""
    SELECT date, optionid, exdate, cp_flag, strike_price, best_bid, best_offer,
           volume, open_interest, impl_volatility, delta,
           am_settlement, expiry_indicator, forward_price
    FROM optionm.opprcd{YEAR}
    WHERE secid = {SECID}
      AND date = '{DATE}'
""", date_cols=["date", "exdate"])

# 2. Index level that day
spot = db.raw_sql(f"""
    SELECT date, close
    FROM optionm.secprd{YEAR}
    WHERE secid = {SECID}
      AND date = '{DATE}'
""", date_cols=["date"])

# 3. Zero curve that day (The risk free rate for different maturities)
zc = db.raw_sql(f"""
    SELECT days, rate
    FROM optionm.zerocd
    WHERE date = '{DATE}'
    ORDER BY days
""")

# 4. The dividend yield that day
div = db.raw_sql(f"""
    SELECT secid, date, rate
    FROM optionm.idxdvd
    WHERE secid = {SECID}
      AND date = '{DATE}'
""")

db.close()

#%%

# ---- Tidy up ----
S = spot["close"].iloc[0]

opts["strike"] = opts["strike_price"] / 1000                  # stored x1000
opts["mid"] = (opts["best_bid"] + opts["best_offer"]) / 2
opts["days"] = (opts["exdate"] - opts["date"]).dt.days
opts["T"] = opts["days"] / 365
opts["S"] = S

# Interpolate risk-free rate to each expiry (zerocd rate is in %, continuously compounded)
opts["r"] = np.interp(opts["days"], zc["days"], zc["rate"]) / 100

opts["q"] = div["rate"].iloc[0] / 100

# Basic cleaning: drop zero bids and missing IVs (-99.99 / NaN)
opts = opts[(opts["best_bid"] > 0) & (opts["impl_volatility"].notna())]

print(f"S&P 500 on {DATE}: {S:.2f}")
print(f"{len(opts)} options, {opts['exdate'].nunique()} expiries")
print(opts.head())

opts.to_csv(f"source/data/multi_option_single_day{DATE}.csv", index=False)
# %%