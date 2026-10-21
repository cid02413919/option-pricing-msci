import pandas as pd
from pricing import black_scholes_call
import numpy as np

df = pd.read_csv("spx_options_2024-06-12.csv", usecols=["cp_flag", "strike_price", "best_bid", "best_offer", "impl_volatility", "T", "S", "r"])
#print(df)

sigma = df["impl_volatility"].mean()

for i in range(200):
    row = df.loc[i]

    cp_flag = row["cp_flag"]
    K       = row["strike_price"] / 1000
    bid     = row["best_bid"]
    ask     = row["best_offer"]
    mid     = (bid + ask) / 2
    T       = row["T"]
    S       = row["S"]
    r       = row["r"]

    call_price_bs = black_scholes_call(S, K, sigma, r, T)
    print(call_price_bs - mid)

    #print(f"black scholes price:{call_price_bs}")
    #print(f"market price:{mid}")