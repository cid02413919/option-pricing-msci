#%%
import pandas as pd
from source.classical_models.black_scholes.pricing import black_scholes_call, black_scholes_merton_call
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("source/data/spx_options_2024-06-12.csv", usecols=["cp_flag", "strike_price", "impl_volatility", "mid", "T", "S", "r", "q", "days"])

#%% Calculating overall MSE for black scholes including and not including dividend
residuals = []

for i in range(len(df)):
    row = df.loc[i]

    cp_flag = row["cp_flag"]
    K       = row["strike_price"] / 1000
    mid     = row["mid"]
    sigma   = row["impl_volatility"]
    T       = row["T"]
    S       = row["S"]
    r       = row["r"]

    if cp_flag == "C":
        call_price_bs = black_scholes_call(S, K, sigma, r, T)
        residuals.append(call_price_bs - mid)

mse = sum(j**2 for j in residuals) / len(residuals)
print(f"MSE of original: {mse:.4f}")

residuals = []

for i in range(len(df)):
    row = df.loc[i]

    cp_flag = row["cp_flag"]
    K       = row["strike_price"] / 1000
    mid     = row["mid"]
    sigma   = row["impl_volatility"]
    T       = row["T"]
    S       = row["S"]
    r       = row["r"]
    q       = row["q"]

    if cp_flag == "C":
        call_price_bs = black_scholes_merton_call(S, K, sigma, r, T, q)
        residuals.append(call_price_bs - mid)

mse = sum(j**2 for j in residuals) / len(residuals)
print(f"MSE of merton: {mse:.4f}")

print(f"number of calls evaulated: {len(residuals)}")

#%% black scholes containing and not containing dividends against expiry

# All calls, with errors for both models computed in one go (no loop needed)
calls = df[df["cp_flag"] == "C"].copy()
K = calls["strike_price"] / 1000

bs  = black_scholes_call(calls["S"], K, calls["impl_volatility"], calls["r"], calls["T"])
bsm = black_scholes_merton_call(calls["S"], K, calls["impl_volatility"], calls["r"], calls["T"], calls["q"])

calls["sq_err_bs"]  = (bs  - calls["mid"]) ** 2
calls["sq_err_bsm"] = (bsm - calls["mid"]) ** 2

# MSE for each expiry (grouped by days to expiry)
mse_by_expiry = calls.groupby("days")[["sq_err_bs", "sq_err_bsm"]].mean()
print(mse_by_expiry)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(mse_by_expiry.index, mse_by_expiry["sq_err_bs"],  "o-", label="Black-Scholes (no dividends)")
ax.plot(mse_by_expiry.index, mse_by_expiry["sq_err_bsm"], "s-", label="Black-Scholes-Merton (with q)")
ax.set_xlabel("Days to expiry")
ax.set_ylabel("MSE (index points²)")
ax.set_yscale("log")
ax.set_title("SPX calls, 2024-06-12: pricing error by expiry")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/mse_by_expiry_log.png", dpi=200)
plt.show()


# %%
