#%%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from pricing import black_scholes_call, black_scholes_put
from vol_methods import historical_volatility

# ---- Load ----
df = pd.read_csv("source/data/single_option_multi_day2023-05-01-2023-12-15.csv",
                 usecols=["cp_flag", "strike", "impl_volatility", "mid", "T", "S", "r", "q", "days"])
df = df[(df["T"] > 0) & df["impl_volatility"].notna()].sort_values("days", ascending=False)

K = df["strike"].iloc[0]
cp = df["cp_flag"].iloc[0]
price_fn = black_scholes_call if cp == "C" else black_scholes_put

df["bs_price"] = price_fn(df["S"], K, df["impl_volatility"], df["r"], df["T"], df["q"])
df["hist_vol"] = historical_volatility(df["S"], df["days"], annualise=True)

# ---- Plot ----
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(df["days"], df["mid"], "o", ms=4, color="blue", alpha=0.8, label="Market mid")
ax.plot(df["days"], df["bs_price"], "-", lw=2, color="red", label="Black–Scholes")
ax.invert_xaxis()                                   # time runs left to right towards expiry
ax.set_xlabel("Days to expiry")
ax.set_ylabel("Option price ($)")
ax.set_title(f"SPX {'call' if cp == 'C' else 'put'}, K = {K:g}", loc="left", fontweight="bold")

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False)

fig.tight_layout()
plt.show()

