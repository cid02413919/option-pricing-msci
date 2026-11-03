#%%
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("source")  # tells python to look in source

from classical_models.black_scholes.bs_funcs import black_scholes_call, black_scholes_put
from data_importing.data_import import multi_option_single_day, single_option_multi_day

# secid -> (ticker, name). The XL... are sector ETFs (American options), SPX is an index (European)
SECURITIES = {
    110007: ("XLB", "Materials"),
    213084: ("XLC", "Communication Services"),
    110011: ("XLE", "Energy"),
    110012: ("XLF", "Financials"),
    110013: ("XLI", "Industrials"),
    110014: ("XLK", "Technology"),
    110009: ("XLP", "Consumer Staples"),
    208181: ("XLRE", "Real Estate"),
    110015: ("XLU", "Utilities"),
    110008: ("XLV", "Health Care"),
    110010: ("XLY", "Consumer Discretionary"),
    108105: ("SPX", "S&P 500 Index"),
}

# WRDS username and dates (same as optionp_wrt_time.py)
username = "kairogers"
start, end = "2023-06-12", "2023-12-15"

#%%
# ---- Choose one comparable option per security ----
def choose_option(secid):
    """On start: calls only, expiry closest to end, then strike closest to spot."""
    chain = multi_option_single_day(username, start, secid=secid, save=False)
    calls = chain[chain["cp_flag"] == "C"].copy()
    if calls.empty:
        raise ValueError(f"no calls on {start}")
    calls["exp_gap"] = (calls["exdate"] - pd.Timestamp(end)).abs()
    calls["k_gap"] = (calls["strike"] - calls["S"]).abs()
    # any remaining tie goes to the most open interest
    calls = calls.sort_values(["exp_gap", "k_gap", "open_interest"], ascending=[True, True, False])
    return calls.iloc[0]

chosen = {}
for secid, (ticker, name) in SECURITIES.items():
    try:
        chosen[secid] = choose_option(secid)
    except Exception as e:
        print(f"Warning: {ticker}: no option chosen ({e})")

summary = pd.DataFrame([
    {"ticker": SECURITIES[s][0], "name": SECURITIES[s][1], "optionid": int(o["optionid"]),
     "strike": o["strike"], "spot": o["S"], "expiry": o["exdate"].date()}
    for s, o in chosen.items()
])
print(summary.to_string(index=False))

#%%
# ---- History and Black–Scholes price with implied vol ----
hist = {}
for secid, o in chosen.items():
    ticker = SECURITIES[secid][0]
    try:
        opt = single_option_multi_day(username, int(o["optionid"]), start, end, secid=secid, save=False)
        opt = opt[(opt["T"] > 0) & opt["impl_volatility"].notna()].sort_values("days", ascending=False)
        if opt.empty:
            raise ValueError("no days with T > 0 and implied vol")

        K = opt["strike"].iloc[0]
        price_fn = black_scholes_call if opt["cp_flag"].iloc[0] == "C" else black_scholes_put
        opt["bs_price_implied"] = price_fn(opt["S"], K, opt["impl_volatility"], opt["r"], opt["T"], opt["q"])
        hist[secid] = opt
    except Exception as e:
        print(f"Warning: {ticker}: no history ({e})")

# how far Black–Scholes sits from the market, relative to the average mid
fit = pd.DataFrame([
    {"ticker": SECURITIES[s][0], "days": len(h), "q_source": h["q_source"].iloc[0],
     "mean_q": h["q"].mean(), "bs_nan": h["bs_price_implied"].isna().sum(),
     "mean_abs_gap": (h["bs_price_implied"] - h["mid"]).abs().mean(),
     "rel_gap": (h["bs_price_implied"] - h["mid"]).abs().mean() / h["mid"].mean()}
    for s, h in hist.items()
])
print(fit.to_string(index=False))

#%%
# ---- Plot: 11 ETFs in table order, SPX in the last panel ----
fig, axes = plt.subplots(4, 3, figsize=(14, 13))

for ax, (secid, (ticker, name)) in zip(axes.flat, SECURITIES.items()):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if secid not in hist:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{ticker} ({name})", loc="left", fontweight="bold", fontsize=10)
        continue

    opt = hist[secid]
    ax.plot(opt["days"], opt["mid"], "o", ms=4, color="blue", alpha=0.8, label="Market mid")
    ax.plot(opt["days"], opt["bs_price_implied"], "-", lw=2, color="pink", label="Black–Scholes implied vol")
    ax.invert_xaxis()                                   # time runs left to right towards expiry
    ax.set_title(f"{ticker} ({name}), K = {opt['strike'].iloc[0]:g}", loc="left", fontweight="bold", fontsize=10)

for ax in axes[-1]:
    ax.set_xlabel("Days to expiry")
for ax in axes[:, 0]:
    ax.set_ylabel("Option price ($)")

fig.suptitle(f"Market vs Black–Scholes (implied vol), ATM calls, {start} to {end}",
             x=0.01, ha="left", fontweight="bold")
fig.text(0.01, 0.958, "ETF options are American; priced with European Black–Scholes",
         ha="left", fontsize=9, color="grey")

# one legend for the figure, taken from the first panel with data
handles, labels = next((ax.get_legend_handles_labels() for ax in axes.flat if ax.lines), ([], []))
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.975), frameon=False, ncol=2)

fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()

# %%
