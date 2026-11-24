#%%
"""Track a single option from `start` to expiry across ALL classical and ML models
(extending option_price_wrt_t.py).

Supports Black-Scholes (implied, historical, GARCH, VIX, lagged IV), Heston,
Random Forest, Gradient Boosted Trees (GBM), Standard MLP, LSTM, and GAN.
All forecasting models (including ML) only use information available BEFORE each day.

Run from the repo root:
    python source/classical_models/comparing_models/option_price_wrt_t_all.py
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

sys.path.append("source")  # tells python to look in source (run from the repo root)

from classical_models.black_scholes.bs_funcs import (
    black_scholes_call, black_scholes_put, historical_volatility_series, garch_vol_series)
from classical_models.Heston.heston import calibrate_heston_by_date, heston_price_chain
from classical_models.comparing_models.compare_funcs import (
    LABELS, COLOURS, SPX, chosen, ask_secid, exercise_style, ticker, cached,
    load_ex_dividend_dates, drop_options_over_dividends)
from data_importing.data_import import (
    multi_option_single_day, multi_option_multi_day, single_option_multi_day,
    return_multi_day, get_vix, wednesdays)
from ML_models.gbm import FEATURES, build_gbm_model, engineer_features
from ML_models.neural_nets import (
    ML_LABELS, ML_COLOURS, decode_predictions, predict_nn_model, target_values, train_nn_model)

# ---- Settings ----
username = "AlexMatijasevic"
start = "2023-06-21"   # starting day to pick an option (data exists through 2025-08-29)
SAVE_FIG = False       # True saves the figure to figures/

ML_TARGET = "iv"       # "iv", "residual", or "price_over_S"
ML_TRAIN_MONTHS = 6    # months of weekly chains strictly before `start` used to train ML
NN_EPOCHS = 12         # training epochs for NN / LSTM / GAN

# ---- Switch any model on (True) or off (False) ----
METHODS = {
    "implied":     True,    # Black-Scholes, same-day implied vol (sits on market mid)
    "hist_vol_zm": False,   # Black-Scholes, zero-mean 60-day historical vol
    "hist_vol_dm": False,   # Black-Scholes, demeaned 60-day historical vol
    "garch_vol":   False,   # Black-Scholes, GJR-GARCH vol forecast
    "vix_dec":     True,    # Black-Scholes, lagged VIX
    "lagged_iv":   True,    # Black-Scholes, previous trading day's implied vol
    "heston":      True,    # Heston, calibrated on previous trading day
    "rf":          True,    # Random Forest (trained strictly before `start`)
    "gbm":         True,    # Gradient Boosted Trees (trained strictly before `start`)
    "nn":          True,    # Standard MLP Neural Network
    "lstm":        True,    # LSTM Neural Network
    "gan":         True,    # GAN Generator-Assessor NN
}
methods = chosen(METHODS)
classical = [m for m in methods if m in LABELS]
ml = [m for m in methods if m in ML_LABELS]
ALL_LABELS = {**LABELS, **ML_LABELS}
ALL_COLOURS = {**COLOURS, **ML_COLOURS}


def pick(prompt, n):
    """Ask for a row number between 0 and n-1 until a valid one is given."""
    while True:
        choice = input(prompt).strip()
        if choice.isdigit() and int(choice) < n:
            return int(choice)
        print(f"Enter a number from 0 to {n - 1}")


# ---- Choose the option ----
secid = ask_secid()
name = ticker(secid)
exercise = exercise_style(secid)
american = exercise == "american"

chain = multi_option_single_day(username, start, secid=secid, save=False)
S0 = chain["S"].iloc[0]
print(f"\n{len(chain)} {exercise} options quoted on {start}, spot = {S0:g}")
if american:
    ex_dates = load_ex_dividend_dates(username, secid, start, start)
    chain = drop_options_over_dividends(chain, ex_dates)
    if chain.empty:
        raise ValueError(f"Every option on {start} has an ex-dividend date before expiry; try another start")

expiries = [pd.Timestamp(e) for e in sorted(chain["exdate"].unique())]
counts = chain["exdate"].value_counts()
print("\n  #  expiry       options")
for i, e in enumerate(expiries):
    print(f"{i:3d}  {e.date()}   {counts[e]}")
exdate = expiries[pick("Choose an expiry # (data ends 2025-08-29): ", len(expiries))]

options = chain[chain["exdate"] == exdate]
near = options[options["strike"].between(0.8 * S0, 1.2 * S0)]
options = (near if not near.empty else options).sort_values(["cp_flag", "strike"]).reset_index(drop=True)
print("\n" + options[["optionid", "cp_flag", "strike", "mid", "impl_volatility", "volume"]].to_string())
choice = options.loc[pick("Choose an option #: ", len(options))]

optionid = int(choice["optionid"])
end = exdate.strftime("%Y-%m-%d")
print(f"\nFollowing option {optionid} from {start} to expiry {end}")

#%%
# ---- Load tracked option & daily returns ----
week_before = (pd.Timestamp(start) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
opt = single_option_multi_day(username, optionid, week_before, end, secid=secid, save=False, calc_q=not american)
opt = opt[(opt["T"] > 0) & opt["impl_volatility"].notna()].sort_values("days", ascending=False)
opt["lagged_iv"] = opt["impl_volatility"].shift(1)
opt = opt[opt["date"] >= start].reset_index(drop=True)
if american or "q" not in opt:
    opt["q"] = 0.0

rtn = return_multi_day(username, start, end, lookback=252, secid=secid, save=False)

# ---- Classical Black-Scholes methods ----
opt["implied"] = opt["impl_volatility"]
for m, how in (("hist_vol_zm", "zero_mean"), ("hist_vol_dm", "demeaned")):
    if m in classical:
        opt[m] = opt["date"].map(historical_volatility_series(rtn["log_return"], window=60, method=how))
if "vix_dec" in classical or (ml and secid == SPX):
    vix_series = get_vix(username, start, end, save=False)["vix_dec"]
    if "vix_dec" in classical:
        opt["vix_dec"] = opt["date"].map(vix_series)
if "garch_vol" in classical:
    opt["garch_vol"] = garch_vol_series(opt["date"], opt["days"], rtn["log_return"])

K = float(opt["strike"].iloc[0])
cp = opt["cp_flag"].iloc[0]
price_fn = black_scholes_call if cp == "C" else black_scholes_put
for m in classical:
    if m != "heston":
        opt[f"price_{m}"] = price_fn(opt["S"], K, opt[m], opt["r"], opt["T"], opt["q"])

# ---- Heston (calibrated to previous trading day's whole chain) ----
if "heston" in classical:
    prev_day = dict(zip(rtn.index[1:], rtn.index[:-1]))
    chains = multi_option_multi_day(username, sorted({prev_day[d] for d in opt["date"]}),
                                    secid=secid, save=False, calc_q=not american)
    if american:
        chains["q"] = 0.0
        chains = drop_options_over_dividends(chains, ex_dates)
    params = calibrate_heston_by_date(chains)
    opt["price_heston"] = heston_price_chain(
        opt, {d: params[prev_day[d]] for d in opt["date"] if prev_day[d] in params}
    )

# ---- ML models (trained strictly on weekly chains BEFORE `start`) ----
if ml:
    train_end = (pd.Timestamp(start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    train_start = (pd.Timestamp(start) - pd.DateOffset(months=ML_TRAIN_MONTHS)).strftime("%Y-%m-%d")
    train_rtn = return_multi_day(username, train_start, train_end, lookback=10, secid=secid, save=False)
    train_dates = wednesdays(train_rtn.index, train_start, train_end)

    train_chains = cached(
        f"source/data/cache_weekly_chains_{secid}_{train_start}_{train_end}.pkl",
        lambda: multi_option_multi_day(username, train_dates, secid=secid, save=False, calc_q=not american),
    )
    if american:
        train_chains = train_chains.assign(q=0.0)
        train_ex = load_ex_dividend_dates(username, secid, train_start, train_end)
        train_chains = drop_options_over_dividends(train_chains, train_ex)
    elif "q" not in train_chains:
        train_chains["q"] = 0.0

    train_feat = engineer_features(train_chains)

    # Build the exact same 11 FEATURES for each daily row of the tracked option `opt`
    opt_feat = opt.copy()
    for col in ("S", "strike", "T", "r", "q", "mid", "lagged_iv"):
        opt_feat[col] = opt_feat[col].astype(float)

    opt_feat["is_call"] = int(cp == "C")
    opt_feat["moneyness"] = opt_feat["strike"] / opt_feat["S"]
    F = opt_feat["S"] * np.exp((opt_feat["r"] - opt_feat["q"]) * opt_feat["T"])
    opt_feat["log_mF"] = np.log(opt_feat["strike"] / F)
    opt_feat["iv_lag"] = opt_feat["lagged_iv"]

    d1 = (-opt_feat["log_mF"] + (opt_feat["iv_lag"] ** 2 / 2.0) * opt_feat["T"]) / (
        opt_feat["iv_lag"] * np.sqrt(opt_feat["T"])
    )
    opt_feat["delta_lag"] = np.where(
        opt_feat["is_call"] == 1,
        np.exp(-opt_feat["q"] * opt_feat["T"]) * norm.cdf(d1),
        -np.exp(-opt_feat["q"] * opt_feat["T"]) * norm.cdf(-d1),
    )
    # Use lagged VIX as the market ATM vol gauge on SPX; fall back to option's lagged IV on other names
    opt_feat["atm_iv_lag"] = opt_feat["date"].map(vix_series).fillna(opt_feat["iv_lag"]) if secid == SPX else opt_feat["iv_lag"]
    opt_feat["spot_ret"] = opt_feat["date"].map(rtn["log_return"].rolling(5).sum())
    opt_feat["is_new"] = 0
    opt_feat["benchmark_price"] = price_fn(opt_feat["S"], K, opt_feat["iv_lag"], opt_feat["r"], opt_feat["T"], opt_feat["q"])

    valid = opt_feat[FEATURES + ["benchmark_price"]].notna().all(axis=1)
    opt_valid = opt_feat[valid]

    for m in ml:
        opt[f"price_{m}"] = np.nan
        if m == "rf":
            model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=0)
            model.fit(train_feat[FEATURES], target_values(train_feat, ML_TARGET))
            preds = decode_predictions(opt_valid, model.predict(opt_valid[FEATURES]), ML_TARGET)
        elif m == "gbm":
            model = build_gbm_model(random_state=0)
            model.fit(train_feat[FEATURES], target_values(train_feat, ML_TARGET))
            preds = decode_predictions(opt_valid, model.predict(opt_valid[FEATURES]), ML_TARGET)
        else:
            bundle = train_nn_model(m, train_feat, target_method=ML_TARGET, epochs=NN_EPOCHS)
            preds = predict_nn_model(bundle, opt_valid, history_df=train_feat)

        opt.loc[opt_valid.index, f"price_{m}"] = preds

#%%
# ---- Plot ----
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(opt["days"], opt["mid"], "o", ms=4, color="blue", alpha=0.8, label="Market mid")
for m in methods:
    label = ALL_LABELS[m]
    if m == "vix_dec" and secid != SPX:
        label += " (S&P 500 vol, only a rough guide here)"
    ax.plot(opt["days"], opt[f"price_{m}"], "-", lw=2, color=ALL_COLOURS[m], label=label)
ax.invert_xaxis()
ax.set_xlabel("Days to expiry")
ax.set_ylabel("Option price ($)")
div_note = " (no dividend before expiry, q = 0)" if american else ""
ax.set_title(f"{name} {exercise} {'call' if cp == 'C' else 'put'}, K = {K:g}, expiry {end}{div_note}",
             loc="left", fontweight="bold")

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False)

fig.tight_layout()
if SAVE_FIG:
    fig.savefig(f"figures/option_price_wrt_t_all_{secid}_{optionid}.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
