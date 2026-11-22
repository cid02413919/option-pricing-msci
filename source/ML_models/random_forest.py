"""A simple random forest that predicts S&P 500 (SPX) call option prices.

The idea:
  - A random forest is a collection of many decision trees.
  - Each tree is trained on a random sample of the data and asks a series of
    yes/no questions ("is moneyness > 1.05?", "is T < 0.1?") to reach a price.
  - The forest's prediction is the average of all the trees' predictions.

Steps in this script:
  1. Load the SPX option data (from source/data if saved, otherwise from WRDS)
  2. Keep only calls and clean the data
  3. Choose the features (inputs, X) and the target (what we predict, y)
  4. Walk-forward test: price each month with a forest trained only on earlier months
  5. Measure how good the predictions are
  6. Plot predicted vs actual prices

Run from the repo root:  python source/ML_models/random_forest.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

sys.path.append("source")
from classical_models.black_scholes.bs_funcs import black_scholes_call

# ---- What the forest predicts: every target set to True is run ----
PREDICT = {
    "price_over_S":    True,   # option price / S
    "bs_correction":   True,   # (market price - Black-Scholes price with last week's IV) / S
    "impl_volatility": True,   # today's implied vol, then priced with Black-Scholes
}


# =============================================================================
# 1. Load the data
# =============================================================================
# Weekly SPX option chains (secid 108105 = SPX). If the data has already been
# downloaded (by this script or compare_funcs.py) it is loaded from source/data,
# otherwise it is downloaded from WRDS and saved there for next time.
USERNAME = "kairogers"     # WRDS username (only needed if the data isn't saved yet)
SECID = 108105
START, END = "2023-01-01", "2023-12-31"

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data",
                         f"cache_weekly_chains_{SECID}_{START}_{END}.pkl")

if os.path.exists(DATA_FILE):
    print("Loading saved option data...")
    df = pd.read_pickle(DATA_FILE)
else:
    print("No saved option data found, downloading from WRDS...")
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # so python can find data_importing
    from data_importing.data_import import multi_option_multi_day, return_multi_day, wednesdays

    trading_days = return_multi_day(USERNAME, START, END, lookback=0, secid=SECID, save=False).index
    dates = wednesdays(trading_days, START, END)
    df = multi_option_multi_day(USERNAME, dates, secid=SECID, save=False, calc_q=True)
    pd.to_pickle(df, DATA_FILE)


# =============================================================================
# 2. Keep only calls and clean the data
# =============================================================================
df = df[df["cp_flag"] == "C"]           # calls only
df = df[df["best_bid"] > 0]             # drop options nobody is bidding on
df = df[df["T"] > 0]                    # drop options expiring today
df = df.dropna(subset=["S", "strike", "T", "r", "q", "mid", "impl_volatility"])

# Turn the columns we use into plain floats
for col in ["S", "strike", "T", "r", "q", "mid", "impl_volatility", "optionid"]:
    df[col] = df[col].astype(float)
df = df.reset_index(drop=True)

print(f"Number of call options: {len(df)}")


# =============================================================================
# 3. Features (X) and target (y)
# =============================================================================
# Trick: instead of predicting the price in dollars, predict price / S.
# A call with strike 4000 when S = 4100 behaves just like a call with
# strike 3900 when S = 4000. Dividing by S makes those two look the same to
# the model, so it learns one pattern instead of one per index level.
df["moneyness"] = df["strike"] / df["S"]     # K / S
df["price_over_S"] = df["mid"] / df["S"]     # C / S

# Moneyness against the forward price F = S e^((r-q)T): 0 = at the money
F = df["S"] * np.exp((df["r"] - df["q"]) * df["T"])
df["log_mF"] = np.log(df["strike"] / F)

# Lagged features: only information from the previous pricing date (a week earlier).
# Today's implied vol and delta are calculated from today's price, so using them
# would give the model the answer.
dates = np.sort(df["date"].unique())
prev = dict(zip(dates[1:], dates[:-1]))
df["prev_date"] = df["date"].map(prev)

# iv_lag: each option's implied vol last week. Options that did not exist last week
# (new strikes and expiries) get the IV of last week's closest option.
df["iv_lag"] = np.nan
df["is_new"] = 0
for date, today in df.groupby("date"):
    if date not in prev:
        continue
    last = df[df["date"] == prev[date]]
    same = today["optionid"].map(last.set_index("optionid")["impl_volatility"])
    tree = cKDTree(np.c_[last["log_mF"], np.sqrt(last["T"])])
    _, nearest = tree.query(np.c_[today["log_mF"], np.sqrt(today["T"])])
    df.loc[today.index, "iv_lag"] = same.fillna(pd.Series(last["impl_volatility"].to_numpy()[nearest], index=today.index))
    df.loc[today.index, "is_new"] = same.isna().astype(int)

# delta_lag: Black-Scholes delta using today's S, K, T, r, q and last week's IV
d1 = (-df["log_mF"] + df["iv_lag"] ** 2 / 2 * df["T"]) / (df["iv_lag"] * np.sqrt(df["T"]))
df["delta_lag"] = np.exp(-df["q"] * df["T"]) * norm.cdf(d1)

# atm_iv_lag: last week's median IV of ~1-month at-the-money options (overall market fear)
atm = df[(df["log_mF"].abs() < 0.02) & df["T"].between(20 / 365, 60 / 365)]
df["atm_iv_lag"] = df["prev_date"].map(atm.groupby("date")["impl_volatility"].median())

# spot_ret: how much the index moved since last week
spot = df.groupby("date")["S"].first()
df["spot_ret"] = np.log(df["S"] / df["prev_date"].map(spot))

FEATURES = ["moneyness", 
            "T", 
            "r", 
            "q",
            "log_mF", 
            "iv_lag", 
            "delta_lag", 
            "atm_iv_lag", 
            "spot_ret", 
            "is_new"
            ]

# The first date has no previous week, so no lagged features
df = df.dropna(subset=FEATURES)

# Black-Scholes price with last week's implied vol, which bs_correction corrects
df["bs_price"] = black_scholes_call(df["S"], df["strike"], df["iv_lag"], df["r"], df["T"], df["q"])
df["bs_correction"] = (df["mid"] - df["bs_price"]) / df["S"]

targets = [t for t, on in PREDICT.items() if on]


def to_price(test, prediction, target):
    """Turn the forest's prediction into a call price."""
    if target == "bs_correction":
        return test["bs_price"] + prediction * test["S"]
    if target == "impl_volatility":
        return black_scholes_call(test["S"], test["strike"], prediction, test["r"], test["T"], test["q"])
    return prediction * test["S"]


# =============================================================================
# 4. Walk-forward test (once for each target switched on)
# =============================================================================
# For each month from March, train on ALL earlier months, then predict that month.
# This mimics real life (train on the past, predict the future) and gives one
# test per month instead of one single test. A random split would put options
# from the same day in both sets, which makes the model look better than it is.
df["month"] = df["date"].dt.to_period("M")
months = sorted(df["month"].unique())[2:]        # first two months are only used for training

# Hyperparameters are tuned on the training-only months (Jan-Feb), so the
# test months play no part in choosing them. TimeSeriesSplit validates on later dates.
tune = df[df["month"] < months[0]].sort_values("date")
param_options = {
    "n_estimators": [50, 100, 200],
    "min_samples_leaf": [1, 5, 20, 50, 200],
    "max_features": [1.0, 0.5, "sqrt"],
    "max_depth": [None, 10, 20],
}

monthly_rmse = pd.DataFrame(index=[str(m) for m in months])
importances = {}
for target in targets:
    print(f"\n---- Target: {target} ----")
    search = RandomizedSearchCV(
        RandomForestRegressor(random_state=0),
        param_options,
        n_iter=20,                               # try 20 random combinations
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=0,
    )
    search.fit(tune[FEATURES], tune[target])
    print("Best hyperparameters:", search.best_params_)

    df[f"price_{target}"] = np.nan
    for month in months:
        train = df[df["month"] < month]
        test = df[df["month"] == month]

        model = RandomForestRegressor(**search.best_params_, n_jobs=-1, random_state=0)
        model.fit(train[FEATURES], train[target])
        df.loc[test.index, f"price_{target}"] = to_price(test, model.predict(test[FEATURES]), target)

        monthly_rmse.loc[str(month), target] = np.sqrt(mean_squared_error(test["mid"], df.loc[test.index, f"price_{target}"]))
        print(f"{month} done")

    importances[target] = model.feature_importances_     # from the last forest (trained on Jan-Nov)


# =============================================================================
# 5. How good are the predictions?
# =============================================================================
tested = df[df["month"].isin(months)]
actual = tested["mid"]

print("\nRMSE (index points) for each test month:")
print(monthly_rmse.round(2).to_string())

print("\nAll test months together (prices in index points):")
summary = pd.DataFrame({
    target: {"RMSE": np.sqrt(mean_squared_error(actual, tested[f"price_{target}"])),
             "R squared": r2_score(actual, tested[f"price_{target}"])}
    for target in targets
}).T
print(summary.round(4).to_string())

print("\nFeature importance (higher = more useful):")
print(pd.DataFrame(importances, index=FEATURES).round(3).to_string())


# =============================================================================
# 6. Plot predicted vs actual prices
# =============================================================================
fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 6), squeeze=False)
max_price = actual.max()
for ax, target in zip(axes[0], targets):
    ax.scatter(actual, tested[f"price_{target}"], s=2, alpha=0.3)
    ax.plot([0, max_price], [0, max_price], color="red", label="perfect prediction")
    ax.set_xlabel("Market price (mid)")
    ax.set_ylabel("Random forest price")
    ax.set_title(f"Target: {target}")
    ax.legend()
fig.suptitle("SPX calls: random forest vs market (Mar-Dec 2023)")
fig.tight_layout()
plt.show()
