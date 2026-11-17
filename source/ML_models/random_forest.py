"""A simple random forest that predicts S&P 500 (SPX) call option prices.

The idea:
  - A random forest is a collection of many decision trees.
  - Each tree is trained on a random sample of the data and asks a series of
    yes/no questions ("is moneyness > 1.05?", "is T < 0.1?") to reach a price.
  - The forest's prediction is the average of all the trees' predictions.

Steps in this script:
  1. Load the SPX option data (already downloaded and cached in source/data)
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append("source")
from classical_models.black_scholes.bs_funcs import black_scholes_call


# =============================================================================
# 1. Load the data
# =============================================================================
# Weekly SPX option chains for 2023 (secid 108105 = SPX), saved by compare_funcs.py
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data",
                         "cache_weekly_chains_108105_2023-01-01_2023-12-31.pkl")
df = pd.read_pickle(DATA_FILE)


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

FEATURES = ["moneyness", "T", "r", "q",                                # old inputs
            "log_mF", "iv_lag", "delta_lag", "atm_iv_lag", "spot_ret", "is_new"]
TARGET = "price_over_S"

# The first date has no previous week, so no lagged features
df = df.dropna(subset=FEATURES)


# =============================================================================
# 4. Walk-forward test
# =============================================================================
# For each month from March, train on ALL earlier months, then predict that month.
# This mimics real life (train on the past, predict the future) and gives one
# test per month instead of one single test. A random split would put options
# from the same day in both sets, which makes the model look better than it is.
df["month"] = df["date"].dt.to_period("M")
months = sorted(df["month"].unique())[2:]        # first two months are only used for training

# Benchmark: Black-Scholes with last week's implied vol (no model at all)
df["benchmark_price"] = black_scholes_call(df["S"], df["strike"], df["iv_lag"], df["r"], df["T"], df["q"])

df["predicted_price"] = np.nan
results = []
for month in months:
    train = df[df["month"] < month]
    test = df[df["month"] == month]

    model = RandomForestRegressor(
        n_estimators=100,      # number of trees in the forest
        min_samples_leaf=5,    # each leaf must contain at least 5 options (stops overfitting)
        n_jobs=-1,             # use all CPU cores
        random_state=0,        # makes the results repeatable
    )
    model.fit(train[FEATURES], train[TARGET])

    # Predict price / S, then multiply by S to get back to a price in index points
    df.loc[test.index, "predicted_price"] = model.predict(test[FEATURES]) * test["S"]

    results.append({
        "month": str(month),
        "train options": len(train),
        "test options": len(test),
        "RMSE forest": np.sqrt(mean_squared_error(test["mid"], df.loc[test.index, "predicted_price"])),
        "RMSE benchmark": np.sqrt(mean_squared_error(test["mid"], test["benchmark_price"])),
    })
    print(f"{month} done")


# =============================================================================
# 5. How good are the predictions?
# =============================================================================
print("\nRMSE (index points) for each test month:")
print(pd.DataFrame(results).set_index("month").round(2).to_string())

tested = df.dropna(subset=["predicted_price"])
actual = tested["mid"]
predicted = tested["predicted_price"]

print("\nAll test months together (prices in index points):")
print(f"  Mean absolute error: {mean_absolute_error(actual, predicted):.2f}")
print(f"  RMSE:                {np.sqrt(mean_squared_error(actual, predicted)):.2f}")
print(f"  R squared:           {r2_score(actual, predicted):.4f}")
print(f"  Benchmark RMSE:      {np.sqrt(mean_squared_error(actual, tested['benchmark_price'])):.2f}")

# Which features did the last forest (trained on Jan-Nov) find most useful?
print("\nFeature importance (higher = more useful):")
for name, importance in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name:<11s} {importance:.3f}")


# =============================================================================
# 6. Plot predicted vs actual prices
# =============================================================================
plt.figure(figsize=(6, 6))
plt.scatter(actual, predicted, s=2, alpha=0.3)
max_price = actual.max()
plt.plot([0, max_price], [0, max_price], color="red", label="perfect prediction")
plt.xlabel("Market price (mid)")
plt.ylabel("Random forest price")
plt.title("SPX calls: random forest vs market (Mar-Dec 2023)")
plt.legend()
plt.tight_layout()
plt.show()
