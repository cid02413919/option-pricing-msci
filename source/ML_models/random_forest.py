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
  4. Split into training and test data
  5. Train the random forest
  6. Measure how good the predictions are on the test data
  7. Plot predicted vs actual prices
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
df = df.dropna(subset=["S", "strike", "T", "r", "q", "mid"])

# Turn the columns we use into plain floats
for col in ["S", "strike", "T", "r", "q", "mid"]:
    df[col] = df[col].astype(float)

print(f"Number of call options: {len(df)}")


# =============================================================================
# 3. Features (X) and target (y)
# =============================================================================
# Trick: instead of predicting the price in dollars, predict price / S.
# A call with strike 4000 when S = 4100 behaves just like a call with
# strike 3900 when S = 4000. Dividing by S makes those two look the same to
# the model, so it learns one pattern instead of one per index level.
# (Random forests can't predict outside the range they were trained on, so
# this also helps when S moves to new levels in the test period.)
df["moneyness"] = df["strike"] / df["S"]     # K / S
df["price_over_S"] = df["mid"] / df["S"]     # C / S

# The same inputs Black-Scholes uses, except volatility
FEATURES = ["moneyness", "T", "r", "q"]
TARGET = "price_over_S"


# =============================================================================
# 4. Split into training and test data
# =============================================================================
# Split by date: train on Jan-Oct, test on Nov-Dec.
# This mimics real life (train on the past, predict the future).
# A random split would put options from the same day in both sets, which
# makes the model look better than it really is.
SPLIT_DATE = "2023-11-01"
train = df[df["date"] < SPLIT_DATE]
test = df[df["date"] >= SPLIT_DATE]

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

print(f"Training options: {len(train)}")
print(f"Test options:     {len(test)}")


# =============================================================================
# 5. Train the random forest
# =============================================================================
model = RandomForestRegressor(
    n_estimators=100,      # number of trees in the forest
    min_samples_leaf=5,    # each leaf must contain at least 5 options (stops overfitting)
    n_jobs=-1,             # use all CPU cores
    random_state=0,        # makes the results repeatable
)
print("Training...")
model.fit(X_train, y_train)


# =============================================================================
# 6. How good are the predictions?
# =============================================================================
# Predict C / S, then multiply by S to get back to a price in index points
test = test.copy()
test["predicted_price"] = model.predict(X_test) * test["S"]

actual = test["mid"]
predicted = test["predicted_price"]

print("\nTest set results (prices in index points):")
print(f"  Mean absolute error: {mean_absolute_error(actual, predicted):.2f}")
print(f"  RMSE:                {np.sqrt(mean_squared_error(actual, predicted)):.2f}")
print(f"  R squared:           {r2_score(actual, predicted):.4f}")

# Which features did the forest find most useful?
print("\nFeature importance (higher = more useful):")
for name, importance in zip(FEATURES, model.feature_importances_):
    print(f"  {name:<10s} {importance:.3f}")


# =============================================================================
# 7. Plot predicted vs actual prices
# =============================================================================
plt.figure(figsize=(6, 6))
plt.scatter(actual, predicted, s=2, alpha=0.3)
max_price = actual.max()
plt.plot([0, max_price], [0, max_price], color="red", label="perfect prediction")
plt.xlabel("Market price (mid)")
plt.ylabel("Random forest price")
plt.title("SPX calls: random forest vs market (Nov-Dec 2023)")
plt.legend()
plt.tight_layout()
plt.show()
