"""Random forest option pricer (Phase 1 ML baseline).

The forest predicts how much each option's implied vol (IV) has changed since the last pricing
date (a week ago); the option is then priced with Black–Scholes using last week's IV + that change.
Only information known on the pricing day is used: today's IV and delta come from today's
market price, so they would give the answer away and are only used in lagged form.

Used by compare_funcs.py (methods "rf" and "iv_lag"). Run this file from the repo root to
test the model on SPX 2023 with walk-forward validation and save a model trained on all of it:
    python source/ML_models/random_forest.py
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

sys.path.append("source")
from classical_models.black_scholes.bs_funcs import black_scholes_call, black_scholes_put

FEATURES = ["is_call", "log_mF", "T", "r", "q", "iv_lag", "delta_lag", "atm_iv_lag", "spot_ret", "is_new"]
SAVE_DIR = "source/ML_models/saved"


# =============================================================================
# 1. Features
# =============================================================================

def add_features(df):
    """Add the model's inputs to a multi-date option chain (columns date, optionid, cp_flag,
    S, strike, T, r, q, impl_volatility). Lagged columns come from the previous date in df,
    so the first date gets NaN."""
    df = df.copy()
    for col in ["S", "strike", "T", "r", "q", "impl_volatility", "optionid"]:
        df[col] = df[col].astype(float)
    df["is_call"] = (df["cp_flag"] == "C").astype(int)
    F = df["S"] * np.exp((df["r"] - df["q"]) * df["T"])                  # forward price
    df["log_mF"] = np.log(df["strike"] / F)                               # 0 = at the money

    dates = np.sort(df["date"].unique())
    prev = dict(zip(dates[1:], dates[:-1]))
    df["prev_date"] = df["date"].map(prev)

    df["iv_lag"], df["is_new"] = _lagged_iv(df, prev)
    df["delta_lag"] = _bs_delta(df, df["iv_lag"])

    # one-month at-the-money IV last week: the overall vol level
    atm = df[(df["log_mF"].abs() < 0.02) & df["T"].between(20 / 365, 60 / 365)]
    atm_iv = atm.groupby("date")["impl_volatility"].median()
    df["atm_iv_lag"] = df["prev_date"].map(atm_iv)

    # market move since last week (vol tends to rise when the market falls)
    spot = df.groupby("date")["S"].first()
    df["spot_ret"] = np.log(df["S"] / df["prev_date"].map(spot))
    return df


def _lagged_iv(df, prev):
    """Each option's IV on the previous date. Options that did not exist then (new strikes and
    expiries) get the IV of the closest option last week, by moneyness and maturity."""
    iv_lag = pd.Series(np.nan, index=df.index)
    is_new = pd.Series(0, index=df.index)
    for date, today in df.groupby("date"):
        if date not in prev:
            continue
        last = df[df["date"] == prev[date]]
        for flag in (0, 1):
            t = today[today["is_call"] == flag]
            l = last[last["is_call"] == flag]
            same = t["optionid"].map(l.set_index("optionid")["impl_volatility"])
            tree = cKDTree(np.c_[l["log_mF"], np.sqrt(l["T"])])
            _, nearest = tree.query(np.c_[t["log_mF"], np.sqrt(t["T"])])
            iv_lag[t.index] = same.fillna(pd.Series(l["impl_volatility"].to_numpy()[nearest], index=t.index))
            is_new[t.index] = same.isna().astype(int)
    return iv_lag, is_new


def _bs_delta(df, sigma):
    """Black–Scholes delta using today's S, K, T, r, q and the given vol."""
    d1 = (-df["log_mF"] + sigma ** 2 / 2 * df["T"]) / (sigma * np.sqrt(df["T"]))
    call_delta = np.exp(-df["q"] * df["T"]) * norm.cdf(d1)
    return np.where(df["is_call"] == 1, call_delta, call_delta - np.exp(-df["q"] * df["T"]))


def bs_price(df, sigma):
    """Black–Scholes call or put price for every row of df with vol sigma."""
    call = black_scholes_call(df["S"], df["strike"], sigma, df["r"], df["T"], df["q"])
    put = black_scholes_put(df["S"], df["strike"], sigma, df["r"], df["T"], df["q"])
    return np.where(df["is_call"] == 1, call, put)


# =============================================================================
# 2. Training and prediction
# =============================================================================

def train(df):
    """Fit a random forest on every row of df (needs add_features) that has a lagged IV."""
    rows = df.dropna(subset=FEATURES)
    model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=200,    # each prediction averages >= 200 options: smooths out quote noise
        max_samples=100_000,     # each tree sees a random 100k options: keeps training fast
        n_jobs=-1,
        random_state=0,
    )
    model.fit(rows[FEATURES], rows["impl_volatility"] - rows["iv_lag"])
    return model


def predict_iv(df, model):
    """Predicted IV for every row of df (needs add_features); NaN where there is no lagged IV."""
    iv = pd.Series(np.nan, index=df.index)
    ok = df[FEATURES].notna().all(axis=1)
    iv[ok] = df.loc[ok, "iv_lag"] + model.predict(df.loc[ok, FEATURES])
    return iv.clip(lower=0.01)


def predict_chain(df, model):
    """Random forest price for every option in df, aligned with df (like heston_price_chain).
    df must include the previous pricing date, which supplies the lagged features."""
    if "iv_lag" not in df:
        df = add_features(df)
    return bs_price(df, predict_iv(df, model))


def walk_forward(df, min_train_months=2):
    """Out-of-sample prices: each month is priced by a forest trained only on earlier months.
    Returns (prices aligned with df, NaN for the first min_train_months; table of results per month)."""
    month = df["date"].dt.to_period("M")
    prices = pd.Series(np.nan, index=df.index)
    folds = []
    for m in sorted(month.unique())[min_train_months:]:
        train_rows, test = df[month < m], df[month == m]
        model = train(train_rows)
        prices[test.index] = bs_price(test, predict_iv(test, model))
        folds.append({"month": str(m), "train options": len(train_rows), "test options": len(test),
                      "RMSE rf": _rmse(prices[test.index], test["mid"]),
                      "RMSE iv_lag": _rmse(bs_price(test, test["iv_lag"]), test["mid"])})
        print(folds[-1]["month"], "done")
    return prices.to_numpy(), pd.DataFrame(folds).set_index("month")


def _rmse(pred, actual):
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    ok = ~np.isnan(pred)
    return np.sqrt(np.mean((pred[ok] - actual[ok]) ** 2))


def save_model(model, name):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{name}.joblib")
    joblib.dump({"model": model, "features": FEATURES}, path)
    return path


def load_model(name):
    saved = joblib.load(os.path.join(SAVE_DIR, f"{name}.joblib"))
    if saved["features"] != FEATURES:
        raise ValueError("Saved model used different features; retrain it")
    return saved["model"]


# =============================================================================
# 3. Test on SPX 2023 and save the model
# =============================================================================

if __name__ == "__main__":
    SECID, START, END = 108105, "2023-01-01", "2023-12-31"

    # weekly chains saved by compare_funcs.load_weekly_prices (run a comparison script first)
    df = pd.read_pickle(f"source/data/cache_weekly_chains_{SECID}_{START}_{END}.pkl")
    df = df[df["T"] > 0]
    df = add_features(df)
    print(f"{len(df)} options on {df['date'].nunique()} dates, "
          f"{df['is_new'].mean():.0%} not quoted the week before")

    df["price_rf"], folds = walk_forward(df)
    print("\nWalk-forward RMSE ($), each month priced by a forest trained on earlier months:")
    print(folds.round(2).to_string())
    tested = df.dropna(subset=["price_rf"])
    print(f"\nOverall: RF {_rmse(tested['price_rf'], tested['mid']):.2f}, "
          f"last week's IV {_rmse(bs_price(tested, tested['iv_lag']), tested['mid']):.2f}")

    model = train(df)
    print(f"\nModel trained on all of {START[:4]} saved to {save_model(model, f'rf_{SECID}')}")
    print("\nFeature importance (share of the forest's error reduction):")
    for name, imp in sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:<11s} {imp:.3f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(tested["mid"], tested["price_rf"], s=2, alpha=0.3)
    top = tested["mid"].max()
    plt.plot([0, top], [0, top], color="red", label="perfect prediction")
    plt.xlabel("Market price (mid)")
    plt.ylabel("Random forest price")
    plt.title("SPX: random forest vs market (out of sample)")
    plt.legend()
    plt.tight_layout()
    plt.show()
