#%%
"""Overall error comparison across ALL classical and ML models (extending error_overall.py).

Supports Black-Scholes (implied, historical, GARCH, VIX, lagged IV), Heston,
Random Forest, Gradient Boosted Trees (GBM), Standard MLP, LSTM, and GAN.
ML models are evaluated walk-forward (train on earlier months, test on next month)
and scored on the exact same test options as the classical models.

Run from the repo root:
    python source/classical_models/comparing_models/error_overall_all.py
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

sys.path.append("source")  # run from the repo root

from classical_models.comparing_models.compare_funcs import (
    LABELS, SPX, chosen, ask_secid, exercise_style, ticker, load_weekly_prices, overall_errors)
from ML_models.gbm import FEATURES, build_gbm_model, engineer_features
from ML_models.neural_nets import (
    ML_LABELS, decode_predictions, predict_nn_model, target_values, train_nn_model)

# ---- Settings ----
username = "AlexMatijasevic"
start, end = "2023-01-01", "2023-12-31"

SHOW_ALL_ERRORS = False   # True also computes median absolute error and MAPE
ML_TARGET = "iv"          # "iv", "residual", or "price_over_S"
NN_EPOCHS = 12            # epochs per walk-forward month for NN / LSTM / GAN

# ---- Switch any model on (True) or off (False) ----
METHODS = {
    "implied":     False,   # Black-Scholes, same-day implied vol (not a forecast)
    "hist_vol_zm": True,    # Black-Scholes, zero-mean 60-day historical vol
    "hist_vol_dm": False,   # Black-Scholes, demeaned 60-day historical vol
    "garch_vol":   False,   # Black-Scholes, GJR-GARCH vol forecast
    "vix_dec":     True,    # Black-Scholes, lagged VIX
    "lagged_iv":   True,    # Black-Scholes, previous trading day's implied vol
    "heston":      True,    # Heston, calibrated on previous trading day
    "rf":          True,    # Random Forest (walk-forward)
    "gbm":         True,    # Gradient Boosted Trees (walk-forward)
    "nn":          True,    # Standard MLP Neural Network (walk-forward)
    "lstm":        True,    # LSTM Neural Network (walk-forward)
    "gan":         True,    # GAN Generator-Assessor NN (walk-forward)
}
methods = chosen(METHODS)
classical = [m for m in methods if m in LABELS]
ml = [m for m in methods if m in ML_LABELS]
ALL_LABELS = {**LABELS, **ML_LABELS}

secid = ask_secid()
name = ticker(secid)
exercise = exercise_style(secid)
print(f"\n{name}, {start} to {end}, {exercise} options")
if "vix_dec" in methods and secid != SPX:
    print("Note: VIX is S&P 500 vol, so the VIX method is only a rough guide here")

#%%
# ---- Classical models via existing pipeline ----
df = load_weekly_prices(username, secid, start, end, classical, exercise=exercise)

# ---- ML models via expanding-window monthly walk-forward ----
if ml:
    # Tag original row indices because engineer_features drops warm-up rows and resets index
    feat_df = engineer_features(df.assign(_orig_idx=df.index))
    feat_df["month"] = feat_df["date"].dt.to_period("M")
    all_months = sorted(feat_df["month"].unique())
    if len(all_months) < 3:
        raise ValueError("ML walk-forward needs at least 3 months (2 warm-up months + >=1 test month).")
    months = all_months[2:]

    for m in ml:
        df[f"price_{m}"] = np.nan

    for month in months:
        train = feat_df[feat_df["month"] < month]
        test = feat_df[feat_df["month"] == month]
        for m in ml:
            if m == "rf":
                model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=0)
                model.fit(train[FEATURES], target_values(train, ML_TARGET))
                preds = decode_predictions(test, model.predict(test[FEATURES]), ML_TARGET)
            elif m == "gbm":
                model = build_gbm_model(random_state=0)
                model.fit(train[FEATURES], target_values(train, ML_TARGET))
                preds = decode_predictions(test, model.predict(test[FEATURES]), ML_TARGET)
            else:
                bundle = train_nn_model(m, train, target_method=ML_TARGET, epochs=NN_EPOCHS)
                preds = predict_nn_model(bundle, test, history_df=train)

            df.loc[test["_orig_idx"].to_numpy(), f"price_{m}"] = preds
        print(f"ML walk-forward: {month} done")

    # Compare all classical and ML models on the exact same evaluated test options
    df = df.dropna(subset=[f"price_{m}" for m in methods])

#%%
# ---- Results ----
errors = overall_errors(df, methods, SHOW_ALL_ERRORS)
names = {"rmse": "RMSE ($)", "mdae": "Median abs error ($)", "mape": "MAPE (%)"}
results = pd.DataFrame({names[k]: v for k, v in errors.items()}).rename(index=ALL_LABELS)
div_note = ", no dividend before expiry, q = 0" if exercise == "american" else ""
print(f"\nOverall pricing error, {name}, {start} to {end} (weekly, Wednesdays{div_note})\n")
print(results.to_string(float_format="%.3f"))
print(f"\n{len(df)} option quotes on {df['date'].nunique()} days")

#%%
# ---- Residual histograms (model price - market price) ----
for m in methods:
    residuals = (df[f"price_{m}"] - df["mid"]).dropna()
    limit = residuals.abs().max()
    bins = np.linspace(-limit, limit, 51)

    plt.figure()
    plt.hist(residuals, bins=bins)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Residual: model price - market price ($)")
    plt.ylabel("Frequency")
    plt.title(f"{ALL_LABELS[m]}, {name}")
    plt.show()

# %%
