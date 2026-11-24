"""Recreate Stage 3 (Table V) of Milyushkov (2025),
"Can Machine Learning Algorithms Outperform Traditional Models for Option Pricing?"
(arXiv:2510.01446).

What the paper does in Stage 3:
  - Underlying : Apple Inc. (AAPL, WRDS OptionMetrics secid = 101594)
  - Date       : 2025-01-02 (first trading day of 2025; paper references Jan 1/2, 2025)
  - Options    : European/American call options on one day (~1,087 filtered quotes in paper)
  - Split      : Random 80% train / 20% test within that single day
  - Models     : Black-Scholes (constant vol), Heston (calibrated on train),
                 Random Forest, HistGradientBoosting (and CatBoost if installed), MLP
  - Metrics    : MSE (primary metric in Table V), RMSE, MAE, R^2 on mid prices

Additionally, this script runs the ML models BOTH:
  (A) WITH today's `impl_volatility` as an input (reproducing Milyushkov's exact feature set)
  (B) WITHOUT `impl_volatility` (removing same-day price leakage, as in our main pipeline)

Run from the repo root:
    python source/ML_models/recreate_milyushkov_aapl.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append("source")
from classical_models.black_scholes.bs_funcs import black_scholes_call, historical_volatility
from classical_models.Heston.heston import calibrate_heston_day, heston_price_chain
from data_importing.data_import import multi_option_single_day, return_multi_day


# =============================================================================
# Settings
# =============================================================================
USERNAME = "AlexMatijasevic"
SECID = 101594           # Apple Inc. (AAPL) in OptionMetrics
DATE = "2025-01-02"      # Paper's Jan 2025 snapshot
USE_DIVIDENDS = False    # Paper uses q = 0 (set True to test with trailing AAPL dividend yield)
TEST_SIZE = 0.20         # 80/20 train-test split
RANDOM_STATE = 42

# Published Table V figures from Milyushkov (2025) for reference
PAPER_TABLE_V = {
    "Black-Scholes": {"MSE": 211.586, "MAE": 8.7308, "R2": 0.9378},
    "Heston":        {"MSE": 10.413,  "MAE": 1.7465, "R2": 0.9969},
    "Random Forest": {"MSE": 4.915,   "MAE": 1.0265, "R2": 0.9986},
    "MLP":           {"MSE": 3.195,   "MAE": 0.6553, "R2": 0.9991},
    "CatBoost":      {"MSE": 1.768,   "MAE": 0.7765, "R2": 0.9995},
}


# =============================================================================
# 1. Load & Cache Data from WRDS
# =============================================================================
def load_aapl_snapshot(username, date, secid):
    """Load AAPL option chain on `date` and 60d prior returns, caching locally."""
    data_dir = os.path.join("source", "data")
    os.makedirs(data_dir, exist_ok=True)

    chain_cache = os.path.join(data_dir, f"cache_aapl_chain_{date}.pkl")
    rtn_cache = os.path.join(data_dir, f"cache_aapl_rtn_{date}.pkl")

    if os.path.exists(chain_cache):
        chain = pd.read_pickle(chain_cache)
    else:
        print(f"Downloading AAPL ({secid}) option chain on {date} from WRDS...")
        chain = multi_option_single_day(username, date, secid=secid, save=False, calc_q=True)
        chain.to_pickle(chain_cache)

    if os.path.exists(rtn_cache):
        rtn = pd.read_pickle(rtn_cache)
    else:
        print(f"Downloading AAPL ({secid}) 60-day prior returns from WRDS...")
        rtn = return_multi_day(username, date, date, lookback=60, secid=secid, save=False)
        rtn.to_pickle(rtn_cache)

    return chain, rtn


def clean_calls(raw_chain, use_dividends=False):
    """Filter to liquid call quotes, matching Stage 3's call-only setup."""
    df = raw_chain.copy()
    df = df[df["cp_flag"] == "C"]
    df = df[df["best_bid"] > 0]
    df = df[df["best_offer"] > df["best_bid"]]
    df = df[df["T"] > 0]

    # Drop completely stale/untraded contracts (matches prepare_wrds_chain liquidity filter)
    traded = (df["volume"].fillna(0) > 0) | (df["open_interest"].fillna(0) > 0)
    df = df[traded]

    df = df.dropna(subset=["S", "strike", "T", "r", "mid", "impl_volatility"]).copy()

    num_cols = ["S", "strike", "T", "r", "mid", "impl_volatility", "days", "strike_price", "forward_price"]
    for col in num_cols:
        if col in df:
            df[col] = df[col].astype(float)

    if not use_dividends or "q" not in df:
        df["q"] = 0.0
    else:
        df["q"] = df["q"].astype(float).fillna(0.0)

    return df.reset_index(drop=True)


# =============================================================================
# 2. Evaluation Helper
# =============================================================================
def evaluate_predictions(name, y_true, y_pred, fit_time=0.0, pred_time=0.0):
    """Compute Table V metrics (MSE, RMSE, MAE, R2) plus timings."""
    return {
        "Model": name,
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Train (s)": fit_time,
        "Pred (s)": pred_time,
    }


# =============================================================================
# 3. Main Experiment
# =============================================================================
def main():
    print("=" * 84)
    print(f"RECREATING MILYUSHKOV (2025) STAGE 3 — AAPL CALLS ON {DATE}")
    print("=" * 84)

    raw_chain, rtn = load_aapl_snapshot(USERNAME, DATE, SECID)
    df = clean_calls(raw_chain, use_dividends=USE_DIVIDENDS)

    spot = float(df["S"].iloc[0])
    hist_vol = float(historical_volatility(rtn.loc[rtn["lookback"], "log_return"], window=60, method="zero_mean"))
    atm_mask = (df["strike"] / spot - 1.0).abs() < 0.05
    atm_iv = float(df.loc[atm_mask, "impl_volatility"].median())

    print(f"Filtered AAPL calls : {len(df)} contracts (paper had 1,087 after Yahoo filtering)")
    print(f"Spot S0             : ${spot:.2f}")
    print(f"60d Historical Vol  : {hist_vol:.4f} ({hist_vol * 100:.2f}%)")
    print(f"Median ATM IV       : {atm_iv:.4f} ({atm_iv * 100:.2f}%)")
    print(f"Dividend yield q    : {'market trailing' if USE_DIVIDENDS else '0.0 (matches paper)'}")

    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    y_train = train["mid"].to_numpy()
    y_test = test["mid"].to_numpy()
    print(f"Train / Test split  : {len(train)} / {len(test)} contracts\n")

    results = []

    # -------------------------------------------------------------------------
    # A. Classical Models (using existing bs_funcs.py & heston.py)
    # -------------------------------------------------------------------------
    # 1. Black-Scholes with constant 60d historical volatility (realistic constant-vol BS)
    t0 = time.perf_counter()
    bs_hist_pred = black_scholes_call(
        test["S"].to_numpy(), test["strike"].to_numpy(), hist_vol,
        test["r"].to_numpy(), test["T"].to_numpy(), test["q"].to_numpy(),
    )
    results.append(evaluate_predictions("BS (60d Hist Vol)", y_test, bs_hist_pred, 0.0, time.perf_counter() - t0))

    # 2. Black-Scholes with single chain-median ATM implied vol
    t0 = time.perf_counter()
    bs_atm_pred = black_scholes_call(
        test["S"].to_numpy(), test["strike"].to_numpy(), atm_iv,
        test["r"].to_numpy(), test["T"].to_numpy(), test["q"].to_numpy(),
    )
    results.append(evaluate_predictions("BS (Const ATM IV)", y_test, bs_atm_pred, 0.0, time.perf_counter() - t0))

    # 3. Heston (calibrated on train split across wide strike/maturity range)
    print("Calibrating Heston on training split (3 starts)...")
    t0 = time.perf_counter()
    heston_params, S0_cal, r_cal, q_cal = calibrate_heston_day(
        train, moneyness=(0.2, 3.0), max_days=1500, n_starts=3
    )
    fit_t = time.perf_counter() - t0
    print("  Calibrated Heston params:", {k: round(v, 4) for k, v in heston_params.items()})

    t0 = time.perf_counter()
    chain_date = df["date"].iloc[0]
    heston_pred = heston_price_chain(test, {chain_date: heston_params})
    pred_t = time.perf_counter() - t0

    valid_h = np.isfinite(heston_pred)
    results.append(evaluate_predictions("Heston (calibrated)", y_test[valid_h], heston_pred[valid_h], fit_t, pred_t))

    # -------------------------------------------------------------------------
    # B. ML Models WITH `impl_volatility` (Reproducing Milyushkov Table V)
    # -------------------------------------------------------------------------
    features_with_iv = ["S", "strike", "T", "r", "impl_volatility"]
    X_tr_iv, X_te_iv = train[features_with_iv].to_numpy(), test[features_with_iv].to_numpy()

    ml_models = {
        "Random Forest (with IV)": lambda: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "HistGBM (with IV)": lambda: HistGradientBoostingRegressor(
            max_iter=400, max_leaf_nodes=63, min_samples_leaf=5, learning_rate=0.05, random_state=RANDOM_STATE
        ),
        "MLP (with IV)": lambda: make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 128, 64), max_iter=1000, early_stopping=True, random_state=RANDOM_STATE),
        ),
    }

    # Optional: include CatBoost if installed in the user's environment
    try:
        from catboost import CatBoostRegressor
        ml_models["CatBoost (with IV)"] = lambda: CatBoostRegressor(
            iterations=1000, learning_rate=0.05, depth=6, verbose=0, random_seed=RANDOM_STATE
        )
    except ImportError:
        pass

    preds_for_plot = {"BS (60d Hist Vol)": bs_hist_pred, "Heston (calibrated)": heston_pred}

    for name, builder in ml_models.items():
        model = builder()
        t0 = time.perf_counter()
        model.fit(X_tr_iv, y_train)
        fit_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = np.maximum(model.predict(X_te_iv), 0.0)
        pred_t = time.perf_counter() - t0

        results.append(evaluate_predictions(name, y_test, pred, fit_t, pred_t))
        preds_for_plot[name] = pred

    # -------------------------------------------------------------------------
    # C. ML Models WITHOUT `impl_volatility` (Leak-Free Control Comparison)
    # -------------------------------------------------------------------------
    # Removes today's IV so models only see contract specs (K, T, r), just like
    # a pricer that does not already know the market price via implied vol.
    features_no_iv = ["S", "strike", "T", "r"]
    X_tr_no, X_te_no = train[features_no_iv].to_numpy(), test[features_no_iv].to_numpy()

    for name, builder in [
        ("Random Forest (no IV)", lambda: RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)),
        ("HistGBM (no IV)", lambda: HistGradientBoostingRegressor(
            max_iter=400, max_leaf_nodes=63, min_samples_leaf=5, learning_rate=0.05, random_state=RANDOM_STATE)),
        ("MLP (no IV)", lambda: make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 128, 64), max_iter=1000, early_stopping=True, random_state=RANDOM_STATE))),
    ]:
        model = builder()
        t0 = time.perf_counter()
        model.fit(X_tr_no, y_train)
        fit_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = np.maximum(model.predict(X_te_no), 0.0)
        pred_t = time.perf_counter() - t0

        results.append(evaluate_predictions(name, y_test, pred, fit_t, pred_t))

    # -------------------------------------------------------------------------
    # 4. Print Comparison Tables
    # -------------------------------------------------------------------------
    res_df = pd.DataFrame(results).set_index("Model")
    print("\n" + "=" * 84)
    print(f"OUR RESULTS ON WRDS AAPL CALLS ({DATE}, n={len(df)}, 80/20 split)")
    print("=" * 84)
    print(res_df.to_string(formatters={
        "MSE": "{:,.4f}".format,
        "RMSE": "{:,.4f}".format,
        "MAE": "{:,.4f}".format,
        "R2": "{:,.5f}".format,
        "Train (s)": "{:,.3f}".format,
        "Pred (s)": "{:,.4f}".format,
    }))

    print("\n" + "=" * 84)
    print("PUBLISHED MILYUSHKOV (2025) TABLE V REFERENCE (AAPL 2025-01-02, n=1,087)")
    print("=" * 84)
    paper_df = pd.DataFrame(PAPER_TABLE_V).T
    paper_df["RMSE"] = np.sqrt(paper_df["MSE"])
    print(paper_df[["MSE", "RMSE", "MAE", "R2"]].to_string(formatters={
        "MSE": "{:,.4f}".format,
        "RMSE": "{:,.4f}".format,
        "MAE": "{:,.4f}".format,
        "R2": "{:,.4f}".format,
    }))

    # -------------------------------------------------------------------------
    # 5. Diagnostic Plot: Predicted vs Actual Mid Price
    # -------------------------------------------------------------------------
    fig_dir = "figures"
    os.makedirs(fig_dir, exist_ok=True)
    plot_path = os.path.join(fig_dir, f"milyushkov_aapl_{DATE}.png")

    plot_keys = [k for k in ["BS (60d Hist Vol)", "Heston (calibrated)", "Random Forest (with IV)", "HistGBM (with IV)", "MLP (with IV)"] if k in preds_for_plot]
    fig, axes = plt.subplots(1, len(plot_keys), figsize=(4 * len(plot_keys), 4), sharex=True, sharey=True)
    max_p = float(y_test.max())

    for ax, label in zip(axes, plot_keys):
        pred = preds_for_plot[label]
        mse = mean_squared_error(y_test[np.isfinite(pred)], pred[np.isfinite(pred)])
        ax.scatter(y_test, pred, s=8, alpha=0.5)
        ax.plot([0, max_p], [0, max_p], "r--", lw=1)
        ax.set_title(f"{label}\nMSE = {mse:.2f}", fontsize=10)
        ax.set_xlabel("Market Mid ($)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Predicted Price ($)")
    fig.suptitle(f"Recreation of Milyushkov (2025) Stage 3 — AAPL Calls on {DATE}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved comparison plot to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
