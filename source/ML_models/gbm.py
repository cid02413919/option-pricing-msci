"""Gradient Boosted Decision Trees (HistGradientBoosting / LightGBM) for Option Pricing.
Part of Phase 2 of the ML Options Pricing Roadmap.

Supports three core target formulations:
  1. "price_over_S" : Direct normalized price (C/S or P/S) from scratch.
  2. "residual"     : Predicts delta over Black-Scholes (Market Price - BS Lag Benchmark).
                      Leverages Black-Scholes for intrinsic value & discounting.
  3. "iv"           : Predicts market Implied Volatility directly, then prices via BS formula.
                      Guarantees put-call parity and no-arbitrage bounds by construction.
  4. "all"          : Evaluates all 3 methods sequentially and prints a side-by-side comparison!

Run from the repo root:
    python source/ML_models/gbm.py
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure source is on the python path
sys.path.append("source")
from classical_models.black_scholes.bs_funcs import black_scholes_call, black_scholes_put

# =============================================================================
# SETTINGS: Choose Method & Contract Types
# =============================================================================
# METHOD options:
#   "all"         : Run all three methods side-by-side to directly compare their errors!
#   "residual"    : Predicts (Market Mid - BS Lagged Benchmark). Recommended for best RMSE.
#   "iv"          : Predicts market Implied Volatility, then prices with Black-Scholes.
#   "price_over_S": Predicts normalized price directly from scratch.
METHOD = "all"

# CP_FILTER options:
#   "both": Evaluates both Calls and Puts (~730k quotes).
#   "C"   : Evaluates Calls only (~350k quotes) - directly matches random_forest.py!
#   "P"   : Evaluates Puts only.
CP_FILTER = "both"

# Standard feature columns used by the models
FEATURES = [
    "moneyness",    # Strike / Spot (K / S)
    "log_mF",        # ln(K / F), moneyness relative to forward price
    "T",            # Time to expiry in years
    "r",            # Risk-free rate
    "q",            # Dividend yield
    "is_call",      # 1 for Call, 0 for Put
    "iv_lag",       # Implied volatility from previous week (matched or k-d tree nearest)
    "delta_lag",    # Black-Scholes delta using last week's IV (+ for call, - for put)
    "atm_iv_lag",   # Overall market volatility regime (ATM 1-month IV last week)
    "spot_ret",     # 1-week log return of the underlying index
    "is_new",       # Indicator: 1 if option was not quoted last week (imputed via surface)
]


# =============================================================================
# 1. Feature Engineering (Strictly No Look-Ahead)
# =============================================================================

def engineer_features(raw_df, cp_filter="both"):
    """Clean the raw option chain and construct non-leaking features.

    Every feature for an option on date t is computed strictly using:
      (a) Observable contract specifications on date t (S, K, T, r, q, cp_flag).
      (b) Information available BEFORE date t (implied volatility and returns from date t-1).
    """
    df = raw_df.copy()

    # Apply call/put filter
    if cp_filter in ("C", "P"):
        df = df[df["cp_flag"] == cp_filter]

    # Drop zero bids, zero maturity, and missing values
    df = df[df["best_bid"] > 0]
    df = df[df["T"] > 0]
    df = df.dropna(subset=["S", "strike", "T", "r", "q", "mid", "impl_volatility", "cp_flag"])

    # Ensure float types
    num_cols = ["S", "strike", "T", "r", "q", "mid", "impl_volatility", "optionid"]
    for col in num_cols:
        df[col] = df[col].astype(float)
    df = df.reset_index(drop=True)

    # 1. Option type indicator
    df["is_call"] = (df["cp_flag"] == "C").astype(int)

    # 2. Moneyness measures
    df["moneyness"] = df["strike"] / df["S"]  # K / S
    F = df["S"] * np.exp((df["r"] - df["q"]) * df["T"])
    df["log_mF"] = np.log(df["strike"] / F)   # ln(K / F)

    # 3. Targets
    df["target_price_over_S"] = df["mid"] / df["S"]
    df["target_iv"] = df["impl_volatility"]

    # 4. Map each date to the previous available weekly date
    dates = np.sort(df["date"].unique())
    prev_map = dict(zip(dates[1:], dates[:-1]))
    df["prev_date"] = df["date"].map(prev_map)

    # 5. Lagged Implied Volatility (iv_lag) with nearest-neighbour surface imputation
    df["iv_lag"] = np.nan
    df["is_new"] = 0

    for date, today in df.groupby("date"):
        if date not in prev_map:
            continue
        last = df[df["date"] == prev_map[date]]

        # Direct optionid matching from previous week
        same_id_iv = today["optionid"].map(last.set_index("optionid")["impl_volatility"])

        # Separate k-d tree interpolation for calls and puts to match exact contract surface
        imputed_iv_series = pd.Series(index=today.index, dtype=float)
        for cp in today["cp_flag"].unique():
            cp_mask_today = today["cp_flag"] == cp
            last_cp = last[last["cp_flag"] == cp]
            if not last_cp.empty:
                tree = cKDTree(np.c_[last_cp["log_mF"], np.sqrt(last_cp["T"])])
                sub_today = today[cp_mask_today]
                _, nearest_idx = tree.query(np.c_[sub_today["log_mF"], np.sqrt(sub_today["T"])])
                imputed_iv_series.loc[sub_today.index] = last_cp["impl_volatility"].to_numpy()[nearest_idx]

        df.loc[today.index, "iv_lag"] = same_id_iv.fillna(imputed_iv_series)
        df.loc[today.index, "is_new"] = same_id_iv.isna().astype(int)

    # 6. Lagged Black-Scholes Delta (signed: in [0, 1] for calls, [-1, 0] for puts)
    d1 = (-df["log_mF"] + (df["iv_lag"] ** 2 / 2.0) * df["T"]) / (df["iv_lag"] * np.sqrt(df["T"]))
    call_delta = np.exp(-df["q"] * df["T"]) * norm.cdf(d1)
    put_delta = -np.exp(-df["q"] * df["T"]) * norm.cdf(-d1)
    df["delta_lag"] = np.where(df["is_call"] == 1, call_delta, put_delta)

    # 7. ATM IV of ~1-month options from last week (market fear indicator)
    atm_mask = (df["log_mF"].abs() < 0.02) & df["T"].between(20.0 / 365.0, 60.0 / 365.0)
    atm_history = df[atm_mask].groupby("date")["impl_volatility"].median()
    df["atm_iv_lag"] = df["prev_date"].map(atm_history)

    # 8. Weekly return of the underlying spot index
    spot_by_date = df.groupby("date")["S"].first()
    df["spot_ret"] = np.log(df["S"] / df["prev_date"].map(spot_by_date))

    # 9. Lagged Black-Scholes benchmark price (no model, just naive projection of last week's IV)
    bench_call = black_scholes_call(df["S"], df["strike"], df["iv_lag"], df["r"], df["T"], df["q"])
    bench_put = black_scholes_put(df["S"], df["strike"], df["iv_lag"], df["r"], df["T"], df["q"])
    df["benchmark_price"] = np.where(df["is_call"] == 1, bench_call, bench_put)

    # Residual target: dollar spread over Black-Scholes benchmark
    df["target_residual"] = df["mid"] - df["benchmark_price"]

    # Drop warm-up date (the very first date has no preceding week)
    df = df.dropna(subset=FEATURES + ["benchmark_price", "target_price_over_S", "target_residual", "target_iv"]).reset_index(drop=True)
    return df


# =============================================================================
# 2. Model Factory & Training
# =============================================================================

def build_gbm_model(random_state=0):
    """Instantiate a histogram-based gradient boosting regressor.

    Hyperparameters:
      - max_iter: 300 boosting rounds.
      - max_leaf_nodes: 63 (increased capacity to resolve local smile details).
      - min_samples_leaf: 10 (finer granularity on option surface).
      - learning_rate: 0.08.
      - l2_regularization: 0.1.
    """
    return HistGradientBoostingRegressor(
        max_iter=300,
        max_leaf_nodes=63,
        min_samples_leaf=10,
        learning_rate=0.08,
        l2_regularization=0.1,
        random_state=random_state,
    )


# =============================================================================
# 3. Walk-Forward Cross-Validation for a Given Method
# =============================================================================

def run_walk_forward_for_method(df, method, verbose=True):
    """Walk-forward monthly expanding window validation for a specific target method.

    Methods:
      - "price_over_S": Fits y = mid / S; pred = y_hat * S
      - "residual"    : Fits y = mid - benchmark_price; pred = benchmark_price + y_hat
      - "iv"          : Fits y = impl_volatility; pred = BS(S, K, y_hat, r, T, q)
    """
    target_col = {
        "price_over_S": "target_price_over_S",
        "residual": "target_residual",
        "iv": "target_iv",
    }[method]

    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")
    months = sorted(df["month"].unique())[2:]  # First two months form initial training pool

    df["predicted_price"] = np.nan
    monthly_stats = []
    last_trained_model = None

    if verbose:
        print(f"\n--- Running Method: '{method.upper()}' over {len(months)} monthly windows ---")

    for m in months:
        train_data = df[df["month"] < m]
        test_data = df[df["month"] == m]

        model = build_gbm_model(random_state=0)
        model.fit(train_data[FEATURES], train_data[target_col])
        last_trained_model = model

        # Transform model output to dollar prices
        raw_pred = model.predict(test_data[FEATURES])

        if method == "price_over_S":
            pred_dollar = np.maximum(raw_pred * test_data["S"], 0.0)
        elif method == "residual":
            pred_dollar = np.maximum(test_data["benchmark_price"] + raw_pred, 0.0)
        elif method == "iv":
            pred_iv = np.clip(raw_pred, 0.01, 3.0)
            c_price = black_scholes_call(test_data["S"], test_data["strike"], pred_iv, test_data["r"], test_data["T"], test_data["q"])
            p_price = black_scholes_put(test_data["S"], test_data["strike"], pred_iv, test_data["r"], test_data["T"], test_data["q"])
            pred_dollar = np.where(test_data["is_call"] == 1, c_price, p_price)

        df.loc[test_data.index, "predicted_price"] = pred_dollar

        actual = test_data["mid"]
        bench = test_data["benchmark_price"]
        rmse_gbm = np.sqrt(mean_squared_error(actual, pred_dollar))
        rmse_bench = np.sqrt(mean_squared_error(actual, bench))

        monthly_stats.append({
            "Month": str(m),
            "Train N": len(train_data),
            "Test N": len(test_data),
            "RMSE GBM": rmse_gbm,
            "RMSE Bench": rmse_bench,
            "Gain vs Bench (%)": (1.0 - rmse_gbm / rmse_bench) * 100.0,
        })
        if verbose:
            gain_sign = "+" if rmse_gbm <= rmse_bench else ""
            print(f"  {m} | Train: {len(train_data):<6d} | Test: {len(test_data):<5d} | "
                  f"GBM RMSE: ${rmse_gbm:6.2f} | Bench RMSE: ${rmse_bench:6.2f} | Gain: {gain_sign}{(1.0 - rmse_gbm/rmse_bench)*100.0:5.1f}%")

    stats_df = pd.DataFrame(monthly_stats)
    return df, stats_df, last_trained_model


# =============================================================================
# 4. Predict Chain Interface (compatible with compare_funcs.py)
# =============================================================================

def predict_chain(df, method="residual", model=None, model_path=None):
    """Predict option prices for an entire dataframe.

    Signature designed as a drop-in sibling to heston_price_chain(df, params_by_date).

    Parameters:
      df : DataFrame containing columns: S, strike, T, r, q, cp_flag, optionid, date, impl_volatility
      method : "residual", "iv", or "price_over_S"
      model : Trained HistGradientBoostingRegressor instance (optional).
      model_path : Path to saved .joblib model (optional, used if model is None).

    Returns:
      np.ndarray of dollar predicted prices aligned with df.index.
    """
    if model is None:
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "saved", f"gbm_{method}_model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at '{model_path}'. Train one first by running gbm.py.")
        model = joblib.load(model_path)

    prepared = engineer_features(df, cp_filter="both")
    raw_pred = model.predict(prepared[FEATURES])

    if method == "price_over_S":
        pred_dollar = np.maximum(raw_pred * prepared["S"], 0.0)
    elif method == "residual":
        pred_dollar = np.maximum(prepared["benchmark_price"] + raw_pred, 0.0)
    elif method == "iv":
        pred_iv = np.clip(raw_pred, 0.01, 3.0)
        c_price = black_scholes_call(prepared["S"], prepared["strike"], pred_iv, prepared["r"], prepared["T"], prepared["q"])
        p_price = black_scholes_put(prepared["S"], prepared["strike"], pred_iv, prepared["r"], prepared["T"], prepared["q"])
        pred_dollar = np.where(prepared["is_call"] == 1, c_price, p_price)

    out = pd.Series(np.nan, index=df.index)
    out.loc[prepared.index] = pred_dollar
    return out.to_numpy()


# =============================================================================
# 5. Main Execution
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                             "cache_weekly_chains_108105_2023-01-01_2023-12-31.pkl")

    print("=" * 78)
    print("PHASE 2: GRADIENT BOOSTED TREES (HistGradientBoosting) FOR OPTION PRICING")
    print(f"Filter: CP_FILTER = '{CP_FILTER}' | Target: METHOD = '{METHOD}'")
    print("=" * 78)

    if not os.path.exists(DATA_PATH):
        print(f"\n[ERROR] Cache file not found: {DATA_PATH}")
        print("Please generate the 2023 SPX weekly data first by running:")
        print("    python source/download_spx_cache.py\n")
        sys.exit(1)

    print(f"Loading cached option data from: {DATA_PATH}")
    raw_chains = pd.read_pickle(DATA_PATH)
    print(f"Loaded {len(raw_chains):,} total quotes.")

    print("\nEngineering features (lagging IV & returns to strictly eliminate leakage)...")
    clean_df = engineer_features(raw_chains, cp_filter=CP_FILTER)
    n_c = (clean_df['is_call'] == 1).sum()
    n_p = (clean_df['is_call'] == 0).sum()
    print(f"Cleaned feature dataset: {len(clean_df):,} options ({n_c:,} calls, {n_p:,} puts).")

    save_dir = os.path.join(os.path.dirname(__file__), "saved")
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    methods_to_run = ["price_over_S", "residual", "iv"] if METHOD == "all" else [METHOD]
    summary_results = []

    for m in methods_to_run:
        eval_df, monthly_df, trained_model = run_walk_forward_for_method(clean_df, method=m, verbose=True)

        tested = eval_df.dropna(subset=["predicted_price"])
        y_true = tested["mid"]
        y_pred = tested["predicted_price"]
        y_bench = tested["benchmark_price"]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        b_rmse = np.sqrt(mean_squared_error(y_true, y_bench))
        r2 = r2_score(y_true, y_pred)
        gain = (1.0 - rmse / b_rmse) * 100.0

        summary_results.append({
            "Method": m,
            "RMSE ($)": rmse,
            "MAE ($)": mae,
            "R2": r2,
            "Bench RMSE ($)": b_rmse,
            "Gain vs Bench (%)": gain,
        })

        # Save model
        save_path = os.path.join(save_dir, f"gbm_{m}_model.joblib")
        joblib.dump(trained_model, save_path)

        # Plot
        plot_path = os.path.join(fig_dir, f"gbm_{m}_vs_market.png")
        plt.figure(figsize=(7, 7))
        c_mask = tested["is_call"] == 1
        p_mask = tested["is_call"] == 0
        if c_mask.any():
            plt.scatter(y_true[c_mask], y_pred[c_mask], s=2, alpha=0.25, color="dodgerblue", label="Calls")
        if p_mask.any():
            plt.scatter(y_true[p_mask], y_pred[p_mask], s=2, alpha=0.25, color="darkorange", label="Puts")
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([0, max_val], [0, max_val], color="crimson", lw=1.5, linestyle="--", label="Ideal Prediction (45 deg)")
        plt.xlabel("Market Mid Price ($)")
        plt.ylabel("Predicted Price ($)")
        plt.title(f"GBM ({m}): Predicted vs Market Mid (2023 Walk-Forward)")
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

    # Final summary display
    print("\n" + "=" * 82)
    print(f"OVERALL PERFORMANCE COMPARISON (March - December 2023 Walk-Forward | Filter: {CP_FILTER})")
    print("=" * 82)
    summary_df = pd.DataFrame(summary_results).set_index("Method")
    print(summary_df.to_string(formatters={
        "RMSE ($)": "${:,.2f}".format,
        "MAE ($)": "${:,.2f}".format,
        "R2": "{:,.4f}".format,
        "Bench RMSE ($)": "${:,.2f}".format,
        "Gain vs Bench (%)": "{:+,.1f}%".format,
    }))
    print("=" * 82)
    print("All models serialized to source/ML_models/saved/")
    print("All diagnostic figures saved to figures/")
