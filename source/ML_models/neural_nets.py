"""Neural Network option pricers: Standard MLP, LSTM, and GAN (Generator-Assessor NN).

Models:
  1. StandardNN : Plain 3-layer MLP mapping option + lagged market features -> target.
  2. LSTMNet    : Reads the last `seq_len` days of market-regime context (ATM IV,
                  spot return, rate) via an LSTM, then combines that with option features.
  3. GAN        : Generator-Assessor ("GAR") adversarial pair. Generator predicts the
                  target given features + noise z; Assessor judges real vs. generated
                  (features, target) pairs. Inference sets z = 0 for deterministic prices.

Target modes (`target_method`, matching gbm.py / random_forest.py):
  - "iv"           : learns implied vol adjustment (target_iv - iv_lag), decoded via BS.
  - "residual"     : learns normalized BS correction (mid - bs_bench) / S.
  - "price_over_S" : learns normalized option price mid / S directly.

Run from the repo root:
    python source/ML_models/neural_nets.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

if "source" not in sys.path:
    sys.path.append("source")

from classical_models.black_scholes.bs_funcs import black_scholes_call, black_scholes_put
from ML_models.gbm import FEATURES, engineer_features

# Shared labels and plot colours for comparison scripts
ML_LABELS = {
    "rf":   "Random Forest",
    "gbm":  "Gradient Boosted Trees (GBM)",
    "nn":   "Neural Network (MLP)",
    "lstm": "LSTM Neural Network",
    "gan":  "GAN (Generator–Assessor NN)",
}
ML_COLOURS = {
    "rf":   "olive",
    "gbm":  "goldenrod",
    "nn":   "navy",
    "lstm": "crimson",
    "gan":  "magenta",
}

SEQ_COLS = ["atm_iv_lag", "spot_ret", "r"]


# =============================================================================
# 1. Target encoding / decoding & sequence builder
# =============================================================================

def target_values(df, target_method="iv"):
    """Training target vector y for the chosen formulation."""
    if target_method == "iv":
        return (df["target_iv"] - df["iv_lag"]).to_numpy(float)
    if target_method == "residual":
        return ((df["mid"] - df["benchmark_price"]) / df["S"]).to_numpy(float)
    if target_method == "price_over_S":
        return (df["mid"] / df["S"]).to_numpy(float)
    raise ValueError(f"Unknown target_method: {target_method!r}")


def decode_predictions(df, raw_pred, target_method="iv"):
    """Convert raw model predictions back into dollar option prices."""
    raw_pred = np.asarray(raw_pred, dtype=float)
    S = df["S"].to_numpy(float)
    if target_method == "price_over_S":
        return np.maximum(raw_pred * S, 0.0)
    if target_method == "residual":
        return np.maximum(df["benchmark_price"].to_numpy(float) + raw_pred * S, 0.0)
    if target_method == "iv":
        pred_iv = np.clip(df["iv_lag"].to_numpy(float) + raw_pred, 0.01, 3.0)
        K, r, T, q = (df[c].to_numpy(float) for c in ("strike", "r", "T", "q"))
        call = black_scholes_call(S, K, pred_iv, r, T, q)
        put = black_scholes_put(S, K, pred_iv, r, T, q)
        return np.where(df["is_call"].to_numpy() == 1, call, put)
    raise ValueError(f"Unknown target_method: {target_method!r}")


def build_market_sequences(df, seq_len=5, history_df=None):
    """Build (N, seq_len, 3) market-regime sequences per row using only dates <= row date."""
    mkt = df.groupby("date")[SEQ_COLS].median()
    if history_df is not None and not history_df.empty:
        hist = history_df[history_df["date"] < df["date"].min()].groupby("date")[SEQ_COLS].median()
        mkt = pd.concat([hist, mkt])
    mkt = mkt.sort_index()

    vals = mkt.to_numpy(np.float32)
    pad = np.repeat(vals[:1], seq_len - 1, axis=0)
    padded = np.vstack([pad, vals])
    windows = np.stack([padded[i:i + seq_len] for i in range(len(mkt))])  # (D, seq_len, 3)

    date_idx = {d: i for i, d in enumerate(mkt.index)}
    return windows[df["date"].map(date_idx).to_numpy()]


# =============================================================================
# 2. PyTorch architectures
# =============================================================================

class StandardNN(nn.Module):
    """Plain Multi-Layer Perceptron with smooth SiLU activations."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LSTMNet(nn.Module):
    """LSTM over recent market-regime days + dense option-feature head."""
    def __init__(self, in_dim, seq_dim=len(SEQ_COLS), hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(in_dim + hidden, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, seq):
        _, (h, _) = self.lstm(seq)
        return self.head(torch.cat([x, h[-1]], dim=-1)).squeeze(-1)


class GANGenerator(nn.Module):
    """Generator: maps standardized option features + latent noise z -> target."""
    def __init__(self, in_dim, noise_dim=4):
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim + noise_dim, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, z):
        return self.net(torch.cat([x, z], dim=-1)).squeeze(-1)


class GANAssessor(nn.Module):
    """Assessor (Discriminator): scores whether (features, target) looks real or generated."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 1, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y.unsqueeze(-1)], dim=-1)).squeeze(-1)


# =============================================================================
# 3. Training & prediction API
# =============================================================================

def train_nn_model(kind, train_df, target_method="iv", epochs=12, batch_size=2048,
                   seq_len=5, lr=1e-3, seed=0):
    """Fit 'nn', 'lstm', or 'gan' on `train_df` with standardized inputs and targets."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_scaler = StandardScaler().fit(train_df[FEATURES].to_numpy(float))
    y_scaler = StandardScaler().fit(target_values(train_df, target_method).reshape(-1, 1))

    X = torch.tensor(x_scaler.transform(train_df[FEATURES].to_numpy(float)), dtype=torch.float32, device=device)
    y = torch.tensor(y_scaler.transform(target_values(train_df, target_method).reshape(-1, 1)).ravel(),
                     dtype=torch.float32, device=device)

    seq_scaler, seq_t = None, None
    if kind == "lstm":
        seq_raw = build_market_sequences(train_df, seq_len=seq_len)
        seq_scaler = StandardScaler().fit(seq_raw.reshape(-1, len(SEQ_COLS)))
        seq_norm = seq_scaler.transform(seq_raw.reshape(-1, len(SEQ_COLS))).reshape(seq_raw.shape)
        seq_t = torch.tensor(seq_norm, dtype=torch.float32, device=device)

    in_dim = len(FEATURES)
    mse = nn.MSELoss()

    if kind == "nn":
        model = StandardNN(in_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif kind == "lstm":
        model = LSTMNet(in_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif kind == "gan":
        model = GANGenerator(in_dim).to(device)
        assessor = GANAssessor(in_dim).to(device)
        opt_g = torch.optim.Adam(model.parameters(), lr=lr)
        opt_d = torch.optim.Adam(assessor.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown model kind: {kind!r}")

    n = len(X)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X[idx], y[idx]

            if kind == "gan":
                # 1) Assessor learns to separate real (x, y) from Generator (x, fake_y)
                with torch.no_grad():
                    fake_d = model(xb, torch.randn(len(xb), model.noise_dim, device=device))
                opt_d.zero_grad()
                loss_d = 0.5 * (bce(assessor(xb, yb), torch.ones_like(yb))
                                + bce(assessor(xb, fake_d), torch.zeros_like(yb)))
                loss_d.backward()
                opt_d.step()

                # 2) Generator matches target (MSE) while fooling the Assessor
                opt_g.zero_grad()
                fake_g = model(xb, torch.randn(len(xb), model.noise_dim, device=device))
                loss_g = mse(fake_g, yb) + 0.01 * bce(assessor(xb, fake_g), torch.ones_like(yb))
                loss_g.backward()
                opt_g.step()
            else:
                opt.zero_grad()
                pred = model(xb, seq_t[idx]) if kind == "lstm" else model(xb)
                mse(pred, yb).backward()
                opt.step()

    model.eval()
    return {
        "kind": kind,
        "target_method": target_method,
        "model": model,
        "x_scaler": x_scaler,
        "seq_scaler": seq_scaler,
        "y_scaler": y_scaler,
        "seq_len": seq_len,
    }


def predict_nn_model(bundle, test_df, history_df=None):
    """Predict dollar option prices on `test_df` using a trained bundle from train_nn_model."""
    kind = bundle["kind"]
    model = bundle["model"]
    device = next(model.parameters()).device

    X = torch.tensor(bundle["x_scaler"].transform(test_df[FEATURES].to_numpy(float)),
                     dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        if kind == "lstm":
            seq_raw = build_market_sequences(test_df, seq_len=bundle["seq_len"], history_df=history_df)
            seq_norm = bundle["seq_scaler"].transform(seq_raw.reshape(-1, len(SEQ_COLS))).reshape(seq_raw.shape)
            seq_t = torch.tensor(seq_norm, dtype=torch.float32, device=device)
            pred_scaled = model(X, seq_t)
        elif kind == "gan":
            z = torch.zeros(len(X), model.noise_dim, device=device)   # deterministic z=0 at inference
            pred_scaled = model(X, z)
        else:
            pred_scaled = model(X)

    raw_pred = bundle["y_scaler"].inverse_transform(pred_scaled.cpu().numpy().reshape(-1, 1)).ravel()
    return decode_predictions(test_df, raw_pred, bundle["target_method"])


# =============================================================================
# 4. Standalone walk-forward test (matching random_forest.py)
# =============================================================================

if __name__ == "__main__":
    MODELS = {"nn": True, "lstm": True, "gan": True}
    TARGET_METHOD = "iv"   # "iv", "residual", or "price_over_S"
    CP_FILTER = "both"     # "both", "C", or "P"
    EPOCHS = 12

    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                             "cache_weekly_chains_108105_2023-01-01_2023-12-31.pkl")
    if not os.path.exists(DATA_PATH):
        print(f"Cache file not found: {DATA_PATH}\nRun: python source/download_spx_cache.py")
        sys.exit(1)

    raw_df = pd.read_pickle(DATA_PATH)
    df = engineer_features(raw_df, cp_filter=CP_FILTER)
    df["month"] = df["date"].dt.to_period("M")
    months = sorted(df["month"].unique())[2:]   # Jan-Feb form initial training window

    active = [k for k, on in MODELS.items() if on]
    monthly_rmse = pd.DataFrame(index=[str(m) for m in months])
    last_bundles = {}

    for kind in active:
        print(f"\n---- Training {ML_LABELS[kind]} (target={TARGET_METHOD}) ----")
        df[f"price_{kind}"] = np.nan
        for m in months:
            train = df[df["month"] < m]
            test = df[df["month"] == m]
            bundle = train_nn_model(kind, train, target_method=TARGET_METHOD, epochs=EPOCHS)
            df.loc[test.index, f"price_{kind}"] = predict_nn_model(bundle, test, history_df=train)
            last_bundles[kind] = bundle
            rmse = np.sqrt(mean_squared_error(test["mid"], df.loc[test.index, f"price_{kind}"]))
            monthly_rmse.loc[str(m), kind] = rmse
            print(f"  {m} | RMSE: ${rmse:.2f}")

    tested = df[df["month"].isin(months)]
    monthly_rmse["bs_lag"] = [
        np.sqrt(mean_squared_error(df.loc[df["month"] == m, "mid"],
                                   df.loc[df["month"] == m, "benchmark_price"]))
        for m in months
    ]
    print("\nMonthly RMSE ($):\n", monthly_rmse.round(2).to_string())

    summary = {
        k: {
            "RMSE ($)": np.sqrt(mean_squared_error(tested["mid"], tested[f"price_{k}"])),
            "R2": r2_score(tested["mid"], tested[f"price_{k}"]),
        }
        for k in active
    }
    summary["bs_lag"] = {
        "RMSE ($)": np.sqrt(mean_squared_error(tested["mid"], tested["benchmark_price"])),
        "R2": r2_score(tested["mid"], tested["benchmark_price"]),
    }
    print("\nOverall summary:\n", pd.DataFrame(summary).T.round(4).to_string())

    save_dir = os.path.join(os.path.dirname(__file__), "saved")
    os.makedirs(save_dir, exist_ok=True)
    for kind, bundle in last_bundles.items():
        bundle["model"] = bundle["model"].cpu()
        torch.save(bundle, os.path.join(save_dir, f"{kind}_{TARGET_METHOD}_model.pt"))

    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(1, len(active), figsize=(5 * len(active), 5), squeeze=False)
    max_p = float(tested["mid"].max())
    for ax, kind in zip(axes[0], active):
        ax.scatter(tested["mid"], tested[f"price_{kind}"], s=2, alpha=0.25, color=ML_COLOURS[kind])
        ax.plot([0, max_p], [0, max_p], "r--", lw=1)
        ax.set_title(ML_LABELS[kind])
        ax.set_xlabel("Market mid ($)")
        ax.set_ylabel("Predicted price ($)")
    fig.suptitle(f"SPX Neural Nets vs Market (target={TARGET_METHOD}, Mar–Dec 2023)")
    fig.tight_layout()
    fig.savefig(f"figures/neural_nets_{TARGET_METHOD}_vs_market.png", dpi=300)
    plt.show()
