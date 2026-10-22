"""
Historical (realised) volatility estimator.

Steps (as on the slide):
    1) choose lookback window n (trading days)
    2) daily log returns r_t = ln(S_t / S_{t-1}), mean r_bar (optional)
    3) daily vol = sample std of r_t over the window (n-1 denominator)
    4) annualise: sigma_annual = s_daily * sqrt(252)

Cells are marked with "# %%" so this runs as a notebook in VS Code.
"""

# %%
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns of the UNDERLYING: r_t = ln(S_t / S_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


def realised_vol(prices: pd.Series, window: int = 60, demean: bool = True) -> float:
    """
    Annualised realised volatility using the last `window` daily returns.

    demean=True : sample std around r_bar, (n-1) denominator  (formula B)
    demean=False: assume zero mean, (n) denominator           (Hull / RiskMetrics convention)
    """
    r = log_returns(prices).iloc[-window:]
    if len(r) < window:
        raise ValueError(f"Need {window + 1} prices, got {len(r) + 1}.")
    if demean:
        s_daily = r.std(ddof=1)
    else:
        s_daily = np.sqrt((r**2).mean())
    return s_daily * np.sqrt(TRADING_DAYS)


def rolling_realised_vol(prices: pd.Series, window: int = 60, demean: bool = True) -> pd.Series:
    """Same estimator, evaluated on every day (value on day t uses returns up to and including t)."""
    r = log_returns(prices)
    if demean:
        s_daily = r.rolling(window).std(ddof=1)
    else:
        s_daily = np.sqrt((r**2).rolling(window).mean())
    return (s_daily * np.sqrt(TRADING_DAYS)).rename(f"rv_{window}d")


def simulate_gbm(S0=100.0, mu=0.05, sigma=0.20, n_days=1000, seed=0) -> pd.Series:
    """GBM price path with known sigma, for checking the estimator."""
    rng = np.random.default_rng(seed)
    dt = 1 / TRADING_DAYS
    r = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_days)
    prices = S0 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    dates = pd.bdate_range("2022-01-03", periods=n_days + 1)
    return pd.Series(prices, index=dates, name="S")


# %% Sanity check on simulated data (true sigma = 20%)
S = simulate_gbm(sigma=0.20)
print(f"60-day RV (demeaned): {realised_vol(S, 60):.4f}")
print(f"60-day RV (zero mean): {realised_vol(S, 60, demean=False):.4f}")
print(f"Full-sample RV:        {realised_vol(S, len(S) - 1):.4f}   (true 0.2000)")

# %% Real data — needs `pip install yfinance` (run locally)
# import yfinance as yf
# S = yf.download("AAPL", start="2023-01-01", auto_adjust=True)["Close"].squeeze()
# print(f"AAPL 60-day RV: {realised_vol(S, 60):.4f}")

# %% Plot rolling RV for several windows
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
for w in [20, 60, 120, 250]:
    rolling_realised_vol(S, w).plot(ax=ax, lw=1.2)
ax.axhline(0.20, color="k", ls="--", lw=1, label="true σ")
ax.set_ylabel("annualised volatility")
ax.set_title("Rolling realised volatility")
ax.legend()
plt.tight_layout()
plt.show()