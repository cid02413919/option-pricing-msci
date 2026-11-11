import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.special import gammaln


# =============================================================================
# 1. Black–Scholes prices
# =============================================================================

def _d1_d2(S, K, sigma, r, t, q):
    d1 = (np.log(S / K) + (r - q + sigma**2 / 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return d1, d2


def black_scholes_call(S, K, sigma, r, t, q=0.0):
    d1, d2 = _d1_d2(S, K, sigma, r, t, q)
    return S * np.exp(-q * t) * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)


def black_scholes_put(S, K, sigma, r, t, q=0.0):
    d1, d2 = _d1_d2(S, K, sigma, r, t, q)
    return K * np.exp(-r * t) * norm.cdf(-d2) - S * np.exp(-q * t) * norm.cdf(-d1)


# =============================================================================
# 2. Historical volatility (annualised std of daily log returns)
# =============================================================================
# method = 'zero_mean' -> mean of r^2         (assumes daily mean return is 0)
# method = 'demeaned'  -> sample variance     (subtracts the window mean)

def historical_volatility(log_returns, window=60, annualization=252, method="zero_mean"):
    """One number: vol over the most recent `window` days."""
    recent = np.asarray(log_returns, dtype=float)[-window:]
    if len(recent) < window:
        raise ValueError(f"Need at least {window} returns, got {len(recent)}")
    if method == "zero_mean":
        daily_var = np.mean(recent**2)
    elif method == "demeaned":
        daily_var = np.var(recent, ddof=1)
    else:
        raise ValueError("method must be 'zero_mean' or 'demeaned'")
    return np.sqrt(daily_var * annualization)


def historical_volatility_series(log_returns, window=60, annualization=252, method="zero_mean"):
    """Rolling version: one vol per day."""
    log_returns = pd.Series(log_returns, dtype=float)
    if method == "zero_mean":
        daily_var = (log_returns**2).rolling(window).mean()
    elif method == "demeaned":
        daily_var = log_returns.rolling(window).var(ddof=1)
    else:
        raise ValueError("method must be 'zero_mean' or 'demeaned'")
    return np.sqrt(daily_var * annualization)


# =============================================================================
# 3. GARCH(1,1) / GJR-GARCH volatility forecast
# =============================================================================
# sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2 [+ gamma*eps_{t-1}^2*1{eps_{t-1}<0}]
# Stationary if alpha + beta + 0.5*gamma < 1.

def fit_garch(returns, model="garch", dist="normal"):
    """Fit by maximum likelihood. model: 'garch' or 'gjr-garch'. dist: 'normal' or 't'."""
    r = np.asarray(returns, dtype=float)
    r = r - r.mean()
    n = len(r)
    var0 = np.var(r)

    def unpack(x):
        omega, alpha, beta = x[:3]
        gamma = x[3] if model == "gjr-garch" else 0.0
        nu = x[-1] if dist == "t" else None
        return omega, alpha, beta, gamma, nu

    def variance_path(omega, alpha, beta, gamma):
        sig2 = np.empty(n)
        sig2[0] = var0
        for t in range(1, n):
            sig2[t] = (omega + alpha * r[t-1]**2 + beta * sig2[t-1]
                       + gamma * r[t-1]**2 * (r[t-1] < 0))
        return sig2

    def neg_loglik(x):
        omega, alpha, beta, gamma, nu = unpack(x)
        if omega <= 0 or min(alpha, beta, gamma) < 0 or alpha + beta + 0.5 * gamma >= 1:
            return 1e10
        sig2 = np.maximum(variance_path(omega, alpha, beta, gamma), 1e-12)
        if dist == "normal":
            ll = -0.5 * np.sum(np.log(2 * np.pi * sig2) + r**2 / sig2)
        else:
            if nu <= 2.0001:
                return 1e10
            c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log((nu - 2) * np.pi)
            ll = np.sum(c - 0.5 * np.log(sig2) - (nu + 1) / 2 * np.log(1 + r**2 / ((nu - 2) * sig2)))
        return -ll

    x0 = [var0 * 0.05, 0.05, 0.90] if model == "garch" else [var0 * 0.05, 0.03, 0.90, 0.05]
    if dist == "t":
        x0 += [8.0]

    res = minimize(neg_loglik, x0, method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})
    omega, alpha, beta, gamma, nu = unpack(res.x)
    sig2 = variance_path(omega, alpha, beta, gamma)

    return dict(omega=omega, alpha=alpha, beta=beta, gamma=gamma, nu=nu,
                last_var=sig2[-1], last_resid=r[-1], loglik=-res.fun)


def garch_forecast_vol(params, horizon_days, annualization=252):
    """Average forecast variance over the next `horizon_days`, annualised."""
    omega, alpha, beta, gamma = params["omega"], params["alpha"], params["beta"], params["gamma"]
    sig2_t, eps_t = params["last_var"], params["last_resid"]

    sig2_next = omega + alpha * eps_t**2 + beta * sig2_t + gamma * eps_t**2 * (eps_t < 0) * 0.5
    persistence = alpha + beta + 0.5 * gamma
    uncond_var = omega / max(1e-10, 1 - persistence)

    h = np.arange(horizon_days)
    path = uncond_var + persistence**h * (sig2_next - uncond_var)   # mean-reverts to long-run var
    return np.sqrt(path.mean() * annualization)


def forecast_realized_vol_garch(log_returns, horizon_days, model="gjr-garch",
                                dist="normal", annualization=252):
    """Log returns in -> (forecast vol, fitted params) out."""
    params = fit_garch(log_returns, model=model, dist=dist)
    return garch_forecast_vol(params, horizon_days, annualization), params


def garch_vol_series(dates, days, log_returns, window=252, model="gjr-garch", annualization=252, min_obs=100):
    """GARCH vol for each option, given its pricing date and calendar days to expiry.

    log_returns : pd.Series of daily log returns indexed by date.
    GARCH is re-fit once per date on the `window` returns BEFORE that date (no look-ahead),
    then forecast over each option's remaining life (calendar days -> trading days).
    Returns an array aligned with `dates` (NaN where there are fewer than `min_obs` past returns).
    """
    log_returns = log_returns.dropna()
    dates = np.asarray(pd.to_datetime(dates))
    days = np.asarray(days)
    out = np.full(len(dates), np.nan)
    for d in np.unique(dates):
        past = log_returns[log_returns.index < d].to_numpy()[-window:]
        if len(past) < min_obs:
            continue
        params = fit_garch(past, model=model)
        rows = dates == d
        for h in np.unique(days[rows]):
            horizon = max(1, round(h * annualization / 365))
            out[rows & (days == h)] = garch_forecast_vol(params, horizon, annualization)
    return out
