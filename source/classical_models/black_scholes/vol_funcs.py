#%%

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from scipy.special import gammaln

#%%

# ===========================================================================
# 1. HISTORICAL / REALIZED VOLATILITY  (physical measure)
# ===========================================================================
# Recipe (matches your notes exactly):
#   1. Pick a lookback window (e.g. 60 trading days to forecast a 30-day option)
#   2. Compute the SAMPLE STANDARD DEVIATION of those daily log returns
#   3. Annualize: multiply by sqrt(annualization_factor), 252 trading days/yr
#
# Strengths: simple, transparent, needs only a price history (no option data).
# Weaknesses: backward-looking (assumes the past window is representative of
# the option's future life), window length is arbitrary (recent 60 days vs
# 252 days can give very different answers, especially right after a vol
# spike or calm period), equal-weights every day in the window (an event-
# driven %move 59 days ago counts as much as one from yesterday -- an EWMA
# or GARCH, below, addresses this), and assumes "similar market conditions"
# persist, which is exactly the assumption that breaks during regime changes
# -- the scenario your adaptive selector is meant to catch.


def historical_volatility(log_returns, window=60, annualization=252, method='zero_mean'):
    """Single number: annualized realized vol over the most recent `window`
    """
    log_returns = np.asarray(log_returns, dtype=float)
    if len(log_returns) < window:
        raise ValueError(f"Need at least {window} returns, got {len(log_returns)}")
    recent = log_returns[-window:]
    if method == 'zero_mean':
        daily_var = np.mean(recent**2)
    elif method == 'demeaned':
        daily_var = np.sum((recent - recent.mean())**2) / (len(recent) - 1)
    else:
        raise ValueError("method must be 'zero_mean' or 'demeaned'")
    return np.sqrt(daily_var * annualization)            # step 4


def historical_volatility_series(log_returns, window=60, annualization=252, method='zero_mean'):
    """Same as above but returns a full rolling TIME SERIES (one vol estimate
    per day)
    """
    log_returns = pd.Series(log_returns, dtype=float)
    if method == 'zero_mean':
        daily_var = (log_returns**2).rolling(window).mean()
    elif method == 'demeaned':
        daily_var = log_returns.rolling(window).var(ddof=1)
    else:
        raise ValueError("method must be 'zero_mean' or 'demeaned'")
    return np.sqrt(daily_var * annualization)

#%%

# ===========================================================================
# 2. IMPLIED VOLATILITY  (risk-neutral measure)
# ===========================================================================
# C_BS(S0, K, T, r, q, sigma_imp) = C_market  --> solve for sigma_imp.
# There's no closed-form inverse of Black-Scholes for sigma, so we solve it
# numerically: given a market price, find the sigma that makes the BS
# formula reproduce that price, using Brent's method (a robust 1D root
# finder -- bisection-like but faster, and it won't diverge the way
# Newton's method sometimes does near-expiry/deep-ITM where BS vega is tiny).
#
# Why build this yourself when WRDS already gives 'impl_volatility'? You need an IV inverter for YOUR model's own output
# prices (Heston, RF, NN) so you can compare everything on the same "vol"
# footing as the market smile, rather than comparing raw dollar prices which
# are dominated by deep-ITM/OTM scale effects.


def black_scholes_price(S, K, sigma, r, T, q=0.0, cp_flag='C'):
    """Black-Scholes price with a continuous dividend yield q (set q=0 to
    match your original black_scholes_call). Handles the sigma->0 / T->0
    edge case by falling back to intrinsic value (avoids div-by-zero)."""
    if sigma <= 0 or T <= 0:
        intrinsic = max(S*np.exp(-q*T) - K*np.exp(-r*T), 0.0)
        return intrinsic if cp_flag == 'C' else intrinsic - S*np.exp(-q*T) + K*np.exp(-r*T)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if cp_flag == 'C':
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)


def implied_volatility(price, S, K, T, r, q=0.0, cp_flag='C', lo=1e-4, hi=5.0):
    """Invert black_scholes_price for sigma given an observed market price.
    Returns np.nan if the price is below intrinsic value (a data error --
    e.g. crossed quotes) or if no root exists in [lo, hi] (5.0 = 500% vol,
    generous enough for virtually any real SPX quote)."""
    intrinsic = max(S*np.exp(-q*T) - K*np.exp(-r*T), 0.0) if cp_flag == 'C' \
        else max(K*np.exp(-r*T) - S*np.exp(-q*T), 0.0)
    if price <= intrinsic + 1e-10:
        return np.nan
    f = lambda sig: black_scholes_price(S, K, sig, r, T, q, cp_flag) - price
    try:
        if f(lo) * f(hi) > 0:          # no sign change -> Brent can't bracket a root
            return np.nan
        return brentq(f, lo, hi, xtol=1e-8, maxiter=200)
    except Exception:
        return np.nan


def add_implied_vol_column(df, spot_col='S', strike_col='K', T_col='T',
                            r_col='r', mid_col='mid', cp_col='cp_flag', q=0.0):
    """Vectorised-by-row helper: adds an 'iv_solved' column to a WRDS-shaped
    chain (after prepare_wrds_chain), useful to cross-check against the
    WRDS-supplied 'impl_volatility' column directly, row by row."""
    out = df.copy()
    out['iv_solved'] = [
        implied_volatility(row[mid_col], row[spot_col], row[strike_col],
                            row[T_col], row[r_col], q, row[cp_col])
        for _, row in out.iterrows()
    ]
    return out

#%%

# ===========================================================================
# 3. GARCH(1,1) / GJR-GARCH(1,1,1)  (physical measure)
# ===========================================================================
# Model:
#   r_t      = mu + eps_t                       (demeaned daily log return)
#   eps_t    = sigma_t * z_t,     z_t ~ N(0,1) or Student-t(nu)
#   sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2        (GARCH)
#   sigma_t^2 = omega + alpha*eps_{t-1}^2 + beta*sigma_{t-1}^2
#               + gamma*eps_{t-1}^2*1{eps_{t-1}<0}                   (GJR, leverage)
#
#   omega : long-run positive baseline (keeps variance from collapsing to 0)
#   alpha : how much yesterday's "surprise" (squared shock) feeds into today's
#           variance -- the ARCH/"shock" term
#   beta  : how much yesterday's variance persists into today -- the
#           "clustering/memory" term
#   gamma : (GJR only) extra kick to variance when yesterday's shock was
#           NEGATIVE -- captures the equity "leverage effect" (down moves
#           raise vol more than up moves of the same size), which plain
#           GARCH(1,1) cannot see since it only uses eps^2 (sign-blind).
#
# Stationarity condition: alpha + beta + 0.5*gamma < 1. If satisfied, there
# is a finite long-run variance omega/(1 - alpha - beta - 0.5*gamma) that
# the forecast mean-reverts toward -- shocks fade out gradually rather than
# compounding forever, matching real markets (this is enforced as a hard
# constraint in the optimizer below).
#
#   1. Use a long history of adjusted close prices (avoid fake shocks from
#      unadjusted splits/dividends).
#   2. Compute daily log returns.
#   3. Fit (omega, alpha, beta[, gamma]) by MAXIMUM LIKELIHOOD, assuming
#      z_t ~ Normal or Student-t (fat tails -- more realistic for equities,
#      selectable via dist='t').
#   4. Use the most recent fitted variance + the last return's residual as
#      the STARTING POINT for forecasting forward (there's no more data
#      after "today", so the model has to roll its own forecasted variance
#      forward recursively -- see garch_forecast_vol).
#   5. Average the forecasted daily variances over the option's remaining
#      life (H days), then annualize -- this gives a SINGLE number
#      compatible with Black-Scholes' constant-vol assumption.
#
# Weaknesses (from your notes): plain GARCH(1,1) treats +/- shocks
# symmetrically, but equity markets exhibit a leverage effect (down moves
# raise vol more) -- use model='gjr-garch' (implemented below) or EGARCH
# (models log-variance, not implemented here) to capture that asymmetry.

def fit_garch(returns, model='garch', dist='normal'):
    """Fit GARCH(1,1) or GJR-GARCH(1,1,1) by maximum likelihood on a 1D
    array of (already demeaned, or not -- we demean internally) daily log
    returns. Returns a dict of fitted parameters plus the final day's
    conditional variance/residual, which garch_forecast_vol() needs to
    roll the forecast forward."""
    r = np.asarray(returns, dtype=float)
    r = r - r.mean()                      # step 1: constant mean model (mu=0 after demeaning)
    n = len(r)
    var0 = np.var(r)                      # used both as x0 seed and as sigma_0^2

    def unpack(x):
        if model == 'garch':
            omega, alpha, beta = x[:3]
            gamma = 0.0
        else:  # 'gjr-garch'
            omega, alpha, beta, gamma = x[:4]
        nu = x[-1] if dist == 't' else None
        return omega, alpha, beta, gamma, nu

    def neg_loglik(x):
        omega, alpha, beta, gamma, nu = unpack(x)
        # Hard-enforce positivity + stationarity (alpha+beta+0.5*gamma < 1):
        if omega <= 0 or alpha < 0 or beta < 0 or gamma < 0 or alpha + beta + 0.5*gamma >= 1:
            return 1e10
        sig2 = np.empty(n)
        sig2[0] = var0
        for t in range(1, n):             # recursive variance filter, step 3
            lev = gamma * r[t-1]**2 * (r[t-1] < 0)
            sig2[t] = omega + alpha*r[t-1]**2 + beta*sig2[t-1] + lev
        sig2 = np.maximum(sig2, 1e-12)
        if dist == 'normal':
            ll = -0.5*np.sum(np.log(2*np.pi*sig2) + r**2/sig2)
        else:  # Student-t log-likelihood, fat tails
            if nu is None or nu <= 2.0001:
                return 1e10
            c = gammaln((nu+1)/2) - gammaln(nu/2) - 0.5*np.log((nu-2)*np.pi)
            ll = np.sum(c - 0.5*np.log(sig2) - ((nu+1)/2)*np.log(1 + r**2/((nu-2)*sig2)))
        return -ll                         # minimize negative log-likelihood

    if model == 'garch':
        x0 = [var0*0.05, 0.05, 0.90]
    else:
        x0 = [var0*0.05, 0.03, 0.90, 0.05]
    if dist == 't':
        x0 = x0 + [8.0]

    res = minimize(neg_loglik, x0, method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    omega, alpha, beta, gamma, nu = unpack(res.x)

    sig2 = np.empty(n); sig2[0] = var0     # re-run filter once more at the optimum
    for t in range(1, n):
        lev = gamma * r[t-1]**2 * (r[t-1] < 0)
        sig2[t] = omega + alpha*r[t-1]**2 + beta*sig2[t-1] + lev

    return dict(omega=omega, alpha=alpha, beta=beta, gamma=gamma, nu=nu,
                last_var=sig2[-1], last_resid=r[-1], loglik=-res.fun)


def garch_forecast_vol(params, horizon_days, annualization=252):
    """Step 4-6 of the recipe: roll the fitted GARCH recursion FORWARD
    (no more real data available), average the path of forecasted daily
    variances over `horizon_days`, annualize -- gives one number directly
    usable as `sigma` in Black-Scholes or as v0/theta-scale inputs for
    Heston, or as your realized_vol() drop-in in the trading workflow."""
    omega, alpha, beta, gamma = params['omega'], params['alpha'], params['beta'], params['gamma']
    sig2_t, eps_t = params['last_var'], params['last_resid']

    # one-step-ahead forecast from the last observed day:
    sig2_next = omega + alpha*eps_t**2 + beta*sig2_t + gamma*eps_t**2*(eps_t < 0)*0.5

    # beyond one step, there's no more eps_t info -- the recursion mean-
    # reverts geometrically toward the unconditional (long-run) variance at
    # rate (alpha+beta+0.5*gamma) per day, which is the standard GARCH
    # multi-step-ahead forecast formula:
    persistence = alpha + beta + 0.5*gamma
    uncond_var = omega / max(1e-10, (1 - persistence))
    path = np.empty(horizon_days)
    path[0] = sig2_next
    for h in range(1, horizon_days):
        path[h] = uncond_var + (persistence**h) * (sig2_next - uncond_var)

    avg_daily_var = path.mean()            # step 5: cumulative/average variance over H days
    return np.sqrt(avg_daily_var * annualization)   # step 6: annualize


def forecast_realized_vol_garch(log_returns, horizon_days, model='gjr-garch',
                                 dist='normal', annualization=252):
    """One-call convenience wrapper: LOG RETURNS in -> forecast vol out.
    (Previously took raw prices and derived returns internally via
    np.diff(np.log(prices)) -- that conversion is removed since your data
    source already provides daily log returns directly.) This is the
    function to plug into realized_vol()/ml_forecast_vol() in
    options_trading_workflow.py as your (classical) baseline forecaster that
    RF/GBM/NN models should be benchmarked against."""
    log_returns = np.asarray(log_returns, dtype=float)
    params = fit_garch(log_returns, model=model, dist=dist)
    vol = garch_forecast_vol(params, horizon_days, annualization)
    return vol, params