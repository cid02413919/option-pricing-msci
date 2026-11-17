"""Heston (1993) stochastic-volatility model: pricing and calibration.

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW1_t
    dv_t = kappa (theta - v_t) dt + sigma_v sqrt(v_t) dW2_t,    corr(dW1, dW2) = rho

    kappa   : mean-reversion speed of the variance
    theta   : long-run variance
    sigma_v : vol of vol
    rho     : correlation between the asset and its variance
    v0      : initial variance
theta and v0 are VARIANCES (vol**2), not vols.

Both pricers invert the same closed-form characteristic function (heston_charfunc):
  - heston_call_price / heston_put_price: numerical integration (scipy quad), one
    strike at a time. Accurate; use for single prices and to check the FFT.
  - heston_fft_call_grid / heston_price_fft: Carr-Madan (1999) FFT, a whole strike
    grid per maturity in one go (~1e-3 interpolation error). Much faster; used
    for calibration and pricing whole chains.

Calibration fits the 5 parameters to one day's option chain by weighted least
squares on mid prices (calibrate_heston). Note that kappa, theta and sigma_v are
often only loosely identified (different combinations fit almost equally well),
while v0 and rho are usually well pinned down, so check the fit residuals too.
"""
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import least_squares

PARAM_NAMES = ["kappa", "theta", "sigma_v", "rho", "v0"]
LOWER_BOUNDS = [1e-3, 1e-4, 1e-3, -0.999, 1e-4]
UPPER_BOUNDS = [20.0, 3.0, 5.0, 0.999, 3.0]


# ---------------------------------------------------------------------------
# Characteristic function
# ---------------------------------------------------------------------------

def heston_charfunc(u, S0, T, r, kappa, theta, sigma_v, rho, v0, q=0.0):
    """Characteristic function of ln(S_T) under the risk-neutral measure,
    phi(u) = E[exp(i u ln S_T)]. u may be a complex array.

    Uses the 'little trap' form (Albrecher et al. 2007), which avoids the jumps
    in the complex log that break the original formula at long maturities."""
    iu = 1j * u
    beta = kappa - rho * sigma_v * iu
    d = np.sqrt(beta**2 + sigma_v**2 * (iu + u**2))
    g = (beta - d) / (beta + d)
    e = np.exp(-d * T)

    C = kappa * theta / sigma_v**2 * ((beta - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
    D = (beta - d) / sigma_v**2 * (1 - e) / (1 - g * e)
    return np.exp(iu * (np.log(S0) + (r - q) * T) + C + D * v0)


def _call_to_put(call, S0, K, T, r, q):
    """Put-call parity: P = C - S0 e^{-qT} + K e^{-rT}."""
    return call - S0 * np.exp(-q * T) + K * np.exp(-r * T)


# ---------------------------------------------------------------------------
# Pricer 1: numerical integration (one strike at a time)
# ---------------------------------------------------------------------------

def heston_call_price(S0, K, T, r, kappa, theta, sigma_v, rho, v0, upper_limit=100, q=0.0):
    """European call via C = S0 e^{-qT} P1 - K e^{-rT} P2, where P1 and P2 are
    the probabilities of finishing in the money (under the stock and
    risk-neutral measures), each found by integrating the characteristic function."""
    def cf(u):
        return heston_charfunc(u, S0, T, r, kappa, theta, sigma_v, rho, v0, q)

    forward = S0 * np.exp((r - q) * T)   # = cf(-i)
    log_K = np.log(K)

    def probability(cf_shifted):
        integrand = lambda u: np.real(np.exp(-1j * u * log_K) * cf_shifted(u) / (1j * u))
        return 0.5 + quad(integrand, 1e-8, upper_limit, limit=200)[0] / np.pi

    P1 = probability(lambda u: cf(u - 1j) / forward)
    P2 = probability(cf)
    return S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2


def heston_put_price(S0, K, T, r, kappa, theta, sigma_v, rho, v0, upper_limit=100, q=0.0):
    """European put via put-call parity."""
    call = heston_call_price(S0, K, T, r, kappa, theta, sigma_v, rho, v0, upper_limit, q)
    return _call_to_put(call, S0, K, T, r, q)


# ---------------------------------------------------------------------------
# Pricer 2: Carr-Madan FFT (a whole strike grid per maturity)
# ---------------------------------------------------------------------------

def heston_fft_call_grid(S0, T, r, kappa, theta, sigma_v, rho, v0, q=0.0,
                         alpha=1.5, N=4096, eta=0.25):
    """Call prices on a grid of strikes for maturity T, from a single FFT.
    Returns (strikes, call_prices).

    alpha : damping factor, makes the call price integrable (1.0-2.0 typical)
    N     : FFT length (power of 2)
    eta   : spacing of the frequency grid (smaller -> wider strike range)"""
    lam = 2 * np.pi / (N * eta)          # log-strike spacing
    b = N * lam / 2                      # log-strikes run from -b to b
    log_strikes = -b + lam * np.arange(N)
    v = eta * np.arange(N)

    cf = heston_charfunc(v - (alpha + 1) * 1j, S0, T, r, kappa, theta, sigma_v, rho, v0, q)
    psi = np.exp(-r * T) * cf / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)

    simpson = np.ones(N)
    simpson[1:-1:2] = 4
    simpson[2:-1:2] = 2
    fft = np.fft.fft(np.exp(1j * b * v) * psi * eta * simpson / 3)

    call_prices = np.exp(-alpha * log_strikes) / np.pi * fft.real
    return np.exp(log_strikes), call_prices


def heston_price_fft(S0, K, T, r, kappa, theta, sigma_v, rho, v0, q=0.0, cp_flag="C",
                     **fft_kwargs):
    """Call ('C') or put ('P') price at strike(s) K, interpolated from the FFT grid."""
    strikes, calls = heston_fft_call_grid(S0, T, r, kappa, theta, sigma_v, rho, v0, q, **fft_kwargs)
    call = np.interp(K, strikes, calls)
    if cp_flag.upper() == "P":
        return _call_to_put(call, S0, K, T, r, q)
    return call


def _price_one_expiry(S0, T, r, q, K, cp_flag, params):
    """FFT prices for many options sharing one expiry (a single FFT).
    K and cp_flag are arrays; params is a dict of the 5 Heston parameters."""
    strikes, calls = heston_fft_call_grid(S0, T, r, q=q, **params)
    call = np.interp(K, strikes, calls)
    return np.where(cp_flag == "P", _call_to_put(call, S0, K, T, r, q), call)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def init_guess_from_market(df):
    """Rough starting point for calibration, read off the market data.
    df needs columns K, T, forward, impl_volatility."""
    moneyness = df["K"] / df["forward"]
    T_min, T_max = df["T"].min(), df["T"].max()

    def atm_variance(T):
        rows = df[df["T"] == T]
        closest = (moneyness[rows.index] - 1).abs().argmin()
        return float(rows["impl_volatility"].iloc[closest] ** 2)

    v0 = atm_variance(T_min)       # short-dated ATM variance
    theta = atm_variance(T_max)    # long-dated ATM variance ~ long-run level

    # kappa: how fast ATM variance moves from v0 towards theta across maturities
    if T_max > T_min:
        kappa = abs(np.log(max(theta, 1e-6) / max(v0, 1e-6))) / (T_max - T_min) + 0.5
    else:
        kappa = 2.0
    kappa = float(np.clip(kappa, 0.3, 8.0))

    # rho: from the short-dated skew (downside IV vs upside IV)
    short = df["T"] == T_min
    put_iv = df.loc[short & (moneyness < 0.95), "impl_volatility"].mean()
    call_iv = df.loc[short & (moneyness > 1.05), "impl_volatility"].mean()
    if np.isfinite(put_iv) and np.isfinite(call_iv):
        rho = float(np.clip(-0.3 - 2.0 * (put_iv - call_iv), -0.95, -0.05))
    else:
        rho = -0.6

    return dict(kappa=kappa, theta=theta, sigma_v=0.5, rho=rho, v0=v0)


def _random_restart(base, rng):
    """A random perturbation of the starting point `base`, for multi-start."""
    return dict(
        kappa=float(np.clip(base["kappa"] * rng.uniform(0.5, 1.5), 0.3, 8.0)),
        theta=float(np.clip(base["theta"] * rng.uniform(0.5, 1.5), 1e-4, 1.0)),
        sigma_v=float(rng.uniform(0.2, 1.2)),
        rho=float(np.clip(base["rho"] * rng.uniform(0.5, 1.5), -0.95, -0.05)),
        v0=float(np.clip(base["v0"] * rng.uniform(0.5, 1.5), 1e-4, 1.0)),
    )


def calibrate_heston(df, r=0.0, q=0.0, x0=None, n_starts=1, verbose=False):
    """Fit Heston to one date's option chain.

    df columns: K, T (years), mid, cp_flag ('C'/'P'), forward, and
    impl_volatility (only used for the starting guess when x0 is None).
    x0       : starting parameters as a dict, or None to guess from the data
    n_starts : number of starting points tried (the best fit is kept)

    Minimises (model - mid) / mid over all options, i.e. relative price errors,
    so cheap OTM options (which carry the smile) count as much as expensive ITM
    ones. Violating the Feller condition 2 kappa theta >= sigma_v^2 is penalised.
    Returns (params dict, scipy OptimizeResult)."""
    S0 = df["forward"].iloc[0] * np.exp(-(r - q) * df["T"].iloc[0])
    K = df["K"].to_numpy(float)
    T = df["T"].to_numpy(float)
    mid = df["mid"].to_numpy(float)
    cp_flag = df["cp_flag"].to_numpy()
    weights = 1.0 / np.maximum(mid, 0.01)

    def residuals(x):
        params = dict(zip(PARAM_NAMES, x))
        model = np.empty(len(df))
        for t in np.unique(T):
            m = T == t
            model[m] = _price_one_expiry(S0, t, r, q, K[m], cp_flag[m], params)
        feller_violation = min(2 * params["kappa"] * params["theta"] - params["sigma_v"]**2, 0.0)
        return np.append((model - mid) * weights, 5.0 * feller_violation)

    starts = [x0 if x0 else init_guess_from_market(df)]
    rng = np.random.default_rng(0)
    starts += [_random_restart(starts[0], rng) for _ in range(n_starts - 1)]

    best = None
    for start in starts:
        x_start = [start[name] for name in PARAM_NAMES]
        result = least_squares(residuals, x_start, bounds=(LOWER_BOUNDS, UPPER_BOUNDS),
                               xtol=1e-12, ftol=1e-12)
        if verbose:
            print("start:", start, "-> cost:", result.cost)
        if best is None or result.cost < best.cost:
            best = result

    return dict(zip(PARAM_NAMES, best.x)), best


# ---------------------------------------------------------------------------
# Working with WRDS option chains
# ---------------------------------------------------------------------------

def prepare_wrds_chain(raw_df, min_days=7):
    """Clean one date's raw WRDS OptionMetrics rows into the format calibrate_heston
    expects. Needs columns cp_flag, strike_price, best_bid, best_offer, volume,
    open_interest, impl_volatility, forward_price, days.

    OptionMetrics quotes strike_price x1000, so it is divided by 1000 when it
    looks that way. Drops untraded, zero-bid, crossed and very short-dated quotes."""
    df = raw_df.copy()
    df["K"] = df["strike_price"] / 1000.0 if df["strike_price"].median() > 1000 else df["strike_price"]
    df["mid"] = (df["best_bid"] + df["best_offer"]) / 2.0
    df["T"] = df["days"] / 365.0
    df["forward"] = df["forward_price"]

    traded = (df["volume"].fillna(0) > 0) | (df["open_interest"].fillna(0) > 0)
    keep = (traded
            & (df["best_bid"] > 0)
            & (df["best_offer"] > df["best_bid"])
            & (df["days"] >= min_days)
            & df["impl_volatility"].notna())

    columns = ["K", "T", "mid", "cp_flag", "forward", "impl_volatility", "days"]
    return df.loc[keep, columns].reset_index(drop=True)


def calibrate_heston_day(chain, moneyness=(0.8, 1.2), max_days=365, n_starts=1):
    """Calibrate Heston to one day's chain as loaded by data_importing.data_import
    (multi_option_single_day, or one date's rows of multi_option_multi_day).

    Uses that day's spot S, the median risk-free rate r and dividend yield q
    (0 if the chain has none). Only options with strike within `moneyness` of
    spot and at most `max_days` to expiry are fitted.
    Returns (params, S0, r, q); params can go straight into heston_price_fft(**params)."""
    S0 = float(chain["S"].iloc[0])
    r = float(chain["r"].median())
    q = float(np.nan_to_num(chain["q"].iloc[0])) if "q" in chain else 0.0

    df = prepare_wrds_chain(chain)
    numeric = ["K", "T", "mid", "forward", "impl_volatility"]
    df[numeric] = df[numeric].astype(float)   # WRDS gives nullable 'Float64' columns
    in_range = df["K"].between(moneyness[0] * S0, moneyness[1] * S0) & (df["days"] <= max_days)
    df = df[in_range].reset_index(drop=True)
    df["forward"] = S0 * np.exp((r - q) * df["T"])   # so calibrate_heston recovers exactly S0

    params, _ = calibrate_heston(df, r=r, q=q, n_starts=n_starts)
    return params, S0, r, q


def calibrate_heston_by_date(chains):
    """calibrate_heston_day for every date in a multi-day chain (multi_option_multi_day).
    Returns {date: params}. Days where the fit fails are skipped with a warning."""
    params = {}
    groups = chains.groupby("date")
    for i, (date, day_chain) in enumerate(groups, 1):
        try:
            params[date] = calibrate_heston_day(day_chain)[0]
        except Exception as e:
            print(f"\nWarning: Heston calibration failed on {date.date()} ({e})")
        print(f"Heston: calibrated {i}/{len(groups)} days", end="\r")
    print()
    return params


def heston_price_chain(df, params_by_date):
    """Heston price for every option in df (columns date, S, strike, T, r, cp_flag,
    and optionally q), using params_by_date[date]. One FFT per (date, expiry).
    Returns an array aligned with df (NaN for dates with no parameters)."""
    prices = pd.Series(np.nan, index=df.index)
    q_all = df["q"].astype(float).fillna(0.0) if "q" in df else pd.Series(0.0, index=df.index)

    for (date, T), group in df.groupby(["date", "T"]):
        params = params_by_date.get(date)
        if params is None:
            continue
        S = float(group["S"].iloc[0])
        r = float(group["r"].iloc[0])
        q = float(q_all[group.index[0]])
        K = group["strike"].to_numpy(float)
        prices[group.index] = _price_one_expiry(S, T, r, q, K, group["cp_flag"].to_numpy(), params)

    return prices.to_numpy()
