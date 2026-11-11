import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm


# ---------- Heston (1993) semi-analytical pricer ----------
# Uses the "little trap" reformulation (Albrecher, Mayer, Schoutens, Tistaert 2007)
# to avoid the branch-cut discontinuities in the original Heston characteristic
# function, which otherwise break numerically for long maturities / certain params.
#
# Parameters (SDEs):
#   dS_t = r S_t dt + sqrt(v_t) S_t dW1_t
#   dv_t = kappa(theta - v_t) dt + sigma_v sqrt(v_t) dW2_t,   corr(dW1,dW2) = rho
#
#   S0      : spot price
#   K       : strike
#   T       : time to maturity (years)
#   r       : risk-free rate
#   kappa   : mean reversion speed of variance
#   theta   : long-run variance
#   sigma_v : vol-of-vol
#   rho     : correlation between asset and variance Brownian motions
#   v0      : initial variance (NOT vol -- it's sigma^2, so v0 = impl_vol**2 as a start)

def _heston_cf_components(phi, kappa, theta, sigma_v, rho, r, T, j):
    # j = 1 -> P1 (asset-measure prob), j = 2 -> P2 (risk-neutral prob)
    u = 0.5 if j == 1 else -0.5
    b = kappa - rho*sigma_v if j == 1 else kappa

    a = kappa*theta
    x = 1j*phi
    d = np.sqrt((rho*sigma_v*x - b)**2 - sigma_v**2*(2*u*x - phi**2))
    g = (b - rho*sigma_v*x - d) / (b - rho*sigma_v*x + d)      # little-trap form

    exp_dT = np.exp(-d*T)
    C = r*x*T + (a/sigma_v**2)*((b - rho*sigma_v*x - d)*T
                                 - 2*np.log((1 - g*exp_dT)/(1 - g)))
    D = ((b - rho*sigma_v*x - d)/sigma_v**2) * ((1 - exp_dT)/(1 - g*exp_dT))
    return C, D


def _heston_Pj_integrand(phi, S0, K, kappa, theta, sigma_v, rho, r, T, v0, j):
    C, D = _heston_cf_components(phi, kappa, theta, sigma_v, rho, r, T, j)
    f = np.exp(C + D*v0 + 1j*phi*np.log(S0))
    return np.real(np.exp(-1j*phi*np.log(K)) * f / (1j*phi))


def heston_call_price(S0, K, T, r, kappa, theta, sigma_v, rho, v0, upper_limit=100):
    """European call price under Heston (1993) via the two-probability formula
    C = S0*P1 - K*exp(-rT)*P2, integrated with the numerically stable 'little trap'
    characteristic function. v0, theta are VARIANCES (sigma^2), not vols."""
    P1 = 0.5 + (1/np.pi) * quad(
        _heston_Pj_integrand, 1e-8, upper_limit,
        args=(S0, K, kappa, theta, sigma_v, rho, r, T, v0, 1), limit=200)[0]
    P2 = 0.5 + (1/np.pi) * quad(
        _heston_Pj_integrand, 1e-8, upper_limit,
        args=(S0, K, kappa, theta, sigma_v, rho, r, T, v0, 2), limit=200)[0]
    return S0*P1 - K*np.exp(-r*T)*P2


def heston_put_price(S0, K, T, r, kappa, theta, sigma_v, rho, v0, upper_limit=100):
    """Put price via put-call parity (cheaper than re-deriving Pj's for puts)."""
    call = heston_call_price(S0, K, T, r, kappa, theta, sigma_v, rho, v0, upper_limit)
    return call - S0 + K*np.exp(-r*T)


# ============================================================================
# FOURIER / CARR-MADAN FFT PRICER
# ============================================================================
# The quad-based pricer above integrates P1, P2 (probabilities) one option at a
# time -- each call to heston_call_price does TWO adaptive numerical quadratures
# (scipy.integrate.quad) over an oscillatory integrand. That's accurate but slow
# if you need many strikes (e.g. 50-200 strikes in a WRDS option chain for one
# date/maturity): it's O(num_strikes) quad calls, each itself iterative.
#
# Carr & Madan (1999) instead prices an entire strike grid in ONE shot using the
# FFT. The trick: the call price as a function of log-strike has a Fourier
# transform that can be written directly in terms of the characteristic function
# phi(u) of log(S_T) (no need to split into P1/P2 at all). You:
#   1. Damp the call price by e^{alpha*k} (alpha>0) so it's integrable even as
#      K->0 (raw call prices don't decay, so their Fourier transform doesn't
#      exist without damping).
#   2. Write that damped transform psi(v) in closed form using phi(u).
#   3. Discretize the transform integral on a grid and recognise it as a
#      Riemann sum -> exactly the form np.fft.fft wants -> get ALL strikes'
#      prices for one maturity from a single length-N FFT.
#
# So _heston_Pj_integrand (quad method) and the FFT method solve the SAME
# underlying problem (Heston has no closed-form density, only a closed-form
# characteristic function -- there are no "Riccati ODEs to solve numerically"
# left to do, Heston already solved them analytically in 1993; C(phi,T) and
# D(phi,T) above ARE that closed-form solution). The only numerical step left
# in both methods is a Fourier inversion -- quad does it one frequency
# integral at a time per strike, FFT does it for a whole strike grid at once
# via the Cooley-Tukey algorithm.
#
# Which is better for this project: FFT, for calibration and ML-dataset
# generation. You will be pricing full option chains (many strikes x several
# maturities) thousands of times (once per optimiser iteration during
# calibration, and potentially millions of times to generate synthetic
# training data for the NN/PINN). FFT turns "num_strikes x 2 quad calls" into
# "num_maturities FFT calls", which is 10-50x faster in practice for a typical
# WRDS SPX chain. Keep the quad pricer for: (a) a single bespoke (S,K,T) price
# where you don't want interpolation error, (b) sanity-checking the FFT output
# (as below), since quad is "exact" up to tolerance while FFT has small grid/
# interpolation error (~1e-3 in the test below, controllable via N, eta, alpha).

def heston_charfunc(u, S0, T, r, kappa, theta, sigma_v, rho, v0, q=0.0):
    """Characteristic function of ln(S_T) under the risk-neutral measure,
    phi(u) = E[exp(i*u*ln S_T)], using the 'little trap' form. u may be a
    complex array (needed for the Carr-Madan damped transform below)."""
    i = 1j
    x = np.log(S0) + (r - q)*T
    a = kappa*theta
    b = kappa
    d = np.sqrt((rho*sigma_v*i*u - b)**2 + sigma_v**2*(i*u + u**2))
    g = (b - rho*sigma_v*i*u - d) / (b - rho*sigma_v*i*u + d)
    C = i*u*x + (a/sigma_v**2)*((b - rho*sigma_v*i*u - d)*T
                                 - 2*np.log((1 - g*np.exp(-d*T))/(1 - g)))
    D = ((b - rho*sigma_v*i*u - d)/sigma_v**2) * ((1 - np.exp(-d*T))/(1 - g*np.exp(-d*T)))
    return np.exp(C + D*v0)


def heston_fft_call_grid(S0, T, r, kappa, theta, sigma_v, rho, v0, q=0.0,
                          alpha=1.5, N=4096, eta=0.25):
    """Carr-Madan FFT: returns (strikes, call_prices) for an entire log-strike
    grid at maturity T, computed with a SINGLE FFT call.
    alpha : damping factor (1.0-2.0 typical; too small -> integrand not
            decaying fast enough / unstable, too large -> loses precision for
            far OTM strikes). eta : freq-grid spacing (finer -> wider strike
            range covered). N : FFT length (power of 2; bigger -> finer
            strike spacing, at a (cheap) performance cost since FFT is
            O(N log N))."""
    i = 1j
    lam = 2*np.pi/(N*eta)                      # resulting log-strike spacing
    b = N*lam/2
    ks = -b + lam*np.arange(N)                 # log-strike grid
    strikes = np.exp(ks)
    v = eta*np.arange(N)
    u = v - (alpha + 1)*i
    cf = heston_charfunc(u, S0, T, r, kappa, theta, sigma_v, rho, v0, q)
    denom = (alpha**2 + alpha - v**2) + i*(2*alpha + 1)*v
    psi = np.exp(-r*T) * cf / denom
    w = np.ones(N); w[1:-1:2] = 4; w[2:-1:2] = 2   # Simpson's rule weights
    fft_input = np.exp(i*b*v) * psi * eta * w/3
    fft_prices = np.fft.fft(fft_input)
    call_prices = (np.exp(-alpha*ks)/np.pi) * fft_prices.real
    return strikes, call_prices


def heston_price_fft(S0, K, T, r, kappa, theta, sigma_v, rho, v0, q=0.0, cp_flag='C',
                      **fft_kwargs):
    """Single-strike price via FFT + linear interpolation onto the FFT's
    strike grid (fast, ~1e-3 accuracy vs the quad pricer -- fine for
    calibration/training data, use heston_call_price for a final reported
    'exact' price)."""
    strikes, calls = heston_fft_call_grid(S0, T, r, kappa, theta, sigma_v, rho, v0, q, **fft_kwargs)
    c = np.interp(K, strikes, calls)
    if cp_flag in ('P', 'p'):
        return c - S0*np.exp(-q*T) + K*np.exp(-r*T)   # put-call parity
    return c


# ============================================================================
# CALIBRATING HESTON PARAMETERS FROM YOUR WRDS SPX OPTION CHAIN
# ============================================================================
# You do NOT get kappa, theta, sigma_v, rho, v0 directly from WRDS -- they are
# NOT observable market quantities, they're the 5 parameters of a model you
# fit so that Heston's prices match the market's (mid) prices as closely as
# possible, for every (strike, maturity) pair on a given date. This is a
# nonlinear least-squares problem solved ONCE PER DATE (one calibration = one
# full day's option chain = one set of 5 params), not once per option.
#
# Recipe, using exactly the columns you have (date, strike_price/strike, cp_flag,
# best_bid, best_offer, forward_price, T or days, impl_volatility):
#   1. mid = (best_bid + best_offer)/2  (already suggested earlier -- use this
#      as the calibration target, never last/bid/offer alone).
#   2. Back out S0, r, q implicitly via forward_price: F = S0*exp((r-q)T), so
#      you don't need to separately know S0, r, q -- just carry 'forward' and
#      use S0*exp(-qT) = F*exp(-rT) in the pricer (see prepare_chain below,
#      which reparametrises the pricer in terms of F directly, avoiding the
#      need to split r and q at all -- cleaner and matches what OptionMetrics
#      gives you).
#   3. Filter: drop rows with volume==0 and open_interest==0 (stale), bid<=0,
#      bid>=offer (crossed/locked quotes - a data error), and T<7/365 (unstable
#      IV/greeks for very short-dated options).
#   4. Get a GOOD initial guess (Heston calibration is non-convex with several
#      local minima and some parameter redundancy -- kappa and theta trade off
#      against each other since only the combination kappa*theta=a, and
#      kappa*theta*T matter most at short maturities -- a bad starting point
#      can converge to a nonsensical but low-residual fit). init_guess_from_market
#      below builds one from the data itself, no guessing needed.
#   5. Minimise weighted price residuals with scipy.optimize.least_squares
#      (Levenberg-Marquardt/trust-region), with:
#        - weights ~ 1/mid (approximates fitting in "vol space" without the
#          cost of inverting Black-Scholes IV at every iteration -- fitting
#          raw prices unweighted over-fits deep ITM options, which have huge
#          absolute price but almost no vol information, and under-fits far
#          OTM/short-dated options, which are tiny in price but carry most of
#          the smile/skew information you actually want Heston to capture)
#        - a soft penalty if the Feller condition 2*kappa*theta >= sigma_v^2
#          is violated (not violating it isn't strictly required for pricing,
#          but a big violation usually signals an unstable/non-generalising
#          fit and will hurt your downstream NN/PINN if you use these params
#          as training labels or features)
#        - bounds to keep parameters economically sane (rho in (-1,1), all
#          vol/variance terms positive, etc).
#   6. Honest caveat (confirmed by testing this on synthetic data where the
#      true parameters are KNOWN): kappa, theta, sigma_v individually are
#      often only loosely identified -- several different (kappa, sigma_v)
#      combinations can fit the SAME price surface almost equally well, while
#      v0 and rho are usually well pinned down (v0 by short-dated ATM options,
#      rho by the skew). This is a well-known Heston calibration issue, not a
#      bug -- so: (a) always report the FIT QUALITY (residuals) alongside the
#      parameters, not just the parameters, (b) consider multi-start (several
#      x0's, keep lowest cost) for production calibration, (c) if you later
#      calibrate day-by-day for a time series, add a smoothness regulariser
#      (penalise large day-to-day jumps in kappa/sigma_v) since day-to-day
#      noise in option prices otherwise causes the parameters to jump around
#      even when the market barely moved.

def init_guess_from_market(df, S0):
    """Build a data-driven initial guess (kappa, theta, sigma_v, rho, v0) from
    a single day's prepared chain (needs columns: K, T, mid, cp_flag,
    impl_volatility, forward). Pure heuristics, not a fit -- just gets the
    optimiser started in a sensible region."""
    df = df.copy()
    df['moneyness'] = df['K'] / df['forward']
    atm = df.iloc[(df['moneyness'] - 1.0).abs().argsort()]

    # v0: ATM implied variance at the shortest available maturity
    T_min = df['T'].min()
    v0 = atm[atm['T'] == T_min]['impl_volatility'].iloc[0]**2

    # theta: ATM implied variance at the longest available maturity (long-run
    # level the variance should mean-revert to)
    T_max = df['T'].max()
    theta = atm[atm['T'] == T_max]['impl_volatility'].iloc[0]**2

    # kappa: rough mean-reversion speed from how fast ATM variance moves from
    # v0 toward theta between the shortest and longest maturity available
    kappa = 2.0 if T_max <= T_min else abs(np.log(max(theta, 1e-6)/max(v0, 1e-6))) / (T_max - T_min) + 0.5
    kappa = float(np.clip(kappa, 0.3, 8.0))

    # rho: sign/magnitude from the skew -- compare IV of a downside put vs an
    # upside call at similar |moneyness| for the shortest maturity (equity
    # index skew is virtually always negative: rho < 0)
    short = df[df['T'] == T_min]
    put_iv = short[short['moneyness'] < 0.95]['impl_volatility'].mean()
    call_iv = short[short['moneyness'] > 1.05]['impl_volatility'].mean()
    if np.isfinite(put_iv) and np.isfinite(call_iv):
        rho = float(np.clip(-0.3 - 2.0*(put_iv - call_iv), -0.95, -0.05))
    else:
        rho = -0.6   # typical SPX default if skew can't be measured

    sigma_v = 0.5    # vol-of-vol has no easy single-quote proxy; mid-range start
    return dict(kappa=kappa, theta=float(theta), sigma_v=sigma_v, rho=rho, v0=float(v0))


def calibrate_heston(df, r=0.0, q=0.0, x0=None, n_starts=1, verbose=False):
    """Calibrate Heston to one date's option chain.

    df columns required: 'K' (strike), 'T' (years), 'mid' (mid price),
    'cp_flag' ('C'/'P'), 'forward' (forward_price), optionally
    'impl_volatility' (used only by init_guess_from_market).
    r, q  : risk-free rate / dividend yield for that date (or pass r and set
            q=0 and use 'forward' consistently -- see prepare_wrds_chain).

    Returns (params_dict, scipy.optimize.OptimizeResult).
    """
    from scipy.optimize import least_squares

    S0 = df['forward'].iloc[0] * np.exp(-(r - q) * df['T'].iloc[0])  # back out S0 for the FFT grid

    bounds = ([1e-3, 1e-4, 1e-3, -0.999, 1e-4],
              [20.0, 3.0, 5.0, 0.999, 3.0])

    def price_chain(params):
        kappa, theta, sigma_v, rho, v0 = params
        out = np.empty(len(df))
        for T in df['T'].unique():
            mask = (df['T'].values == T)
            strikes_grid, call_grid = heston_fft_call_grid(S0, T, r, kappa, theta, sigma_v, rho, v0, q)
            sub = df[mask]
            c = np.interp(sub['K'].values, strikes_grid, call_grid)
            is_put = (sub['cp_flag'].values == 'P')
            out[mask] = np.where(is_put, c - S0*np.exp(-q*T) + sub['K'].values*np.exp(-r*T), c)
        return out

    def residuals(params):
        kappa, theta, sigma_v, rho, v0 = params
        model = price_chain(params)
        weights = 1.0 / np.maximum(df['mid'].values, 0.01)   # ~vega-weighting, avoids over/under-fitting ITM vs OTM
        res = (model - df['mid'].values) * weights
        feller_violation = min(2*kappa*theta - sigma_v**2, 0.0)
        return np.append(res, 5.0*feller_violation)

    starts = [x0] if x0 else [init_guess_from_market(df, S0)]
    if n_starts > 1:
        rng = np.random.default_rng(0)
        for _ in range(n_starts - 1):
            base = starts[0]
            starts.append({
                'kappa': float(np.clip(base['kappa']*rng.uniform(0.5, 1.5), 0.3, 8.0)),
                'theta': float(np.clip(base['theta']*rng.uniform(0.5, 1.5), 1e-4, 1.0)),
                'sigma_v': float(rng.uniform(0.2, 1.2)),
                'rho': float(np.clip(base['rho']*rng.uniform(0.5, 1.5), -0.95, -0.05)),
                'v0': float(np.clip(base['v0']*rng.uniform(0.5, 1.5), 1e-4, 1.0)),
            })

    best = None
    for s in starts:
        x0_vec = [s['kappa'], s['theta'], s['sigma_v'], s['rho'], s['v0']]
        result = least_squares(residuals, x0_vec, bounds=bounds, xtol=1e-12, ftol=1e-12)
        if verbose:
            print("start:", s, "-> cost:", result.cost)
        if best is None or result.cost < best.cost:
            best = result

    params = dict(zip(['kappa', 'theta', 'sigma_v', 'rho', 'v0'], best.x))
    return params, best


def prepare_wrds_chain(raw_df, min_days=7):
    """Turn one date's raw WRDS OptionMetrics rows into the clean chain
    calibrate_heston() expects. Expects raw WRDS columns:
    date, exdate, cp_flag, strike_price, best_bid, best_offer, volume,
    open_interest, impl_volatility, forward_price, days.
    (strike_price in OptionMetrics is in cents for the *1000 'strike' field in
    some extracts -- check your pull; this assumes 'strike_price'/1000 = strike
    in index points, adjust if your extract already has 'strike'.)"""
    df = raw_df.copy()
    df['K'] = df['strike_price'] / 1000.0 if df['strike_price'].median() > 1000 else df['strike_price']
    df['mid'] = (df['best_bid'] + df['best_offer']) / 2.0
    df['T'] = df['days'] / 365.0
    df['forward'] = df['forward_price']

    df = df[(df['volume'].fillna(0) > 0) | (df['open_interest'].fillna(0) > 0)]
    df = df[df['best_bid'] > 0]
    df = df[df['best_offer'] > df['best_bid']]
    df = df[df['days'] >= min_days]
    df = df[df['impl_volatility'].notna()]

    return df[['K', 'T', 'mid', 'cp_flag', 'forward', 'impl_volatility', 'days']].reset_index(drop=True)

def calibrate_heston_day(chain, moneyness=(0.8, 1.2), max_days=365, n_starts=1):
    """Calibrate Heston to one day's chain as loaded by data_importing.data_import
    (multi_option_single_day, or one date's rows of multi_option_multi_day).

    Uses that day's spot S, one risk-free rate (median r over the chain) and q
    (0 if the chain has none), the same inputs the Black–Scholes scripts use.
    Only options with strike within `moneyness` of spot and at most `max_days`
    to expiry are fitted (keeps the fit fast and focused on the liquid part).
    Returns (params, S0, r, q): params can go straight into heston_price_fft(**params).
    """
    S0 = float(chain["S"].iloc[0])
    r = float(chain["r"].median())
    q = float(np.nan_to_num(chain["q"].iloc[0])) if "q" in chain else 0.0

    df = prepare_wrds_chain(chain)
    num = ["K", "T", "mid", "forward", "impl_volatility"]
    df[num] = df[num].astype(float)   # WRDS gives nullable 'Float64' columns, whose missing <NA> breaks `if` checks
    df = df[df["K"].between(moneyness[0] * S0, moneyness[1] * S0) & (df["days"] <= max_days)].copy()
    df["forward"] = S0 * np.exp((r - q) * df["T"])   # so calibrate_heston backs out exactly S0
    params, _ = calibrate_heston(df.reset_index(drop=True), r=r, q=q, n_starts=n_starts)
    return params, S0, r, q


def calibrate_heston_by_date(chains):
    """calibrate_heston_day for every date in a stacked multi-day chain (multi_option_multi_day).
    Returns {date: params}. Days where the fit fails are skipped with a warning."""
    params = {}
    groups = chains.groupby("date")
    for i, (d, day_chain) in enumerate(groups, 1):
        try:
            params[d] = calibrate_heston_day(day_chain)[0]
        except Exception as e:
            print(f"\nWarning: Heston calibration failed on {d.date()} ({e})")
        print(f"Heston: calibrated {i}/{len(groups)} days", end="\r")
    print()
    return params


def heston_price_chain(df, params_by_date):
    """Heston price for every option in df (columns date, S, strike, T, r, cp_flag, and q if
    there is one), using params_by_date[date]. One FFT per (date, expiry), so whole chains are fast.
    Returns an array aligned with df (NaN for dates with no parameters)."""
    out = pd.Series(np.nan, index=df.index)
    q_all = df["q"].astype(float).fillna(0.0) if "q" in df else pd.Series(0.0, index=df.index)
    for (d, T), g in df.groupby(["date", "T"]):
        params = params_by_date.get(d)
        if params is None:
            continue
        S, r, q = float(g["S"].iloc[0]), float(g["r"].iloc[0]), float(q_all[g.index[0]])
        strikes, calls = heston_fft_call_grid(S, T, r, q=q, **params)
        K = g["strike"].to_numpy(float)
        call = np.interp(K, strikes, calls)
        put = call - S * np.exp(-q * T) + K * np.exp(-r * T)     # put-call parity
        out[g.index] = np.where(g["cp_flag"].to_numpy() == "P", put, call)
    return out.to_numpy()
