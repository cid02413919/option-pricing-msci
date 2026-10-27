import numpy as np
from scipy.stats import norm
import pandas as pd


def black_scholes_call(S, K, sigma, r, t):
    d1 = (np.log(S/K) + (r + ((sigma**2)/2))*t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    C = S * norm.cdf(d1) - K * np.exp(-r*t) * norm.cdf(d2)
    return C

def black_scholes_merton_call(S, K, sigma, r, t, q=0.0):
    d1 = (np.log(S/K) + (r - q + ((sigma**2)/2))*t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    C = S * np.exp(-q*t) * norm.cdf(d1) - K * np.exp(-r*t) * norm.cdf(d2)
    return C

def black_scholes_put(S, K, sigma, r, t):
    d1 = (np.log(S/K) + (r + ((sigma**2)/2))*t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    P = K * np.exp(-r*t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return P

def black_scholes_merton_put(S, K, sigma, r, t, q=0.0):
    d1 = (np.log(S/K) + (r - q + ((sigma**2)/2))*t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    P = K * np.exp(-r*t) * norm.cdf(-d2) - S * np.exp(-q*t) * norm.cdf(-d1)
    return P

def realised_vol(date, returns, N=60, annual=252):
    """Annualised zero-mean realised vol from the N log returns strictly before `date`."""
    date = pd.Timestamp(date)
    past = returns.loc[returns.index < date].dropna()
    if len(past) < N:
        raise ValueError(f"only {len(past)} returns before {date.date()}, need {N}")
    window = past.iloc[-N:]
    return np.sqrt(annual * np.mean(window ** 2))