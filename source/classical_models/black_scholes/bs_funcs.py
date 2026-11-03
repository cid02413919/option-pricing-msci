import numpy as np
from scipy.stats import norm
import pandas as pd


def black_scholes_call(S, K, sigma, r, t, q=0.0):
    d1 = (np.log(S/K) + (r - q + ((sigma**2)/2))*t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    C = S * np.exp(-q*t) * norm.cdf(d1) - K * np.exp(-r*t) * norm.cdf(d2)
    return C

def black_scholes_put(S, K, sigma, r, t, q=0.0):
    d1 = (np.log(S/K) + (r - q + ((sigma**2)/2))*t) / (sigma * np.sqrt(t))
    d2 = d1 - (sigma * np.sqrt(t))
    P = K * np.exp(-r*t) * norm.cdf(-d2) - S * np.exp(-q*t) * norm.cdf(-d1)
    return P

