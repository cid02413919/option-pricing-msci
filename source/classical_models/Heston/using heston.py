import pandas as pd
from heston_pricer import prepare_wrds_chain, calibrate_heston, heston_price_fft, heston_call_price

# 1. Load one day's raw WRDS pull (the CSV format from your earlier script)
raw = pd.read_csv("spx_options_2024-06-12.csv", parse_dates=["date", "exdate"])

# 2. Clean it into calibration-ready form
chain = prepare_wrds_chain(raw)

# 3. Calibrate Heston to that day's whole chain
#    r: use that day's rate (you already have a column 'r' from the zero curve)
#    q=0 is fine here since 'forward' already embeds the real dividend yield
r_today = raw['r'].iloc[0]
params, fit_result = calibrate_heston(chain, r=r_today, q=0.0, n_starts=3, verbose=True)

print("Calibrated params:", params)
print("Fit cost (lower = better):", fit_result.cost)

# 4. Price ANY option (on or off the chain) with the calibrated params
S0 = chain['forward'].iloc[0] * np.exp(-r_today * chain['T'].iloc[0])  # back out spot

price_fast = heston_price_fft(S0, K=5400, T=30/365, r=r_today, cp_flag='C', **params)
price_exact = heston_call_price(S0, K=5400, T=30/365, r=r_today,
                                 kappa=params['kappa'], theta=params['theta'],
                                 sigma_v=params['sigma_v'], rho=params['rho'], v0=params['v0'])