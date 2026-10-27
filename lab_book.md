## 15/10/26
- Imported data from WRDS: data_import_wrds.py
- Added black scholes call function `black_scholes_call(S, K, sigma, r, t)`
- Applied `black_scholes_call(S, K, sigma, r, t)` to call data using mean implied volatility
- Notices that using the exact implied volatility of a given call, doesnt lead to a perfects priced call. Potentially due to not taking dividends into account.

## 21/10/26
**Aims:**
- Understand why using exact implied volatility doesnt lead to perfect option price, and fix if possible.

A potential cause of the implied volatility not matching is due to not taking into account the dividends yield. The merton model takes into accound dividends into its calculation. When i searched it up i found that european options use the black scholed merton model to calculate the implied volatility.

**Results:**
The merton model decreased the MSE significantly, especially for longer expiry options, however a slight missmatch remains.
![MSE by expiry, log scale](figures/mse_by_expiry.png)

