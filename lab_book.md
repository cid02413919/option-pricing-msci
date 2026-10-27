## 15/10/26 - Kai
- Imported data from WRDS: data_import_wrds.py
- Added black scholes call function `black_scholes_call(S, K, sigma, r, t)`
- Applied `black_scholes_call(S, K, sigma, r, t)` to call data using mean implied volatility
- Notices that using the exact implied volatility of a given call, doesnt lead to a perfects priced call. Potentially due to not taking dividends into account.

## 21/10/26
**Aims:**
- Understand why using exact implied volatility doesnt lead to perfect option price, and fix if possible.