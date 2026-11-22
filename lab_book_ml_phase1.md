---

## 17/11/2026 – ML Phase 1: Random forest baseline

**Aims:**
- Improve the first random forest (RF) for SPX calls: add volatility-based inputs and test it properly over time.

**Files changed:**
- `source/ML_models/random_forest.py`: same structure, target (predict call price ÷ S) and calls-only data as before, with new inputs and walk-forward testing instead of one train/test split.

---

### 1. Key idea in plain words

An option's price is fixed by a few contract terms (strike K, time to expiry T, today's index level S, interest rate r, dividend yield q) plus one unknown: how much the market expects the index to move, the **volatility**. Black–Scholes (BS) turns a volatility into a price. The volatility that makes BS give exactly the market price is the **implied volatility (IV)**.

The RF learns the price directly from examples. It gets the contract terms plus some information about recent volatility, and predicts price ÷ S.

### 2. Why start with a random forest

A random forest is many decision trees. Each tree splits the data with yes/no questions ("is T < 0.1?", "is the option far out of the money?"). Each tree is trained on a random sample, and the forest averages all of them (Breiman, 2001).

| Alternative | Why not first |
|---|---|
| Linear regression | Option prices depend on the inputs in strongly curved, interacting ways; a straight-line model cannot capture this without hand-made features. |
| Gradient boosting (LightGBM, XGBoost) | Usually more accurate than RF, but has more settings to tune and overfits more easily. A better *second* model (Phase 2). |
| Neural network | Smooth and flexible, but needs feature scaling, careful tuning and much more data handling (Phase 3). |
| Gaussian process | Training cost grows with n³; impossible for ~850k options. |

The RF was chosen first because it:
- has few settings and works well with defaults
- needs no feature scaling and tolerates outliers
- gives feature importances
- is a standard benchmark in ML-in-finance work (e.g. Gu, Kelly & Xiu, 2020)

### 3. The most important rule: only use what is known at pricing time (data leakage)

**Leakage** means the model is given information it would not have in real use, which makes results look far better than they are.

- **Today's IV and delta are leakage.** OptionMetrics computes `impl_volatility` and `delta` *from today's market price*. Giving them to the model gives it the answer: BS with today's IV reprices the options almost perfectly (RMSE $0.65 on calls and puts). So IV and delta are used in **lagged** form only, from the previous pricing date a week earlier.
- **This matches the other methods.** Heston is calibrated on the previous trading day, and the historical, GARCH and VIX vols only use earlier data.
- **Today's S is allowed.** It is known when pricing, and every other method uses it.

### 4. What the model predicts (the target): price ÷ S

As in the original version, the RF predicts **option price ÷ S** and then multiplies by S. An option with strike 4000 when S = 4100 behaves like one with strike 3900 when S = 4000. Dividing by S makes them look the same to the model, so it learns one pattern instead of one per index level.

Alternatives considered:
- **Price in dollars.** It scales with the index level, and trees cannot predict outside the range seen in training.
- **IV, then price with BS.** IV is on the same scale for every option (≈ 0.05–1.5).
- **Change in IV since last week, then price with BS.** IV barely moves week to week, so only the small change needs predicting.

In an earlier test on calls and puts, predicting the change in IV scored a lower RMSE than price ÷ S ($7.76 vs $9.69 on the same options). This could be a future improvement.

### 5. Features (inputs)

| Feature | Meaning | Why |
|---|---|---|
| `moneyness` | K / S | Original input: how far in or out of the money |
| `T` | Years to expiry | Original input |
| `r`, `q` | Interest rate, dividend yield | Original inputs (standard BS inputs) |
| `log_mF` | ln(K / F), F = S·e^((r−q)T) is the forward price | **New.** Moneyness measured against the forward; 0 = at the money. |
| `iv_lag` | This option's IV last week | **New.** Recent volatility of this option. |
| `delta_lag` | BS delta using today's S, K, T, r, q and last week's IV | **New.** Leak-free version of delta; combines moneyness, maturity and vol in one number. |
| `atm_iv_lag` | Last week's median IV of ~1-month at-the-money options | **New.** Overall market fear level. |
| `spot_ret` | ln(S today / S last week) | **New.** Vol tends to rise when the market falls (the "leverage effect"). |
| `is_new` | 1 if the contract was not quoted last week | **New.** Marks options whose `iv_lag` is borrowed (below). |

Notes:
- **New contracts.** New weekly expiries and strikes have no IV last week. They take the IV of last week's closest call, measured by moneyness and √T. (On calls and puts, this was off by a median of 1.4 vol points, against 0.8 for contracts with their own history.) This is better than discarding them.
- **Calls only.** Puts were tried and dropped to keep the study simple. With puts, the model also needs an `is_call` input, and much of `delta_lag`'s importance came just from telling calls and puts apart.
- **The first date (4 Jan) is dropped**, because it has no previous week.
- **No transformed copies.** `sqrt(T)` and `T·IV` were *not* added. A tree only cares about the order of a feature's values, and √T has the same order as T, so these add nothing for a tree. (They matter for neural networks.)

### 6. Walk-forward testing

The old code trained on Jan–Oct and tested once on Nov–Dec. One split gives one number, which may be luck.

Walk-forward (expanding window): for each month from March to December, train a new forest on **all earlier months**, then price that month. This gives 10 out-of-sample test months, all priced by a model that never saw them. Jan–Feb are only used for training.

Why not a random split? Options on the same day are almost copies of each other, and future data would leak into training.

**Benchmark:** each month also reports BS priced with last week's IV (`iv_lag`), with no model at all. An ML model is only useful if it beats this.

### 7. Model settings (unchanged from the original)

| Setting | Value | Why |
|---|---|---|
| `n_estimators` | 100 | More trees beyond ~100 barely change results |
| `min_samples_leaf` | 5 | Each leaf holds at least 5 options, which limits overfitting |
| `random_state` | 0 | Repeatable results |

The full run takes a couple of minutes (10 forests on up to 385k options each).

### 8. How the code works

1. **Load** the cached weekly SPX 2023 chains (52 Wednesdays).
2. **Clean:** keep calls only (430k quotes) with a bid, T > 0, and no missing values.
3. **Features:**
   - `moneyness`, `log_mF`, and the target `price_over_S`.
   - For each date, look up each option on the previous date by `optionid` to get `iv_lag`. Missing options get the nearest neighbour (`cKDTree`).
   - `delta_lag`, `atm_iv_lag` and `spot_ret` are then built from the lagged data.
4. **Walk-forward:** for each test month, fit the forest on all earlier months, predict price ÷ S, multiply by S. Also price the benchmark.
5. **Results:** RMSE per month for the forest and the benchmark; overall MAE, RMSE, R²; feature importance of the last forest.
6. **Plot:** predicted vs market prices.

### 9. Results (SPX calls, 2023, Wednesdays, Mar–Dec, 372,888 options, all out of sample)

RMSE (index points ≈ $) per test month:

| Month | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|
| Random forest | 15.10 | 8.25 | 5.51 | **18.38** | 6.77 | 7.00 | 4.97 | 5.43 | 6.45 | 5.65 |
| Benchmark (`iv_lag`) | 7.31 | 3.85 | 4.21 | 6.36 | 5.82 | 6.20 | 3.39 | 4.71 | 5.93 | 5.94 |

All test months together:

| Method | RMSE ($) |
|---|---:|
| BS with last week's IV (benchmark) | **5.55** |
| Random forest (MAE 5.28, R² 0.9998) | 9.24 |

Feature importance (forest trained on Jan–Nov):

| Feature | Importance |
|---|---:|
| `log_mF` | 0.97 |
| `moneyness` | 0.02 |
| `T` | 0.005 |
| `delta_lag` | 0.003 |
| All others | ≈ 0 |

**Interpretation:**
- **The RF does not beat the benchmark.** "Use last week's IV" is very hard to beat, because IV changes from week to week are mostly unpredictable market news. The RF only beats it in December (5.65 vs 5.94), with the most training data.
- **June 2023 fails (18.4).** Market vol fell to levels not seen in training, and trees cannot predict outside the range they were trained on.
- **The importances show the forest has learned the curve's *shape*.**
  - **Moneyness dominates.** Most of the variation in call price ÷ S is intrinsic value: deep in the money ≈ 1 − K/S, far out of the money ≈ 0. That is set by moneyness, so `log_mF` takes almost all the credit.
  - **`log_mF` crowds out its look-alikes.** `moneyness` is nearly the same feature, so it gets little extra credit. `delta_lag` mostly re-describes moneyness, so it also gets little.
  - **The vol level is almost ignored.** `iv_lag` is ≈ 0, so the forest cannot follow shifts in the overall vol level (June).
- **The importance scores measure training error, not prediction.** They show how much each feature reduced the training error and split credit between correlated features. Permutation importance on test months would show what actually helps prediction.
- **R² of 0.9998 is misleading.** Prices range from cents to thousands of dollars, so any sensible model explains almost all of the variance. RMSE against a benchmark is the meaningful measure.

### 10. Traps and intricacies

- **Leakage through IV and delta** (section 3). This is a common mistake in the literature; see Ruf & Wang (2020).
- **Random train/test splits** leak same-day information. Use time-ordered splits.
- **Trees cannot extrapolate.** If the target can drift outside the training range (prices, vol levels), predictions fail when the market regime changes (June 2023).
- **R² looks excellent for any option pricing model.** Compare RMSE against a benchmark instead.
- **SPX has duplicate-looking quotes.** Monthly (AM-settled) and weekly (PM-settled) options can share a strike and expiry. They have different `optionid`s, so the lagged IV matches the right one.
- **Some IVs are absurd.** OptionMetrics caps IV at 3.0, and ~10k quotes have IV > 1 (very short-dated, far out of the money).
- **The lag is a week, not a day.** Only weekly chains are stored, while Heston uses the previous trading day.

### 11. Limitations

- One year (2023), one underlying (SPX), weekly data. 2023 was a calm, falling-vol year.
- The lag is one week.
- The RF price surface is piecewise flat. It is not smooth and does not guarantee no-arbitrage.
- The RF is not yet in `compare_funcs.py`, so it is not in the comparison plots with the other methods.

### 12. Next steps

1. **Predict the IV change** instead of price ÷ S. It was tested: $7.76 vs $9.69, and no June failure.
2. **Daily lag.** Store the previous trading day's chains (already downloaded for the Heston calibration), to match Heston's information.
3. **More data:** several years (e.g. 2019–2023) so the model sees different vol regimes.
4. **Phase 2:** gradient boosting (LightGBM / HistGradientBoosting) with the same inputs and walk-forward test, always reporting the benchmark.
5. **Add the RF to `compare_funcs.py`**, so it appears in the maturity, moneyness and sector plots.

**References:**
- Breiman, L. (2001). Random forests. *Machine Learning* 45, 5–32.
- Gu, S., Kelly, B., Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies* 33(5).
- Ruf, J., Wang, W. (2020). Neural networks for option pricing and hedging: a literature review. *Journal of Computational Finance* 24(1).
