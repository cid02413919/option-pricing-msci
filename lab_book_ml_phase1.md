---

## 17/11/2026 – ML Phase 1: Random forest baseline

**Aims:**
- Turn the first random forest (RF) into a fair, comparison-ready model that plugs into `compare_funcs.py`.
- Price calls and puts, add volatility-based features, test with walk-forward validation, save the trained model.

**Files changed:**
- `source/ML_models/random_forest.py`: rewritten as functions (`add_features`, `train`, `predict_chain`, `walk_forward`, `save_model`, `load_model`) plus a test run on SPX 2023.
- `source/classical_models/comparing_models/compare_funcs.py`: two new methods, `"rf"` and `"iv_lag"`.
- `source/classical_models/comparing_models/error_overall.py`: switches for the two new methods.
- `.gitignore`: ignores `source/ML_models/saved/` (models can be regenerated).

---

### 1. Key idea in plain words

An option's price is fixed by a few contract terms (strike K, time to expiry T, today's price S, interest rate r, dividend yield q) plus one unknown: how much the market expects the price to move, the **volatility**. Black–Scholes (BS) turns a volatility into a price. The volatility that makes BS give exactly the market price is the **implied volatility (IV)**. Every pricing method in this project is really a way of guessing the IV (or, for Heston, the whole volatility process).

The RF here learns to guess each option's IV from information available on the pricing day. BS then turns that IV into a price.

### 2. Why start with a random forest

A random forest is many decision trees. Each tree splits the data with yes/no questions ("is T < 0.1?", "is the option far out of the money?"). Each tree is trained on a random sample, and the forest averages all of them (Breiman, 2001).

| Option | Why not first |
|---|---|
| Linear regression | Option prices depend on the inputs in strongly curved, interacting ways; a straight-line model cannot capture this without hand-made features. |
| Gradient boosting (LightGBM, XGBoost) | Usually more accurate than RF, but has more settings to tune and overfits more easily. A better *second* model, once a baseline exists (Phase 2). |
| Neural network | Smooth and flexible, but needs feature scaling, careful tuning and much more data handling. Phase 3. |
| Gaussian process | Training cost grows with n³; impossible for ~850k options. |

The RF was chosen first because it:
- has few settings and works well with defaults
- needs no feature scaling and tolerates outliers
- is fast to train (~5 s per fit here)
- gives feature importances
- is a standard benchmark in ML-in-finance work (e.g. Gu, Kelly & Xiu, 2020)

It is the "first sensible ML model" that later models must beat.

### 3. The most important design decision: what is known when (data leakage)

**Leakage** means the model is given information it would not have in real use, which makes results look far better than they really are.

- **Today's IV and delta are leakage.** OptionMetrics computes `impl_volatility` and `delta` *from today's market price*. Giving them to the model means giving it the answer: BS with today's IV reprices the option with RMSE $0.65, almost perfectly (see Results). The roadmap's suggestion of adding `impl_volatility` and `delta` was therefore implemented in **lagged** form only:
  - `iv_lag` is the option's IV on the previous pricing date (one week earlier).
  - `delta_lag` is the BS delta recomputed from `iv_lag`.
- **This matches the other methods.** Heston is calibrated on the previous trading day, and the historical, GARCH and VIX vols only use earlier data. The RF is held to the same rule.
- **Today's spot price S is allowed.** It is known at pricing time, and every other method uses it too.

### 4. What the model predicts (the target)

Candidates, tested on SPX 2023 with the walk-forward set-up in section 6. The prototype was slightly simpler than the final code, so these numbers differ a little from the Results.

| Target | Avg monthly RMSE ($) | Comment |
|---|---:|---|
| Price / S (the old approach) | 9.3, 8.9, **28.1** (Apr, May, Jun) | Fails when the market regime changes |
| IV level, contract features only | 8.7, 9.3, **27.7** (Apr, May, Jun) | Same problem |
| IV level, + lagged IV features | 8.93 (Apr–Dec) | Better, still poor |
| **IV change since last week**, + lagged features | **6.53** (Apr–Dec) | Chosen |
| No model: BS with last week's IV (`iv_lag`) | 5.02 (Apr–Dec) | Naive benchmark |

Why IV rather than price:
- **Price scale.** Prices run from $0.05 to $6,000+, so a price model spends its effort on the expensive options. IV sits in roughly 0.05–1.5 for every option.
- **No drift.** IV does not rise with the index level, whereas prices do.
- **BS does the conversion.** Mapping IV to a price is handled exactly by the BS formula, including the put–call relationship.
- **Readable errors.** Errors can be read in "vol points".

Why the *change* in IV rather than its level:
- **IV persists.** Most of today's IV is explained by last week's IV, so predicting only the change is a much easier job.
- **Trees cannot extrapolate.** A tree can only output values it saw in training. When market vol fell to new lows in June 2023, the level models were stuck at old levels (RMSE ≈ 28). The change model is not affected, because "last week's IV" carries the level.

### 5. Features (inputs)

| Feature | Meaning | Why |
|---|---|---|
| `is_call` | 1 = call, 0 = put | Calls and puts are now both used (twice the data). Their IVs are nearly equal but not identical. |
| `log_mF` | ln(K / F), F = S·e^((r−q)T) is the forward price | Moneyness measured against the forward; 0 = at the money. Better than K/S when r, q ≠ 0. |
| `T` | Years to expiry | The vol smile and its changes depend strongly on maturity. |
| `r`, `q` | Interest rate, dividend yield | Standard BS inputs (little importance in practice). |
| `iv_lag` | This option's IV last week | The starting point the model corrects. |
| `delta_lag` | BS delta using `iv_lag` | Combines moneyness, maturity and vol in one number (roadmap request, leak-free version). |
| `atm_iv_lag` | Last week's median IV of ~1-month at-the-money options | Overall market fear level. |
| `spot_ret` | ln(S today / S last week) | Vol tends to rise when the market falls (the "leverage effect"), so this explains part of the IV change. |
| `is_new` | 1 if the contract was not quoted last week | Marks options whose `iv_lag` is borrowed from a neighbour (below). |

Notes:
- **New contracts (12% of quotes).** New weekly expiries and strikes have no IV last week. They take the IV of last week's closest option, measured by moneyness and √T. Its median error is 1.4 vol points, against 0.8 for contracts with their own history. This is better than discarding them.
- **No transformed copies.** `sqrt(T)` and `T·IV` were *not* added. A tree only cares about the order of a feature's values, and √T has the same order as T, so these add nothing for a tree. (They will matter for the neural network.)
- **Features tried and dropped:** the change in each option's moneyness since last week made April much worse (RMSE 20.7), through overfitting.

### 6. Walk-forward validation

The old code trained on Jan–Oct and tested once on Nov–Dec. One split gives one number, which may just be luck.

Walk-forward (expanding window): for each month from March to December, train on **all earlier months**, then price that month. This gives 10 out-of-sample test months, and every RF price in the comparison comes from a model that never saw that month.

Why not ordinary random K-fold cross-validation? Options on the same day are almost copies of each other, and future data would leak into training.

### 7. Model settings

| Setting | Value | Why |
|---|---|---|
| `n_estimators` | 100 | More trees beyond ~100 barely change results |
| `min_samples_leaf` | 200 | Each prediction averages at least 200 options. This smooths out bid–ask noise; it beat 5 (average monthly RMSE 6.53 vs 6.72) |
| `max_samples` | 100,000 | Each tree trains on a random 100k options: fast, and adds variety between trees |
| `random_state` | 0 | Repeatable results |

Also tried and not kept:
- **Weighting training by vega** (price sensitivity to vol), so that the training loss matches the dollar RMSE: 6.63 → 6.57, a negligible gain.
- **Shallower trees** (depth 6): 6.48, also negligible.

### 8. How the code works

1. `add_features(df)` takes the weekly chains and adds the features above. For each date it looks up each option on the previous date by `optionid`, using a nearest-neighbour search (`cKDTree`) for new contracts.
2. `train(df)` fits the forest with target = IV today − `iv_lag`.
3. `predict_iv` computes `iv_lag` + predicted change, and `bs_price` turns it into a call or put price.
4. `predict_chain(df, model)` returns a price for every row of `df`, aligned with `df`. This is the same interface as `heston_price_chain`. `df` must include the previous week, which supplies the lagged features.
5. `walk_forward(df)` loops over the months as in section 6 and returns out-of-sample prices plus a per-month table.
6. `save_model` / `load_model` store the model and its feature list with `joblib` in `source/ML_models/saved/rf_<secid>.joblib`. Loading checks that the features match.
7. In `compare_funcs.load_weekly_prices`:
   - `"iv_lag"` is just another BS vol column.
   - `"rf"` runs `walk_forward` on the same options the other methods price, cached in `source/data/cache_rf_prices_*.pkl` (delete to retrain). Sector ETFs get their own forest trained on their own data.

**Note:** switching on `"rf"` removes Jan–Feb from *every* method's comparison, because the RF has no out-of-sample prices for its first two training months.

### 9. Results (SPX, 2023, Wednesdays, Mar–Dec, 732,904 options, all out of sample)

| Method | RMSE ($) |
|---|---:|
| BS with today's implied vol (uses the answer, not a forecast) | 0.65 |
| BS with last week's IV (`iv_lag`) | **6.31** |
| Random forest | 7.76 |
| Heston (calibrated on previous day) | 15.85 |

RMSE ($) by days to expiry:

| Days | `iv_lag` | RF | Heston |
|---|---:|---:|---:|
| 0–7 | 2.38 | 4.42 | 2.44 |
| 7–30 | 4.10 | 4.75 | **3.38** |
| 30–90 | 4.76 | 5.24 | **4.15** |
| 90–365 | **5.97** | 7.46 | 7.99 |
| 365+ | **16.01** | 20.13 | 57.24 |

Walk-forward RMSE ($) per test month (from running `random_forest.py`):

| Month | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|
| RF | 12.55 | 7.59 | 5.98 | 7.27 | 6.18 | 11.46 | 5.51 | 5.25 | 5.48 | 4.93 |
| `iv_lag` | 11.71 | 5.81 | 4.65 | 6.12 | 5.86 | 5.45 | 3.96 | 4.90 | 5.64 | 4.97 |

Feature importance (final model): `delta_lag` 0.69, `log_mF` 0.17, `T` 0.10, the rest ≤ 0.02. Impurity importance splits credit between correlated features: `delta_lag` already contains moneyness and maturity, so these numbers show *where* the forest splits, not cause and effect.

**Interpretation:**
- **The RF beats Heston overall (7.8 vs 15.9), but not the naive benchmark (6.3).** "Use last week's IV" is very hard to beat because IV changes from week to week are mostly unpredictable market news. The RF adds noise when it tries to forecast them.
- **The RF closes the gap with more training data.** It gets close to, or slightly beats, `iv_lag` in Nov–Dec, when it has ~700k training options. With only ~2 months of data (March) it is worst. So data quantity matters, which supports using more years.
- **Heston is better than both for options under 90 days.** It only loses overall because of very large errors on options over 1 year, which dominate the dollar RMSE.
- **A few options dominate RMSE.** The worst 1% of options cause ~45% of the squared error for both `iv_lag` and RF (long-dated, expensive options). Dollar RMSE mostly measures long-dated options; percentage errors would tell a different story.

### 10. Traps and intricacies

- **Leakage through IV and delta** (section 3). This is the most common mistake in the literature; see Ruf & Wang (2020) for a review.
- **Random train/test splits** leak same-day information. Use time-ordered splits.
- **Trees cannot extrapolate.** Predict changes or ratios, not levels that can drift.
- **Dollar RMSE vs training loss.** The model minimises error in IV, but is judged in dollars, where long-dated options carry most of the weight.
- **SPX has duplicate-looking quotes.** Monthly (AM-settled) and weekly (PM-settled) options can share a strike and expiry. They have different `optionid`s, so the lagged IV matches the right one.
- **Some IVs are absurd.** OptionMetrics caps IV at 3.0, and ~10k quotes have IV > 1 (very short-dated, far out of the money). They are kept, because the other methods are tested on them too.
- **The lag is a week, not a day.** Only weekly chains are stored, while Heston uses the previous trading day. This handicaps the RF and `iv_lag`.

### 11. Limitations

- One year (2023), one underlying (SPX), weekly data. 2023 was a calm, falling-vol year.
- The lag is one week (see above).
- The RF price surface is piecewise flat. It is not smooth and does not guarantee no-arbitrage (e.g. prices that fall smoothly as the strike rises).
- ETF options are American; pricing them with BS (here and in the other methods) is only exact for dividend-free calls.
- The prototype tests in sections 4 and 7 used a simpler set-up (200k training sample, own-contract lags only), so their numbers are indicative.

### 12. Next steps

1. **Daily lag.** Store the previous trading day's chains (they are already downloaded for the Heston calibration) so `iv_lag` is one day old, like Heston's input.
2. **More data.** Several years (e.g. 2019–2023, including the 2020 crash and the 2022 bear market), so the model sees different vol regimes.
3. **Phase 2.** Gradient boosting (LightGBM / HistGradientBoosting) with the same target, features and walk-forward test, so it can be compared directly with this RF. Always report the `iv_lag` benchmark.
4. **Percentage errors.** Report errors by moneyness and maturity buckets and in percentage terms, not just overall dollar RMSE.

**References:**
- Breiman, L. (2001). Random forests. *Machine Learning* 45, 5–32.
- Gu, S., Kelly, B., Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies* 33(5).
- Ruf, J., Wang, W. (2020). Neural networks for option pricing and hedging: a literature review. *Journal of Computational Finance* 24(1).
