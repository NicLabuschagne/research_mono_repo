# Overnight crush-residual reversion (ZS vs ZM / ZL)

Soybeans should move with their crushed products, meal and oil. Overnight, beans can drift out of line
with the products for a few hours and then snap back. The strategy trades that snap-back.

## Pipeline

| File | What it does |
|---|---|
| `data_ingest.py` | 1-minute ZS/ZM/ZL data -> sessions, switch-free returns, contract-switch detection, back-adjusted prices, session-aligned 5-minute bars |
| `kalman_filter.py` | Kalman filter for `log ZS = alpha + b_meal log ZM + b_oil log ZL`; hedge frozen at the previous session close; residual |
| `features.py` | Residual z-score (previous 20 days, same session type) and the evening-session signal |
| `strategy.py` | Trade rules: next-open entry, entry cutoff, triple-barrier exits with a session-close time stop, costs |
| `config.yaml` | Every param, the split dates, the validation grid and the gates |
| `validate.py` | Walk-forward validation over the grid on development only; results, tear sheet and gates saved to `runs/` |
| `holdout.py` | One-time holdout, locked until the latest validation passes every gate on the current config |

Run: put the raw 1-minute parquet at `data/raw/latest_df_backup.parquet`, then `python -m strategies.crush_strategy.validate`.
Every run saves a tear sheet (`tearsheet.png`, metrics in `metrics.json`, `monthly_returns.csv`).

## Specification

- **Hedge:** Kalman filter on 5-minute bars, hedge frozen at the previous session's close
- **Signal:** evening session (19:00-07:45 CT), first bar with |z| >= 1.5 on the residual; re-arms below 0.5
- **Entry:** next bar's open; hedge frozen at the signal bar; **no entries with less than 3 h to the session close**
- **Exits (first touched, on bar closes):** profit 1 sigma, disaster stop 4 sigma, or the **close of the entry
  session** (12 h cap). Trades are never carried into the day session or the next evening.
- **Costs:** 1 tick per leg round trip (0.58 c/bu); one trade at a time

## Results (1 tick per leg; returns in % of bean notional)

| Sample | Trades | Win rate | Net / trade | Return | Max DD | Sharpe | Calmar | Profit factor |
|---|---|---|---|---|---|---|---|---|
| Development, Jan 2021 - Nov 2024 | 292 | 69% | 0.87 c/bu | +18.3% | -1.1% | 2.90 | 4.20 | 2.38 |
| Holdout, Dec 2024 - Dec 2025 (see note) | 57 | 67% | 0.47 c/bu | +2.6% | -1.4% | 1.33 | 1.82 | 1.59 |

Development walk-forward: out-of-sample Sharpe 2.53, all 6 folds positive, every gate passed
(`runs/20261029-112032_validation`). Holdout: `runs/20261029-112059_holdout`.

**The holdout is a contaminated check, not a clean test.** The same period was used to test the original
version of this strategy (Sharpe 0.07) before this recalibration. No choice below was made on holdout data,
but its weakness was known. A clean test needs data after Dec 2025.

About 6 trades a month (1-16), median hold ~5 h. Capacity is set by meal: ~10 bean contracts per trade keeps
the hedge at ~4% of evening meal volume in the 30 minutes after entry (p90 10%).

## What the research found (development only: chosen on 2021-22, confirmed on 2023 - Nov 2024)

The full research (scripts and every run) is in commit `29c1cfd` on branch `crush/kalman-calibration`.

1. **Exits mattered most.** About 60% of the original trades carried past the session close into the day
   session and lost money on average. Closing at the session end doubled the Sharpe and halved the drawdown,
   mostly by removing variance. Tight stops hurt (they exit at the biggest dislocation); a 0.5-1 sigma target helps.
2. **Time left in the session is the strongest feature** (IC 0.34, then 0.32 out of sample). Reversion builds
   over hours, so late signals run out of time. A 3 h entry cutoff (plateau 3-8 h) cut ~120 losing trades.
3. **Kalman filter:** maximum likelihood says the hand-set noise levels are wrong, but only the drift-to-noise
   ratio moves the hedge, and the hand-set ratio already sits where the residual reverts best. An MLE-fitted
   filter lost at trade level, and the hand-set speed held at every bar size.
4. **z-score:** the tails are fat (|z| >= 2 fires on ~7% of bars, not 4.6%). A percentile z fixes the
   calibration but did not trade better.
5. **Day session:** real reversion but below cost. 180 trade-level variants (frozen or live hedge, thresholds,
   exits); none held in both halves, and adding the best one hurt the strategy.
6. **Bar size:** 5 minutes beats 10, 15 and 30 on every validation measure; longer bars enter later and lose
   edge in quiet years. 1 minute was not tested (three-leg execution and capacity).
7. **Not confirmed:** a volatility-regime filter (strong in 2021-22, flat in 2023-24), and the earlier
   meta-model (failed out of sample).

**Open problem:** the edge scales with volatility. In 2023-24 it was ~0.3 c/bu per trade after costs
(+/- 0.16), against 1.4 c in 2021-22.

**Multiple testing:** many variants were tried (Kalman specs, speeds, z-scores, ~560 exit settings, feature
screen, day session, 4 bar sizes, several validation grids). Discount the development Sharpe accordingly.

Raw market data is not included in this repository.
