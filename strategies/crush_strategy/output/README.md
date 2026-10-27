# Overnight crush-residual reversion (ZS vs ZM / ZL)

Research on how soybeans trade against meal and oil intraday, and whether overnight
dislocations in that relationship are tradeable.

## Pipeline

| File | What it does |
|---|---|
| `data_ingest.py` | 1-minute ZS/ZM/ZL data -> sessions, switch-free returns, contract-switch detection, back-adjusted prices, session-aligned 5-minute bars |
| `kalman_filter.py` | Kalman filter for `log ZS = alpha + b_meal log ZM + b_oil log ZL`; hedge frozen at the previous session close; residual |
| `features.py` | Residual z-score (previous 20 days, same session type) and the evening-session signal |
| `strategy.py` | Final strategy: next-open entry, triple-barrier exits, costs, performance, equity/drawdown chart |
| `config.yaml` | Every param, the split dates, the validation grid and the gates |
| `validate.py` | Walk-forward validation over the grid on development only; results, plots and gates saved to `runs/` |
| `holdout.py` | One-time holdout, locked until the latest validation passes every gate on the current config |
| `meta_model_reference.py` | Reference only: feature engineering, IC table, meta-model walk-forward (did not survive the holdout) |

Run: put the raw 1-minute parquet at `data/raw/latest_df_backup.parquet`, then `python -m strategies.crush_strategy.validate`.

## Final specification

- **Signal:** evening session (19:00-07:45 CT), first 5-minute bar with |z| >= 2 on the Kalman residual; re-arms below 0.5
- **Entry:** next bar's open; hedge frozen at the signal bar
- **Exits (first touched):** profit 1 sigma, stop 3 sigma, time 12 hours
- **Costs:** 1 tick per leg round trip (0.58 c/bu on the spread); every signal taken, one trade at a time

## Results (1 tick per leg; % of bean notional, USD per 5,000 bu hedged)

| Sample | Trades | Hit rate | Net / trade | Return | Max DD | Sharpe |
|---|---|---|---|---|---|---|
| Development, Jan 2021 - Nov 2024 | 191 | 79% | 0.84 c | +11.5% | -2.8% | 1.57 |
| Holdout, Dec 2024 - Dec 2025 (run once) | 36 | 69% | 0.06 c | +0.1% | -2.5% | 0.07 |

## Parameter sensitivity (development only)

![Parameter sensitivity](../figures/parameter_sensitivity.png)

Sensitivity maps, not an optimisation: settings were fixed before the holdout. How to read them:
- **Entry threshold x time stop:** a broad plateau (Sharpe ~1.3-1.7 for |z| 1.5-2.5 at 12-24h); only 6h is clearly weaker.
  The result does not hinge on one exact threshold.
- **Entry threshold x cost:** smooth and monotonic; every extra half tick per leg costs ~0.5-0.7 Sharpe at every threshold.
- **Exit grid:** the final setting (profit 1 sigma, stop 3 sigma) is the best cell but sits in the grid's corner, and
  performance rises steadily toward small profit targets and wide stops. The edge of the grid is not a confirmed
  optimum; that direction was not explored further to avoid tuning after the holdout.

## What was learned

- **Relationship:** beans hedge against products at about 0.59 meal / 0.36 oil, below board-crush value shares.
  The Kalman filter gave the tightest residual and the most stable hedge ratios.
- **Dynamics:** dislocations are partly transient (revert within hours, mostly overnight) and partly permanent.
  The day session shows no reversion; on daily bars large dislocations tend to continue.
- **Economics:** the edge is roughly 0.4-0.7 sigma per trade and larger in high-volatility regimes; costs are fixed
  in cents, so profitability depends on volatility and execution. The holdout year lacked high-volatility dislocations.
- **Method notes:** daily refits fake mean reversion (always measure with the hedge fixed); the z-score window was
  chosen on calibration, not results; meta-model features picked in development did not hold out of sample.

Raw market data is not included in this repository.
