# Wheat microstructure: order-flow imbalance

## 0. Hypothesis (write BEFORE looking at results)
- **Thesis:** short bursts of one-sided aggressive flow in ZW overshoot, then partly revert (or: persist - `direction` tests both)
- **Why it should pay:** liquidity providers get paid for absorbing impatient flow
- **Kill criteria:** no window/threshold plateau beats half-spread costs across folds

## Process
| Gate | Where | Pass when |
|---|---|---|
| 1. Data | `research.ipynb`, `core_quant.utilities.frame_summary` | no gaps / duplicates / look-ahead |
| 2. Features & signal | `core_quant.feature_registry`, `core_quant.validation` (IC, NW t-stat, block permutation) | signal has IC with an honest p-value |
| 3. Walk-forward validation | `python -m strategies.<name>.validate` | every gate in `config.yaml` passes |
| 4. Holdout (once) | `python -m strategies.<name>.holdout` | result close to the walk-forward OOS |

Every validation / holdout run is saved in `runs/` with its config, git commit, metrics and plots.

## Log
| Date | Run | Change | Result |
|---|---|---|---|
| 2026-10-27 | – | Loaded 1 year of ticks (2025-11-20 → 2026-10-27, 2,387,069 trades, parquet). Holdout locked at 2026-08-01 | Step 2 done |

## Data notes (step 3, in progress)
- Prices: 0 off-grid (all on the 0.25 c tick). Only `kind == "trade"`. Times stored as UTC (tz removed in `load_ticks`).
- Largest jumps fall in three groups:
  - session opens (00:00/01:00 UTC): roll candidates, e.g. 2026-08-17 18.75 c (Sep → Dec?)
  - 16:00 UTC (11:00 CT) on 2026-07-10 and 2026-09-11: WASDE releases, real events
  - 2026-09-02 01:55:14: a jump out and back within 10 ms, possible bad print

## Next session: where we left off
Loop: 1 data ✅ · 2 holdout ✅ · **3 data checks (in progress)** · 4 features · 5 IC exploration · 6 rule strategy · 7 ML model · 8 holdout

To finish step 3 in `research.ipynb`:
1. Fix the session-hours check: assign the Chicago index (`chicago = ticks.index.tz_localize("UTC").tz_convert("America/Chicago")`), then check the 07:45–08:30 CT break and the 13:20 CT close.
2. Redo the jumps with the **signed** `diff()` and `.nlargest(20)` (skips the NaN). Rolls in contango should be jumps **up** that stay up.
3. `frame_summary(ticks)` plus a count of duplicate rows (same time, price and size).
4. Ticks **per date** and the 10 quietest dates (missing days / outages).
5. `ticks["size"].describe()` and the largest trades.
6. Write the findings list here, under Data notes.

Then decide: does `load_ticks` need roll handling before step 4?
