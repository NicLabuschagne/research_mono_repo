# Bean fair value (Kalman crush) - current state

Two layers. **Layer 1 (pricing)**: a fair value for soybeans from meal and oil, judged by statistics only.
**Layer 2 (execution)**: which divergences to trade, in which instrument, with what exits. Full history:
`RESEARCH_LOG.md`. Development Jan 2021 - Nov 2024; **holdout from 2024-12-01 stays locked** until the whole
process is fixed.

## Code (one file per stage; new tests go into the stage's file, not new files)

| File | What it does | Run |
|---|---|---|
| `config.yaml` | data path, split, trigger level, z half-life, cost | |
| `fair_value.py` | Kalman crush fair value, the gap in ZS ticks (live / prior close), adaptive z, sessions | `python -m strategies.bean_divergence.fair_value` |
| `triggers.py` | one trigger per move: first bar with \|z\| >= 1.645, side, z and gap kept as features | `... .triggers` |
| `outcomes.py` | forward moves after triggers vs random: horizons, win / loss tails, race | `... .outcomes` |
| `charts.py` | every trigger on the last 4 development days, beans vs fair value | `... .charts` |
| `labels.py` | triple-barrier labels, net of costs (next) | |
| `meta_model.py` | meta-label features + logistic baseline (after labels) | |

Add `prior_close` to any run for the prior-close gap; the default is the live gap. Units are ZS ticks
throughout (1 tick = 0.25 c/bu = $12.50; 4 ticks = 1 c).

## What we know (layer 1)

- **The crush is the fair value.** It explains 64-89% of bean moves over 30 min - 4 h; corn, wheat and USD/CNH add
  almost nothing.
- **Live gap is the base** (hedge from the bar before = the filter's prediction error). Fast: typical size
  3.9 ticks, correlation with itself 0.75 after 5 min, 0.24 after 30 min, 0.05 after 1 h. The prior-close gap is
  slow (17.2 ticks; 0.84 after 1 h), trends within a session and jumps when re-frozen at session starts.
- **z** = gap / its own recent size (4-hour exponential RMS, past bars only) - not the filter's own variance.
- **Triggers** (live, \|z\| >= 1.645): ~40 a week (US ~20, Europe ~13, Asia ~7), both halves.
- **Raw edge, before costs:** beans +0.3 to +0.6 ticks per trigger from 5 min to 1 h (t 2-3); the spread
  +0.6 to +1.0 tick at 5 min. After a 1-tick cost, taking every trigger loses.
- **The edge is in the tails, tilted not one-sided:**

  | 30 min after, 2023-24 | >= +5 | <= -5 | >= +10 | <= -10 |
  |---|---|---|---|---|
  | triggers, beans | 32.8% | 29.4% | 19.2% | 17.8% |
  | triggers, spread | 19.2% | 13.2% | 5.8% | 4.8% |
  | random, beans | 27.8% | 27.5% | 14.7% | 14.5% |

  The moves after a trigger are large (5-10+ ticks often); the net mean is small because wins and losses nearly
  cancel. Separating them is the meta-model's job.
- Higher z thresholds do not help (worse early: beans keep going first), so z is a feature, not a filter level.

## Decisions

- Live Kalman gap is the outright base; one trigger per move at \|z\| >= 1.645; z and gap size are features.
- Every trigger is labelled **both** as outright beans and as the crush spread; the meta-model picks the
  instrument or skips. "Which side moved into the divergence" is the main input for that choice.
- **Labels net of costs**, so a predicted win probability above 0.5 means positive expected value. Outright: 1 tick
  per round trip. Spread: three cost cases - (a) all legs crossed (ZS $12.50, ZM $10, ZL $6 per tick),
  (b) autospreader with one passive leg, (c) the exchange-listed crush (fixed ratios; measure its tracking).
- Spread target: 4-5 ticks per trade after costs, from the meta-model's selection.

## Next: layer 2

1. **Triple-barrier labels** (`labels.py`). Time limit from the IC decay: 30-60 min (the live gap is gone by 1 h).
   Barriers in volatility over that horizon, not the 14-hour ATR. The stop no tighter than the target - triggers
   often move against you first. Rough maths for symmetric barriers of R ticks and a 1-tick cost: net per trade =
   (2p - 1) x R - 1, so each +1 point of win rate adds 2R/100 ticks; 5 ticks net needs R of 15-20 ticks and a win
   rate of 65-70%.
2. **Meta-label model** (`meta_model.py`): labels only where the primary fired (side +1 / -1); baseline
   L2-regularised logistic regression (ridge's classifier twin, gives probabilities). Walk-forward with purge and
   embargo, since labels overlap in time. Take trades above the threshold chosen on development data.
3. Assess the model and the strategy as a whole; only then the holdout.

Features (all known at the trigger bar):
- |z| and gap size at the trigger (|z| has outliers up to ~50 - cap or log it).
- **Which side moved**: over the last 5 / 15 / 30 min, with the trigger bar's hedge held fixed:
  beans move = (log ZS_t - log ZS_t-k) x ticks; fair move = (beta_meal x d log ZM + beta_oil x d log ZL) x ticks;
  beans share = |beans move| / (|beans move| + |fair move|). Beans moved -> outright; products moved -> spread.
- **Position in the day's range**: (close - running low) / (running high - running low), from the RUNNING high /
  low up to the trigger (the full day's would leak the future); also for the current session. The minute data has
  true high / low per leg (`build_bars` drops them; add them back).
- Volume of the trigger bar vs its usual level at that time of day; a new high / low of the day on this bar.
- **Rolling hit rate** of recent triggers - counting ONLY trades whose barrier was hit before this trigger's entry.
  Earlier entries still open would leak the future (with ~40 triggers a week and 30-60 min holds, that is common).
- Session, time of day, the reopen bar after the 13:20-19:00 break, speed of the move into the trigger, ATR from
  true high / low. Later, with tick data: trade imbalance, quote depletion.
