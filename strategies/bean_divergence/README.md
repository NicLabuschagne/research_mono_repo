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
| `labels.py` | triple-barrier labels, net of costs, for every trigger: beans and spread, a fixed grid of barriers, plus random entries | `... .labels` |
| `meta_model.py` | meta-label features (leakage-checked, screened) + walk-forward logistic per instrument; add `fit` to skip the feature rebuild | `... .meta_model` |
| `volume_bars.py` | the signal on volume bars (458 ZS contracts): triggers, clock-time labels, the same model, a cut-off leakage check | `... .volume_bars` |

Add `prior_close` to any run for the prior-close gap; the default is the live gap. Numbers below are from the
roll-fixed data (2026-11-02). Units are ZS ticks
throughout (1 tick = 0.25 c/bu = $12.50; 4 ticks = 1 c).

## What we know (layer 1)

- **The crush is the fair value.** It explains 64-89% of bean moves over 30 min - 4 h; corn, wheat and USD/CNH add
  almost nothing.
- **Live gap is the base** (hedge from the bar before = the filter's prediction error). Fast: typical size
  3.9 ticks, correlation with itself 0.75 after 5 min, 0.24 after 30 min, 0.05 after 1 h. The prior-close gap is
  slow (17.2 ticks; 0.84 after 1 h), trends within a session and jumps when re-frozen at session starts.
- **z** = gap / its own recent size (4-hour exponential RMS, past bars only) - not the filter's own variance.
- **Triggers** (live, \|z\| >= 1.645): ~40 a week (US ~20, Europe ~13, Asia ~7), both halves.
- **Raw edge, before costs:** beans +0.2 to +0.5 ticks per trigger from 5 min to 1 h (t 1.5-2.8); the spread
  +0.6 to +0.9 tick at 5 min. After a 1-tick cost, taking every trigger loses.
- **The edge is in the tails, tilted not one-sided:**

  | 30 min after, 2023-24 | >= +5 | <= -5 | >= +10 | <= -10 |
  |---|---|---|---|---|
  | triggers, beans | 32.9% | 29.6% | 19.3% | 18.0% |
  | triggers, spread | 19.3% | 13.2% | 5.8% | 4.9% |
  | random, beans | 27.4% | 28.2% | 14.5% | 15.0% |

  The moves after a trigger are large (5-10+ ticks often); the net mean is small because wins and losses nearly
  cancel. Separating them is the meta-model's job.
- **Best move in the trade's favour** (not net; 5-min closes; median, with random entries in brackets):

  | | 30 min | 1 h | 2 h | time to the best (1 h window) |
  |---|---|---|---|---|
  | Europe beans | 4.1 (3.1) | 6.5 (5.0) | 11.3 (9.0) | median 35 min |
  | US beans | 6.2 (5.1) | 9.9 (8.3) | 14.2 (12.4) | median 35 min |
  | Europe spread | 3.1 (1.7) | 4.5 (2.7) | 6.5 (4.2) | median 35 min |
  | US spread | 3.4 (2.4) | 4.9 (3.8) | 6.7 (5.7) | median 35 min |

  Within 1 h a quarter of Europe triggers reach 14.5+ ticks on beans and 10% reach 26.5+. The Europe spread runs
  almost twice as far as random - the clearest edge, if its costs can be managed.
- Higher z thresholds do not help (worse early: beans keep going first), so z is a feature, not a filter level.

## Labels (layer 2, step 1 - `labels.py`, 2026-11-02)

- **Barrier unit = sigma_1h**, the expected 1-hour move: exponential RMS of 5-min returns over the last 4 h (bars
  before the trigger) x sqrt(12). Median ~11 ticks Europe, ~15 US (spread: 6 / 7). Of recent, same-time-of-day and
  daily (Lopez de Prado's default) volatility, this made Europe, US and both halves most alike.
- Target = k x sigma_1h (k 0.5 / 0.75 / 1.0), stop = 1x or 1.25x the target, time limit 60 min; entry at the next
  open; beans touches on 1-min high / low (same minute -> stop), spread on 1-min closes; fills AT the barrier.
  Random entries now come out at ~0 gross, as they should.
- **Every trigger, no meta-model yet** (ticks per trade, k 0.75 - 1.0, both halves similar):

  | | gross | net | random net | win% after costs |
  |---|---|---|---|---|
  | beans, Europe | +0.4 to +0.7 | -0.3 to -0.6 | -1.1 | 50-53% |
  | beans, US | 0.0 to +0.1 | -0.9 to -1.0 | -1.1 | 50-54% |
  | spread, beans passive (1.1 tick), Europe | +1.1 to +1.2 | **0.0 to +0.1** | -1.1 | 52-58% |
  | spread, beans passive, US | +0.2 to +0.4 | -0.7 to -0.8 | -1.1 | 49-54% |

  All legs crossed costs 2.1 ticks: 60-99% of spread trades fall below the 4 x cost minimum barrier. Labels are
  balanced (49-57% ones), so no class weights are needed. 19% of triggers hold through a session break.
- Reading: the triggers beat random by 0.4-1.2 ticks in every case; Europe spread is already breakeven untouched.
  The meta-model has to lift the win rate by a few points - the label set gives it ~8,000 trades to learn from.

## Features (layer 2, step 2a - `meta_model.py`, 2026-11-02)

25 pre-registered features per trigger (list in the file's docstring), all known at the trigger close. **Leakage
check:** every input is cut at a date and the features before it must not change - passes. Screen at the main label
(k 1.0, stop 1.25x), per half: rank correlation (IC) with net ticks and AUC vs the label. Standard error of the IC
is ~0.016 per half, so |IC| > 0.03 is about 2 standard errors.

- **Every feature is weak alone** (|IC| <= 0.06, AUC 0.49-0.53) - normal for a meta-model; it has to combine them.
- **Spread - consistent in both halves:** Europe session (+), time of day (tod_sin +), oil-driven fair-value move
  (oil_share_30 +), first 30 min of a session (-), ZS volume vs 20 days (+), new high / low of the day (-).
- **Beans - consistent:** ZS volume vs today so far (+0.033 / +0.035) and vs the previous day's session (+0.021 /
  +0.041) - both better than vs 20 days - and the volatility regime (sigma_ratio +). Beans results are noisier
  (per-trade std ~13 ticks vs ~6 for the spread).
- **No information yet:** range position (flips sign between halves), hit rate, hedge drift, side, beans share
  (for both instruments). Kept for the model as pre-registered; the model's coefficient stability will judge them.
- Redundant: session dummies with tod_sin / tod_cos (0.74 / -0.87); the three "volume vs today" features (0.72-0.76).
  vol_*_today is missing for the first 6 bars of a trading day (6.7%).

## Meta-model, first fit (layer 2, step 2b - `meta_model.py fit`, 2026-11-02)

L2 logistic per instrument, all 25 features, walk-forward in 6 half-year blocks 2022 - Nov 2024 (train only on
trades that exited a day before the block), C picked inside each training window.

- **Barely better than a coin, out of sample:** AUC 0.52 (beans) / 0.53 (spread); log loss equal to the base
  rate. The inner validation chose the strongest shrinkage on the grid (C 0.003) in 11 of 12 blocks - the data
  says "trust the features as little as possible". Predicted p spans only 0.45 - 0.58, but it is ordered: top
  decile wins 56-57%, bottom 50-51%.
- **Breakeven win rate is ~55%, not 50%:** the stop is 1.25x the target, so mean win +10.8 / loss -13.1 ticks
  (beans), +5.1 / -6.3 (spread). Win share rises with the threshold, net turns positive only above p 0.58:
  +0.2 / +0.4 ticks at 1.6 / 2.0 trades a week - not a high-frequency book.
- **It does not beat the simple rule:** "Europe spread only" = +0.21 ticks per trade at 11.5 a week (daily t 1.4).
  The router (higher expected value of beans / spread) is negative at every threshold.
- **Coefficients stable in all 6 blocks** (|mean / std| > 2): spread - crosses_break +, oil_share_30 +,
  hedge_drift -, time of day (tod_sin + / tod_cos -), first_30min -, log_abs_z +, session_us -, hit_rate -,
  new_extreme -, sigma_ratio +, vol_zs_20d +, vol_zm_today +. Beans - range_pos - (entering at the day's extreme
  is WORSE: the move keeps going), crosses_break +, beans_share_15 +, gap_change_15 +, vol_zs_today +,
  log_abs_z -, oil_share_30 +. Beans block 2023-07 picked C 1.0 and its coefficients are outliers.
- **Weak in both instruments** (|mean / std| < 1.6, or sign flips): side, gap_sigma, range_sigma, beans_share_30,
  session_europe (overlaps the time-of-day pair), vol_zs_prev, vol_zl_today, bars_since_calm.
- **Refit without those 8** (17 features): the same - AUC 0.523 / 0.528, log loss still equal to the base rate,
  coefficients steadier. The 8 carried nothing. Net still turns positive only above p 0.58 (1-2 trades a week).
- **Which stop** (same model refitted per label, target 1 sigma_1h, compared at the same trade count):

  | stop x target | beans: every trigger / top 10% by p | spread: every trigger / top 10% by p | breakeven win% |
  |---|---|---|---|
  | 0.5 (wins 2x losses) | -0.74 / -0.62 | -0.49 / -0.40 | 42-46 |
  | 0.75 | -0.64 / -0.77 | -0.44 / -0.21 | 49-51 |
  | **1.0 (symmetric)** | **-0.58 / -0.23** | **-0.36 / -0.05** | 52-53 |
  | 1.25 | -0.63 / -0.63 | -0.41 / -0.02 | 55 |

  Tight stops lose: the median worst move after a trigger is ~0.7 sigma_1h, so a 0.5x stop is hit before the
  reversion - the win share falls to 38% against 42% needed. Symmetric is best or tied, and with it the win share
  maps straight to expected value (breakeven 52.5%). Differences between 1.0x and 1.25x are small (0.05-0.4 ticks).

## Confirmation entry (2026-11-02, `labels.py` entry rules) - rejected

Instead of entering at the trigger: wait (at most 30 min, same session) for the first bar whose gap narrows
(gap_turn, ~6 min wait) or for |z| back inside 1.645 (z_back, ~7-9 min). 90-96% of triggers confirm.

- **It does not change the payoff shape.** The worst move within the hour barely moves (beans Europe 7.8 -> 7.7
  ticks, US 12.9 -> 12.2) and the best move shrinks a little. The move against you is not concentrated in the
  first bars, so one bar of confirmation does not dodge it.
- **Beans:** mixed by half (US better in 2021-22, worse in 2023-24) - no consistent gain. **Spread: waiting hurts**
  (Europe, symmetric stop: +0.04 / -0.05 at the trigger vs -0.33 / -0.48 after gap_turn) - its edge is front-loaded.
- **Beans after a trigger is nearly symmetric:** median best move = median worst (Europe 8.4 / 7.8, US 12.8 /
  12.9 ticks). Widening the beans barrier cannot help by itself. The asymmetric edge is the Europe spread
  (best 4.9 / worst 3.4 ticks).

## Tick probe, beans only (2026-11-02) - failed the bar, one lead

ZS trades Nov 2022 - Nov 2023 (time, price, size; no aggressor flag - fills with one timestamp = one order, side by
the tick rule). The feed carries ~72% of the minute file's volume; 15 roll days on another contract dropped.
1,759 triggers. Pass bar set beforehand: |IC| >= 0.05, same sign in both periods, or AUC +0.03.

- Flow imbalance (who pushed the gap open): no information (IC -0.02 to +0.03). Adding all 7 tick features to the
  bar model LOWERED out-of-sample AUC (0.532 -> 0.519 beans) - too little training data for 7 noisy inputs.
- **Lead: big orders in the push = fade works better (beans).** large_share_5 IC +0.029 / +0.076, order_size_5
  -0.012 / +0.086 (Nov 22 - May 23 / Jun - Nov 23); top minus bottom fifth +3.8 ticks in the second period.

**Pre-registered confirmation (fixed before seeing the new data):** beans ticks for the rest of development
(Jan 2021 - Nov 2022, Nov 2023 - Nov 2024). Only two features, definitions unchanged (large = an order of 20+
contracts; windows 5 min before the trigger close vs the 60 min before that): large_share_5 and order_size_5,
expected sign POSITIVE. Label: beans, k 1.0, stop 1.25x (symmetric reported alongside). Passes if, on the new data
only, IC >= +0.05 pooled and positive in each of 2021, 2022 and 2024. Then meal / oil ticks and a proper study;
otherwise microstructure stops here.

**Result (2026-11-02): FAILED - microstructure stops.** New data 26 Jan 2021 - 1 Aug 2022 (2,876 beans triggers,
IC standard error ~0.019): large_share_5 pooled IC -0.010 (2021 +0.012, 2022 -0.022); order_size_5 +0.013
(+0.011 / +0.021). Both far below +0.05; 2024 cannot rescue a pooled 0.05 (it would need ~+0.13 alone), so it was
not downloaded. The Jun - Nov 2023 effect was chance. (Europe-only ICs of +0.03 to +0.055 are a post-hoc subgroup -
not followed up.)

## Decisions

- **Two books (target design, 2026-11-02):** (1) the **intraday book** - 5-min triggers traded in the outright,
  short holds, high frequency: the bread and butter (this README's current work); (2) a **multi-day book** on 1H
  where size is built - traded as the SPREAD, because the longer a position is held the less outright directional
  exposure we want (overnight / weekend / USDA-report gaps). The multi-day book is only researched if the intraday
  book passes (step 4 below).

- Live Kalman gap is the outright base; one trigger per move at \|z\| >= 1.645; z and gap size are features.
- Every trigger is labelled **both** as outright beans and as the crush spread; the meta-model picks the
  instrument or skips. "Which side moved into the divergence" is the main input for that choice.
- **Labels net of costs**, so a predicted win probability above 0.5 means positive expected value. Outright: 1 tick
  per round trip. Spread: three cost cases - (a) all legs crossed (ZS $12.50, ZM $10, ZL $6 per tick),
  (b) autospreader with one passive leg, (c) the exchange-listed crush (fixed ratios; measure its tracking).
- Spread target: 4-5 ticks per trade after costs, from the meta-model's selection.
- **The meta-model trains on all sessions (2026-11-02).** A model trained on Europe triggers only ranked the same
  Europe trades slightly worse (5-min spread, top half +0.29 vs +0.33 ticks; AUC 0.54 vs 0.55): a quarter of the
  examples, and the shared effects (volume, oil-led moves, session crossings) are better estimated on all sessions.
  If trading is restricted to a session, score with the all-session model and filter afterwards. Session results
  (scratch, 2022 - Nov 2024 out of sample): US loses in every variant; 5-min Europe spread, all-session model's top
  two thirds = +0.34 ticks at ~7.7 a week (t 1.9); 1H Europe spread +4.7 per trade but ~0.3 a week.

## Volume bars (2026-11-03, `volume_bars.py`) - the current best

A new bar every 458 ZS contracts (the mean 5-min bar volume in Jul - Dec 2020, set before development), restarting
at each session open. Same Kalman drift per bar, 48-bar z, trigger rule, clock-time labels (60 min, target 1
sigma_1h), 17 features and walk-forward logistic. Quiet hours get fewer, longer bars, so Europe fires ~5.4 triggers
a week instead of ~12.7 - and they are better ones.

With the z half-life fixed at 192 bars (see "z lookback" below; the first version used 48):

| Europe spread (beans passive), stop 1.25x | net / trade | t | per week | 2022 / 2023-24 |
|---|---|---|---|---|
| every trigger, 2021-24 | +0.36 | | ~5.1 | |
| model's top two thirds (out of sample 2022 on) | +0.94 | 3.1 | ~3.1 | +0.84 / +1.00 |
| **model's top half** | **+1.44** | **4.0** | **~2.4** | **+1.37 / +1.50** |
| model's top half, symmetric stop | +1.19 | 3.5 | ~2.4 | +1.57 / +0.94 |

Asia spread: top half +0.26 (2022 +2.15, 2023-24 -0.51) - not robust. US and beans outright: negative.
(At z = 48 the top half was +0.99, t 2.8, 2022 +1.41 / 2023-24 +0.73.)

- **Leakage:** the whole pipeline rebuilt from data cut at 1 Mar 2023 gives identical bars, hedge, z, triggers and
  features before the cut (part of the run).
- **Bar size:** the first scratch run used 379 (the all-development mean - a look-ahead) and showed +1.25; 458 is
  the honest number. 300 gives +0.83 - robust to the size, AUC 0.57 - 0.63.
- **Barriers (scratch grid, size 379):** a plateau, not a peak - any target >= 1 sigma with stop >= 1x gives
  +0.8 to +1.25 for the top half; targets of 0.5 sigma lose (cost is a third of the win). A 2-hour hold is about the
  same but steadier in 2023-24.
- **Trees vs logistic (scratch):** gradient-boosted trees stop after 7 - 20 rounds and do not beat the logistic
  (top half +1.01 vs +1.25 at size 379).
- Beans outright, Asia and US: still negative. ~2.5 ticks a week per spread at a 1.1-tick cost; limit orders on
  the meal / oil legs are the next lever. Europe, the stop, the bar type and the size were all chosen on
  development data - the holdout (with size 458 fixed) decides.

Also tried 2026-11-03 (scratch, recorded here): Delphic Alpha signals (core_quant/feature_registry.py) - nothing
across sessions; in Europe the mean-reversion family is consistent (IC 0.06 - 0.10) but adds ~0.01 to the model;
quick take-profits at +2 to +8 ticks make every version worse (the edge builds over 30 - 60 min); the trigger's order
in its session (3rd+ better in Europe) adds +0.02 to +0.07 as a Europe-only feature.

**z lookback on volume bars (2026-11-03, scratch).** Only the trigger z's half-life changed; full refit each time.
Europe spread, model's top half, stop 1.25x (symmetric in brackets), out of sample 2022 - Nov 2024, ~2.3 - 2.6 a week:

| half-life (bars) | 24 | 48 (current) | 96 | 192 | 384 | 768 |
|---|---|---|---|---|---|---|
| top half | +0.87 (+0.69) | +0.99 (+0.82) | +1.23 (+0.94) | +1.44 (+1.19) | +1.60 (+1.26) | +1.37 (+1.09) |
| 2022 / 2023-24 | +1.36 / +0.64 | +1.41 / +0.73 | +1.24 / +1.22 | +1.37 / +1.50 | +2.38 / +1.09 | +1.46 / +1.30 |
| every Europe trigger | +0.24 | +0.17 | +0.33 | +0.36 | +0.42 | +0.45 |

A plateau from 192 to 768 bars (+1.37 to +1.60), with 2023-24 now as strong as 2022 - the short lookback was part of
the weak recent half (a Europe volume bar is ~13 min, a US one ~2 min, so 48 bars is little clock time when busy).
Beans: negative at every lookback. Not yet fixed in config - six values were tried on development data.

**Order flow inside the volume bar (2026-11-03) - lead found, then FAILED its pre-registered test.** Screen (scratch,
beans ticks Jan 2021 - Jul 2022 and Nov 2022 - Nov 2023): aggressive buy - sell volume inside the trigger's volume
bar, flipped by side, had spread IC 0.109 in Europe (0.110 / 0.125 by period) and 0.054 over all sessions.
Pre-registered test on fresh ticks 27 Nov 2023 - 29 Nov 2024 (1,795 triggers; pass = IC >= +0.05 all sessions and
> 0 in Europe): all sessions -0.008, Europe +0.014 -> FAIL. Second tick lead in a row that did not survive fresh
data; beans order flow from trades alone is treated as exhausted. Bar speed and VWAP position within the bar (minute
data) looked useful in Europe but faded in 2023-24 and lowered the model's top half (+0.91 -> +0.84).

## Holdout RESULT (run 2026-11-03): FAILED

Europe spread, the frozen model's top half (p >= 0.556): **-0.97 ticks a trade, t -2.06, 151 trades (2.8 a week),
win 54%, mean win +3.7 / mean loss -6.5** - against +1.44 (t 4.0) in development. Every Europe spread trigger: -0.87
(268 trades). Before costs the taken trades made only ~+0.1 tick: the signal had no edge in Dec 2024 - Dec 2025,
and smaller barriers in a quieter year made the ~1.1-tick cost a bigger share. US: -0.64 (every trigger), beans
outright negative everywhere; Asia's small positive picks (25 - 32 trades) are noise. Full table:
`output/holdout_result.csv`; `output/holdout_LOCK` stops a re-run.

Reading: the development result came from many choices made on the same data (Europe, volume bars, bar size,
z lookback, stop, model) - the holdout says that combination did not generalise to 2025. The holdout is now spent;
re-tuning on it would not be a test. This volume-bar setup is not a tradeable strategy as it stands.

## Holdout - pre-registered 2026-11-03 (`holdout.py`)

Fixed before any holdout result was seen: volume bars of 458, **z half-life 192** (`config.yaml`), trigger 1.645 /
0.6745, clock-time labels (60 min, target 1 sigma_1h, stop 1.25x), spread with beans passive. The 17-feature
logistic is fitted once on all development triggers and frozen; trades are taken in Europe when p >= the median p
of the development Europe spread triggers under that model. Holdout = 1 Dec 2024 to the end of the minute file
(16 Dec 2025). **Judged:** net ticks per trade on those trades - PASS if > 0, strong if t >= 1.65; everything else is
reported only. One run (a lock file stops a second). After it, 2026 data becomes the next untouched test.

## The crush strategy compared (2026-11-03, scratch)

`strategies/crush_strategy` trades the SLOW overnight gap (hedge frozen at the previous close, 20-day same-session z,
held to the session close, median ~5 h): +2.9 ZS ticks a trade all legs crossed (+3.9 beans passive), ~1.5 a week,
t 5.8, 2021-22 +4.4 / 2023-24 +1.1, Asia +3.6 / Europe +1.4. Swapping in this study's pieces one at a time made it
worse every time: live hedge -0.7, adaptive z +1.1 (2023-24 negative), 60-min exits -1.0. On volume bars (150 /
300 / 458) 2021-22 improves but 2023-24 falls to ~0 or below. So the two are different signals - a slow
dislocation held for hours vs a fast one held for an hour - and each keeps its own modelling. Its Dec 2024 - Dec
2025 holdout is already contaminated; its clean test is 2026 data.

## Where we stopped (2026-11-02, paused)

The signal has edge; the open problem is filtering the right trades. Out of sample 2022 - Nov 2024, net of costs:
- **5-min:** only the Europe spread is positive (every trigger +0.21; all-session model's top two thirds +0.34 at
  ~7.7 a week, t 1.9). The 5-min Asia spread is negative (-0.24), the US loses everywhere.
- **1H** (Kalman per bar, spread held 24 h): Europe +4.7 and Asia +1.9 per trade, ~1.6 a week together (t 1.75);
  US -1.3. The 1H meta-model adds nothing (AUC 0.51 - 0.53).
- **Beans outright:** the moves are large (10+ ticks often) but after a trigger they go both ways about equally
  (median best ~ median worst). Nothing tested so far predicts the direction - not the bar features, a tree model,
  confirmation entry, tilted or longer barriers, or beans trade flow. Capturing it needs direction information,
  not a better exit.

Tested and rejected today (details above / in RESEARCH_LOG): confirmation entry, stops tighter than the target,
Europe-only training, gradient-boosted trees, beans tick features (pre-registered test failed). The 1H, session,
time-limit, tree and tick scripts were scratch (not kept); their results are recorded here.

Open directions:
1. Europe spread execution: the gross edge is ~1.2 ticks, so costs decide it (passive legs, the listed board crush).
2. Europe spread with a 2-hour hold (scratch: +0.13 vs -0.04 at 60 min, both halves >= 0).
3. 1H non-US spread as a rule (overnight / weekend gap risk; Asia is in it).
4. USDA report-window feature - the calendar is in `data/raw/usda_release_times_utc.csv`.
5. Beans outright: what tells the direction of the 10-tick move (new information, not new transforms).

## Next: layer 2

1. ~~**Triple-barrier labels**~~ (`labels.py`, done - see Labels above). Time limit from the IC decay: 30-60 min (the live gap is gone by 1 h).
   Barriers in volatility over that horizon, not the 14-hour ATR. The stop no tighter than the target - triggers
   often move against you first. Rough maths for symmetric barriers of R ticks and a 1-tick cost: net per trade =
   (2p - 1) x R - 1, so each +1 point of win rate adds 2R/100 ticks; 5 ticks net needs R of 15-20 ticks and a win
   rate of 65-70%.
2. **Meta-label model** (`meta_model.py`; features done - see Features above): labels only where the primary fired (side +1 / -1); baseline
   L2-regularised logistic regression (ridge's classifier twin, gives probabilities). Walk-forward with purge and
   embargo, since labels overlap in time. Take trades above the threshold chosen on development data.
3. Assess the model and the strategy as a whole.
4. If it passes, before the holdout: capacity. Can it scale beyond a sub-book? Revisit the 1H signal (z lookback and
   filter speed per bar; a 2026-11-02 scratch test found 1H with a 2-day z marginal: net +0.2 to +1.6 ticks at 2-4 h,
   t <= 2, both halves; 4H nothing) - mainly as a SPREAD trade, which held up better on 1H (gross +0.6 to +2.0
   ticks, both halves) - and size the 5-min version with `supportable_lots` from the crush strategy.
5. Only then the holdout.

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
- Scheduled report window (USDA releases at 11:00 CT carry the largest real moves in the data).
- Session, time of day, the reopen bar after the 13:20-19:00 break, speed of the move into the trigger, ATR from
  true high / low. Later, with tick data: trade imbalance, quote depletion.
