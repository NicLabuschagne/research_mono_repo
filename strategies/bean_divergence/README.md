# Bean divergence: relative strength vs products leads outright beans

## 0. Hypothesis (written 2026-10-29, BEFORE looking at results)

- **Origin:** a discretionary trade (London open to NY close, flat before session breaks), read off two
  TradingView lines: *crush* = 10 ZS - 11 ZM - 9 ZL and *skew* = 6 ZS - 11 ZM - 9 ZL (raw price units).
- **What the lines measure:** crush - skew = 4 ZS, so the pair carries only beans and products. In raw units
  6 ZS is close to meal-neutral for the Kalman hedge (beans move ~0.57 x meal + 0.36 x oil), so *skew* is
  roughly the hedged residual (beans vs products) and *crush* is roughly that residual plus beans' direction.
- **Thesis:** when beans strengthen against products (residual rising) while the outright bean price falls
  (the drop is product-led), beans' outright price catches up with its own relative strength: **go long
  beans**; the mirror (residual falling, beans rising) is a short. Extremes help: beans near the day's low
  while the residual is off its highs.
- **Why it could pay:** outright bean traders react to the headline price move (often driven by meal / oil)
  and miss that beans are bid relative to products; the relative strength is real demand.
- **Contrast:** the crush strategy found overnight dislocations REVERT (mostly through beans). This thesis is
  the opposite bet and is expected, if anywhere, in London / US hours, where the crush research saw no
  reversion and daily dislocations tended to continue.

**Kill criteria** (any one stops it):
1. The divergence move (residual up and beans down, or the mirror) is under ~1 tick net of a 1-tick cost at
   every horizon from 15 min to 4 h.
2. It does not hold in both halves of development (2021-22 and 2023 - Nov 2024).
3. It is no better than a beans-only control (beans fell -> beans bounce), i.e. the residual adds nothing.

## Data and split

5-minute ZS / ZM / ZL bars from the crush pipeline (`strategies/crush_strategy/data_ingest.py`), hedge from
its Kalman filter, frozen at the previous session close. Development Jan 2021 - Nov 2024; **holdout from
Dec 2024 locked** (used for the crush spread, never for this outright hypothesis). Sessions reported apart:
Asia (19:00-02:00 CT), Europe (02:00-07:45 CT), US (08:30-13:20 CT).

## Log
| Date | Run | Change | Result |
|---|---|---|---|
| 2026-10-29 | `runs/20261029-124015_feature_study` | Gate 2: IC, event study, regression (development only) | **Thesis killed** (criteria 1 and 2); see below |

## Result of gate 2 (development only; the holdout was not opened)

- **Direction is the opposite of the thesis.** Overnight (Asia, Europe) a rising residual is followed by beans
  FALLING (IC -0.02 to -0.07, t to -6 in 2021-22; weaker in 2023-24): the relative-value reversion the crush
  strategy trades, not continuation. US session: nothing either way (|t| < 2.6, signs flip between halves).
- **The divergence pattern** (residual up and beans down -> long, and the mirror) is rare (30-280 events per
  session and half), noisy (standard errors 1-4 ticks) and changes sign between 2021-22 and 2023-24. It never
  clears ~2 ticks gross (1 tick net) in both halves. Kill criteria 1 and 2.
- **Regression** (future bean move ~ residual move + bean move): with the bean move controlled, the residual
  still predicts reversion in Europe 2021-22 (-0.3 to -1.3 ticks per 1 z, t -2.3 to -4.0), weaker in 2023-24.
  The relative information is real but small, and points the RV way. R-squared under 0.5% (1.2% at best).
- **Trading only the bean leg of the crush spread's own signals:** +2.2 / +0.8 ticks net per trade, but
  per-trade spread 19-29 ticks (t ~ 1). The same trades hedged: +5.5 / +1.2 ticks at t ~ 7 / 2. The hedge is
  what makes the edge usable: outright beans add bean-price noise far larger than the relative edge.
- **The TradingView skew (6 ZS - 11 ZM - 9 ZL)** tells the same story as the Kalman residual, but it carries more
  of beans' own move (correlation with the bean move 0.73 vs 0.37), so it mixes the two effects.

- **Level vs change** (sign so + = beans expected up): the richness LEVEL (-z, the crush signal) is strongest in
  Europe 2021-22 (IC +0.05 to +0.08, t 3.5-4.9) but zero in 2023-24; the recent residual CHANGE (-zres, 1 h) is
  weaker but holds in Asia in both halves (t 2.9-3.9, then 2.2 / 1.7). Both are small for outright beans.
- **Does fair value lead beans? No, not at 5 minutes.** Beans and fair value move together within the bar
  (correlation 0.87) and not after it (|correlation| < 0.01 at 1-6 bars, either direction). A fair-value rally
  over the last 30-60 min is followed by slightly WEAKER beans overnight (IC -0.02 to -0.05, 2021-22): the
  whole complex mean-reverts short term; beans do not catch up. Any lead would have to be under 5 minutes.

- **Why outright fails** (`failure_study.py`, `runs/20261029-125909_failure_study`). Outright fade-richness
  trades (|z| >= 1.5, 1-2 h, all sessions) split exactly into residual part + market part. The RESIDUAL part is
  positive in almost every session and half (overnight 2021-22 +1.7 to +3.2 ticks): the thesis works. The
  MARKET part (meal / oil through the hedge) is 87% of the variance and, overnight in 2021-22, systematically
  against the trade (-1.8 to -2.8 ticks), cancelling it. Which leg created the dislocation does not explain it.
- **Entry-time conditions:** only the complex's prior 2 h trend lines up in both halves and both session groups:
  trading WITH the complex trend (e.g. short rich beans while meal / oil are falling, which is the discretionary
  setup with the direction flipped) averages +1 to +2 ticks net; AGAINST it loses. But standard errors are 2-3
  ticks on ~40-60 trades per bucket (t < 1): a hint, not a result. Volatility and sigma buckets are inconsistent.

- **Europe-only trading** (`europe_session_study.py`, `runs/20261029-132318_europe_session_study`): entries
  02:00 CT onward, every trade out by the 07:45 break. With the session hedge it makes +2.5 to +6.1 ticks per
  spread trade in 2021-22 but LOSES (-0.8 to -1.5) in 2023-24 at every cutoff and threshold; re-anchoring the
  hedge at the Europe open is weaker still (-0.3 to -1.0 in 2023-24). The window is too short for a reversion
  that takes hours, and the 2023-24 Europe reversion (~1 tick per 2 h) is below the 2.3-tick spread cost.
  Asia-only entries (19:00-02:00) did best in 2023-24 (+2.3 ticks, Sharpe 2.0), which is the longer end of the
  crush strategy's 3-8 h entry-cutoff plateau, not a new rule.

- **Blog method ("Spread Alpha Atlas")** (`blog_spread_test.py`, `runs/20261029-134710_blog_spread_test`): 30-min
  bars, hedge frozen daily (and per session for comparison), position = sign of a feature every bar, flat across
  breaks, our real cost (~4.8 bps round trip on the bean notional for three legs). GROSS, mean-reversion features
  work in Asia AND Europe in both halves (Sharpe +4 to +7), momentum loses (-2 to -7), US is weak (~+1): the blog's
  pattern reproduces. NET, nearly everything is negative: 3-20 position changes a day at 1-3 bps gross per change
  (US 0.4-0.9) against ~2.4 bps cost per change. Daily vs session hedge makes no difference.

- **Blog method traded discretely** (`blog_threshold_test.py`, `runs/20261029-135125_blog_threshold_test`): same
  30-min daily-hedged spread, rolling z over 10/20/50 bars, entry at |z| >= 2 / 2.5 in Asia or Europe only (no US),
  exit at |z| <= 0.5 or 07:45, 2.3-tick cost. 2021-22: positive for 20-50 bar lookbacks (+0.9 to +3.7 ticks per
  trade). 2023-24: NEGATIVE in every one of 18 variants, Asia and Europe alike (-0.1 to -2.1). Weaker than the
  crush strategy's own construction (20-day z, profit target, cutoff: +1.2 ticks all evening, +2.3 Asia-only in
  2023-24). A short rolling mean re-anchors within hours, so it fires on smaller dislocations.

## Layer 1: pricing (2026-10-30 onward)

Pricing and execution are now separate. Layer 1 judges the fair value only on statistics (no thresholds, no trades).

- **Step 1, look** (`layer1_event_view.py`): around baseline triggers beans do the moving beforehand (rich
  triggers mostly at the day's high); afterwards the gap closes slowly, partly via products catching up.
- **Step 2, crush scorecard** (`layer1_scorecard.py`, `runs/20261030-161614_layer1_scorecard`):
  - The crush fair value explains 64-89% of soybean moves over 30 min - 4 h (69-81% at 5 min).
  - The residual itself reverts strongly in Asia and Europe in both halves (IC vs next gap move -0.05 to -0.20,
    t to -19); weaker in the US.
  - Main test 1, residual vs next SOYBEAN move: with the hedge fixed at the prior close (identical to the session
    hedge in Asia / Europe, which start the trading day) it is significant only in Europe 2021-22 (-0.05 to -0.09).
    With the LIVE hedge (the filter's prediction error) it is negative in Asia and Europe in both halves, measured
    from the next bar's open to remove bid-ask bounce: -0.02 to -0.05 at 30 min - 1 h (t -1.8 to -5), fading by
    2 h, smaller in 2023-24. The products also move toward beans in the same window.
  - Normalization does not matter here: the Kalman's own std sqrt(F) is almost constant (14-21 ticks), so the
    Kalman-normalized gap has the same IC as the raw gap; the 20-day z is also about the same.
  - Momentum angle fails: a fair-value move over the last hour is followed by beans going the OTHER way, not
    following (Asia 2021-22 -0.05, t -4.1; about 0 elsewhere).

- **Step 3, day-to-day behaviour** (`layer1_week_view.py`): spread IC re-checked from the next bar's open (no
  bounce): Europe strong in BOTH halves (live residual -0.07 to -0.11, t -4.8 to -11.7; prior-close -0.06 to
  -0.15, t -3.4 to -9.4); US weak (-0.02 to -0.05, t -1.4 to -2.8). Inside a session the prior-close gap ranges
  ~12-17 ticks (median; 21-30 at the 90th pct) in Asia / Europe and ~24-27 (48) in the US, and ends near 0 on the
  median day. Continuous-week charts: the gap moves in plateaus and steps, often jumping at the US open.
- **Step 4, corn and wheat** (`layer1_corn_wheat.py`, `runs/20261030-165205_layer1_corn_wheat`; 15-min bars,
  all five legs as switch-free indices because the grain files are not roll-adjusted). Adding corn + wheat to the
  outright fair value raises the share of the soybean move explained by only 0.5-2 points (e.g. Europe 1 h
  81.6% -> 83.6%; US flat) and leaves the residual's soybean IC unchanged (within about +/-0.005). As
  explanations of the crush residual, grains account for 2-7% of its same-time moves and their last-hour move
  does not predict its next move (|t| < 2.3), except US 2023-24 at 2 h (corn +0.13, t 3.7), which flips sign in
  2021-22. Verdict: the crush alone is the fair value; corn and wheat add nothing worth the extra legs.
- **Step 5, USD/CNH** (`layer1_fx.py`, `runs/20261030-165839_layer1_fx`; 5-min bars, last CLOSED FX bar carried):
  adding it to the outright fair value changes the share of the soybean move explained by at most 0.2 points and
  the residual's soybean IC by at most 0.002. It explains under 1% of the crush residual's same-time moves, and its
  last-hour move does not predict the residual's next move (|t| <= 2.0). No BRL data was available. Fair-value
  inputs are now settled: the crush (meal + oil) alone.
- **Step 6, extremes** (`layer1_extremes.py`, `runs/20261030-170242_layer1_extremes`): extreme = the gap beyond
  its own 90th / 95th / 99th percentile of the previous 20 days of the same session, once per excursion; moves
  measured from the next bar's open, + = back toward fair value, no costs.
  - Standout: Europe, prior-close residual, beyond the 95th pct: about 1 a week; spread +3.5 / +1.8 ticks at 1 h
    (2021-22 / 2023-24, se 0.9 / 0.6), outright beans +5.2 / +2.1 (se 2.1 / 1.0); adverse move before it comes back
    ~2-3 ticks (median, 2 h), favourable ~4-6.
  - Asia (prior close) 90-95th: spread +2.4 to +3.0 in 2021-22, ~+0.3 in 2023-24, ~2-4 a week.
  - US: no reversion. After 95th-pct extremes outright beans keep going the extreme's way (-6.7 / -1.0 ticks at
    2 h, both halves negative): a continuation hint, not a result.
  - Live residual: 10-34 extremes a week, follow-through +0.3 to +1.3 ticks (spread): real but small vs costs.
  - Hit rates are often under 50% with positive means: small losses are common, the gains are larger.
- **Step 7, extremes in today's terms** (`layer1_adaptive_extremes.py`, `runs/20261030-171351_layer1_adaptive_extremes`).
  Today's live-gap size differs from its 20-day average by more than 30% on 35-49% of days, so the scale was made
  adaptive (exponential, bars before t only): live residual / the filter's adaptive forecast std (half-life 1 h,
  4 h, 1 day); prior-close gap / its recent RMS (4 h, 1 day). Extremes at |z| 1.28 / 1.645 / 2.33.
  - Europe, prior close: the 20-day percentiles still give the biggest moves per event at the 95th (spread +3.5 /
    +1.8, outright +5.2 / +2.1, ~1 a week); adaptive 4 h gives twice as many (~2 a week) and the most even spread
    result across halves (+1.6 / +1.5, se 0.5 / 0.4), outright +0.3 / +1.5.
  - US, prior close: adaptive scaling removes the "continuation" seen with 20-day levels (that came from busy days
    flagging ordinary moves as extreme) and turns it mildly positive in both halves (4 h, 95th: spread +1.0 / +0.8,
    outright +0.9 / +1.4, ~4-5 a week; standard errors ~0.5-1.5).
  - Live residual: about the same as with 20-day levels (+0.6 to +1.7 ticks spread).
  - Units checked: values are ZS ticks (1 tick = 0.25 c/bu), asserted in the script.
- **Size of the outright moves after signals A / B / C** (all 1,483 events): the edge is a small tilt in a wide
  distribution. Average +1.4 ticks at 1 h, +2.1 to the session close (0.35-0.5 c), against a typical move of
  10-14 ticks either way (2.6-3.5 c; 10th / 90th pct about -21 / +24 at 1 h, -31 / +38 to the close). Moves over
  10 c happen on 6-16% of events within the session and on 41% only if held into the next session. The large moves
  in the signal-day charts were mostly next-session moves on volatile days, which the measurements (and a
  session-close exit) do not include.
- **Barrier preview vs random entries** (`layer2_barrier_preview.py`; sessions ignored, holds up to 12 h; 10 random
  entries per signal from the same session and half, random direction). Moves after signals are NOT unusually large:
  typical size 11.3 vs 9.5 ticks at 1 h, 21.6 vs 20.8 at 4 h, equal at 12 h. What the signal adds is a direction
  tilt that grows with time held: +1.0 / +0.8 ticks at 1 h, +3.2 / +1.9 at 4 h, +5.2 / +2.2 at 12 h (2021-22 /
  2023-24; random entries ~0). Fixed-tick triple barriers (targets / stops 10-40 ticks, 4 h / 12 h) do not amplify
  it: best net +0.9 / -0.0 ticks per trade (T40 / S40 / 12 h), signal minus random +0.4 to +1.9, i.e. the tilt
  itself. The barrier shapes the payoff around a small tilt; it does not create edge here.
  ATR-scaled barriers (ATR = 14-hour average of the 1-hour range of 5-min closes; median 13.9 ticks, 10th-90th
  9.7-22.7) give the same picture: the best settings (targets and stops of 2-3 ATR, 4-24 h) net about +0.5 to
  +0.9 ticks per trade in both halves with standard errors of 1-2.3 ticks; tight stops (0.5 ATR) are the worst
  (-1.4 to -1.7), consistent with signals moving against you first. Signal minus random: -0.5 to +3.0.
- Mispricing at the trigger and how much comes back (`layer1_mispricing_capture.py`; ZS ticks, 4 ticks = 1 c).
  The band IS the mispricing: median |gap| at the trigger is A 18.7 / 14.4 ticks (4.7 / 3.6 c; 2021-22 / 2023-24),
  B 14.2 / 10.0, C 16.3 / 13.0; 90th pct 19-30 ticks. Random bars in the same sessions sit at 7-10 ticks, so a
  trigger is roughly 1.5-2x the normal distance from fair value. What comes back on the spread (average,
  inside the session, from the next open): at 1 h A +3.5 / +1.8 (11-17% of the gap), B +1.6 / +1.5 (10-13%),
  C +1.0 / +0.8 (5-6%); random ~0. The "best spread move within 2 h" (median 4-7 ticks) is mostly noise: random
  entries reach 2.4-5.1 ticks with no signal, so the extra over random is only ~1-2 ticks. Bigger mispricings
  give more ticks back (A large: +4.3 at 1 h, +7.7 at 4 h; B large +2.3 / +7.2) but not a bigger SHARE: the market
  returns a roughly constant ~10-15% of the gap within an hour in Europe, ~5% in the US. The rest of the gap
  persists (or is fair value moving), which is why the edge per trade is a few ticks, not the size of the band.
  Large Europe triggers, A + B combined (same dislocation counted once): gap >= 15 ticks fires 1.4 / 0.6 a week
  (2021-22 / 2023-24), >= 20 ticks 0.7 / 0.3, >= 25 ticks 0.3 / 0.1. Spread at 1 h for gap >= 15: +3.1 (t 4.0) /
  +2.4 (t 2.5). The big 4 h numbers (+9 to +11 ticks) are 2021-22 only and come from a different subset: the
  measurement stops at the 07:45 session close, so only triggers at 02:00-03:45 CT have a 4 h value (35 of 139);
  2023-24 at 4 h: -0.3 (n 13). Signal to noise (mean / se) peaks at 30 m - 1 h and falls after, which is what the
  IC showed: the average move keeps growing, the noise around it grows faster.
- Frequent triggers, forward returns from the trigger close, no session cut (`layer1_trigger_harvest.py`):
  adaptive |z| (4 h) in any session; >= 1.96 gives ~5 a week (US 3.1-3.7, Europe ~1, Asia ~0.6), >= 2.33 ~2.5.
  Outright beans in the first 30 min: +0.0 to +0.7 ticks, t < 2.6, hit rate ~49-51% (random 44-48%, lower because
  of zero moves). Higher z is worse early: at >= 2.58, -1 to -3 ticks at 15 min (beans keep going first).
  The SPREAD reverts at once (+0.5 to +1.0 tick at 5 min at every level and half) but beans do not: the gap closes
  through meal / oil moving toward beans, which a beans-only order does not capture.
  Race +3 before -3 ticks within 30 min: 53% vs random 49%; needed 57% at 0.4 tick cost, 67% at 1 tick.
  Asia / Europe 55-60%, US ~50%. Short-horizon outright harvesting is not there on 5-minute closes.
- Seen on the last 4 development days (`layer1_trigger_window.py`): the prior-close gap JUMPS at session starts,
  because the hedge is re-frozen there. Median |gap change| on a session's first bar 10.8 ticks (90th pct 32.8) vs
  1.2 inside a session; 9.7 of it is the hedge update alone. Consequences: 122 of 979 |z| >= 1.96 triggers fall in
  a session's first 15 min (55 on the first bar), and the adaptive scale (RMS of the gap LEVEL) inflates after a
  jump, so the bands widen and nothing fires for a day or two (26-27 Nov 2024: bands at +/-40-50 ticks).
  The jumps also show the filter's fair value moving toward beans during a session (the persistent part of the gap).
- Prior-close vs live gap: the prior-close gap is persistent (std 17.2 ticks; autocorrelation 0.98 at 5 min, 0.84
  at 1 h, 0.47 at 4 h) - a slow mispricing that trends within a session and jumps at re-freezes. The live gap (hedge
  from the bar before = the filter's prediction error) is fast (std 3.9 ticks; 0.75 at 5 min, 0.24 at 30 min,
  0.05 at 1 h). DECISION 2026-10-30: the live Kalman gap is the locked base for outright beans, with a LARGE trigger
  sample (tens a week) for a later meta-model rather than a high threshold.
- Live-gap triggers (`layer1_trigger_harvest.py live`, `layer1_trigger_window.py live`): the live z has fat tails,
  so |z| >= 1.645 fires ~40 a week, 1.96 ~26, 2.33 ~17, 2.58 ~12 (US about half, then Europe, Asia).
  Outright beans from the trigger close: a small, steady +0.3 to +0.6 ticks from 5 min to 1 h in BOTH halves
  (t 2-3 at 1.645 with the large sample), hit rate 49-53% vs random 43-48%. The spread again snaps back
  +0.6 to +1.0 tick at 5 min. Race +3 before -3 in 30 min: 52-54% vs random ~50%; needed 57% at 0.4 tick cost.
  So the live gap carries a real but small tilt on beans per trigger - raw, it does not clear costs; it is the
  candidate base for the meta-model (which triggers to take) and the barrier design.
  Higher levels do not help: |z| >= 3.0 (~8 a week) and 3.5 (~5 a week) are no better than 1.645 and worse in
  2023-24 (3.5: -0.2 to -0.9 ticks in the first 30 min; beans keep going first). The large-sample choice stands.
  Spread tail (|z| >= 1.645, ~40 triggers a week): big spread moves are TILTED, not one-sided. At 30 min,
  >= +10 ticks on 7.8% / 5.8% of triggers (2021-22 / 2023-24) vs <= -10 on 5.5% / 4.8% (random: 4.3 vs 4.4, flat);
  >= +5 on 23.4% / 19.2% vs <= -5 on 14.0% / 13.2%. In 2023-24 that is ~2.2 ten-tick wins a week next to ~1.8
  ten-tick losses: the meta-model's job is to tell those apart before entry.
  Seen on the chart: the first bar after the 13:20-19:00 break can carry a large live gap (25 Nov 19:04,
  -24.5 ticks, z -6.9): the legs reopen differently. Worth checking separately (reopen effects, contract rolls).
- 5-minute bars stay the base: 15 / 30-minute bars leave the prior-close residual unchanged and lose the live
  residual's 2023-24 signal (the quiet-year reversion happens inside a 30-minute bar).

**Conclusion (outright study, before layer 1):** no outright directional signal from relative strength vs products in this data. The discretionary
edge more likely came from context and execution the features do not capture (news, order flow, which level
you faded). The relative-value information is best traded as the hedged spread (`strategies/crush_strategy`).

## Next session: layer 2 on the live gap (planned 2026-10-30)

Base: live-gap triggers, adaptive |z| >= 1.645 (~40 a week), entry at the trigger close. Then triple-barrier labels,
then a meta-model that learns WHICH triggers to take. Holdout stays locked until the whole process is fixed.

Instrument per trigger (decided 2026-10-30): some divergences belong in the outright, some in the spread. Label every
trigger BOTH ways (outright beans; the crush spread with the trigger bar's hedge, all three legs, with three legs'
costs), and let the meta-model choose: take it in beans, in the spread, or skip. The "which side moved" feature is
the main input for that choice.
Spread target (user, 2026-10-30): 4-5 ZS ticks per trade AFTER costs, reliably - the average snap-back is ~1 tick at
5 min, so this has to come from the meta-model's selection, not the raw trigger. Cost scenarios to label with:
(a) all three legs crossed (one tick each: ZS $12.50, ZM $10, ZL $6 per contract); (b) autospreader - work one leg
passively with a limit (the leg expected to move, often meal / oil), hedge the other two on fill; (c) the exchange-
listed crush spread (fixed board ratios, so it tracks the Kalman hedge only approximately - measure that tracking).

Meta-model features to build (all known at the trigger bar, nothing from later in the day):
- **Position in the day's range** (user idea): where beans sit between the day's low and high SO FAR,
  (close - running low) / (running high - running low), 0 = at the low, 1 = at the high; plus the same for the
  current session. Hypothesis: a divergence at the day's high / low behaves differently from one mid-range
  (exhaustion vs continuation). Use the RUNNING high / low up to the trigger - the full day's high / low would leak
  the future. The minute data has true high / low per leg (`build_bars` currently drops them; add them back).
- Paired with a volume / microstructure event: volume of the trigger bar vs its usual level for that time of day
  (bars keep `ZS_volume`), a range break (new high / low of the day on this bar), later tick-data events
  (trade imbalance, quote depletion) once tick data is in.
- **Which side moved into the divergence** (return decomposition): over the last k bars into the trigger
  (k = 1, 3, 6: 5 / 15 / 30 min), with the trigger bar's hedge held fixed so the filter's own update is left out:
      beans move   = (log ZS_t - log ZS_t-k) x ticks
      fair move    = (beta_meal x (log ZM_t - log ZM_t-k) + beta_oil x (log ZL_t - log ZL_t-k)) x ticks
      gap change  ~= beans move - fair move
  Features: both moves signed in the trade's direction, and beans share = |beans move| / (|beans move| + |fair move|)
  (1 = beans pulled away alone, 0 = the products moved alone). Hypothesis: when beans did the moving, beans come
  back (the outright trade works, e.g. 29 Nov 2024 US open); when the products moved, the gap closes through the
  products and beans need not move (the outright trade fails, e.g. the 25 Nov 2024 19:00 reopen: products ~8 c
  higher, beans flat). Fits the layer-1 finding that the spread snaps back mostly through meal / oil.
  The reopen bar after the 13:20-19:00 break is flagged separately.
- Session, time of day, |z| size, speed of the move into the trigger, ATR (from true high / low).
