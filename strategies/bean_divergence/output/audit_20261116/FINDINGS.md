# Bean divergence audit, 16 Nov 2026

Development data only (Jan 2021 to Nov 2024, 5-minute bars). The Dec 2024 to Dec 2025 holdout is spent and was
not touched. Everything here is a lead from one sample after many variants, not a tested result.
Scripts are in `scripts/` (copies of the session scratchpad; they chain through `runpy`, so keep them in one folder
and run from the repo root). Charts are in this folder.

## Purpose
A semi-systematic alert for discretionary intraday trading of soybeans. The fair value is accurate; the nuance is
that it can come back to price without price moving. Knowing how it behaves gives the positioning edge; location
and exits are the trader's.

## Settings agreed
- **Live fair value:** Kalman crush hedge, 10x slower than before (`DRIFT_TO_NOISE_PER_15_MINUTES = 0.01`).
- **Session fair value:** hedge frozen at the previous session close (`crush_hedge(bars, live=False)`), the orange line.
- **Yardstick:** expected 1-hour beans move = `adaptive_scale(ZS_ret in ticks, 48) * sqrt(12)`. Gaps are divided by it,
  so a level means the same thing in every year and session.
- **Live stretch:** |live gap| / yardstick >= 0.75 (about 10 a week) or 0.95 (about 5 a week); re-arm inside half the level.
- **Turn:** first close within 30 minutes where the live gap is no wider than the bar before AND the fair value's own
  15-minute move (meal and oil at the hedge ratios) is flat or in the trade's favour.
- **Session-gap state at the turn:** against / same side small (< 0.75) / same side large (0.75 to 1.5) / same side extreme (>= 1.5).
- **Spread rule:** session-anchored gap, z = gap / its 20-day size, enter at |z| >= 1.5, exit at fair value or the session close.

## What the fair values are
- The live Kalman is effectively fixed hedge ratios (meal ~0.56, oil ~0.36) plus a moving level. At the original
  speed it refits away 18% of each error per bar.
- The three legs are NOT cointegrated with fixed ratios (Engle-Granger p 0.70; 1 of 16 sixty-day windows). The spread's
  level is a random walk plus a short-lived reverting piece. Fair value must be re-anchored; multi-day means fail.
- z calibration is fine at 1.645 (90% of bars inside, every year); tails are fatter than normal.

## Gap sizes (ticks)
- Live gap, slower line: median 2.5, 90% 8.4, 95% 11.5. US is twice Asia and Europe. 2022 and 2023 are much larger than 2021 and 2024.
- Session gap: median 7, 20+ ticks on 12% of bars, about 5 episodes a week; the day session reaches 20 on 52% of days.

## Outright beans
- Direction after any signal is about 50%. Any gain comes from winners being slightly larger than losers.
- Stretch + turn, by session-gap state (looser level, beans at 1 hour, before costs): against -3.3 (t -2.5),
  small +0.1, large +0.6, extreme +1.6 (t 1.6; +5.4 by the session close, t 2.8; about 1.4 a week). Same order at the stricter level.
- Do not fade the live gap when beans are on the other side of the session line.
- **Exit at the live fair value loses** (-1.5 net, t -3.2, no time stop): the line comes to the price. Trades that reach
  it inside 30 minutes make about +3; after that it gets steadily worse. Use it as a quick target only.
- The session fair value is NOT a constant directional bias. Counting every bar equally, beans show no lean at
  20 to 30 ticks; overnight they lean back past 30 ticks (small samples); the US shows nothing. An earlier
  day-weighted table overstated this and was corrected.
- Fixed-distance resting limit entries did not reproduce a better entry (the missed fills were the winners).
- Bar size (5 vs 15 vs 30 minutes) and the choice of z do not change the picture.

## Spread (beans hedged with meal and oil)
- Fixed holds: the move back is under a tick within hours; costs are 1.1 (beans passive) to 2.1 ticks (all crossed).
- Traded back to a moving average (2 or 8 hours): positive before costs (t 3 to 7), marginal after, large left tail.
- **Session-anchored, z 1.5:** 4.7 trades a week, +1.43 net of 1.1, Sharpe 1.49, worst drawdown -328 ticks. Two thirds
  of the profit is 2022, three quarters is Asia, overnight flat since mid-2023, US nothing. Only 15% reach fair value.
  The move peaks about 4 hours after the signal; the session-close exit is not too early. Robust to z from 1.25 to 2.0.
- Spread and outright signals overlap in time about half the time but their daily profits are uncorrelated (-0.01).

## Open
- Live dashboard for the signals (next session).
- Realistic overnight and autospreader costs; Europe spread trades carried into the US session.
- A one-time test on 2026 data (needs a data pull).
- Nothing from this audit is in the README or committed.
