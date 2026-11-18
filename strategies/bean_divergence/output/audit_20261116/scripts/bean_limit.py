"""Two questions on the 'turn' signals (development data only, outright beans, ZS ticks):
A. Can the LIVE fair value be slowed down, and does a slower one give a better signal / a reachable exit?
   Same Kalman, slower drift. Each speed's stretch level is set so that 4% of bars are stretched (as now), so
   the speeds are compared at the same signal frequency, not at the same tick size.
B. Entering and exiting with LIMIT orders, the way the user trades.
Fixed before the run:
  signal   live turn / both turn exactly as drawn (bean_signal_charts2.py); the session fair value is unchanged.
  entry    market: the next bar's open. limit: resting at the signal bar's close plus d x the expected 1-hour move
           in our favour (d = 0, 0.15, 0.25), good for 30 minutes, then cancelled. A limit counts as filled only
           when a later 5-minute bar trades THROUGH it by a tick (or opens beyond it: filled at that open).
  exit     at each close a limit is placed at that bar's fair value (live or session) for the next bar; filled
           when the next bar trades through it by a tick, or at its open if it opens beyond. Otherwise the
           session's last close.
  cost     1 tick round trip with a market entry, 0.5 tick with a limit entry (fees plus a little slippage).
  pass     net > 0 with t >= 2 and positive in both halves."""
import numpy as np
import pandas as pd

import strategies.crush_strategy.kalman_filter as kalman_filter
from core_quant.config import ROOT
from strategies.bean_divergence.fair_value import (TICK_CENTS, adaptive_scale, crush_gap, crush_hedge, half_label, load,
                                                   load_bars)
from strategies.bean_divergence.triggers import find_triggers

MIN_GAP_TICKS, SESSION_AGREE, REARM_SHARE = 5, 1.0, 0.4
TURN_WAIT_BARS, FAIR_LOOKBACK_BARS, LIMIT_LIFE_BARS = 6, 3, 6
STRETCHED_SHARE = 0.04

pd.set_option("display.width", 250)
cfg = load()
bars = load_bars(cfg)
in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
session_id = bars["session_id"].values
half = half_label(bars["trading_date"])
dates = bars["trading_date"].values
open_s, close_s = bars["ZS_adj_open"].values, bars["ZS_adj_close"].values
to_ticks = bars["ZS_adj_close"] / TICK_CENTS
yardstick = adaptive_scale(bars["ZS_ret"] * to_ticks, 48) * np.sqrt(12)           # expected 1-hour beans move, ticks
weeks = bars.loc[in_dev, "trading_date"].nunique() / 5

# 5-minute highs and lows from the minute file, on the roll-adjusted scale of the bars
minutes = pd.read_parquet(ROOT / cfg.data_path, columns=["ZS_high", "ZS_low", "ZS_close"])
minutes = minutes[minutes.index <= bars.index[-1]]                                 # nothing from the holdout
bar_of_minute = np.searchsorted(bars.index, minutes.index, side="left")            # a bar is stamped with its last minute
factor = (bars["ZS_adj_close"] / minutes["ZS_close"].reindex(bars.index)).values
high_s = minutes["ZS_high"].groupby(bar_of_minute).max().reindex(range(len(bars))).values * factor
low_s = minutes["ZS_low"].groupby(bar_of_minute).min().reindex(range(len(bars))).values * factor
assert (high_s >= close_s - 1e-6).all() and (low_s <= close_s + 1e-6).all(), "highs / lows do not bracket the closes"

hedge_now = crush_hedge(bars, True, 5)
session_gap = crush_gap(bars, crush_hedge(bars, False, 5))
session_ratio = (session_gap / yardstick).values
fair_move = ((hedge_now["beta_meal"] * bars["ZM_ret"].rolling(FAIR_LOOKBACK_BARS).sum()
              + hedge_now["beta_oil"] * bars["ZL_ret"].rolling(FAIR_LOOKBACK_BARS).sum()) * to_ticks).values


def turn_signals(live_gap: pd.Series, level: float) -> dict[str, list[tuple[int, int]]]:
    """(bar, side) of the live turn and both turn signals for one live fair value. side +1 = beans rich -> sell."""
    ratio = live_gap / yardstick
    held = np.sign(ratio) * np.minimum(ratio.abs(), level * 0.99)                  # under the tick floor: cannot fire
    stretches = find_triggers(ratio.where(live_gap.abs() >= MIN_GAP_TICKS, held).values, session_id, level, REARM_SHARE * level)
    gap = live_gap.values
    result = {"live turn": [], "both turn": []}
    for t, side in stretches:
        for bar in range(t, min(t + TURN_WAIT_BARS + 1, len(gap))):
            if session_id[bar] != session_id[t] or side * gap[bar] < MIN_GAP_TICKS:
                break
            if abs(gap[bar]) <= abs(gap[bar - 1]) and side * fair_move[bar] <= 0:
                result["live turn"].append((bar, side))
                if side * session_ratio[t] >= SESSION_AGREE:
                    result["both turn"].append((bar, side))
                break
    return result


def simulate(events, exit_gap: np.ndarray, distance: float | None) -> pd.DataFrame:
    """One row per signal with filled True / False. distance None = market at the next open."""
    cost = 1.0 if distance is None else 0.5
    yard = yardstick.values
    rows = []
    for t, rich in events:
        if not in_dev[t] or t + 1 >= len(bars) or session_id[t + 1] != session_id[t]:
            continue
        position = -rich                                                           # rich -> short
        entry_bar, entry_price = None, np.nan
        if distance is None:
            entry_bar, entry_price = t + 1, open_s[t + 1]
        else:
            limit = round((close_s[t] + rich * distance * yard[t] * TICK_CENTS) / TICK_CENTS) * TICK_CENTS
            for bar in range(t + 1, min(t + 1 + LIMIT_LIFE_BARS, len(bars))):
                if session_id[bar] != session_id[t]:
                    break
                if position * (limit - open_s[bar]) >= 0:                          # opened beyond the limit: better fill
                    entry_bar, entry_price = bar, open_s[bar]
                    break
                traded_through = low_s[bar] <= limit - TICK_CENTS if position > 0 else high_s[bar] >= limit + TICK_CENTS
                if traded_through:
                    entry_bar, entry_price = bar, limit
                    break
        row = {"date": dates[t], "half": half[t], "filled": entry_bar is not None}
        if entry_bar is not None:
            bar, reason = entry_bar, "session end"
            while bar + 1 < len(bars) and session_id[bar + 1] == session_id[entry_bar]:
                target = round((close_s[bar] - exit_gap[bar] * TICK_CENTS) / TICK_CENTS) * TICK_CENTS   # fair value now
                if position * (open_s[bar + 1] - target) >= 0:                     # already at / beyond fair value
                    exit_price, reason = open_s[bar + 1], "fair value"
                    break
                through = high_s[bar + 1] >= target + TICK_CENTS if position > 0 else low_s[bar + 1] <= target - TICK_CENTS
                if through:
                    exit_price, reason = target, "fair value"
                    break
                bar += 1
            if reason == "session end":
                exit_price = close_s[bar]
            row.update({"net": position * (exit_price - entry_price) / TICK_CENTS - cost, "reached": reason == "fair value",
                        "minutes": 5 * (bar - entry_bar + 1),
                        "better_entry": position * (open_s[t + 1] - entry_price) / TICK_CENTS})   # vs the market entry
        rows.append(row)
    return pd.DataFrame(rows)


def describe(frame: pd.DataFrame) -> dict:
    filled = frame[frame["filled"]]
    if len(filled) < 30:
        return {"fills a week": len(filled) / weeks}
    daily = filled.groupby("date")["net"].sum()
    net = filled["net"]
    by_half = filled.groupby("half")["net"].mean()
    return {"fill rate": len(filled) / len(frame), "fills a week": len(filled) / weeks,
            "net (t)": f"{net.mean():+.2f} ({daily.mean() / (daily.std() / np.sqrt(len(daily))):+.1f})",
            "win rate": (net > 0).mean(), "winner / loser": f"{net[net > 0].mean():+.1f} / {net[net <= 0].mean():+.1f}",
            "at fair value": filled["reached"].mean(), "minutes": filled["minutes"].mean(),
            "ticks better than market entry": filled["better_entry"].mean(),
            "2021-22 / 2023-24": " / ".join(f"{by_half.get(key, np.nan):+.2f}" for key in ["2021-22", "2023-Nov24"])}


# ---------------------------------------------------------------- A: the live fair value at slower speeds
print("== A. The live Kalman at slower speeds; market entry at the next open, 1 tick cost")
speed_rows, live_gaps, levels = [], {}, {}
for drift in [0.1, 0.03, 0.01, 0.003]:
    kalman_filter.DRIFT_TO_NOISE_PER_15_MINUTES = drift
    gap = crush_gap(bars, crush_hedge(bars, True, 5))
    level = float((gap.abs() / yardstick)[in_dev].quantile(1 - STRETCHED_SHARE))
    live_gaps[drift], levels[drift] = gap, level
    signals = turn_signals(gap, level)
    row = {"drift": f"{drift:g}" + (" (now)" if drift == 0.1 else ""), "typical gap (ticks)": np.sqrt((gap[in_dev] ** 2).mean()),
           "gap half-life (min)": 5 * np.log(0.5) / np.log(gap[in_dev].autocorr(1)), "stretch level (x 1-hour move)": level}
    for name, events in signals.items():
        for exit_name, exit_gap in [("live", gap.values), ("session", session_gap.values)]:
            d = describe(simulate(events, exit_gap, None))
            row[f"{name}, exit {exit_name}"] = f"{d['net (t)']}, {d['fills a week']:.1f}/wk, at fv {d['at fair value']:.0%}, {d['minutes']:.0f} min"
    speed_rows.append(row)
kalman_filter.DRIFT_TO_NOISE_PER_15_MINUTES = 0.1
print(pd.DataFrame(speed_rows).set_index("drift").round(2).T.to_string())

# ---------------------------------------------------------------- B: limit entries and exits
for drift in [0.1, 0.01]:
    gap = live_gaps[drift]
    signals = turn_signals(gap, levels[drift])
    for name, events in signals.items():
        for exit_name, exit_gap in [("live", gap.values), ("session", session_gap.values)]:
            table = {}
            for label, distance in [("market, next open", None), ("limit at the signal close", 0.0),
                                    ("limit 0.15 x 1-hour move better", 0.15), ("limit 0.25 x better", 0.25)]:
                table[label] = describe(simulate(events, exit_gap, distance))
            print(f"\n== B. {name}, live drift {drift:g}, exit by limit at the {exit_name.upper()} fair value")
            print(pd.DataFrame(table).round(2).to_string())
