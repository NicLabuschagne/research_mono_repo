"""STATISTICS for the signals drawn in bean_signal_charts2.py, definitions frozen (development data only).
Entry: the open of the bar after the signal bar. Everything is the beans move in the FADE direction, in ZS ticks.
  ahead N min  : from the entry to the close N minutes later (cut at the session's last close), BEFORE costs
  best / worst : furthest 5-minute close for / against the trade within 60 minutes (median), before costs
  exit live    : out at the next open after the first close at or through the LIVE fair value, else the session's
                 last close; AFTER a 1-tick cost
  exit session : the same against the SESSION fair value; after a 1-tick cost
t-statistics come from daily sums (signals on one day are not independent).
Pass mark, fixed before the run: a 'turn' set passes if exit-session net > 0 with t >= 2 and positive in both halves."""
import numpy as np
import pandas as pd

from strategies.bean_divergence.fair_value import (TICK_CENTS, adaptive_scale, crush_gap, crush_hedge, half_label, load,
                                                   load_bars, session_label)
from strategies.bean_divergence.triggers import find_triggers

MIN_GAP_TICKS = 5
LIVE_LEVEL, SESSION_LEVEL, AGREE_LEVEL, REARM_SHARE = 0.5, 1.5, 1.0, 0.4
TURN_WAIT_BARS, FAIR_LOOKBACK_BARS = 6, 3
COST_TICKS = 1.0

pd.set_option("display.width", 250)
cfg = load()
bars = load_bars(cfg)
in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
session_id = bars["session_id"].values
session = session_label(bars.index, bars["session_type"])
half = half_label(bars["trading_date"])
dates = bars["trading_date"].values
open_s, close_s = bars["ZS_adj_open"].values, bars["ZS_adj_close"].values
to_ticks = bars["ZS_adj_close"] / TICK_CENTS
yardstick = adaptive_scale(bars["ZS_ret"] * to_ticks, 48) * np.sqrt(12)
hedge = crush_hedge(bars, True, 5)
live_gap = crush_gap(bars, hedge)
session_gap = crush_gap(bars, crush_hedge(bars, False, 5))
live_ratio, session_ratio = live_gap / yardstick, session_gap / yardstick
fair_move = ((hedge["beta_meal"] * bars["ZM_ret"].rolling(FAIR_LOOKBACK_BARS).sum()
              + hedge["beta_oil"] * bars["ZL_ret"].rolling(FAIR_LOOKBACK_BARS).sum()) * to_ticks).values
weeks = bars.loc[in_dev, "trading_date"].nunique() / 5


def stretches(ratio, gap, level):
    held = np.sign(ratio) * np.minimum(ratio.abs(), level * 0.99)
    return find_triggers(ratio.where(gap.abs() >= MIN_GAP_TICKS, held).values, session_id, level, REARM_SHARE * level)


def turns(events, gap):
    found = []
    for t, side in events:
        for bar in range(t, min(t + TURN_WAIT_BARS + 1, len(gap))):
            if session_id[bar] != session_id[t] or side * gap[bar] < MIN_GAP_TICKS:
                break
            if abs(gap[bar]) <= abs(gap[bar - 1]) and side * fair_move[bar] <= 0:
                found.append((bar, side))
                break
    return found


def session_last_bar(entry: int) -> int:
    last = entry
    while last + 1 < len(bars) and session_id[last + 1] == session_id[entry]:
        last += 1
    return last


def outcomes(events: list[tuple[int, int]]) -> pd.DataFrame:
    """One row per signal. side +1 = beans rich -> the fade is a SALE, so a fall in beans is a gain."""
    rows = []
    for t, rich in events:
        entry = t + 1
        if not in_dev[t] or entry >= len(bars) or session_id[entry] != session_id[t]:
            continue
        end = session_last_bar(entry)
        row = {"date": dates[t], "session": session[t], "half": half[t]}
        for minutes in [15, 30, 60, 120]:
            stop = min(entry + minutes // 5 - 1, end)
            row[f"ahead {minutes}m"] = -rich * (close_s[stop] - open_s[entry]) / TICK_CENTS
        path = -rich * (close_s[entry:min(entry + 12, end + 1)] - open_s[entry]) / TICK_CENTS
        row["best 60m"], row["worst 60m"] = max(path.max(), 0), min(path.min(), 0)
        for name, gap in [("live", live_gap.values), ("session", session_gap.values)]:
            bar = entry
            while rich * gap[bar] > 0 and bar < end:
                bar += 1
            reached = rich * gap[bar] <= 0
            fill = open_s[bar + 1] if reached and bar < end else close_s[bar]
            row[f"exit {name}"] = -rich * (fill - open_s[entry]) / TICK_CENTS - COST_TICKS
            row[f"{name} reached"] = reached
            row[f"{name} minutes"] = 5 * (bar - entry + 1)
        rows.append(row)
    return pd.DataFrame(rows)


def mean_t(frame: pd.DataFrame, column: str) -> str:
    if len(frame) < 20:
        return f"n={len(frame)}"
    daily = frame.groupby("date")[column].sum()
    return f"{frame[column].mean():+.2f} ({daily.mean() / (daily.std() / np.sqrt(len(daily))):+.1f})"


live_signals = stretches(live_ratio, live_gap, LIVE_LEVEL)
session_signals = stretches(session_ratio, session_gap, SESSION_LEVEL)
both = [(t, side) for t, side in live_signals if side * session_ratio.values[t] >= AGREE_LEVEL]
SETS = {"live stretch": live_signals, "live turn": turns(live_signals, live_gap.values),
        "session stretch": session_signals, "session turn": turns(session_signals, session_gap.values),
        "both stretch": both, "both turn": turns(both, live_gap.values)}
results = {name: outcomes(events) for name, events in SETS.items()}

rows = []
for name, frame in results.items():
    row = {"signal": name, "a week": len(frame) / weeks}
    row.update({column: mean_t(frame, column) for column in ["ahead 15m", "ahead 30m", "ahead 60m", "ahead 120m"]})
    row["best / worst 60m (median)"] = f"{frame['best 60m'].median():.1f} / {frame['worst 60m'].median():.1f}"
    for exit_name in ["live", "session"]:
        column = f"exit {exit_name}"
        row[f"{column}: net (t)"] = mean_t(frame, column)
        row[f"{column}: win rate"] = (frame[column] > 0).mean()
        row[f"{column}: avg winner / loser"] = f"{frame.loc[frame[column] > 0, column].mean():+.1f} / {frame.loc[frame[column] <= 0, column].mean():+.1f}"
        row[f"{column}: reached fair value"] = frame[f"{exit_name} reached"].mean()
        row[f"{column}: minutes held"] = frame[f"{exit_name} minutes"].mean()
    rows.append(row)
print("== development 2021 - Nov 2024. 'ahead' and best / worst are BEFORE costs; the two exits are AFTER a 1-tick cost. mean (t)")
print(pd.DataFrame(rows).set_index("signal").round(2).T.to_string())

for column in ["ahead 60m", "exit live", "exit session"]:
    table = []
    for name, frame in results.items():
        row = {"signal": name}
        for group_column in ["half", "session"]:
            for key, part in frame.groupby(group_column):
                row[key] = f"{mean_t(part, column)} {len(part) / weeks:.1f}/wk"
        table.append(row)
    print(f"\n-- {column}{' (before costs)' if column.startswith('ahead') else ' (after costs)'}: mean (t) and signals a week, by half and by session")
    print(pd.DataFrame(table).set_index("signal").to_string())
