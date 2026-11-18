"""CHARTS ONLY (development data), second version: a gap counts as stretched against BEANS' OWN MOVEMENT, so a signal
is a stretch you can see on the chart in a quiet week and in a wild one. No trades, no exits.
  yardstick      : the expected 1-hour beans move (labels.py sigma_1h: 4-hour exponential size of 5-minute moves x sqrt 12)
  live stretch   : first close with |live gap| >= 0.5 x yardstick      (4% of bars)
  session stretch: first close with |session gap| >= 1.5 x yardstick   (11% of bars)
  both           : a live stretch while the session gap is on the same side and >= 1.0 x yardstick
  every signal also needs a gap of 5 ticks; a side fires again only after its gap is back inside 0.4 x its level.
  turn (filled)  : the first close within 30 minutes of a stretch where that gap is no wider than the bar before AND
                   the fair value has not moved against the trade over the last 15 minutes (meal and oil at the
                   bar's hedge ratios: flat or already going the trade's way).
The levels were set from how OFTEN they occur, before looking at what happens after them."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import ROOT
from strategies.bean_divergence.fair_value import (STRATEGY_DIR, TICK_CENTS, adaptive_scale, crush_gap, crush_hedge,
                                                   load, load_bars)
from strategies.bean_divergence.triggers import find_triggers

MIN_GAP_TICKS = 5
LIVE_LEVEL, SESSION_LEVEL, AGREE_LEVEL, REARM_SHARE = 0.5, 1.5, 1.0, 0.4
TURN_WAIT_BARS, FAIR_LOOKBACK_BARS = 6, 3
WINDOW_STARTS = ["2021-03-15", "2022-09-12", "2023-06-12", "2024-02-12", "2024-11-21"]

cfg = load()
bars = load_bars(cfg)
in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
session_id = bars["session_id"].values
to_ticks = bars["ZS_adj_close"] / TICK_CENTS
yardstick = adaptive_scale(bars["ZS_ret"] * to_ticks, 48) * np.sqrt(12)           # expected 1-hour beans move, ticks
hedge = crush_hedge(bars, True, 5)
live_gap = crush_gap(bars, hedge)
session_gap = crush_gap(bars, crush_hedge(bars, False, 5))
live_ratio, session_ratio = live_gap / yardstick, session_gap / yardstick
# fair value's own move over the last 15 minutes, in ticks (inside-session returns, the bar's hedge ratios)
fair_move = ((hedge["beta_meal"] * bars["ZM_ret"].rolling(FAIR_LOOKBACK_BARS).sum()
              + hedge["beta_oil"] * bars["ZL_ret"].rolling(FAIR_LOOKBACK_BARS).sum()) * to_ticks).values


def stretches(ratio: pd.Series, gap: pd.Series, level: float) -> list[tuple[int, int]]:
    """First bar of each stretch beyond the level (triggers.find_triggers); a bar under the tick floor cannot fire."""
    held = np.sign(ratio) * np.minimum(ratio.abs(), level * 0.99)
    return find_triggers(ratio.where(gap.abs() >= MIN_GAP_TICKS, held).values, session_id, level, REARM_SHARE * level)


def turns(events: list[tuple[int, int]], gap: np.ndarray) -> list[tuple[int, int]]:
    """For each stretch, the first bar (from the stretch bar itself) where the gap stopped widening and the fair
    value is flat or moving the trade's way. side +1 = beans rich: we want fair value flat or falling."""
    found = []
    for t, side in events:
        for bar in range(t, min(t + TURN_WAIT_BARS + 1, len(gap))):
            if session_id[bar] != session_id[t] or side * gap[bar] < MIN_GAP_TICKS:
                break                                                              # the stretch is gone
            if abs(gap[bar]) <= abs(gap[bar - 1]) and side * fair_move[bar] <= 0:
                found.append((bar, side))
                break
    return found


live_signals = stretches(live_ratio, live_gap, LIVE_LEVEL)
session_signals = stretches(session_ratio, session_gap, SESSION_LEVEL)
both = [(t, side) for t, side in live_signals if side * session_ratio.values[t] >= AGREE_LEVEL]
SETS = [("live stretch", live_signals, live_gap.values, "white", 110),
        ("session stretch", session_signals, session_gap.values, "darkorange", 150),
        ("both", both, live_gap.values, "#1f77b4", 260)]
weeks = bars.loc[in_dev, "trading_date"].nunique() / 5
print("signals a week over development (stretch / of which turn within 30 minutes):")
for name, events, gap, _, _ in SETS:
    print(f"  {name:16s} {sum(in_dev[t] for t, _ in events) / weeks:5.1f} / {sum(in_dev[t] for t, _ in turns(events, gap)) / weeks:5.1f}")

minutes = pd.read_parquet(ROOT / cfg.data_path, columns=["ZS_high", "ZS_low", "ZS_close"])
trading_dates = np.array(sorted(bars["trading_date"].unique()))
out_dir = STRATEGY_DIR / "output" / "audit_20261116"

for window_start in WINDOW_STARTS:
    days = trading_dates[trading_dates >= pd.Timestamp(window_start)][:3]
    shown = np.flatnonzero(bars["trading_date"].isin(days).values)
    first, last = shown[0], shown[-1] + 1
    part = bars.iloc[first:last]
    inside = minutes[(minutes.index > bars.index[first - 1]) & (minutes.index <= part.index[-1])]
    bar_of_minute = np.searchsorted(part.index, inside.index, side="left")        # a bar is stamped with its last minute
    factor = (part["ZS_adj_close"] / inside["ZS_close"].reindex(part.index)).values
    high = inside["ZS_high"].groupby(bar_of_minute).max().values * factor
    low = inside["ZS_low"].groupby(bar_of_minute).min().values * factor
    open_, close = part["ZS_adj_open"].values, part["ZS_adj_close"].values
    x = np.arange(last - first)
    colours = np.where(close >= open_, "#2ca02c", "#d62728")
    live_g, session_g, yard = live_gap.values[first:last], session_gap.values[first:last], yardstick.values[first:last]

    figure, (top, middle, bottom) = plt.subplots(3, 1, figsize=(24, 15), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    top.vlines(x, low, high, color=colours, lw=0.7, alpha=0.6)
    top.bar(x, np.maximum(np.abs(close - open_), 0.05), bottom=np.minimum(open_, close), width=0.7, color=colours, alpha=0.6)
    top.plot(x, close - live_g * TICK_CENTS, color="black", lw=1.1, label="live fair value")
    top.plot(x, close - session_g * TICK_CENTS, color="darkorange", lw=1.3, label="session fair value")
    reach = (high.max() - low.min()) * 0.03
    counts = {}
    for name, events, gap, colour, size in SETS:
        for kind, chosen_events, filled in [("stretch", events, False), ("turn", turns(events, gap), True)]:
            chosen = [(t - first, side) for t, side in chosen_events if first <= t < last]
            counts[f"{name} {kind}"] = len(chosen)
            for sign, marker in [(1, "v"), (-1, "^")]:                             # rich -> sell (down), cheap -> buy (up)
                xs = np.array([i for i, side in chosen if side == sign], dtype=int)
                distance = reach * (2.2 if filled else 1)                          # turns sit one row further out
                ys = high[xs] + distance if sign == 1 else low[xs] - distance
                face = ("black" if colour == "white" else colour) if filled else "none"
                edge = "black" if filled or colour == "white" else colour
                top.scatter(xs, ys, marker=marker, s=size, facecolor=face, edgecolor=edge, linewidth=1.8, zorder=5,
                            label=f"{name}, {kind}: {len(chosen)}" if sign == 1 else None)
    top.set_ylabel("soybeans, cents per bushel (roll-adjusted)")
    top.set_title(f"{pd.Timestamp(days[0]):%d %b %Y} to {pd.Timestamp(days[-1]):%d %b %Y}, 5-minute bars. Hollow = gap first stretched, "
                  f"filled = stopped widening with fair value flat or turned. Triangle down = beans rich (sell), up = beans cheap (buy)", loc="left")
    top.legend(loc="upper left", ncol=4)

    middle.bar(x, live_g, width=0.8, color="#1f77b4", alpha=0.7, label="live gap (ticks)")
    middle.plot(x, session_g, color="darkorange", lw=1.3, label="session gap (ticks)")
    for sign in [1, -1]:
        middle.plot(x, sign * LIVE_LEVEL * yard, color="#1f77b4", ls="--", lw=1, label="live level (0.5 x expected 1-hour move)" if sign == 1 else None)
        middle.plot(x, sign * SESSION_LEVEL * yard, color="darkorange", ls="--", lw=1, label="session level (1.5 x)" if sign == 1 else None)
    middle.axhline(0, color="grey", lw=0.8)
    middle.set_ylabel("gap in ticks: + = beans above fair value")
    middle.legend(loc="upper left", ncol=4)

    bottom.plot(x, fair_move[first:last], color="purple", lw=1.0, label="fair value's own move over the last 15 minutes (ticks)")
    bottom.plot(x, (bars["ZS_ret"].rolling(FAIR_LOOKBACK_BARS).sum() * to_ticks).values[first:last], color="grey", lw=0.8, alpha=0.8,
                label="beans move over the last 15 minutes (ticks)")
    bottom.axhline(0, color="grey", lw=0.8)
    bottom.set_ylabel("ticks")
    bottom.legend(loc="upper left", ncol=2)

    ids = session_id[first:last]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    local = part.index.tz_convert("Europe/London")
    ticks = np.arange(0, len(x), 24)                                               # every two hours
    bottom.set_xticks(ticks, [f"{local[i]:%d %b\n%H:%M}" for i in ticks], fontsize=8)
    for axis in (top, middle, bottom):
        for session_start in starts:
            axis.axvline(session_start - 0.5, color="grey", lw=0.8, alpha=0.7)
        axis.grid(axis="y", alpha=0.3)
    top.set_xlim(-1, len(x) + 1)
    figure.tight_layout()
    path = out_dir / f"signals_v2_{window_start}.png"
    figure.savefig(path, dpi=100)
    plt.close(figure)
    print(path, counts)
