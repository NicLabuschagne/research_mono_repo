"""CHARTS ONLY (development data): where the live and the session gap signal, with a 5-tick floor. No trades, no exits.
  live signal    : first 5-minute close with |live z| >= 1.645 AND |live gap| >= 5 ticks (z = gap / its 4-hour size)
  session signal : first close with |session z| >= 1.645 AND |session gap| >= 5 ticks (z = gap / its 20-day size)
  both           : a live signal while the session gap is on the same side and also >= 5 ticks
  A side can signal again only after its |z| is back inside 0.6745; a new session starts fresh.
Windows of three trading days, fixed before drawing: the one the user marked up plus one per year."""
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
WINDOW_STARTS = ["2021-03-15", "2022-09-12", "2023-06-12", "2024-02-12", "2024-11-21"]

cfg = load()
r = cfg.research
bars = load_bars(cfg)
in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
session_id = bars["session_id"].values
window = 20 * 211
live_gap = crush_gap(bars, crush_hedge(bars, True, 5))
session_gap = crush_gap(bars, crush_hedge(bars, False, 5))
live_z = live_gap / adaptive_scale(live_gap, r.z_half_life_bars)
session_z = session_gap / np.sqrt((session_gap ** 2).rolling(window, min_periods=window // 2).mean()).shift(1)


def signals(z: pd.Series, gap: pd.Series) -> list[tuple[int, int]]:
    """find_triggers with the floor: a bar whose gap is under the floor cannot fire (its z is held just under the
    trigger level), but still counts for re-arming."""
    held = np.sign(z) * np.minimum(z.abs(), r.trigger_z - 0.01)
    return find_triggers(z.where(gap.abs() >= MIN_GAP_TICKS, held).values, session_id, r.trigger_z, r.rearm_z)


live_signals = signals(live_z, live_gap)
session_signals = signals(session_z, session_gap)
both = [(t, side) for t, side in live_signals if side * session_gap.values[t] >= MIN_GAP_TICKS]
weeks = bars.loc[in_dev, "trading_date"].nunique() / 5
print("signals a week over development:",
      {name: round(sum(in_dev[t] for t, _ in events) / weeks, 1)
       for name, events in [("live", live_signals), ("session", session_signals), ("both", both)]})

minutes = pd.read_parquet(ROOT / cfg.data_path, columns=["ZS_high", "ZS_low", "ZS_close"])
trading_dates = np.array(sorted(bars["trading_date"].unique()))
out_dir = STRATEGY_DIR / "output" / "audit_20261116"
out_dir.mkdir(exist_ok=True)

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
    live_g, session_g = live_gap.values[first:last], session_gap.values[first:last]

    figure, (top, middle, bottom) = plt.subplots(3, 1, figsize=(24, 15), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1.2]})
    top.vlines(x, low, high, color=colours, lw=0.7, alpha=0.6)
    top.bar(x, np.maximum(np.abs(close - open_), 0.05), bottom=np.minimum(open_, close), width=0.7, color=colours, alpha=0.6)
    top.plot(x, close - live_g * TICK_CENTS, color="black", lw=1.1, label="live fair value")
    top.plot(x, close - session_g * TICK_CENTS, color="darkorange", lw=1.3, label="session fair value")
    reach = (high.max() - low.min()) * 0.035                                       # markers sit clear of the candles
    counts = {}
    for name, events, colour, size in [("live signal", live_signals, "white", 110), ("session signal", session_signals, "darkorange", 150),
                                       ("both (live signal, session gap the same side)", both, "#1f77b4", 260)]:
        chosen = [(t - first, side) for t, side in events if first <= t < last]
        counts[name] = len(chosen)
        for sign, marker in [(1, "v"), (-1, "^")]:                                 # rich -> sell (down), cheap -> buy (up)
            xs = np.array([i for i, side in chosen if side == sign], dtype=int)
            ys = high[xs] + reach if sign == 1 else low[xs] - reach
            top.scatter(xs, ys, marker=marker, s=size, color=colour, edgecolor="black", linewidth=1.2, zorder=5 if size < 200 else 4,
                        label=f"{name}: {len(chosen)}" if sign == 1 else None)
    top.set_ylabel("soybeans, cents per bushel (roll-adjusted)")
    top.set_title(f"{pd.Timestamp(days[0]):%d %b %Y} to {pd.Timestamp(days[-1]):%d %b %Y}, 5-minute bars. Signals need |z| >= {r.trigger_z} "
                  f"and a gap of at least {MIN_GAP_TICKS} ticks. Triangle down = beans rich (sell), up = beans cheap (buy)", loc="left")
    top.legend(loc="upper left", ncol=5)

    middle.bar(x, live_g, width=0.8, color="#1f77b4", alpha=0.7, label="live gap (ticks)")
    middle.plot(x, session_g, color="darkorange", lw=1.3, label="session gap (ticks)")
    for level in [MIN_GAP_TICKS, -MIN_GAP_TICKS]:
        middle.axhline(level, color="grey", ls="--", lw=1)
    middle.axhline(0, color="grey", lw=0.8)
    middle.set_ylabel("gap in ticks: + = beans above fair value")
    middle.legend(loc="upper left", ncol=2)

    bottom.plot(x, live_z.values[first:last], color="black", lw=0.9, label="live z (gap / its 4-hour size)")
    bottom.plot(x, session_z.values[first:last], color="darkorange", lw=1.3, label="session z (gap / its 20-day size)")
    for level in [r.trigger_z, -r.trigger_z]:
        bottom.axhline(level, color="grey", ls="--", lw=1)
    bottom.axhline(0, color="grey", lw=0.8)
    bottom.set_ylim(-5, 5)
    bottom.set_ylabel("z")
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
    path = out_dir / f"signals_{window_start}.png"
    figure.savefig(path, dpi=100)
    plt.close(figure)
    print(path, counts)
