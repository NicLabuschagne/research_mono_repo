"""CHARTS ONLY: the alert as it stands (live turn / both turn) with the live fair value at four speeds, one panel
per speed, same three trading days. Signals and data come from bean_limit.py unchanged. No trades, no exits."""
import contextlib, io, runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(__file__.replace("bean_speed_charts.py", "bean_limit.py"))
bars, live_gaps, levels, turn_signals = g["bars"], g["live_gaps"], g["levels"], g["turn_signals"]
high_s, low_s, session_gap, session_id, TICK = g["high_s"], g["low_s"], g["session_gap"].values, g["session_id"], g["TICK_CENTS"]
open_s, close_s = g["open_s"], g["close_s"]
NAMES = {0.1: "live speed as now (gap halves in 12 min)", 0.03: "3x slower (20 min)",
         0.01: "10x slower (34 min)", 0.003: "30x slower (60 min)"}
signals = {drift: turn_signals(live_gaps[drift], levels[drift]) for drift in NAMES}
trading_dates = np.array(sorted(bars["trading_date"].unique()))
out_dir = g["ROOT"] / "strategies" / "bean_divergence" / "output" / "audit_20261116"

for window_start in ["2021-03-15", "2022-09-12", "2023-06-12", "2024-02-12", "2024-11-21"]:
    days = trading_dates[trading_dates >= pd.Timestamp(window_start)][:3]
    shown = np.flatnonzero(bars["trading_date"].isin(days).values)
    first, last = shown[0], shown[-1] + 1
    x = np.arange(last - first)
    open_, close, high, low = open_s[first:last], close_s[first:last], high_s[first:last], low_s[first:last]
    colours = np.where(close >= open_, "#2ca02c", "#d62728")
    reach = (high.max() - low.min()) * 0.04
    ids = session_id[first:last]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    figure, axes = plt.subplots(4, 1, figsize=(24, 30), sharex=True)
    for axis, drift in zip(axes, NAMES):
        axis.vlines(x, low, high, color=colours, lw=0.7, alpha=0.6)
        axis.bar(x, np.maximum(np.abs(close - open_), 0.05), bottom=np.minimum(open_, close), width=0.7, color=colours, alpha=0.6)
        axis.plot(x, close - live_gaps[drift].values[first:last] * TICK, color="black", lw=1.3, label="live fair value")
        axis.plot(x, close - session_gap[first:last] * TICK, color="darkorange", lw=1.3, label="session fair value")
        for name, colour, size in [("live turn", "white", 130), ("both turn", "#1f77b4", 330)]:
            chosen = [(t - first, side) for t, side in signals[drift][name] if first <= t < last]
            for sign, marker in [(1, "v"), (-1, "^")]:
                xs = np.array([i for i, side in chosen if side == sign], dtype=int)
                ys = high[xs] + reach if sign == 1 else low[xs] - reach
                axis.scatter(xs, ys, marker=marker, s=size, color=colour, edgecolor="black", linewidth=1.5,
                             zorder=5 if size < 200 else 4, label=f"{name}: {len(chosen)}" if sign == 1 else None)
        for session_start in starts:
            axis.axvline(session_start - 0.5, color="grey", lw=0.8, alpha=0.7)
        axis.grid(axis="y", alpha=0.3)
        axis.set_ylabel("soybeans, cents per bushel")
        axis.set_title(f"{NAMES[drift]}: live gap stretched at {levels[drift]:.2f} x the expected 1-hour move", loc="left", fontsize=13)
        axis.legend(loc="upper left", ncol=4)
    local = bars.index[first:last].tz_convert("Europe/London")
    ticks = np.arange(0, len(x), 24)
    axes[-1].set_xticks(ticks, [f"{local[i]:%d %b\n%H:%M}" for i in ticks], fontsize=8)
    axes[0].set_xlim(-1, len(x) + 1)
    figure.suptitle(f"{pd.Timestamp(days[0]):%d %b %Y} to {pd.Timestamp(days[-1]):%d %b %Y}, 5-minute bars. White = live turn, blue = both turn "
                    f"(live turn with the session gap stretched the same way). Triangle down = sell, up = buy", x=0.01, ha="left", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    path = out_dir / f"speeds_{window_start}.png"
    figure.savefig(path, dpi=90)
    plt.close(figure)
    print(path, {NAMES[d].split(" (")[0]: {k: sum(first <= t < last for t, _ in v) for k, v in signals[d].items()} for d in NAMES})
