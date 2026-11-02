"""
charts.py
---------
Every trigger over the LAST 4 trading days of the development data - not chosen for activity, so a typical stretch.
Panels: beans with the crush fair value and the triggers; the gap with the trigger band (+/- trigger_z x recent
size), marker size = |z| at the trigger; the spread and beans move 30 min after each trigger (ticks).

    python -m strategies.bean_divergence.charts                   # live gap (the base)
    python -m strategies.bean_divergence.charts prior_close       # prior-close gap
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from core_quant.theme import LIGHT, PALETTE
from strategies.bean_divergence.fair_value import (LEGS, STRATEGY_DIR, TICK_CENTS, adaptive_scale, load,
                                                   session_label)
from strategies.bean_divergence.triggers import build

SESSION_COLOUR = {"asia": LIGHT["neutral"], "europe": "#dce9f8", "us": "#fde6da"}
WINDOW_DAYS = 4
AFTER_BARS = 6                      # 30 min

if __name__ == "__main__":
    cfg = load()
    live = "prior_close" not in sys.argv[1:]
    gap_name = "live" if live else "prior-close"
    bars, hedge, gap_s, z_s, triggers = build(cfg, live)
    a, bm, bo = (hedge[c].values for c in ["alpha", "beta_meal", "beta_oil"])
    log_c = {leg: np.log(bars[f"{leg}_adj_close"]).values for leg in LEGS}
    ticks = (bars["ZS_adj_close"] / TICK_CENTS).values
    gap = gap_s.values
    band = cfg.research.trigger_z * adaptive_scale(gap_s, cfg.research.z_half_life_bars).values
    session_id = bars["session_id"].values
    session = session_label(bars.index, bars["session_type"])
    dates = bars["trading_date"].values

    last_days = np.sort(np.unique(dates))[-WINDOW_DAYS:]
    idx = np.flatnonzero(np.isin(dates, last_days))
    idx = idx[idx + AFTER_BARS < len(bars)]            # the last 30 min of the data have no 30-minute outcome
    position = {bar: k for k, bar in enumerate(idx)}
    shown = triggers[triggers["bar"].isin(idx)].copy()
    after_beans, after_spread = [], []
    for tr in shown.itertuples():
        s, e = tr.bar, tr.bar + AFTER_BARS
        held = lambda i: log_c["ZS"][i] - (a[s] + bm[s] * log_c["ZM"][i] + bo[s] * log_c["ZL"][i])
        after_beans.append(-tr.side * (log_c["ZS"][e] - log_c["ZS"][s]) * ticks[s])
        after_spread.append(-tr.side * (held(e) - held(s)) * ticks[s])
    shown["beans_30m"], shown["spread_30m"] = after_beans, after_spread

    x = np.arange(len(idx))
    fig, axes = plt.subplots(3, 1, figsize=(18, 11), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.2, 0.8]})
    start = 0
    for i in range(1, len(x) + 1):                          # shade each run of bars in one session
        if i == len(x) or session[idx[i]] != session[idx[start]] or session_id[idx[i]] != session_id[idx[start]]:
            for ax in axes:
                ax.axvspan(start, i - 1, color=SESSION_COLOUR[session[idx[start]]], linewidth=0, zorder=0)
            start = i
    fair_value = bars["ZS_adj_close"].values * np.exp(-gap / ticks)      # gap = (log beans - log fair) x ticks
    axes[0].plot(x, bars["ZS_adj_close"].values[idx], color=LIGHT["ink"], linewidth=0.9, label="beans (ZS)")
    axes[0].plot(x, fair_value[idx], color=PALETTE[1], linewidth=0.9, alpha=0.9, label=f"crush fair value ({gap_name})")
    axes[1].plot(x, gap[idx], color=LIGHT["ink"], linewidth=0.9, label="gap: beans - crush fair value")
    axes[1].fill_between(x, -band[idx], band[idx], color=PALETTE[0], alpha=0.12, linewidth=0,
                         label=f"trigger band +/- {cfg.research.trigger_z} x recent size")
    axes[1].axhline(0, color=LIGHT["muted"], linewidth=0.8)
    for tr in shown.itertuples():
        k = position[tr.bar]
        colour = PALETTE[7] if tr.side > 0 else PALETTE[5]        # red: beans rich (sell), green: beans cheap (buy)
        size = 25 * min(abs(tr.z), 6) ** 1.5                      # bigger marker = bigger |z| (capped for the plot)
        axes[0].scatter(k, bars["ZS_adj_close"].values[tr.bar], s=size, color=colour, zorder=3)
        axes[1].scatter(k, tr.gap_ticks, s=size, color=colour, zorder=3)
        axes[2].bar(k - 3, tr.spread_30m, width=6, color=PALETTE[0], alpha=0.8)
        axes[2].bar(k + 3, tr.beans_30m, width=6, color=PALETTE[3], alpha=0.8)
    axes[2].legend(handles=[Patch(color=PALETTE[0], label="spread move 30 min after (+ = toward fair value)"),
                            Patch(color=PALETTE[3], label="beans move 30 min after (+ = in the trade's favour)")],
                   fontsize=8, loc="upper left")
    axes[2].axhline(0, color=LIGHT["muted"], linewidth=0.8)
    axes[0].set_ylabel("c/bu"); axes[1].set_ylabel("ZS ticks (1 tick = 0.25 c/bu)"); axes[2].set_ylabel("ZS ticks")
    axes[0].set_title(f"Last {WINDOW_DAYS} development days, {gap_name} gap: every trigger, marker size = |z| "
                      "(grey Asia, blue Europe, orange US; red = beans rich -> sell, green = beans cheap -> buy)", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper left"); axes[1].legend(fontsize=8, loc="upper left")
    day_starts = [k for k in range(len(idx)) if k == 0 or dates[idx[k]] != dates[idx[k - 1]]]
    axes[2].set_xticks(day_starts, [f"{pd.Timestamp(dates[idx[k]]):%a %d %b}" for k in day_starts], fontsize=8)
    fig.tight_layout()
    out = STRATEGY_DIR / "output" / f"trigger_window{'' if live else '_prior_close'}.png"
    fig.savefig(out, dpi=110)
    shown["time_ct"] = pd.to_datetime(shown["time"]).dt.tz_convert("America/Chicago").dt.strftime("%a %d %b %H:%M")
    pd.set_option("display.width", 200)
    print(shown[["time_ct", "session", "side", "gap_ticks", "z", "beans_30m", "spread_30m"]].round(1).to_string(index=False))
    print("saved", out)
