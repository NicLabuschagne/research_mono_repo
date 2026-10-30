"""
layer1_trigger_window.py
------------------------
Every trigger of layer1_trigger_harvest (adaptive |z| at 1.96 / 2.58 / 3.5, any session) over the LAST 4 trading
days of the development data - not chosen for activity, so a typical stretch.
Panels: beans with the triggers; the gap with the three adaptive bands (+/- level x recent size); the spread move
and the beans move 30 min after each trigger (from the trigger close, ticks).

    python -m strategies.bean_divergence.layer1_trigger_window           # prior-close gap
    python -m strategies.bean_divergence.layer1_trigger_window live      # live gap
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from core_quant.config import ROOT
from core_quant.theme import LIGHT, PALETTE
from strategies.bean_divergence.failure_study import session_label
from strategies.bean_divergence.feature_study import STRATEGY_DIR, TICK_CENTS
from strategies.bean_divergence.layer1_adaptive_extremes import REARM, adaptive_scale
from strategies.bean_divergence.layer1_extremes import find_extremes
from strategies.bean_divergence.layer1_signal_days import HALF_LIFE_4H, SESSION_COLOUR, WINDOW_DAYS
from strategies.crush_strategy.data_ingest import build_bars
from strategies.bean_divergence.layer1_trigger_harvest import GAP, SUFFIX, crush_hedge
from strategies.crush_strategy.strategy import STRATEGY_DIR as CRUSH_DIR, load

LEVEL_STYLE = {1.96: ("o", 40), 2.58: ("s", 60), 3.5: ("*", 160)}
AFTER_BARS = 6                      # 30 min

if __name__ == "__main__":
    cfg = load(CRUSH_DIR / "config.yaml")
    bars = build_bars(ROOT / cfg.data_path, 5)
    bars = bars[bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start)]        # holdout stays locked
    hedge = crush_hedge(bars)
    a, bm, bo = (hedge[c].values for c in ["alpha", "beta_meal", "beta_oil"])
    log_c = {leg: np.log(bars[f"{leg}_adj_close"]).values for leg in ["ZS", "ZM", "ZL"]}
    ticks = (bars["ZS_adj_close"] / TICK_CENTS).values
    gap = (log_c["ZS"] - (a + bm * log_c["ZM"] + bo * log_c["ZL"])) * ticks
    scale = adaptive_scale(pd.Series(gap), HALF_LIFE_4H).values
    session_id = bars["session_id"].values
    session = session_label(bars.index, bars["session_type"])
    dates = bars["trading_date"].values

    last_days = np.sort(np.unique(dates))[-WINDOW_DAYS:]
    idx = np.flatnonzero(np.isin(dates, last_days))
    idx = idx[idx + AFTER_BARS < len(bars)]            # the last 30 min of the data have no 30-minute outcome
    triggers = []
    for level in LEVEL_STYLE:
        constant = pd.DataFrame({"hi": level, "lo": -level, 0.75: REARM, 0.25: -REARM}, index=bars.index)
        for s, side in find_extremes(gap / scale, constant, session_id, "hi", "lo"):
            if s in set(idx):
                held = lambda i: log_c["ZS"][i] - (a[s] + bm[s] * log_c["ZM"][i] + bo[s] * log_c["ZL"][i])
                triggers.append({"bar": s, "time_ct": bars.index[s].tz_convert("America/Chicago").strftime("%a %d %b %H:%M"),
                                 "session": session[s], "level": level, "side": "rich -> sell" if side > 0 else "cheap -> buy",
                                 "gap_ticks": gap[s], "z": gap[s] / scale[s],
                                 "beans_30m": -side * (log_c["ZS"][s + AFTER_BARS] - log_c["ZS"][s]) * ticks[s],
                                 "spread_30m": -side * (held(s + AFTER_BARS) - held(s)) * ticks[s], "sign": side})
    triggers = pd.DataFrame(triggers).sort_values("bar").reset_index(drop=True)

    x = np.arange(len(idx))
    position = {bar: k for k, bar in enumerate(idx)}
    fig, axes = plt.subplots(3, 1, figsize=(18, 11), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.2, 0.8]})
    start = 0
    for i in range(1, len(x) + 1):                          # shade each run of bars in one session
        if i == len(x) or session[idx[i]] != session[idx[start]] or session_id[idx[i]] != session_id[idx[start]]:
            for ax in axes:
                ax.axvspan(start, i - 1, color=SESSION_COLOUR[session[idx[start]]], linewidth=0, zorder=0)
            start = i
    fair_value = bars["ZS_adj_close"].values * np.exp(-gap / ticks)      # gap = (log beans - log fair) x ticks
    axes[0].plot(x, bars["ZS_adj_close"].values[idx], color=LIGHT["ink"], linewidth=0.9, label="beans (ZS)")
    axes[0].plot(x, fair_value[idx], color=PALETTE[1], linewidth=0.9, alpha=0.9, label=f"crush fair value ({GAP.replace('_', '-')})")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].plot(x, gap[idx], color=LIGHT["ink"], linewidth=0.9, label="gap: beans - crush fair value")
    for colour, level in zip([PALETTE[2], PALETTE[0], PALETTE[1]], LEVEL_STYLE):
        axes[1].plot(x, level * scale[idx], color=colour, linewidth=0.8, linestyle="--", label=f"+/- {level} x recent size")
        axes[1].plot(x, -level * scale[idx], color=colour, linewidth=0.8, linestyle="--")
    axes[1].axhline(0, color=LIGHT["muted"], linewidth=0.8)
    for tr in triggers.itertuples():
        k = position[tr.bar]
        colour = PALETTE[7] if tr.sign > 0 else PALETTE[5]        # red: beans rich (sell), green: beans cheap (buy)
        marker, size = LEVEL_STYLE[tr.level]
        axes[0].scatter(k, bars["ZS_adj_close"].values[tr.bar], marker=marker, s=size, color=colour, zorder=3)
        axes[1].scatter(k, tr.gap_ticks, marker=marker, s=size, color=colour, zorder=3)
        axes[2].bar(k - 3, tr.spread_30m, width=6, color=PALETTE[0], alpha=0.8)
        axes[2].bar(k + 3, tr.beans_30m, width=6, color=PALETTE[3], alpha=0.8)
    for level, (marker, size) in LEVEL_STYLE.items():
        axes[1].scatter([], [], marker=marker, s=size, color=LIGHT["ink_secondary"], label=f"trigger |z| >= {level}")
    axes[2].legend(handles=[Patch(color=PALETTE[0], label="spread move 30 min after (+ = toward fair value)"),
                            Patch(color=PALETTE[3], label="beans move 30 min after (+ = in the trade's favour)")],
                   fontsize=8, loc="upper left")
    axes[2].axhline(0, color=LIGHT["muted"], linewidth=0.8)
    axes[0].set_ylabel("c/bu"); axes[1].set_ylabel("ZS ticks (1 tick = 0.25 c/bu)"); axes[2].set_ylabel("ZS ticks")
    axes[0].set_title(f"Last {WINDOW_DAYS} development days, {GAP.replace('_', '-')} gap: every adaptive-z trigger (grey Asia, blue Europe, orange US; "
                      "red = beans rich -> sell, green = beans cheap -> buy)", fontsize=10)
    axes[1].legend(fontsize=8, loc="upper left", ncol=4)
    day_starts = [k for k in range(len(idx)) if k == 0 or dates[idx[k]] != dates[idx[k - 1]]]
    axes[2].set_xticks(day_starts, [f"{pd.Timestamp(dates[idx[k]]):%a %d %b}" for k in day_starts], fontsize=8)
    fig.tight_layout()
    out = STRATEGY_DIR / "output" / f"layer1_trigger_window{SUFFIX}.png"
    fig.savefig(out, dpi=110)
    pd.set_option("display.width", 200)
    print(triggers.drop(columns=["bar", "sign"]).round(1).to_string(index=False))
    print("saved", out)
