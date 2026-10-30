"""
layer1_signal_days.py
---------------------
Look at the candidate signals on continuous days, and count how many DISTINCT events they give together.

Candidates (prior-close crush gap, ZS ticks):
    A  Europe, beyond the 20-day 95th / 5th percentile               (layer1_extremes)
    B  Europe, |adaptive z| >= 1.645, 4-hour half-life               (layer1_adaptive_extremes)
    C  US,     |adaptive z| >= 1.645, 4-hour half-life
Combined count: events from all three, with any event within one hour of an earlier one in the SAME session and
direction counted once (the same dislocation flagged twice).

Charts (output/layer1_signal_days_<year>.png): the 4 trading days with the most candidate signals in 2022 and in
2024 (chosen for having something to look at, so not typical days). Panels: beans with the signals (outright view),
the gap with the 20-day 95th / 5th levels and the adaptive band (+/-1.645 x recent size), sessions shaded.
Development only.

    python -m strategies.bean_divergence.layer1_signal_days
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import ROOT
from core_quant.theme import LIGHT, PALETTE
from strategies.bean_divergence.failure_study import session_label
from strategies.bean_divergence.feature_study import STRATEGY_DIR, TICK_CENTS
from strategies.bean_divergence.layer1_adaptive_extremes import REARM, adaptive_scale
from strategies.bean_divergence.layer1_extremes import find_extremes, past_percentiles
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.kalman_filter import freeze_at_previous_session_close, run_kalman_filter
from strategies.crush_strategy.strategy import STRATEGY_DIR as CRUSH_DIR, load

Z_95 = 1.6449
HALF_LIFE_4H = 48
SESSION_COLOUR = {"asia": LIGHT["neutral"], "europe": "#dce9f8", "us": "#fde6da"}
MARKER = {"A": "^", "B": "o", "C": "s"}
WINDOW_DAYS = 4


if __name__ == "__main__":
    cfg = load(CRUSH_DIR / "config.yaml")
    bars = build_bars(ROOT / cfg.data_path, 5)
    bars = bars[bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start)]        # holdout stays locked
    states = run_kalman_filter(bars, 5)
    hedge = freeze_at_previous_session_close(bars, states)
    log_c = {leg: np.log(bars[f"{leg}_adj_close"]) for leg in ["ZS", "ZM", "ZL"]}
    ticks = bars["ZS_adj_close"] / TICK_CENTS
    gap = (log_c["ZS"] - (hedge["alpha"] + hedge["beta_meal"] * log_c["ZM"] + hedge["beta_oil"] * log_c["ZL"])) * ticks
    session = pd.Series(session_label(bars.index, bars["session_type"]), index=bars.index)
    session_id = bars["session_id"].values
    dates = bars["trading_date"]

    # the two kinds of levels
    pct = past_percentiles(gap, session, dates, [0.05, 0.25, 0.75, 0.95])
    scale = adaptive_scale(gap, HALF_LIFE_4H)
    z = gap / scale
    constant = pd.DataFrame({"hi": Z_95, "lo": -Z_95, 0.75: REARM, 0.25: -REARM}, index=bars.index)

    events = []
    for name, found in [("A", find_extremes(gap.values, pct, session_id, 0.95, 0.05)),
                        ("adaptive", find_extremes(z.values, constant, session_id, "hi", "lo"))]:
        for s, side in found:
            sess = session.iloc[s]
            label = name if name == "A" else ("B" if sess == "europe" else "C" if sess == "us" else None)
            if label is None or (label == "A" and sess != "europe"):
                continue
            if dates.iloc[s] < pd.Timestamp(cfg.split.development_start):
                continue
            events.append({"bar": s, "time": bars.index[s], "trading_date": dates.iloc[s], "session": sess,
                           "signal": label, "side": side, "gap_ticks": gap.iloc[s]})
    events = pd.DataFrame(events).sort_values("bar").reset_index(drop=True)

    # distinct events: the same session and direction within one hour of a kept event counts once
    kept_bars, distinct = [], []
    for ev in events.itertuples():
        duplicate = any(k_sess == session_id[ev.bar] and k_side == ev.side and 0 <= ev.bar - k_bar <= 12
                        for k_bar, k_sess, k_side in kept_bars)
        distinct.append(not duplicate)
        if not duplicate:
            kept_bars.append((ev.bar, session_id[ev.bar], ev.side))
    events["distinct"] = distinct
    events.to_csv(STRATEGY_DIR / "output" / "layer1_signal_days_events.csv", index=False)
    weeks = dates[dates >= pd.Timestamp(cfg.split.development_start)].nunique() / 5
    print(f"per week: A {(events.signal == 'A').sum() / weeks:.1f}, B {(events.signal == 'B').sum() / weeks:.1f}, "
          f"C {(events.signal == 'C').sum() / weeks:.1f}; all {len(events) / weeks:.1f}; distinct {events.distinct.sum() / weeks:.1f}")
    both_ab = events[events.signal.isin(["A", "B"])]
    print(f"Europe A and B on the same dislocation: {(~both_ab.distinct).sum()} of {len(both_ab)} Europe events")

    # charts: 4 consecutive trading days with the most signals, one window in 2022 and one in 2024
    trading_days = pd.Series(sorted(dates.unique()))
    per_day = events.groupby("trading_date").size().reindex(trading_days, fill_value=0)
    for year in [2022, 2024]:
        in_year = trading_days[trading_days.dt.year == year].index
        best_start = max(in_year[:-WINDOW_DAYS], key=lambda i: per_day.iloc[i:i + WINDOW_DAYS].sum())
        window_days = trading_days.iloc[best_start:best_start + WINDOW_DAYS]
        idx = np.flatnonzero(dates.isin(window_days).values)
        x = np.arange(len(idx))
        fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
        sess = session.values[idx]
        start = 0
        for i in range(1, len(x) + 1):                          # shade each run of bars in one session
            if i == len(x) or sess[i] != sess[start] or session_id[idx[i]] != session_id[idx[start]]:
                for ax in axes:
                    ax.axvspan(start, i - 1, color=SESSION_COLOUR[sess[start]], linewidth=0, zorder=0)
                start = i
        axes[0].plot(x, bars["ZS_adj_close"].values[idx], color=LIGHT["ink"], linewidth=0.9, label="beans (ZS)")
        axes[1].plot(x, gap.values[idx], color=LIGHT["ink"], linewidth=0.9, label="gap: beans - crush fair value")
        axes[1].plot(x, pct[0.95].values[idx], color=PALETTE[0], linestyle="--", linewidth=0.9, label="20-day 95th / 5th pct (A)")
        axes[1].plot(x, pct[0.05].values[idx], color=PALETTE[0], linestyle="--", linewidth=0.9)
        axes[1].fill_between(x, -Z_95 * scale.values[idx], Z_95 * scale.values[idx], color=PALETTE[1], alpha=0.15,
                             linewidth=0, label="adaptive band +/-1.645 x recent size (B, C)")
        axes[1].axhline(0, color=LIGHT["muted"], linewidth=0.8)
        position = {bar: k for k, bar in enumerate(idx)}
        for ev in events[events["bar"].isin(idx)].itertuples():
            k = position[ev.bar]
            colour = PALETTE[7] if ev.side > 0 else PALETTE[5]        # red: beans rich (sell), green: beans cheap (buy)
            axes[0].scatter(k, bars["ZS_adj_close"].values[ev.bar], marker=MARKER[ev.signal], color=colour, s=60, zorder=3)
            axes[1].scatter(k, ev.gap_ticks, marker=MARKER[ev.signal], color=colour, s=60, zorder=3)
        for sig, mk in MARKER.items():
            axes[1].scatter([], [], marker=mk, color=LIGHT["ink_secondary"], label=f"signal {sig}")
        axes[0].set_ylabel("c/bu"); axes[1].set_ylabel("ZS ticks (1 tick = 0.25 c/bu)")
        axes[0].set_title(f"{year}: {window_days.iloc[0]:%a %d %b} - {window_days.iloc[-1]:%a %d %b}  "
                          "(grey Asia, blue Europe, orange US; red = beans rich -> sell, green = beans cheap -> buy)", fontsize=10)
        axes[0].legend(fontsize=8, loc="upper left"); axes[1].legend(fontsize=8, loc="upper left", ncol=3)
        day_starts = [k for k in range(len(idx)) if k == 0 or dates.values[idx[k]] != dates.values[idx[k - 1]]]
        axes[1].set_xticks(day_starts, [f"{pd.Timestamp(dates.values[idx[k]]):%a %d %b}" for k in day_starts], fontsize=8)
        fig.tight_layout(); fig.savefig(STRATEGY_DIR / "output" / f"layer1_signal_days_{year}.png", dpi=110)
    print("charts saved")
