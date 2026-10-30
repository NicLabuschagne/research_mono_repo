"""
layer1_event_view.py
--------------------
Layer 1, step 1: LOOK before modelling. No trades, no exits.

A "baseline trigger" is the crush model's own signal: the first 5-minute bar where the residual's
z reaches |z| >= 1.5 (hedge frozen at the previous session close). For every trigger in the
development period we record, from 2 h before to 6 h after the trigger bar, staying inside the
same session:
    beans path : how far beans moved from the trigger bar's close, in ZS ticks
    fair path  : how far the fair value (meal + oil through the frozen hedge) moved, in ZS ticks
    gap path   : beans path - fair path (the residual, in ticks)
plus where beans sat in the day's range at the trigger (0 = session low so far, 1 = high).

Charts (strategies/bean_divergence/output/):
    layer1_event_paths.png  average paths by session (Asia / Europe / US), rich vs cheap triggers
    layer1_range_split.png  rich triggers split by where beans sat in the day's range
    layer1_gallery.png      eight real Europe / US triggers from 2023-24, for the eye

    python -m strategies.bean_divergence.layer1_event_view
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.theme import LIGHT, PALETTE
from strategies.bean_divergence.failure_study import session_label
from strategies.crush_strategy.features import detect_signals
from strategies.crush_strategy.strategy import STRATEGY_DIR as CRUSH_DIR, load, prepare_bars
from strategies.bean_divergence.feature_study import STRATEGY_DIR, TICK_CENTS

BEFORE, AFTER = 24, 72          # bars: 2 h before, 6 h after (5-minute bars)
THRESHOLD = 1.5
SESSIONS = ["asia", "europe", "us"]


def collect_events(bars: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """One row per trigger, plus the beans and fair-value paths (rows = triggers, columns = offsets)."""
    log_beans = np.log(bars["ZS_adj_close"].values)
    log_fair = (bars["frozen_alpha"] + bars["frozen_beta_meal"] * np.log(bars["ZM_adj_close"])
                + bars["frozen_beta_oil"] * np.log(bars["ZL_adj_close"])).values
    session_id = bars["session_id"].values
    label = session_label(bars.index, bars["session_type"])

    # where beans sit in today's range so far (running low / high within the session)
    low = pd.Series(log_beans).groupby(session_id).cummin().values
    high = pd.Series(log_beans).groupby(session_id).cummax().values

    signals = []
    for session_type in ["evening", "day"]:
        signals.append(detect_signals(bars, THRESHOLD, session=session_type))
    signals = pd.concat(signals).sort_values("signal_position")

    rows, beans_paths, fair_paths = [], [], []
    offsets = range(-BEFORE, AFTER + 1)
    for signal in signals.itertuples():
        s = signal.signal_position
        ticks_per_log = bars["ZS_adj_close"].values[s] / TICK_CENTS        # turns a log move into ZS ticks
        beans_path, fair_path = [], []
        for k in offsets:
            i = s + k
            # outside the data or outside this session: no value (we never look across a break)
            if i < 0 or i >= len(bars) or session_id[i] != session_id[s]:
                beans_path.append(np.nan)
                fair_path.append(np.nan)
            else:
                beans_path.append((log_beans[i] - log_beans[s]) * ticks_per_log)
                fair_path.append((log_fair[i] - log_fair[s]) * ticks_per_log)
        beans_paths.append(beans_path)
        fair_paths.append(fair_path)
        day_range = high[s] - low[s]
        rows.append({"bar": s, "time": bars.index[s], "trading_date": bars["trading_date"].values[s],
                     "session": label[s], "side": "rich" if signal.entry_z > 0 else "cheap", "z": signal.entry_z,
                     "range_pos": (log_beans[s] - low[s]) / day_range if day_range > 0 else np.nan,
                     "gap_ticks": bars["residual"].values[s] * ticks_per_log})
    return pd.DataFrame(rows), np.array(beans_paths), np.array(fair_paths)


def plot_paths(ax, beans: np.ndarray, fair: np.ndarray, title: str) -> None:
    hours = np.arange(-BEFORE, AFTER + 1) * 5 / 60
    ax.fill_between(hours, np.nanpercentile(beans, 25, axis=0), np.nanpercentile(beans, 75, axis=0),
                    color=PALETTE[0], alpha=0.15, linewidth=0, label="beans, middle 50%")
    ax.plot(hours, np.nanmean(beans, axis=0), color=PALETTE[0], label="beans, average")
    ax.plot(hours, np.nanmean(fair, axis=0), color=PALETTE[1], linestyle="--", label="fair value, average")
    ax.plot(hours, np.nanmean(beans - fair, axis=0), color=LIGHT["ink"], linewidth=1, label="gap, average")
    ax.axvline(0, color=LIGHT["muted"], linewidth=0.8)
    ax.axhline(0, color=LIGHT["muted"], linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("hours from trigger")


if __name__ == "__main__":
    cfg = load(CRUSH_DIR / "config.yaml")
    bars = prepare_bars(cfg)
    bars = bars[(bars["trading_date"] >= pd.Timestamp(cfg.split.development_start))
                & (bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start))]           # holdout stays locked
    events, beans, fair = collect_events(bars)
    out = STRATEGY_DIR / "output"
    events.to_csv(out / "layer1_events.csv", index=False)

    # 1. average paths, by session and side
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for row, side in enumerate(["rich", "cheap"]):
        for col, session in enumerate(SESSIONS):
            pick = ((events["side"] == side) & (events["session"] == session)).values
            plot_paths(axes[row, col], beans[pick], fair[pick], f"{session}: beans {side} (n = {pick.sum()})")
    axes[0, 0].set_ylabel("ZS ticks from trigger"); axes[1, 0].set_ylabel("ZS ticks from trigger")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("What beans and fair value do around a baseline trigger (|z| >= 1.5), development 2021 - Nov 2024",
                 x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out / "layer1_event_paths.png", dpi=110)

    # 2. rich triggers split by where beans sat in the day's range
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for col, session in enumerate(SESSIONS):
        for i, (name, lo, hi) in enumerate([("near day LOW (bottom third)", 0, 1 / 3), ("middle", 1 / 3, 2 / 3),
                                            ("near day HIGH (top third)", 2 / 3, 1.01)]):
            pick = ((events["side"] == "rich") & (events["session"] == session)
                    & (events["range_pos"] >= lo) & (events["range_pos"] < hi)).values
            hours = np.arange(-BEFORE, AFTER + 1) * 5 / 60
            axes[col].plot(hours, np.nanmean(beans[pick], axis=0), color=PALETTE[i], label=f"{name} (n = {pick.sum()})")
        axes[col].axvline(0, color=LIGHT["muted"], linewidth=0.8); axes[col].axhline(0, color=LIGHT["muted"], linewidth=0.8)
        axes[col].set_title(f"{session}: beans rich, average beans path", fontsize=10)
        axes[col].set_xlabel("hours from trigger"); axes[col].legend(fontsize=8)
    axes[0].set_ylabel("ZS ticks from trigger")
    fig.tight_layout(); fig.savefig(out / "layer1_range_split.png", dpi=110)

    # 3. gallery: eight real Europe / US triggers from 2023-24 (fixed seed so it is reproducible)
    recent = events[(events["session"].isin(["europe", "us"])) & (events["trading_date"] >= pd.Timestamp("2023-01-01"))]
    sample = recent.sample(8, random_state=7).sort_values("time")
    fair_cents = np.exp(bars["frozen_alpha"] + bars["frozen_beta_meal"] * np.log(bars["ZM_adj_close"])
                        + bars["frozen_beta_oil"] * np.log(bars["ZL_adj_close"]))
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
    for ax, ev in zip(axes.ravel(), sample.itertuples()):
        same_session = bars["session_id"].values == bars["session_id"].values[ev.bar]
        x = np.flatnonzero(same_session)
        local = bars.index[x].tz_convert("America/Chicago")
        ax.plot(range(len(x)), bars["ZS_adj_close"].values[x], color=PALETTE[0], label="beans")
        ax.plot(range(len(x)), fair_cents.values[x], color=PALETTE[1], label="fair value")
        trigger = int(np.flatnonzero(x == ev.bar)[0])
        ax.axvline(trigger, color=LIGHT["ink"], linestyle=":", linewidth=1)
        ax.set_title(f"{ev.session} {local[trigger]:%d %b %Y %H:%M} CT\nbeans {ev.side}, z {ev.z:+.2f}, "
                     f"day-range pos {ev.range_pos:.2f}", fontsize=9)
        ticks_at = list(range(0, len(x), 24))
        ax.set_xticks(ticks_at, [f"{local[t]:%H:%M}" for t in ticks_at], fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "layer1_gallery.png", dpi=110)

    print(events.groupby(["session", "side"]).agg(n=("z", "size"), median_range_pos=("range_pos", "median"),
                                                  median_gap_ticks=("gap_ticks", "median")).round(2).to_string())
