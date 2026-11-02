"""
signal_charts.py
----------------
Pictures of the v2 signal, to check by eye that it does what the spec says. Development data only.
Saved to runs/<time>_signal_charts/.

    python -m strategies.wheat_corn_lag.signal_charts

Signal shown: lookback L = 32 bars (8 h), |corn_z| > 1 and |own_z| > 1 in the same direction (the probe's
"with corn (overshoot)" episodes), outcome over h = 71 bars (about one session).

1. event_study      average path around every entry, flipped so the overshoot always points UP:
                    wheat (solid) and corn x beta (dashed, the part of the move corn explains).
                    Before entry, wheat should rise further than corn x beta (that gap IS own_z).
                    After entry, a falling wheat line = the fade makes money. Wheat-alone and lag episodes
                    are drawn for comparison.
2. example_episodes 6 random overshoot episodes: wheat against the path corn implied for it.
3. outcomes         every episode's result (bp) and the running total through time.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import save_run
from core_quant.theme import set_theme
from strategies.wheat_corn_lag.features import build_features, build_targets
from strategies.wheat_corn_lag.probe import episodes
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load, load_bars

LOOKBACK, HORIZON = 32, 71
CONTEXTS = ["with corn (overshoot)", "wheat alone", "against corn (lag)"]


def paths(bars, features, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Wheat and corn-x-beta paths in bp from bar -LOOKBACK to +HORIZON around each entry, overshoot pointing up.
    A path point is NaN when it lies in another segment (a possible roll in between)."""
    log_w, log_c, segment = bars["log_w"].values, bars["log_c"].values, bars["segment"].values
    offsets = np.arange(-LOOKBACK, HORIZON + 1)
    wheat = np.full((len(table), len(offsets)), np.nan)
    corn = np.full_like(wheat, np.nan)
    positions = bars.index.get_indexer(table.index)
    for row, (position, own) in enumerate(zip(positions, table["own_z"])):
        direction = np.sign(own)
        beta = features["beta"].iloc[position]
        for column, offset in enumerate(offsets):
            other = position + offset
            if 0 <= other < len(bars) and segment[other] == segment[position]:
                wheat[row, column] = direction * (log_w[other] - log_w[position]) * 1e4
                corn[row, column] = direction * beta * (log_c[other] - log_c[position]) * 1e4
    return wheat, corn


def event_study(bars, features, tables: dict) -> plt.Figure:
    offsets = np.arange(-LOOKBACK, HORIZON + 1)
    figure, axes = plt.subplots(1, len(CONTEXTS), figsize=(15, 4.6), sharey=True)
    for axis, context in zip(axes, CONTEXTS):
        table = tables[context]
        wheat, corn = paths(bars, features, table)
        mean = np.nanmean(wheat, axis=0)
        error = 1.96 * np.nanstd(wheat, axis=0) / np.sqrt(np.sum(np.isfinite(wheat), axis=0))
        axis.plot(offsets, mean, lw=2, label="wheat")
        axis.fill_between(offsets, mean - error, mean + error, alpha=0.2)
        axis.plot(offsets, np.nanmean(corn, axis=0), lw=1.5, ls="--", label="corn x beta")
        axis.axvline(0, color="grey", lw=0.8)
        axis.axhline(0, color="grey", lw=0.8)
        axis.set_title(f"{context}: {len(table)} episodes\nafter entry: wheat {mean[-1]:+.0f} bp", loc="left")
        axis.set_xlabel("bars from entry (15 min)")
    axes[0].set_ylabel("bp, overshoot direction = up")
    axes[0].legend(loc="upper left")
    figure.suptitle(f"Average path around entry (L={LOOKBACK}, h={HORIZON}); band = 95% for the mean", x=0.01, ha="left")
    figure.tight_layout()
    return figure


def example_episodes(bars, features, table: pd.DataFrame, count: int = 6, seed: int = 3) -> plt.Figure:
    picks = table.sample(count, random_state=seed).sort_index()
    figure, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for axis, (time, episode) in zip(axes.flat, picks.iterrows()):
        position = bars.index.get_loc(time)
        start, end = position - LOOKBACK, min(position + HORIZON, len(bars) - 1)
        window = bars.iloc[start:end + 1]
        window = window[window["segment"] == bars["segment"].iloc[position]]
        beta = features["beta"].iloc[position]
        anchor = bars.iloc[start]
        # where wheat "should" be if it had moved exactly beta x corn since the start of the lookback
        implied = anchor["W"] * np.exp(beta * (window["log_c"] - anchor["log_c"]))
        axis.plot(window.index, window["W"], lw=1.6, label="wheat")
        axis.plot(window.index, implied, lw=1.3, ls="--", label=f"corn-implied wheat (beta {beta:.2f})")
        axis.axvline(time, color="grey", lw=1)
        side = "SELL" if episode["own_z"] > 0 else "BUY"
        axis.set_title(f"{time:%Y-%m-%d %H:%M} UTC  {side}\ncorn {episode['corn_z']:+.1f} sd, own {episode['own_z']:+.1f} sd,"
                       f" result {episode['fade_bp']:+.0f} bp", loc="left", fontsize=9)
        axis.tick_params(axis="x", labelrotation=30, labelsize=7)
    axes.flat[0].legend(loc="best", fontsize=8)
    figure.suptitle("Random overshoot episodes: wheat vs the path corn implied, entry = grey line, cents/bu", x=0.01, ha="left")
    figure.tight_layout()
    return figure


def outcomes(tables: dict) -> plt.Figure:
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 4.6))
    for context in CONTEXTS:
        result = tables[context]["fade_bp"]
        # step outlines and density, so a group of 111 is not hidden behind a group of 324
        left.hist(result.clip(-400, 400), bins=32, density=True, histtype="step", lw=1.8,
                  label=f"{context}: median {result.median():+.0f}, mean {result.mean():+.0f}")
        right.plot(result.index, result.cumsum(), lw=1.5, label=context)
    left.axvline(0, color="grey", lw=0.8)
    left.set_xlabel("result per episode, bp (clipped at +-400, so the end bins hold the tails)")
    left.set_ylabel("share of episodes (density)")
    left.set_title("Spread of episode results", loc="left")
    left.legend(fontsize=8)
    right.axhline(0, color="grey", lw=0.8)
    right.set_ylabel("cumulative bp, one unit per episode, before costs")
    right.set_title("Running total through time", loc="left")
    right.legend(fontsize=8)
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    bars = load_bars(cfg)                                     # development data only
    features = build_features(bars, cfg.features)
    frame = features.join(build_targets(bars, cfg.features))
    table = episodes(frame, LOOKBACK, HORIZON)
    tables = {context: table[table["context"] == context] for context in CONTEXTS}

    figures = {"event_study": event_study(bars, features, tables),
               "example_episodes": example_episodes(bars, features, tables["with corn (overshoot)"]),
               "outcomes": outcomes(tables)}
    run_dir = save_run(STRATEGY_DIR, "signal_charts", cfg, {"lookback": LOOKBACK, "horizon": HORIZON}, {}, figures)
    print(f"saved to {run_dir}")
