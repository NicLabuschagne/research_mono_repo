"""
robustness.py
-------------
Is the signal robust to HOW the divergence is measured? Development data, IN-SAMPLE diagnostic: it picks
nothing; the walk-forward stays the test. Saved to runs/<time>_robustness/.

    python -m strategies.wheat_corn_lag.robustness

Three ways to measure "wheat's own move vs corn" (all use past data only; all trade the same rule:
|corn| > 1 and |own| > 1 in the same direction, fade wheat, fill at the next bar's open, outright)
  window, sqrt scale   the current one: own move over L bars / (sigma_e x sqrt L), corn likewise
  window, empirical    same move, divided by the actual sd of L-bar moves over the last 20 sessions
                       (sqrt L assumes a random walk; 15-minute moves partly reverse, so it overstates the sd)
  EWMA                 no hard window: cumulative move minus its exponentially weighted average with
                       half-life L bars, divided by that gap's own past sd (like an OU spread z-score)

Plateau test: each is run at L (or half-life) = 8 .. 142 bars. A real effect is a broad plateau; a spike at
one L is noise.

Exits compared (time stop 1 session in all three)
  time only        no target, no stop: the raw edge
  divergence       target 0.5 x D, safety stop 2 x D   (D = how far wheat overshot corn)
  ATR-style        target 0.5 x ATR, safety stop 2 x ATR (ATR = sigma_w x sqrt L, wheat's typical move)
"""
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import save_run
from core_quant.theme import set_theme
from strategies.wheat_corn_lag.backtest import session_returns, simulate
from strategies.wheat_corn_lag.features import build_features, change_over
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load, load_bars

LOOKBACKS = [8, 16, 24, 32, 48, 71, 106, 142, 177, 213, 284]   # 2 h .. 4 sessions
EXITS = {
    "time only": {"target_share": 5.0, "stop_share": 5.0, "level_unit": "divergence"},
    "divergence 0.5 / 2": {"target_share": 0.5, "stop_share": 2.0, "level_unit": "divergence"},
    "ATR-style 0.5 / 2": {"target_share": 0.5, "stop_share": 2.0, "level_unit": "atr"},
}


def lagged_sd(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window // 2).std().shift(1)


def variants(bars, features, lookback: int, window: int) -> dict[str, pd.DataFrame]:
    """Feature frames that simulate() can read under key `lookback`: own_z, corn_z and div (D, log units)."""
    segment, beta = bars["segment"], features["beta"]
    move_w = change_over(bars["log_w"], segment, lookback)
    move_c = change_over(bars["log_c"], segment, lookback)
    own_move = move_w - beta * move_c
    root = np.sqrt(lookback)

    # EWMA: cumulative roll-free moves (returns across a cut count as 0) minus their EWMA
    cumulative_w = bars["ret_w"].fillna(0).cumsum()
    cumulative_c = bars["ret_c"].fillna(0).cumsum()
    gap_w = cumulative_w - cumulative_w.ewm(halflife=lookback).mean()
    gap_c = cumulative_c - cumulative_c.ewm(halflife=lookback).mean()
    gap_own = gap_w - beta * gap_c

    frames = {
        "window, sqrt scale": (own_move / (features["sigma_e"] * root), move_c / (features["sigma_c"] * root), own_move.abs()),
        "window, empirical": (own_move / lagged_sd(own_move, window), move_c / lagged_sd(move_c, window), own_move.abs()),
        "EWMA": (gap_own / lagged_sd(gap_own, window), gap_c / lagged_sd(gap_c, window), gap_own.abs()),
    }
    out = {}
    for name, (own_z, corn_z, divergence) in frames.items():
        out[name] = features[["sigma_w", "sigma_e", "beta"]].assign(
            **{f"own_z_{lookback}": own_z, f"corn_z_{lookback}": corn_z, f"div_{lookback}": divergence})
    return out


def measure(bars, frame, params, weeks: float) -> dict:
    trades = simulate(bars, frame, params)
    daily = session_returns(bars, trades)
    per_episode = trades.groupby("episode")["ret"].sum() * 1e4
    by_year = trades.groupby(trades["exit_time"].dt.year)["ret"].mean()
    return {"trades": len(trades), "per_week": len(trades) / weeks, "mean_bp": trades["ret"].mean() * 1e4,
            "t_episode": per_episode.mean() / (per_episode.std() / np.sqrt(len(per_episode))),
            "sharpe": daily["ret"].mean() / daily["ret"].std() * np.sqrt(252),
            "years_positive": int((by_year > 0).sum()), "hit_rate": (trades["ret"] > 0).mean()}


def plateau_chart(table: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(3, len(EXITS), figsize=(16, 10), sharex=True)
    for column, exit_name in enumerate(EXITS):
        block = table[table["exit"] == exit_name]
        for variant, rows in block.groupby("variant"):
            hours = rows["L"] / 4
            axes[0, column].plot(hours, rows["mean_bp"], marker="o", ms=3, label=variant)
            axes[1, column].plot(hours, rows["sharpe"], marker="o", ms=3, label=variant)
            axes[2, column].plot(hours, rows["per_week"], marker="o", ms=3, label=variant)
        axes[0, column].set_title(f"exit: {exit_name} (1-session time stop)", loc="left")
        for row in range(3):
            axes[row, column].axhline(0, color="grey", lw=0.8)
            axes[row, column].set_xscale("log")
            ticks = [2, 4, 8, 17.75, 35.5, 71]
            axes[row, column].set_xticks(ticks, [f"{t:g}" for t in ticks])
            axes[row, column].minorticks_off()
        axes[2, column].set_xlabel("lookback / EWMA half-life, hours (17.75 h = 1 session)")
    axes[0, 0].set_ylabel("bp per trade, after costs")
    axes[1, 0].set_ylabel("Sharpe (per session, annualised)")
    axes[2, 0].set_ylabel("trades per week")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Plateau test, development period IN-SAMPLE: a robust signal is flat across lookbacks, not a spike",
                    x=0.01, ha="left")
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    bars = load_bars(cfg)                                     # development data only
    features = build_features(bars, cfg.features)
    weeks = (bars.index[-1] - bars.index[0]).days / 7
    base = replace(cfg.strategy, scaling="sqrt", corn_sd=1.0, own_sd=1.0, max_hold_bars=71, instrument="outright", reentry=False)

    rows = []
    for lookback in LOOKBACKS:
        for variant, frame in variants(bars, features, lookback, cfg.features.stats_window_bars).items():
            for exit_name, levels in EXITS.items():
                params = replace(base, lookback=lookback, **levels)
                rows.append({"L": lookback, "variant": variant, "exit": exit_name, **measure(bars, frame, params, weeks)})
    table = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    for exit_name in EXITS:
        print(f"\n== exit: {exit_name}   (mean bp per trade after costs | Sharpe | trades/week)")
        block = table[table["exit"] == exit_name]
        summary = block.pivot(index="L", columns="variant", values=["mean_bp", "sharpe", "per_week"])
        print(summary.round(2).to_string())
    print("\n== years positive (of 6), divergence exit")
    print(table[table["exit"] == "divergence 0.5 / 2"].pivot(index="L", columns="variant", values="years_positive").to_string())

    run_dir = save_run(STRATEGY_DIR, "robustness", cfg, {"lookbacks": LOOKBACKS, "exits": list(EXITS)},
                       {"plateau": table}, {"plateau": plateau_chart(table)})
    print(f"\nsaved to {run_dir}")
