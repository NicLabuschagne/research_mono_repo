"""
horizon_sweep.py
----------------
Enter on the signal, exit exactly H bars later: no target, no stop. One curve per instrument and corn
condition, so you can see how the result builds with time. Development data, IN-SAMPLE diagnostic (the
walk-forward is the test). Uses backtest.simulate, so fills (next bar's open), costs and roll cuts are the
real ones. Saved to runs/<time>_horizon_sweep/.

    python -m strategies.wheat_corn_lag.horizon_sweep

The t-stat is per EPISODE (all trades of one divergence summed), because trades inside one episode are
not independent.
"""
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import save_run
from core_quant.theme import set_theme
from strategies.wheat_corn_lag.backtest import simulate
from strategies.wheat_corn_lag.features import build_features
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load, load_bars

HORIZONS = [1, 2, 4, 8, 16, 24, 32, 48, 71, 106, 142]
NO_LEVEL = 5.0                     # target and stop this far away are never hit


def sweep(bars, features, base) -> pd.DataFrame:
    weeks = (bars.index[-1] - bars.index[0]).days / 7
    rows = []
    for instrument in ["outright", "spread"]:
        for corn_sd in [0.0, 1.0]:
            for horizon in HORIZONS:
                params = replace(base, instrument=instrument, corn_sd=corn_sd, max_hold_bars=horizon,
                                 target_share=NO_LEVEL, stop_share=NO_LEVEL, reentry=False)
                trades = simulate(bars, features, params)
                per_episode = trades.groupby("episode")["ret"].sum() * 1e4
                rows.append({"instrument": instrument, "corn": "corn > 1 sd" if corn_sd else "no corn condition",
                             "H_bars": horizon, "trades": len(trades), "per_week": len(trades) / weeks,
                             "mean_bp_per_trade": trades["ret"].mean() * 1e4,
                             "episodes": len(per_episode), "mean_bp_per_episode": per_episode.mean(),
                             "t_episode": per_episode.mean() / (per_episode.std() / np.sqrt(len(per_episode))),
                             "hit_rate": (trades["ret"] > 0).mean()})
    return pd.DataFrame(rows)


def sweep_chart(table: pd.DataFrame, lookback: int) -> plt.Figure:
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 4.8))
    for (instrument, corn), block in table.groupby(["instrument", "corn"]):
        style = "-" if instrument == "outright" else "--"
        label = f"{instrument}, {corn}"
        left.plot(block["H_bars"] / 4, block["mean_bp_per_trade"], ls=style, marker="o", ms=3, label=label)
        right.plot(block["H_bars"] / 4, block["t_episode"], ls=style, marker="o", ms=3, label=label)
    left.axhline(0, color="grey", lw=0.8)
    left.set_xscale("log")
    left.set_xlabel("H, hours held (log scale; 17.75 h = 1 session)")
    left.set_ylabel("bp per trade, after costs")
    left.set_title(f"Exit exactly H bars after entry (lookback {lookback} bars, wheat's own move > 1 sd)", loc="left")
    left.legend(fontsize=8)
    right.axhline(0, color="grey", lw=0.8)
    for level in (2, -2):
        right.axhline(level, color="grey", ls=":", lw=0.9)
    right.set_xscale("log")
    right.set_xlabel("H, hours held (log scale)")
    right.set_ylabel("t-stat per episode")
    right.set_title("Is it distinguishable from zero? (dotted = +-2)", loc="left")
    for axis in (left, right):
        ticks = [0.25, 1, 4, 8, 17.75, 35.5]
        axis.set_xticks(ticks, [f"{t:g}" for t in ticks])
        axis.minorticks_off()
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    bars = load_bars(cfg)                                     # development data only
    features = build_features(bars, cfg.features)
    table = sweep(bars, features, cfg.strategy)
    pd.set_option("display.width", 200)
    print(table.round(2).to_string(index=False))
    run_dir = save_run(STRATEGY_DIR, "horizon_sweep", cfg, {"horizons": HORIZONS}, {"sweep": table},
                       {"horizon_sweep": sweep_chart(table, cfg.strategy.lookback)})
    print(f"\nsaved to {run_dir}")
