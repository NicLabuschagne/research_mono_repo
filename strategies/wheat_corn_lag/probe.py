"""
probe.py
--------
Step 5b: probe the lead from ic.py: "wheat's own move (the part corn does not explain) is given back".
Development data only. Results saved to runs/<time>_probe/.

    python -m strategies.wheat_corn_lag.probe

Counting EPISODES, not bars: bars inside one move are near copies of each other, so 300 bars may be
only 30 independent events. An episode starts on the first bar where |own_z| crosses `ENTRY_SD`; after an
episode, no new one IN THE SAME CONTEXT is counted for h bars, so episodes in one row never share a window.
Trade measured: fade wheat's own move, i.e. forward wheat move x -sign(own_z), in bp, before ~6 bp costs.

Splits (all fixed before running):
  corn context  with corn     corn moved > 1 sd the SAME way as wheat's own move (wheat overshot corn)
                wheat alone   |corn| < 0.5 sd (wheat moved by itself)
                against corn  corn moved > 1 sd the OTHER way (wheat lagged or went the wrong way)
                other         everything in between
  direction     own move up / own move down
  session       day (08:30-13:20 CT) / overnight
  year
Cells: (L=8, h=16) and (L=32, h=71), the two heatmap cells fixed in ic.py, plus (L=71, h=284) where the
regression found its strongest multi-day result.
"""
import numpy as np
import pandas as pd

from core_quant.config import save_run
from strategies.wheat_corn_lag.features import build_features, build_targets
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load, load_bars

CELLS = [(8, 16), (32, 71), (71, 284)]
ENTRY_SD = 1.0


def episodes(frame: pd.DataFrame, lookback: int, horizon: int) -> pd.DataFrame:
    """First bar of every |own_z| > ENTRY_SD move, spaced at least `horizon` bars apart."""
    own = frame[f"own_z_{lookback}"]
    corn = frame[f"corn_z_{lookback}"]
    forward = frame[f"fwd_{horizon}"]
    inside = own.abs() > ENTRY_SD
    starts = np.flatnonzero((inside & ~inside.shift(1, fill_value=False)).values)

    # spacing is applied WITHIN each context: spacing across all contexts (the first version) skipped
    # overshoots that came soon after any other crossing and inflated the overshoot result (+57 bp, n=111)
    rows, next_allowed = [], {}
    for position in starts:
        if not np.isfinite(forward.iloc[position]):
            continue
        own_value, corn_value = own.iloc[position], corn.iloc[position]
        if abs(corn_value) < 0.5:
            context = "wheat alone"
        elif abs(corn_value) > 1 and np.sign(corn_value) == np.sign(own_value):
            context = "with corn (overshoot)"
        elif abs(corn_value) > 1:
            context = "against corn (lag)"
        else:
            context = "other"
        if position < next_allowed.get(context, 0):
            continue
        next_allowed[context] = position + horizon
        rows.append({"time": frame.index[position], "own_z": own_value, "corn_z": corn_value, "context": context,
                     "fade_bp": -np.sign(own_value) * forward.iloc[position] * 1e4})
    table = pd.DataFrame(rows).set_index("time")
    central = table.index.tz_convert("America/Chicago")
    minutes = central.hour * 60 + central.minute
    table["session"] = np.where((minutes > 8 * 60 + 30) & (minutes <= 13 * 60 + 30), "day", "overnight")
    table["direction"] = np.where(table["own_z"] > 0, "own move up", "own move down")
    table["year"] = table.index.year
    return table


def literal_episodes(frame: pd.DataFrame, lookback: int, horizon: int) -> pd.DataFrame:
    """The trader's own two cases, traded in the trader's direction, as non-overlapping episodes.

    Case 1   (f0 = +-1)  W-1.5C up and 1.5W-C flat/down -> buy wheat (and the mirror: sell)
    Case 1.1 (f0 = +-2)  both spreads down while wheat is flat/up -> sell wheat (and the mirror: buy)
    Each episode is also tagged by where it sits in the (corn, wheat's own move) picture.
    """
    rule = frame[f"f0_{lookback}"]
    forward = frame[f"fwd_{horizon}"]
    own, corn = frame[f"own_z_{lookback}"], frame[f"corn_z_{lookback}"]
    starts = np.flatnonzero((rule.notna() & (rule != 0) & (rule != rule.shift())).values)
    rows, next_allowed = [], 0
    for position in starts:
        if position < next_allowed or not np.isfinite(forward.iloc[position]):
            continue
        next_allowed = position + horizon
        signal = rule.iloc[position]
        same_way = np.sign(own.iloc[position]) == np.sign(corn.iloc[position])
        rows.append({"time": frame.index[position], "case": "1.1 (both spreads vs wheat)" if abs(signal) == 2 else "1 (spreads diverge)",
                     "wheat_vs_corn": "wheat beyond its usual ratio" if same_way else "wheat short of its usual ratio",
                     "trade_bp": np.sign(signal) * forward.iloc[position] * 1e4})
    return pd.DataFrame(rows).set_index("time")


def summarise(table: pd.DataFrame, by: str, column: str = "fade_bp") -> pd.DataFrame:
    grouped = table.groupby(by)[column]
    out = pd.DataFrame({"episodes": grouped.size(), "mean_bp": grouped.mean(),
                        "t": grouped.mean() / (grouped.std() / np.sqrt(grouped.size())),
                        "hit_rate": grouped.apply(lambda x: (x > 0).mean())})
    total = table[column]
    out.loc["ALL"] = [len(total), total.mean(), total.mean() / (total.std() / np.sqrt(len(total))), (total > 0).mean()]
    return out


if __name__ == "__main__":
    cfg = load()
    bars = load_bars(cfg)                                     # development data only
    frame = build_features(bars, cfg.features).join(build_targets(bars, cfg.features))
    pd.set_option("display.width", 200)

    tables = {}
    for lookback, horizon in CELLS:
        table = episodes(frame, lookback, horizon)
        tables[f"episodes_L{lookback}_h{horizon}"] = table
        print(f"\n######## L={lookback} bars, h={horizon} bars: fade wheat's own move after |own_z| > {ENTRY_SD}")
        for by in ["context", "direction", "session", "year"]:
            print(f"\n-- by {by}")
            print(summarise(table, by).round(2).to_string())
        print("\n-- context x direction (mean bp / episodes)")
        cross = table.groupby(["context", "direction"])["fade_bp"].agg(["mean", "size"]).unstack()
        print(cross.round(1).to_string())

        literal = literal_episodes(frame, lookback, horizon)
        tables[f"literal_L{lookback}_h{horizon}"] = literal
        print(f"\n-- the trader's cases, traded in the trader's direction (bp per episode)")
        print(summarise(literal, ["case", "wheat_vs_corn"], "trade_bp").round(2).to_string())

    run_dir = save_run(STRATEGY_DIR, "probe", cfg, {"cells": CELLS, "entry_sd": ENTRY_SD}, tables)
    print(f"\nsaved to {run_dir}")
