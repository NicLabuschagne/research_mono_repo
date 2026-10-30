"""
layer2_barrier_preview.py
-------------------------
Does a triple barrier capture moves the SIGNALS create, or moves beans make anyway?

Signals: A / B / C from layer1_signal_days (prior-close crush gap extremes). Outright beans, entry at the next
bar's open, direction = sell when beans rich, buy when cheap. Sessions are ignored: a position may be held across
breaks (overnight gaps included), up to 12 h of trading bars.
Baseline: for every signal, 10 random bars from the same session (Asia / Europe / US) and half, with a random
direction (fixed seed). Same entry, same barriers. What the signals add = signals minus baseline.

    1. move size: typical absolute move at 1 h / 4 h / 12 h, signals vs baseline
    2. excursions over 12 h: best move for the trade (favourable) and worst against it (adverse)
    3. triple barrier on bar closes: target T and stop S in ZS ticks (10 / 20 / 40), time limit 4 h or 12 h;
       exit at the close that crosses a barrier (so the realised move can overshoot it); 1 tick cost
    4. the same with barriers in ATR multiples (0.5 / 1 / 2 / 3 x ATR), time limit 4 h / 12 h / 24 h.
       ATR = average over the last 14 hours of the 1-hour range (highest - lowest 5-minute close in each hour),
       known at the signal bar. The bars have no intrabar high / low, so it runs a little below a true ATR(14).
Development only; the holdout stays locked.

    python -m strategies.bean_divergence.layer2_barrier_preview
"""
from itertools import product

import numpy as np
import pandas as pd

from core_quant.config import ROOT
from strategies.bean_divergence.failure_study import session_label
from strategies.bean_divergence.feature_study import STRATEGY_DIR, TICK_CENTS
from strategies.crush_strategy.data_ingest import build_bars

MAX_BARS = 288                      # 24 h of 5-minute trading bars
SIZE_HORIZONS = {"1h": 12, "4h": 48, "12h": 144}
TARGETS, STOPS, TIME_LIMITS = [10, 20, 40], [10, 20, 40], {"4h": 48, "12h": 144}
ATR_MULTIPLES, ATR_TIME_LIMITS = [0.5, 1.0, 2.0, 3.0], {"4h": 48, "12h": 144, "24h": 288}
RANDOM_PER_SIGNAL = 10
COST_TICKS = 1.0


def path_in_ticks(log_open: np.ndarray, log_close: np.ndarray, ticks: np.ndarray, s: int, direction: int) -> np.ndarray:
    """Move of a trade entered at bar s+1's open, at each later bar close (ticks, + = in the trade's favour)."""
    e = s + 1
    end = min(e + MAX_BARS, len(log_close))
    return direction * (log_close[e:end] - log_open[e]) * ticks[s]


def barrier_result(path: np.ndarray, target: float, stop: float, limit: int) -> float:
    """Exit at the first close at or beyond the target or the stop, else at the time limit. Net of costs."""
    for k in range(min(limit, len(path))):
        if path[k] >= target or path[k] <= -stop:
            return path[k] - COST_TICKS
    return path[min(limit, len(path)) - 1] - COST_TICKS


if __name__ == "__main__":
    bars = build_bars(ROOT / "data/raw/latest_df_backup.parquet", 5)
    bars = bars[bars["trading_date"] < pd.Timestamp("2024-12-01")]          # the same bars layer1_signal_days used
    log_open, log_close = np.log(bars["ZS_adj_open"].values), np.log(bars["ZS_adj_close"].values)
    ticks = (bars["ZS_adj_close"] / TICK_CENTS).values
    # ATR in ticks: the 1-hour range of closes (12 bars), averaged over the last 14 hours (168 bars), up to bar t
    close_ticks = pd.Series(bars["ZS_adj_close"].values / TICK_CENTS)
    hourly_range = close_ticks.rolling(12).max() - close_ticks.rolling(12).min()
    atr = hourly_range.rolling(168).mean().values
    session = session_label(bars.index, bars["session_type"])
    half = np.where(bars["trading_date"] < pd.Timestamp("2023-01-01"), "2021-22", "2023-Nov24")
    in_dev = (bars["trading_date"] >= pd.Timestamp("2021-01-01")).values

    signals = pd.read_csv(STRATEGY_DIR / "output" / "layer1_signal_days_events.csv")
    signals = signals[signals["bar"] + 1 + MAX_BARS < len(bars)]
    assert (session[signals["bar"].values] == signals["session"].values).all()   # same bar numbering as before

    # matched random entries: same session label and half as each signal, random direction
    rng = np.random.default_rng(42)
    pool = {}
    for key in set(zip(signals["session"], np.where(pd.to_datetime(signals["trading_date"]) < "2023-01-01", "2021-22", "2023-Nov24"))):
        pool[key] = np.flatnonzero((session == key[0]) & (half == key[1]) & in_dev & (np.arange(len(bars)) + 1 + MAX_BARS < len(bars)))
    trades = []
    for ev in signals.itertuples():
        hf = "2021-22" if pd.Timestamp(ev.trading_date) < pd.Timestamp("2023-01-01") else "2023-Nov24"
        trades.append({"kind": "signal", "signal": ev.signal, "half": hf, "bar": ev.bar, "direction": -ev.side})
        for s in rng.choice(pool[(ev.session, hf)], RANDOM_PER_SIGNAL):
            trades.append({"kind": "random", "signal": ev.signal, "half": hf, "bar": int(s), "direction": int(rng.choice([-1, 1]))})
    trades = pd.DataFrame(trades)

    rows = []
    for t in trades.itertuples():
        path = path_in_ticks(log_open, log_close, ticks, t.bar, t.direction)
        row = {"kind": t.kind, "signal": t.signal, "half": t.half,
               "favourable_12h": max(0.0, path.max()), "adverse_12h": -min(0.0, path.min())}
        for label, h in SIZE_HORIZONS.items():
            row[f"move_{label}"] = path[h - 1]
        for target, stop, (limit_name, limit) in product(TARGETS, STOPS, TIME_LIMITS.items()):
            row[f"T{target}_S{stop}_{limit_name}"] = barrier_result(path, target, stop, limit)
        row["atr_ticks"] = atr[t.bar]
        for k_target, k_stop, (limit_name, limit) in product(ATR_MULTIPLES, ATR_MULTIPLES, ATR_TIME_LIMITS.items()):
            row[f"ATR_T{k_target:g}_S{k_stop:g}_{limit_name}"] = barrier_result(path, k_target * atr[t.bar], k_stop * atr[t.bar], limit)
        rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(STRATEGY_DIR / "output" / "layer2_barrier_preview.csv", index=False)

    pd.set_option("display.width", 250)
    size = results.groupby("kind")[[f"move_{h}" for h in SIZE_HORIZONS]].agg(lambda x: x.abs().median()).round(1)
    print("typical absolute outright move (ticks), median:\n", size.to_string())
    mean_move = results.groupby(["kind", "half"])[[f"move_{h}" for h in SIZE_HORIZONS]].mean().round(2)
    print("\naverage move in the trade's direction (ticks):\n", mean_move.to_string())
    exc = results.groupby(["kind", "half"])[["favourable_12h", "adverse_12h"]].median().round(1)
    print("\nexcursions over 12 h (median ticks):\n", exc.to_string())
    print("\nATR at the signal bar (ticks): median", round(results["atr_ticks"].median(), 1),
          "| 10th-90th pct", results["atr_ticks"].quantile([0.1, 0.9]).round(1).tolist())
    barrier_cols = [c for c in results.columns if c.startswith("T") or c.startswith("ATR_T")]
    grid = []
    for col in barrier_cols:
        for hf in ["2021-22", "2023-Nov24"]:
            sig = results[(results["kind"] == "signal") & (results["half"] == hf)][col]
            rnd = results[(results["kind"] == "random") & (results["half"] == hf)][col]
            grid.append({"barrier": col, "half": hf, "signal_net": sig.mean(), "signal_se": sig.sem(),
                         "random_net": rnd.mean(), "signal_minus_random": sig.mean() - rnd.mean()})
    grid = pd.DataFrame(grid)
    grid.to_csv(STRATEGY_DIR / "output" / "layer2_barrier_grid.csv", index=False)
    print("\ntick barriers, net ticks per trade (1 tick cost):\n",
          grid[~grid["barrier"].str.startswith("ATR")].pivot(index="barrier", columns="half").round(2).to_string())
