"""
strategy.py
-----------
Wheat order-flow imbalance. A worked EXAMPLE of the workflow: swap the signal for your own idea.

Data   : every data/raw/WHEAT_futures_tick_*.csv, joined. Download more days -> more data, no code change.
Sign   : tick rule. Up-tick = buyer-initiated (+1), down-tick = seller (-1), unchanged = same as the last trade.
Bars   : ticks -> `bar_seconds` bars with total volume and signed volume (buys minus sells).
Signal : imbalance = signed volume / volume over the last `window` bars, in [-1, 1].
         |imbalance| >= threshold -> position = direction * sign(imbalance)
         direction +1 follows the flow (momentum), -1 fades it (reversion).
Trade  : the position is held from the NEXT bar (shift(1)), so a signal never trades the bar it was computed on.
Costs  : cost_ticks x 0.25 c for every unit the position changes.
Returns: basis points of price, per bar.
"""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from core_quant.config import ROOT, GateConfig, SplitConfig, ValidationConfig, load_config

STRATEGY_DIR = Path(__file__).parent
TICK_SIZE = 0.25          # cents per bushel


@dataclass(frozen=True)
class StrategyConfig:
    bar_seconds: int = 10, 20, 40
    window: int = 5           # bars of flow in the imbalance
    threshold: float = 0.4    # |imbalance| needed to take a position
    direction: int = -1       # +1 follow the flow, -1 fade it
    cost_ticks: float = 0.5   # per unit of position change (0.5 = half the spread)


@dataclass(frozen=True)
class Config:
    data_glob: str
    split: SplitConfig
    strategy: StrategyConfig
    validation: ValidationConfig
    gates: GateConfig


def load(path: Path = STRATEGY_DIR / "config.yaml") -> Config:
    return load_config(path, Config)


def load_ticks(data_glob: str) -> pd.DataFrame:
    """All trade ticks, time-ordered, each signed by the tick rule."""
    files = sorted((ROOT / "data" / "raw").glob(data_glob))
    if not files:
        raise FileNotFoundError(f"no data/raw/{data_glob}")
    # shortcut: assumes the files do not overlap in time; de-duplicate here if your downloads do
    ticks = pd.concat(pd.read_parquet(file) for file in files)
    ticks = ticks[ticks["kind"] == "trade"].sort_values("time", kind="stable").set_index("time")
    ticks.index = ticks.index.tz_convert(None)
    ticks["side"] = np.sign(ticks["price"].diff()).replace(0, np.nan).ffill().fillna(0)
    return ticks


def make_bars(ticks: pd.DataFrame, bar_seconds: int) -> pd.DataFrame:
    """Time bars; bars with no trades are dropped (shortcut: positions carry across those gaps)."""
    rule = f"{bar_seconds}s"
    bars = pd.DataFrame({
        "close": ticks["price"].resample(rule).last(),
        "volume": ticks["size"].resample(rule).sum(),
        "signed_volume": (ticks["side"] * ticks["size"]).resample(rule).sum(),
    })
    return bars.dropna(subset=["close"])


def make_evaluator(cfg: Config):
    """Load ticks once; return evaluate(params, start, end) and the bar timestamps the folds are cut on.

    How to read this: a function that builds and returns another function.
      - The outer part (here) runs ONCE: load the data, do the slow param-free work.
      - The inner `evaluate` runs once PER PARAM COMBO, and still sees the data loaded above
        (Python keeps it alive for the inner function; this is called a closure).
    The validation calls evaluate(params, start, end) dozens of times without reloading anything.
    """
    ticks = load_ticks(cfg.data_glob)
    bars_by_size = {}

    def bars_for(bar_seconds: int) -> pd.DataFrame:
        if bar_seconds not in bars_by_size:
            bars_by_size[bar_seconds] = make_bars(ticks, bar_seconds)
        return bars_by_size[bar_seconds]

    def evaluate(params: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        config = replace(cfg.strategy, **params)
        bars = bars_for(config.bar_seconds)
        # built on ALL bars so the rolling window is warm at `start`, then cut to [start, end)
        imbalance = (bars["signed_volume"].rolling(config.window).sum()
                     / bars["volume"].rolling(config.window).sum())
        signal = np.sign(imbalance).where(imbalance.abs() >= config.threshold, 0.0) * config.direction
        held = signal.shift(1).fillna(0.0)
        turnover = held.diff().abs().fillna(held.abs())
        cost = turnover * config.cost_ticks * TICK_SIZE / bars["close"].shift(1)
        ret_bps = (held * bars["close"].pct_change().fillna(0.0) - cost.fillna(0.0)) * 1e4
        out = pd.DataFrame({"ret": ret_bps, "trades": (turnover > 0).astype(int)})
        return out[(out.index >= start) & (out.index < end)]

    return evaluate, bars_for(cfg.strategy.bar_seconds).index
