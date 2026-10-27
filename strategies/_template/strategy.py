"""
strategy.py
-----------
<one paragraph: signal, entry, exit, sizing, costs>
"""
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from core_quant.config import ROOT, GateConfig, SplitConfig, ValidationConfig, load_config

STRATEGY_DIR = Path(__file__).parent


@dataclass(frozen=True)
class StrategyConfig:
    lookback: int = 20
    threshold: float = 2.0
    cost_bps: float = 1.0


@dataclass(frozen=True)
class Config:
    data_path: str
    split: SplitConfig
    strategy: StrategyConfig
    validation: ValidationConfig
    gates: GateConfig


def load(path: Path = STRATEGY_DIR / "config.yaml") -> Config:
    return load_config(path, Config)


def make_evaluator(cfg: Config):
    """Load data once; return evaluate(params, start, end) and every trading date.

    How to read this: a function that builds and returns another function.
      - The outer part (here) runs ONCE: load the data, do the slow param-free work.
      - The inner `evaluate` runs once PER PARAM COMBO, and still sees the data loaded above
        (Python keeps it alive for the inner function; this is called a closure).
    The validation calls evaluate(params, start, end) dozens of times without reloading anything.
    """
    data = pd.read_parquet(ROOT / cfg.data_path)
    trading_dates = pd.DatetimeIndex(data.index.normalize().unique())

    def evaluate(params: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        config = replace(cfg.strategy, **params)
        # build the signal on ALL data (so lookbacks are warm), then cut to [start, end)
        raise NotImplementedError("return DataFrame(index=date, columns=[ret, trades])")

    return evaluate, trading_dates
