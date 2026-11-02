"""
fair_value.py
-------------
Layer 1: the Kalman crush fair value of soybeans and the gap to it.

    fair value (log) = alpha + beta_meal x log ZM + beta_oil x log ZL     (Kalman filter, strategies/crush_strategy)
    gap              = (log ZS - fair value) x ZS price / 0.25            ZS ticks (1 tick = 0.25 c/bu, 4 ticks = 1 c)
    live             : hedge from the bar before = the filter's one-step prediction error. The locked outright base.
    prior close      : hedge frozen at the previous session close. Trends within a session, jumps at each re-freeze.
    z                : gap / its own recent size (exponential RMS, 4-hour half-life, bars BEFORE t only). Not the
                       filter's own variance, which barely moves.
Only bars before the holdout start are ever loaded.

    python -m strategies.bean_divergence.fair_value          # self-checks + how persistent each gap is
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core_quant.config import ROOT, SplitConfig, load_config
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.kalman_filter import freeze_at_previous_session_close, run_kalman_filter

STRATEGY_DIR = Path(__file__).parent
TICK_CENTS = 0.25
LEGS = ["ZS", "ZM", "ZL"]


@dataclass(frozen=True)
class ResearchConfig:
    bar_minutes: int = 5
    z_half_life_bars: int = 48
    trigger_z: float = 1.645
    rearm_z: float = 0.6745
    cost_ticks: float = 1.0


@dataclass(frozen=True)
class BeanConfig:
    data_path: str
    split: SplitConfig
    research: ResearchConfig


def load(path: Path = STRATEGY_DIR / "config.yaml") -> BeanConfig:
    return load_config(path, BeanConfig)


def load_bars(cfg: BeanConfig) -> pd.DataFrame:
    """5-minute ZS / ZM / ZL bars up to the holdout start (the holdout stays locked)."""
    bars = build_bars(ROOT / cfg.data_path, cfg.research.bar_minutes)
    return bars[bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start)]


def session_label(index: pd.DatetimeIndex, session_type: pd.Series) -> np.ndarray:
    """asia 19:00-02:00 CT, europe 02:00-08:00 CT (overnight session), us = the day session."""
    hour = index.tz_convert("America/Chicago").hour
    return np.where(session_type == "day", "us", np.where((hour >= 2) & (hour < 8), "europe", "asia"))


def half_label(trading_date: pd.Series) -> np.ndarray:
    """The two development halves every result is checked in."""
    return np.where(trading_date < pd.Timestamp("2023-01-01"), "2021-22", "2023-Nov24")


def crush_hedge(bars: pd.DataFrame, live: bool, bar_minutes: int = 5) -> pd.DataFrame:
    """alpha / beta_meal / beta_oil per bar: live (from the bar before) or frozen at the previous session close."""
    states = run_kalman_filter(bars, bar_minutes)
    return states.shift(1) if live else freeze_at_previous_session_close(bars, states)


def crush_gap(bars: pd.DataFrame, hedge: pd.DataFrame) -> pd.Series:
    """Beans minus crush fair value, in ZS ticks (+ = beans rich)."""
    log_c = {leg: np.log(bars[f"{leg}_adj_close"]) for leg in LEGS}
    fair = hedge["alpha"] + hedge["beta_meal"] * log_c["ZM"] + hedge["beta_oil"] * log_c["ZL"]
    return (log_c["ZS"] - fair) * bars["ZS_adj_close"] / TICK_CENTS


def adaptive_scale(values: pd.Series, half_life_bars: int) -> pd.Series:
    """Exponential RMS of `values` using bars BEFORE t only (shifted one bar), so bar t is not in its own scale."""
    return np.sqrt((values ** 2).ewm(halflife=half_life_bars, min_periods=half_life_bars).mean()).shift(1)


if __name__ == "__main__":
    assert np.isclose(np.log(1201 / 1200) * 1200 / TICK_CENTS, 4.0, rtol=1e-3)     # a 1 cent move at 1200 c = 4 ticks
    cfg = load()
    bars = load_bars(cfg)
    in_dev = bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)
    for name, live in [("prior close", False), ("live", True)]:
        gap = crush_gap(bars, crush_hedge(bars, live, cfg.research.bar_minutes))
        if live:                                     # self-check: the scale at bar i uses only bars before i
            i = len(gap) // 2
            altered = gap.copy()
            altered.iloc[i:] *= 10
            assert np.isclose(adaptive_scale(gap, 48).iloc[i], adaptive_scale(altered, 48).iloc[i])
            assert not np.isclose(adaptive_scale(gap, 48).iloc[i + 1], adaptive_scale(altered, 48).iloc[i + 1])
        g = gap[in_dev]
        persistence = {f"{k * 5} min": round(g.autocorr(k), 2) for k in [1, 6, 12, 48]}
        print(f"{name:12s} typical size (std) {g.std():5.1f} ticks | correlation with itself after {persistence}")
