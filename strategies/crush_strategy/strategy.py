"""
strategy.py
-----------
Overnight crush-residual reversion with triple-barrier exits (final specification).

Signal : evening session, Kalman residual z-score (features.py) crosses |z| >= threshold.
Trade rules
  Entry  : next bar's open after the signal bar closes; hedge (betas) frozen at the signal bar.
           No entries with less than min_hours_left (3 h) to the session close: with the
           session exit they get too little time to revert (the strongest trade-level feature).
  Exits  : first barrier touched, checked on bar closes
             profit : favourable move >= 1 sigma   (about half the gap to the mean;
                      part of every dislocation is permanent, so waiting for the full
                      mean often fails)
             stop   : adverse move     >= 4 sigma  (a disaster stop: tighter stops exit at the
                      largest dislocation, exactly when reversion is most likely)
             time   : the close of the entry session (evening: 07:45 CT), capped at 12 h.
                      Never carried into the day session or the next evening: ~60% of trades
                      used to carry, and those lost money on average.
           sigma = residual std at the signal (z_std), so barriers scale with the regime.
  Sizing : 1 bushel of beans against beta-weighted meal and oil. The residual change is
           a log return on the beans notional; x bean price gives cents per bushel.
  Costs  : 1 tick per leg, round trip (= 0.58 c/bu on the spread).
  One trade at a time; every signal that passes the entry cutoff is taken (no meta-model).

Reporting: returns as % of bean notional (independent of contract count), USD per 5,000 bu.
All params live in config.yaml; validate.py tunes them, holdout.py tests them once.
"""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from core_quant.config import ROOT, GateConfig, SplitConfig, ValidationConfig, load_config
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.kalman_filter import add_kalman_residual
from strategies.crush_strategy.features import add_session_zscore, detect_signals

TICK_COST_CENTS = 0.58          # ZS 0.25 + ZM 0.22 + ZL 0.11 c/bu, one tick each, round trip
BUSHELS_PER_CONTRACT = 5000
STRATEGY_DIR = Path(__file__).parent


@dataclass(frozen=True)
class StrategyConfig:
    threshold: float = 1.5      # entry |z|
    profit_sigma: float = 1.0
    stop_sigma: float = 4.0
    holding_hours: float = 12    # clock-time cap; the session close usually comes first
    bar_minutes: int = 5
    cost_ticks: float = 1.0      # per leg, round trip
    min_hours_left: float = 3.0  # skip signals with less time than this to the session close


@dataclass(frozen=True)
class CrushConfig:
    data_path: str
    split: SplitConfig
    strategy: StrategyConfig
    validation: ValidationConfig
    gates: GateConfig


def load(path: Path = STRATEGY_DIR / "config.yaml") -> CrushConfig:
    return load_config(path, CrushConfig)


def simulate_trades(bars: pd.DataFrame, signals: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """Walk each signal forward with the hedge frozen at the signal bar."""
    log_open = {leg: np.log(bars[f"{leg}_adj_open"].values) for leg in ["ZS", "ZM", "ZL"]}
    log_close = {leg: np.log(bars[f"{leg}_adj_close"].values) for leg in ["ZS", "ZM", "ZL"]}
    session = bars["session_id"].values
    bar_time = bars.index.values
    session_last_bar = pd.Series(np.arange(len(bars))).groupby(session).transform("max").values

    trades, next_free_bar = [], -1
    for signal in signals.itertuples():
        signal_bar, entry_bar = signal.signal_position, signal.signal_position + 1
        if entry_bar <= next_free_bar:                     # still in the previous trade
            continue
        if entry_bar >= len(bars) or session[entry_bar] != session[signal_bar]:
            continue                                       # no fill in the same session
        # session close from the data: matches the exchange schedule, incl. known early closes
        hours_left = (bar_time[session_last_bar[signal_bar]] - bar_time[signal_bar]) / np.timedelta64(1, "h")
        if hours_left < config.min_hours_left:
            continue                                       # too late in the session to revert
        # clock-time cap, never past the session close
        cap = bar_time[entry_bar] + np.timedelta64(int(config.holding_hours * 60), "m")
        last_bar = min(int(np.searchsorted(bar_time, cap, side="right")) - 1, session_last_bar[entry_bar])

        alpha = bars["frozen_alpha"].values[signal_bar]
        beta_meal = bars["frozen_beta_meal"].values[signal_bar]
        beta_oil = bars["frozen_beta_oil"].values[signal_bar]

        def hedged_residual(prices, index):
            return prices["ZS"][index] - (alpha + beta_meal * prices["ZM"][index] + beta_oil * prices["ZL"][index])

        entry_residual = hedged_residual(log_open, entry_bar)
        path = np.arange(entry_bar, last_bar + 1)
        favourable = signal.direction * (hedged_residual(log_close, path) - entry_residual)

        sigma = bars["z_std"].values[signal_bar]
        profit_hits = np.flatnonzero(favourable >= config.profit_sigma * sigma)
        stop_hits = np.flatnonzero(favourable <= -config.stop_sigma * sigma)
        first_profit = profit_hits[0] if len(profit_hits) else np.inf
        first_stop = stop_hits[0] if len(stop_hits) else np.inf
        exit_step = int(min(first_profit, first_stop, len(path) - 1))
        exit_reason = "profit" if exit_step == first_profit else "stop" if exit_step == first_stop else "time"

        bean_price = bars["ZS_adj_close"].values[signal_bar]
        trades.append(dict(
            signal_time=bars.index[signal_bar], exit_time=bars.index[path[exit_step]],
            signal_date=bars["trading_date"].values[signal_bar],
            exit_date=bars["trading_date"].values[path[exit_step]],
            entry_bar=entry_bar, exit_bar=path[exit_step],
            side="short beans / long products" if signal.direction < 0 else "long beans / short products",
            entry_z=signal.entry_z, beta_meal=beta_meal, beta_oil=beta_oil,
            sigma_cents=sigma * bean_price, bean_price_cents=bean_price,
            exit_reason=exit_reason, holding_minutes=(exit_step + 1) * config.bar_minutes,
            gross_cents=favourable[exit_step] * bean_price,
        ))
        next_free_bar = path[exit_step]
    return pd.DataFrame(trades)


def daily_returns(trades: pd.DataFrame, trading_dates: pd.DatetimeIndex, cost_ticks: float) -> pd.DataFrame:
    """Net return (% of bean notional) and trades closed, per day; every trading day included (zero when flat)."""
    if trades.empty:
        return pd.DataFrame({"ret": 0.0, "trades": 0}, index=trading_dates)
    net_cents = trades["gross_cents"] - cost_ticks * TICK_COST_CENTS
    return_pct = net_cents / trades["bean_price_cents"] * 100
    exit_dates = pd.to_datetime(trades["exit_date"])
    # include exit dates past the sample end, so a trade that closes after the cut-off still counts
    all_days = trading_dates.union(pd.DatetimeIndex(exit_dates.unique()))
    return pd.DataFrame({"ret": return_pct.groupby(exit_dates).sum().reindex(all_days, fill_value=0.0),
                         "trades": exit_dates.value_counts().reindex(all_days, fill_value=0)})


def trade_table(trades: pd.DataFrame, cost_ticks: float) -> pd.DataFrame:
    """Per-trade rows in the tear sheet's format: ret in % of bean notional, pnl in c/bu after costs."""
    net_cents = trades["gross_cents"] - cost_ticks * TICK_COST_CENTS
    table = pd.DataFrame({"exit_date": pd.to_datetime(trades["exit_date"]),
                          "ret": net_cents / trades["bean_price_cents"] * 100, "pnl": net_cents,
                          "holding_minutes": trades["holding_minutes"], "exit_reason": trades["exit_reason"]})
    table.attrs["pnl_unit"] = "c_per_bu"
    return table.reset_index(drop=True)


def performance(trades: pd.DataFrame, trading_dates: pd.DatetimeIndex, cost_ticks: float) -> tuple[dict, pd.Series]:
    """Net results after costs; daily series over every trading day (zero when flat) for Sharpe and drawdown."""
    net_cents = trades["gross_cents"] - cost_ticks * TICK_COST_CENTS
    daily = daily_returns(trades, trading_dates, cost_ticks)["ret"]
    equity = daily.cumsum()
    summary = dict(
        trades=len(trades), trades_per_year=len(trades) / (len(trading_dates) / 252),
        hit_rate=(net_cents > 0).mean(), mean_net_cents=net_cents.mean(),
        total_return_pct=equity.iloc[-1], max_drawdown_pct=(equity - equity.cummax()).min(),
        sharpe=daily.mean() / daily.std() * np.sqrt(252),
        total_usd=net_cents.sum() / 100 * BUSHELS_PER_CONTRACT,
    )
    return summary, equity


def prepare_bars(cfg: CrushConfig) -> pd.DataFrame:
    """Ingest -> Kalman residual -> z-score. Param-free, so it runs once per validation."""
    bars = build_bars(ROOT / cfg.data_path, cfg.strategy.bar_minutes)
    bars = add_kalman_residual(bars, cfg.strategy.bar_minutes)
    return add_session_zscore(bars)


def make_evaluator(cfg: CrushConfig):
    """evaluate(params, start, end) for core_quant.walk_forward, plus every trading date.

    How to read this: a function that builds and returns another function.
      - The outer part (here) runs ONCE: load the data, do the slow param-free work.
      - The inner `evaluate` runs once PER PARAM COMBO, and still sees the data loaded above
        (Python keeps it alive for the inner function; this is called a closure).
    The validation calls evaluate(params, start, end) dozens of times without reloading anything.
    """
    if "bar_minutes" in cfg.validation.grid:
        raise ValueError("bar_minutes cannot be in the grid: bars are built once")
    bars = prepare_bars(cfg)
    trading_dates = pd.DatetimeIndex(sorted(bars["trading_date"].unique()))
    signals_by_threshold = {}

    def evaluate(params: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        config = replace(cfg.strategy, **params)
        if config.threshold not in signals_by_threshold:
            signals_by_threshold[config.threshold] = detect_signals(bars, config.threshold)
        trades = simulate_trades(bars, signals_by_threshold[config.threshold], config)
        signal_dates = pd.to_datetime(trades["signal_date"])
        trades = trades[(signal_dates >= start) & (signal_dates < end)]
        dates = trading_dates[(trading_dates >= start) & (trading_dates < end)]
        daily = daily_returns(trades, dates, config.cost_ticks)
        daily.attrs["trades"] = trade_table(trades, config.cost_ticks)      # for the tear sheet
        return daily

    return evaluate, trading_dates


def run(cfg: CrushConfig, start: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline with the selected params: bars and every trade from `start`, labelled development/holdout."""
    start = start or pd.Timestamp(cfg.split.development_start)
    bars = prepare_bars(cfg)
    trades = simulate_trades(bars, detect_signals(bars, cfg.strategy.threshold), cfg.strategy)
    signal_dates = pd.to_datetime(trades["signal_date"])
    trades = trades[signal_dates >= start].reset_index(drop=True)
    trades["sample"] = np.where(pd.to_datetime(trades["signal_date"]) >= pd.Timestamp(cfg.split.holdout_start),
                                "holdout", "development")
    return bars, trades
