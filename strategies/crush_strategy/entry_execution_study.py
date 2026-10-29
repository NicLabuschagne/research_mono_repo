"""
entry_execution_study.py
------------------------
How much edge does working the entry cost, and how many lots does it buy?

Signals and every rule stay as in config.yaml (5-minute bars). Only the entry changes:
    delay : one fill, N minutes after the next bar's open (worst case of a slow fill)
    twap  : equal slices at each bar's open over W minutes (how an order is really worked)

Capacity for a TWAP window: per trade, the lots each leg's volume during the build window
supports at a given participation (lots = share x volume / contracts per bean contract),
the smallest over the three legs. Median and 25th percentile over trades.

Development only for decisions (2021-22 / 2023 - Nov 2024); 2025 shown as a contaminated check.

    python -m strategies.crush_strategy.entry_execution_study
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from core_quant.config import save_run
from strategies.crush_strategy.features import detect_signals
from strategies.crush_strategy.strategy import STRATEGY_DIR, TICK_COST_CENTS, daily_returns, load, prepare_bars, simulate_trades

VARIANTS = [("delay", m) for m in [0, 15, 30, 60, 90]] + [("twap", m) for m in [15, 30, 60, 90]]
PARTICIPATION = [0.05, 0.10]
BUSHELS, MEAL_TONS, OIL_LB = 5000, 100, 60000
WINDOWS = {"2021-22": ("2021-01-01", "2023-01-01"), "2023-Nov24": ("2023-01-01", "2024-12-01"),
           "2025 (contaminated)": ("2024-12-01", "2026-01-01")}


def window_stats(trades: pd.DataFrame, dates: pd.DatetimeIndex, start: str, end: str, cost_ticks: float) -> dict:
    t = trades[(pd.to_datetime(trades["signal_date"]) >= start) & (pd.to_datetime(trades["signal_date"]) < end)]
    d = dates[(dates >= start) & (dates < end)]
    daily = daily_returns(t, d, cost_ticks)["ret"]
    net = t["gross_cents"] - cost_ticks * TICK_COST_CENTS
    return {"trades": len(t), "net_c": net.mean(), "se": net.sem(), "sharpe": daily.mean() / daily.std() * np.sqrt(252)}


def supportable_lots(bars: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Lots per trade each participation level allows, limited by the thinnest leg."""
    signal = trades["entry_bar"].values - 1        # hedge ratios are known at the signal; close enough for sizing
    bean_dollars = trades["bean_price_cents"].values / 100 * BUSHELS
    per_bean = {"ZS": np.ones(len(trades)),
                "ZM": trades["beta_meal"].values * bean_dollars / (bars["ZM_adj_close"].values[signal] * MEAL_TONS),
                "ZL": trades["beta_oil"].values * bean_dollars / (bars["ZL_adj_close"].values[signal] / 100 * OIL_LB)}
    start, n = trades["entry_bar"].values, trades["entry_slices"].values
    lots = {}
    for leg, ratio in per_bean.items():
        cum = np.r_[0, np.cumsum(bars[f"{leg}_volume"].values)]
        lots[leg] = (cum[start + n] - cum[start]) / ratio
    legs = pd.DataFrame(lots)
    return pd.DataFrame({"thinnest_leg": legs.idxmin(axis=1), "lots_at_100pct": legs.min(axis=1)})


if __name__ == "__main__":
    cfg = load()
    bars = prepare_bars(cfg)
    dates = pd.DatetimeIndex(sorted(bars["trading_date"].unique()))
    signals = detect_signals(bars, cfg.strategy.threshold)
    rows = []
    for kind, minutes in VARIANTS:
        config = replace(cfg.strategy, **({"entry_delay_minutes": minutes} if kind == "delay" else {"entry_window_minutes": minutes}))
        trades = simulate_trades(bars, signals, config)
        row = {"entry": kind, "minutes": minutes}
        for w, (a, b) in WINDOWS.items():
            row |= {f"{w}_{k}": v for k, v in window_stats(trades, dates, a, b, config.cost_ticks).items()}
        dev = trades[(pd.to_datetime(trades["signal_date"]) >= "2021-01-01") & (pd.to_datetime(trades["signal_date"]) < "2024-12-01")]
        if kind == "twap" or minutes == 0:
            lots = supportable_lots(bars, dev.reset_index(drop=True))
            row["thinnest_leg"] = lots["thinnest_leg"].mode()[0]
            for p in PARTICIPATION:
                row[f"lots_median_{int(p * 100)}pct"] = (p * lots["lots_at_100pct"]).median()
                row[f"lots_p25_{int(p * 100)}pct"] = (p * lots["lots_at_100pct"]).quantile(0.25)
        rows.append(row)
        print(f"{kind} {minutes} min: done", flush=True)
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(table.round(2).to_string(index=False))
    print("saved", save_run(STRATEGY_DIR, "entry_execution_study", cfg, {"variants": [list(v) for v in VARIANTS]},
                            tables={"entry_execution": table}))
