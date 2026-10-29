"""
exit_study.py
-------------
Step 2: how should a trade end? Time stop x profit target x stop loss, for each of the
four pipelines (hand_set / mle filter x roll20 / pct20 z), on development only.

    time_stop  : bars 12 h (the original, carries across sessions) or session + clock cap
                 of 1-12 hours (never past the close of the entry session)
    profit     : 0.5 - 2 sigma, or none (inf)
    stop       : 2 - 4 sigma, or none (inf)
    threshold  : 1.5, 2.0

Choose on 2021-22, confirm on 2023 - Nov 2024; full development for reference.
Sharpe uses daily returns over every trading day of the window (zero when flat).

    python -m strategies.crush_strategy.exit_study
"""
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from core_quant.config import save_run
from strategies.crush_strategy.features import detect_signals
from strategies.crush_strategy.kalman_calibration import FIT_SPLIT
from strategies.crush_strategy.strategy import (
    STRATEGY_DIR, TICK_COST_CENTS, daily_returns, load, prepare_bars, simulate_trades)

PIPELINES = {"hand_set+roll20": STRATEGY_DIR / "output" / "config_handset_roll20.yaml",
             "mle+roll20": STRATEGY_DIR / "output" / "config_mle_roll20.yaml",
             "hand_set+pct20": STRATEGY_DIR / "output" / "config_handset_pct20.yaml",
             "mle+pct20": STRATEGY_DIR / "output" / "config_mle_pct20.yaml"}
THRESHOLDS = [1.5, 2.0]
PROFITS = [0.5, 1.0, 1.5, 2.0, np.inf]
STOPS = [2.0, 3.0, 4.0, np.inf]
TIME_STOPS = [("bars", 12)] + [("session", h) for h in [1, 2, 4, 6, 8, 12]]


def window_stats(trades: pd.DataFrame, dates: pd.DatetimeIndex, start, end, cost_ticks: float) -> dict:
    signal_dates = pd.to_datetime(trades["signal_date"]) if len(trades) else pd.Series(dtype="datetime64[ns]")
    t = trades[(signal_dates >= start) & (signal_dates < end)]
    d = dates[(dates >= start) & (dates < end)]
    daily = daily_returns(t, d, cost_ticks)["ret"]
    net = t["gross_cents"] - cost_ticks * TICK_COST_CENTS if len(t) else pd.Series(dtype=float)
    return {"trades": len(t), "net_c": net.mean(), "net_se": net.std() / np.sqrt(max(len(t), 1)),
            "hit": (net > 0).mean(), "sharpe": daily.mean() / daily.std() * np.sqrt(252)}


if __name__ == "__main__":
    rows = []
    for name, path in PIPELINES.items():
        cfg = load(Path(path))
        bars = prepare_bars(cfg)
        dates = pd.DatetimeIndex(sorted(bars["trading_date"].unique()))
        windows = {"choose": (pd.Timestamp(cfg.split.development_start), pd.Timestamp(FIT_SPLIT)),
                   "confirm": (pd.Timestamp(FIT_SPLIT), pd.Timestamp(cfg.split.holdout_start)),
                   "dev": (pd.Timestamp(cfg.split.development_start), pd.Timestamp(cfg.split.holdout_start))}
        signals = {th: detect_signals(bars, th) for th in THRESHOLDS}
        for th, (mode, hours), profit, stop in product(THRESHOLDS, TIME_STOPS, PROFITS, STOPS):
            config = replace(cfg.strategy, threshold=th, time_stop=mode, holding_hours=hours,
                             profit_sigma=profit, stop_sigma=stop)
            trades = simulate_trades(bars, signals[th], config)
            row = {"pipeline": name, "threshold": th, "time_stop": mode, "hours": hours,
                   "profit_sigma": profit, "stop_sigma": stop}
            for w, (a, b) in windows.items():
                row |= {f"{w}_{k}": v for k, v in window_stats(trades, dates, a, b, config.cost_ticks).items()}
            if len(trades):
                row["exit_mix"] = trades["exit_reason"].value_counts(normalize=True).round(2).to_dict()
            rows.append(row)
        print(f"{name}: done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(STRATEGY_DIR / "output" / "exit_study.csv", index=False)
    cfg = load()
    print("saved", save_run(STRATEGY_DIR, "exit_study", cfg, {"pipelines": list(PIPELINES), "n_rows": len(table)},
                            tables={"exit_study": table}))
