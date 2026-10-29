"""
day_session_study.py
--------------------
Step 3b: is there a tradeable day-session (08:30-13:20 CT) version of the signal, and does
it add to the evening strategy?

Lead from kalman_zscore.py: in the day session a LIVE hedge reverted best. The day session
opens right after the relationship resets (MLE: ~44x normal drift at session opens), so a
hedge frozen at the evening close is stale by 08:30.

    session hedge : state at the previous session's close (production, the control)
    live hedge    : the filter's prediction for bar t = its state after bar t-1 (the state
                    is a random walk, so prediction = last filtered state). Bar t's own price
                    is not in it; a trade then holds that hedge fixed from entry.

Day trades use the session exit (out by 13:20 CT, never carried). Grid: hedge x threshold x
profit x stop x min_hours_left. Choose on 2021-22, confirm on 2023 - Nov 2024. Then the best
2021-22 day variant is added to the evening baseline (they never overlap: evening trades
are out by 07:45, the day session opens at 08:30).

    python -m strategies.crush_strategy.day_session_study
"""
from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

from core_quant.config import ROOT, save_run
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.exit_study import window_stats
from strategies.crush_strategy.features import add_session_zscore, detect_signals
from strategies.crush_strategy.kalman_calibration import FIT_SPLIT
from strategies.crush_strategy.kalman_filter import design_matrix, freeze_at_previous_session_close, run_kalman_filter
from strategies.crush_strategy.strategy import STRATEGY_DIR, daily_returns, load, prepare_bars, simulate_trades

HEDGES = ["session", "live"]
THRESHOLDS = [1.5, 2.0, 2.5]
PROFITS = [0.5, 1.0, 1.5, 2.0, np.inf]
STOPS = [4.0, np.inf]
MIN_HOURS_LEFT = [0.0, 1.0, 2.0]


def bars_with_hedge(cfg, hedge: str) -> pd.DataFrame:
    bars = build_bars(ROOT / cfg.data_path, cfg.strategy.bar_minutes)
    states = run_kalman_filter(bars, cfg.strategy.bar_minutes, cfg.kalman)
    held = freeze_at_previous_session_close(bars, states) if hedge == "session" else states.shift(1)
    y, X = design_matrix(bars)
    bars = bars.join(held.add_prefix("frozen_")).assign(residual=y - np.sum(X * held.values, axis=1))
    return add_session_zscore(bars, method=cfg.signal.z_method)


def sharpe(daily: pd.Series) -> float:
    return daily.mean() / daily.std() * np.sqrt(252)


if __name__ == "__main__":
    cfg = load()
    windows = {"choose": (pd.Timestamp(cfg.split.development_start), pd.Timestamp(FIT_SPLIT)),
               "confirm": (pd.Timestamp(FIT_SPLIT), pd.Timestamp(cfg.split.holdout_start))}

    # self-check: the session hedge rebuilt here must equal production's residual
    production = prepare_bars(cfg)
    session_bars = bars_with_hedge(cfg, "session")
    assert np.allclose(session_bars["residual"], production["residual"], equal_nan=True)

    rows, day_trades = [], {}
    for hedge in HEDGES:
        bars = session_bars if hedge == "session" else bars_with_hedge(cfg, hedge)
        dates = pd.DatetimeIndex(sorted(bars["trading_date"].unique()))
        signals = {th: detect_signals(bars, th, session="day") for th in THRESHOLDS}
        for th, profit, stop, cutoff in product(THRESHOLDS, PROFITS, STOPS, MIN_HOURS_LEFT):
            config = replace(cfg.strategy, threshold=th, profit_sigma=profit, stop_sigma=stop, min_hours_left=cutoff)
            trades = simulate_trades(bars, signals[th], config)
            key = (hedge, th, profit, stop, cutoff)
            day_trades[key] = trades
            row = dict(zip(["hedge", "threshold", "profit_sigma", "stop_sigma", "min_hours_left"], key))
            for w, (a, b) in windows.items():
                row |= {f"{w}_{k}": v for k, v in window_stats(trades, dates, a, b, config.cost_ticks).items()}
            rows.append(row)
        print(f"{hedge} hedge: done", flush=True)
    table = pd.DataFrame(rows)

    # combine: evening baseline + the day variant with the best 2021-22 Sharpe (among >= 40 trades)
    evening = simulate_trades(production, detect_signals(production, cfg.strategy.threshold), cfg.strategy)
    dates = pd.DatetimeIndex(sorted(production["trading_date"].unique()))
    eligible = table[table["choose_trades"] >= 40]
    best = eligible.loc[eligible["choose_sharpe"].idxmax()]
    key = tuple(best[["hedge", "threshold", "profit_sigma", "stop_sigma", "min_hours_left"]])
    combined_rows = []
    for w, (a, b) in windows.items():
        d = dates[(dates >= a) & (dates < b)]
        ev = daily_returns(evening[(pd.to_datetime(evening["signal_date"]) >= a) & (pd.to_datetime(evening["signal_date"]) < b)], d, cfg.strategy.cost_ticks)["ret"]
        dt = day_trades[key]
        dy = daily_returns(dt[(pd.to_datetime(dt["signal_date"]) >= a) & (pd.to_datetime(dt["signal_date"]) < b)], d, cfg.strategy.cost_ticks)["ret"]
        both = ev.add(dy, fill_value=0.0)
        combined_rows.append({"window": w, "evening_sharpe": sharpe(ev), "day_sharpe": sharpe(dy),
                              "combined_sharpe": sharpe(both), "corr_daily": ev.corr(dy),
                              "evening_return_pct": ev.sum(), "day_return_pct": dy.sum()})
    combined = pd.DataFrame(combined_rows).set_index("window")

    table.to_csv(STRATEGY_DIR / "output" / "day_session_study.csv", index=False)
    pd.set_option("display.width", 250)
    print("best day variant on 2021-22:", dict(zip(["hedge", "threshold", "profit_sigma", "stop_sigma", "min_hours_left"], key)))
    print(combined.round(3).to_string())
    print("saved", save_run(STRATEGY_DIR, "day_session_study", cfg, {"best_choose_variant": [str(k) for k in key]},
                            tables={"day_session_study": table, "combined": combined}))
