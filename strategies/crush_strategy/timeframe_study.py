"""
timeframe_study.py
------------------
Step 4: which bar size? 5, 10, 15, 30 minutes, everything else fixed in CLOCK time:
the z window (20 days), the 3 h entry cutoff, the session exit, the 12 h cap, and the
Kalman adaptation speed (the hand-set filter defines its drift per 15 minutes, so Q per
bar scales with the bar size; hand_set_speed 0.3 / 1 / 3 checks that this still fits).
1 minute is left out on purpose: three-leg execution and capacity (see README).

    A. choose / confirm at the selected params, per bar size x filter speed x threshold
    B. walk-forward validation per bar size on the full config grid (speed 1)
    C. capacity: contracts per leg for N bean contracts, as a share of each leg's volume
       in the 30 minutes after entry and the 30 minutes up to the exit

    python -m strategies.crush_strategy.timeframe_study
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from core_quant.config import save_run
from core_quant.walk_forward import run_validation
from strategies.crush_strategy.exit_study import window_stats
from strategies.crush_strategy.features import detect_signals
from strategies.crush_strategy.kalman_calibration import FIT_SPLIT
from strategies.crush_strategy.strategy import STRATEGY_DIR, load, make_evaluator, prepare_bars, simulate_trades

BAR_SIZES = [30, 15, 10, 5]           # 5 last, so the latest validation run is the current config
SPEEDS = [0.3, 1.0, 3.0]
THRESHOLDS = [1.5, 2.0, 2.5]
SIZES = [10, 25, 50]                  # bean contracts per trade
WINDOW_MINUTES = 30
BUSHELS, MEAL_TONS, OIL_LB = 5000, 100, 60000


def with_bar_size(cfg, minutes: int, speed: float = 1.0):
    return replace(cfg, strategy=replace(cfg.strategy, bar_minutes=minutes),
                   kalman=replace(cfg.kalman, hand_set_speed=speed))


def capacity(bars: pd.DataFrame, trades: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Participation (contracts / leg volume) at entry and exit for N bean contracts."""
    k = max(1, WINDOW_MINUTES // minutes)
    signal = trades["entry_bar"].values - 1
    bean_dollars = trades["bean_price_cents"].values / 100 * BUSHELS
    per_bean = {"ZS": np.ones(len(trades)),
                "ZM": trades["beta_meal"].values * bean_dollars / (bars["ZM_adj_close"].values[signal] * MEAL_TONS),
                "ZL": trades["beta_oil"].values * bean_dollars / (bars["ZL_adj_close"].values[signal] / 100 * OIL_LB)}
    rows = []
    for leg, ratio in per_bean.items():
        volume = bars[f"{leg}_volume"].values
        cum = np.r_[0, np.cumsum(volume)]
        entry = trades["entry_bar"].values
        exit_ = trades["exit_bar"].values
        at_entry = cum[np.minimum(entry + k, len(volume))] - cum[entry]
        at_exit = cum[exit_ + 1] - cum[np.maximum(exit_ + 1 - k, 0)]
        for n in SIZES:
            for when, vol in [("entry", at_entry), ("exit", at_exit)]:
                share = n * ratio / np.where(vol > 0, vol, np.nan)
                rows.append({"leg": leg, "contracts_per_bean": np.median(ratio), "bean_contracts": n, "when": when,
                             "median_volume_30m": np.nanmedian(vol), "participation_median": np.nanmedian(share),
                             "participation_p90": np.nanquantile(share, 0.9)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = load()
    windows = {"choose": (pd.Timestamp(cfg.split.development_start), pd.Timestamp(FIT_SPLIT)),
               "confirm": (pd.Timestamp(FIT_SPLIT), pd.Timestamp(cfg.split.holdout_start))}
    study, validations, capacities = [], [], []
    for minutes in BAR_SIZES:
        for speed in SPEEDS:
            cfg_m = with_bar_size(cfg, minutes, speed)
            bars = prepare_bars(cfg_m)
            dates = pd.DatetimeIndex(sorted(bars["trading_date"].unique()))
            for threshold in THRESHOLDS:
                config = replace(cfg_m.strategy, threshold=threshold)
                trades = simulate_trades(bars, detect_signals(bars, threshold), config)
                row = {"bar_minutes": minutes, "speed": speed, "threshold": threshold}
                for w, (a, b) in windows.items():
                    row |= {f"{w}_{k}": v for k, v in window_stats(trades, dates, a, b, config.cost_ticks).items()}
                if len(trades):
                    dev = trades[pd.to_datetime(trades["signal_date"]) < windows["confirm"][1]]
                    row["median_hold_min"] = dev["holding_minutes"].median()
                study.append(row)
                if speed == 1.0 and threshold == cfg.strategy.threshold:
                    dev = trades[(pd.to_datetime(trades["signal_date"]) >= windows["choose"][0])
                                 & (pd.to_datetime(trades["signal_date"]) < windows["confirm"][1])]
                    capacities.append(capacity(bars, dev.reset_index(drop=True), minutes).assign(bar_minutes=minutes))
            print(f"{minutes} min, speed {speed}: done", flush=True)

        cfg_m = with_bar_size(cfg, minutes)
        evaluate, trading_dates = make_evaluator(cfg_m)
        metrics = run_validation(cfg_m, evaluate, trading_dates, STRATEGY_DIR)["metrics"]
        validations.append({"bar_minutes": minutes, "dev_sharpe": metrics["development"]["sharpe"],
                            "trades": metrics["development"]["trades"],
                            "max_dd": metrics["development"]["max_drawdown"],
                            **{f"fold_{k}": v for k, v in metrics["selected_across_folds"].items()},
                            "wf_oos_sharpe": metrics["walk_forward_oos"]["sharpe"], "gates_passed": metrics["gates_passed"]})
        print(f"{minutes} min: validated", flush=True)

    tables = {"study": pd.DataFrame(study), "validation": pd.DataFrame(validations), "capacity": pd.concat(capacities)}
    for name, table in tables.items():
        table.to_csv(STRATEGY_DIR / "output" / f"timeframe_{name}.csv", index=False)
    print("saved", save_run(STRATEGY_DIR, "timeframe_study", cfg, {"bar_sizes": BAR_SIZES, "speeds": SPEEDS}, tables=tables))
