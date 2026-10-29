"""
kalman_speed_sweep.py
---------------------
Step 1b: tune the filter's adaptation speed for the trading thesis, not for forecasting.

The noise STRUCTURE comes from the MLE fit (mle_diag in the latest kalman_calibration run):
per-coefficient drift shape, R per session type, extra drift at session opens.
One knob is swept: speed s multiplies every drift variance, so s = 1 is the MLE filter
and small s approaches the old hand-set filter (about 1/5000 of MLE).

Two ways to hold the hedge when measuring the residual at bar t:
    session : state at the previous session's close (production)
    live    : the filter's prediction for bar t, i.e. its state after bar t-1. It uses
              every bar up to t-1 but not t itself, so bar t's surprise is not absorbed
              before we measure it. A trade then holds that hedge fixed from entry, so
              any reversion is captured by prices converging, not by the hedge moving.

Scored with kalman_reversion.score on 2021-22 (choose) and 2023 - Nov 2024 (confirm).

    python -m strategies.crush_strategy.kalman_speed_sweep
"""
import numpy as np
import pandas as pd

from core_quant.config import ROOT, latest_run, save_run
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.features import add_session_zscore
from strategies.crush_strategy.kalman_calibration import BAR_MINUTES, FIT_SPLIT, CrushKalman
from strategies.crush_strategy.kalman_filter import freeze_at_previous_session_close
from strategies.crush_strategy.kalman_reversion import score
from strategies.crush_strategy.strategy import STRATEGY_DIR, load

SPEEDS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
HEDGE_MODES = ["session", "live"]


def residual_bars(bars: pd.DataFrame, model: CrushKalman, params: np.ndarray, mode: str) -> pd.DataFrame:
    result = model.filter(params)
    if mode == "session":
        states = result.filtered_state.T[:, :3]
    else:
        states = result.predicted_state.T[:-1, :3]   # predicted_state[t] = state given bars < t
    states = pd.DataFrame(states, index=bars.index, columns=["alpha", "beta_meal", "beta_oil"])
    states["alpha"] -= states["beta_meal"] * model.centre[0] + states["beta_oil"] * model.centre[1]
    hedge = freeze_at_previous_session_close(bars, states) if mode == "session" else states
    y = np.log(bars["ZS_adj_close"].values)
    X = np.column_stack([np.ones(len(bars)), np.log(bars["ZM_adj_close"].values), np.log(bars["ZL_adj_close"].values)])
    out = bars.join(hedge.add_prefix("frozen_")).assign(residual=y - np.sum(X * hedge.values, axis=1))
    return add_session_zscore(out)


if __name__ == "__main__":
    cfg = load()
    run_dir, metrics = latest_run(STRATEGY_DIR, "kalman_calibration")
    mle = metrics["params"]["mle_diag"]
    bars = build_bars(ROOT / cfg.data_path, BAR_MINUTES)
    dev = bars[bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start)]
    burn = int((dev["trading_date"] < pd.Timestamp(cfg.split.development_start)).sum())
    model = CrushKalman(dev, "mle_diag", burn)
    base = np.array([mle[name] for name in model.param_names])
    windows = {"choose 2021-22": (cfg.split.development_start, FIT_SPLIT), "confirm 2023-Nov24": (FIT_SPLIT, cfg.split.holdout_start)}

    tables = []
    for mode in HEDGE_MODES:
        for speed in SPEEDS:
            params = base.copy()
            params[:3] *= speed
            resid = residual_bars(dev, model, params, mode)
            session_sd = resid.groupby("session_id")["frozen_beta_meal"].first().diff().std()
            for window, (a, b) in windows.items():
                w = resid[(resid["trading_date"] >= pd.Timestamp(a)) & (resid["trading_date"] < pd.Timestamp(b))]
                tables.append(score(w).assign(mode=mode, speed=speed, window=window, beta_meal_session_sd=session_sd))
            print(f"{mode} speed {speed:g}: scored", flush=True)
    table = pd.concat(tables).set_index(["window", "session", "horizon_bars", "mode", "speed"]).sort_index()

    pd.set_option("display.width", 250)
    show = table[table.index.get_level_values("horizon_bars").isin([12, 48])][["ic", "t_nw", "cents_z2.0", "n_z2.0"]]
    print(show.round(3).to_string())
    print("saved", save_run(STRATEGY_DIR, "kalman_speed_sweep", cfg, {"params_from": run_dir.name, "speeds": SPEEDS},
                            tables={"speed_sweep": table}))
