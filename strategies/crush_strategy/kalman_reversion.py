"""
kalman_reversion.py
-------------------
Does the residual of each Kalman spec actually mean-revert, and in which session?

For every bar t (not a session's first bar), with the hedge frozen at t exactly as a
trade would hold it:
    entry    : residual at the open of bar t+1
    outcome  : residual change from that entry to the close of bar t+h, same session only
    IC       : Spearman rank correlation of -z_t with that change (> 0 = reversion)
plus, for |z| >= 1.5 and >= 2 (where trades happen), the average favourable move in
cents per bushel against the 0.58 c round-trip cost.

Development period only; params come from the latest kalman_calibration run.

    python -m strategies.crush_strategy.kalman_reversion
"""
import numpy as np
import pandas as pd

from core_quant.config import ROOT, latest_run, save_run
from core_quant.validation import nw_t_stat
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.kalman_calibration import BAR_MINUTES, CrushKalman, frozen_residual
from strategies.crush_strategy.strategy import STRATEGY_DIR, TICK_COST_CENTS, load

HORIZONS = [6, 12, 24, 48]          # bars: 30 min, 1 h, 2 h, 4 h
Z_LEVELS = [1.5, 2.0]


def reversion_targets(bars: pd.DataFrame, h: int) -> pd.Series:
    """Residual change (log, hedge frozen at t) from open of t+1 to close of t+h; NaN if it leaves the session."""
    a, bm, bo = (bars[f"frozen_{c}"].values for c in ["alpha", "beta_meal", "beta_oil"])

    def resid(field, shift):
        p = {leg: np.log(bars[f"{leg}_adj_{field}"].shift(-shift).values) for leg in ["ZS", "ZM", "ZL"]}
        return p["ZS"] - (a + bm * p["ZM"] + bo * p["ZL"])

    same_session = bars["session_id"].shift(-h).values == bars["session_id"].values
    return pd.Series(np.where(same_session, resid("close", h) - resid("open", 1), np.nan), index=bars.index)


def score(bars: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # targets on the full bar sequence first (they shift), then drop session-open signal bars
    bars = bars.assign(rev_signal=-bars["z"], **{f"d{h}": reversion_targets(bars, h) for h in HORIZONS})
    bars = bars[bars["bar_number"] != 0]
    for session in ["evening", "day"]:
        s = bars[bars["session_type"] == session]
        for h in HORIZONS:
            target = f"d{h}"
            nw = nw_t_stat(s, ["rev_signal"], target, lags=h).iloc[0]
            row = {"session": session, "horizon_bars": h, "n": int(nw["n"]),
                   "ic": s[["rev_signal", target]].corr(method="spearman").iloc[0, 1], "t_nw": nw["t_nw"]}
            for z in Z_LEVELS:
                hit = s[s["z"].abs() >= z]
                favourable_cents = (-np.sign(hit["z"]) * hit[target] * hit["ZS_adj_close"]).dropna()
                row[f"n_z{z}"] = len(favourable_cents)
                row[f"cents_z{z}"] = favourable_cents.mean()
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = load()
    run_dir, metrics = latest_run(STRATEGY_DIR, "kalman_calibration")
    bars = build_bars(ROOT / cfg.data_path, BAR_MINUTES)
    dev = bars[bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start)]
    burn = int((dev["trading_date"] < pd.Timestamp(cfg.split.development_start)).sum())

    tables = []
    for spec, params in metrics["params"].items():
        model = CrushKalman(dev, spec, burn)
        p = np.array([params[name] for name in model.param_names])
        resid = frozen_residual(dev, model, p)
        resid = resid[resid["trading_date"] >= pd.Timestamp(cfg.split.development_start)]
        tables.append(score(resid).assign(spec=spec))
        print(f"{spec}: scored", flush=True)
    table = pd.concat(tables).set_index(["spec", "session", "horizon_bars"])
    pd.set_option("display.width", 250)
    print(f"params from {run_dir.name}; cost = {TICK_COST_CENTS} c/bu round trip\n")
    print(table.round(3).to_string())
    print("saved", save_run(STRATEGY_DIR, "kalman_reversion", cfg, {"params_from": run_dir.name},
                            tables={"reversion_by_spec": table}))
