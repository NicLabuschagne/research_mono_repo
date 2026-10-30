"""
failure_study.py
----------------
When does fading bean richness, traded OUTRIGHT, fail, and why?

Trade: first bar of a session where |z| >= 1.5 on the crush residual (re-arms below 0.5, the
crush strategy's detector), sell beans if rich / buy if cheap, enter at the next bar's open,
exit after H bars (1 h, 2 h) or at the session close if sooner. 1 tick cost. All sessions.

Exact split of every trade (ticks, signed in the trade direction):
    outright = market part + residual part
    market part   : the fair-value move (meal and oil through the frozen hedge) - the beta
    residual part : beans minus fair value - did the dislocation actually correct?

Then every trade is bucketed by what was known at entry (bucket edges from 2021-22 only):
    prior_trend_2h  : complex (fair value) move over the 2 h before the signal, + = already moving
                      in the trade's direction
    vol_regime      : 20-day RMS of 5-minute fair-value moves (how busy the complex is)
    recent_vol_1h   : the same over the last hour
    beans_led_share : share of the last hour's dislocation that came from beans moving (vs products)
    sigma_cents     : the residual's usual size (z_std) in c/bu
    session, hour, weekday, month

Development only (2021-22, 2023 - Nov 2024); the holdout stays locked.

    python -m strategies.bean_divergence.failure_study
"""
from pathlib import Path

import numpy as np
import pandas as pd

from core_quant.config import ROOT, load_config, save_run
from strategies.bean_divergence.feature_study import STRATEGY_DIR, TICK_CENTS, WINDOWS, StudyConfig, past_rms, same_session_lag
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.features import add_session_zscore, detect_signals
from strategies.crush_strategy.kalman_filter import add_kalman_residual

THRESHOLD = 1.5
HOLD_BARS = [12, 24]
BUCKETED = ["prior_trend_2h", "vol_regime", "recent_vol_1h", "beans_led_share", "sigma_cents"]


def session_label(index: pd.DatetimeIndex, session_type: pd.Series) -> np.ndarray:
    hour = index.tz_convert("America/Chicago").hour
    return np.where(session_type == "day", "us", np.where((hour >= 2) & (hour < 8), "europe", "asia"))


def build_trades(bars: pd.DataFrame) -> pd.DataFrame:
    log = {leg: (np.log(bars[f"{leg}_adj_open"].values), np.log(bars[f"{leg}_adj_close"].values)) for leg in ["ZS", "ZM", "ZL"]}
    fair = bars["frozen_alpha"] + bars["frozen_beta_meal"] * np.log(bars["ZM_adj_close"]) + bars["frozen_beta_oil"] * np.log(bars["ZL_adj_close"])
    fair_ret = fair.diff().where(bars["session_id"] == bars["session_id"].shift())
    price = bars["ZS_adj_close"].values
    ticks = price / TICK_CENTS                                   # log move x ticks = move in ZS ticks
    prior_2h = (fair - same_session_lag(bars, fair, 24)).values
    vol_regime = (past_rms(bars, fair_ret, 20) * ticks).values
    recent_vol = (fair_ret.rolling(12).std() * np.sqrt(1) * ticks).where(bars["session_id"].shift(11) == bars["session_id"]).values
    d_beans_1h = (np.log(bars["ZS_adj_close"]) - same_session_lag(bars, np.log(bars["ZS_adj_close"]), 12)).values
    d_fair_1h = (fair - same_session_lag(bars, fair, 12)).values
    session = bars["session_id"].values
    last_bar = pd.Series(np.arange(len(bars))).groupby(session).transform("max").values
    labels = session_label(bars.index, bars["session_type"])

    signals = pd.concat([detect_signals(bars, THRESHOLD, session=s) for s in ["evening", "day"]]).sort_values("signal_position")
    rows = []
    for sig in signals.itertuples():
        s, e = sig.signal_position, sig.signal_position + 1
        if e >= len(bars) or session[e] != session[s]:
            continue
        direction = sig.direction                                  # -1 = beans rich -> sell
        bm, bo = bars["frozen_beta_meal"].values[s], bars["frozen_beta_oil"].values[s]
        beans_part = -direction * d_beans_1h[s]                    # + = beans moved INTO the dislocation
        products_part = direction * d_fair_1h[s]                   # + = products moved away, creating it
        row = {"signal_time": bars.index[s], "trading_date": bars["trading_date"].values[s], "session": labels[s],
               "direction": direction, "entry_z": sig.entry_z, "sigma_cents": bars["z_std"].values[s] * price[s],
               "prior_trend_2h": direction * prior_2h[s] * ticks[s], "vol_regime": vol_regime[s],
               "recent_vol_1h": recent_vol[s],
               "beans_led_share": beans_part / (abs(beans_part) + abs(products_part)) if beans_part or products_part else np.nan}
        for hold in HOLD_BARS:
            x = min(e + hold - 1, last_bar[e])
            d_beans = log["ZS"][1][x] - log["ZS"][0][e]
            d_fair = bm * (log["ZM"][1][x] - log["ZM"][0][e]) + bo * (log["ZL"][1][x] - log["ZL"][0][e])
            row[f"outright_{hold}"] = direction * d_beans * ticks[s] - 1.0          # net of the 1-tick cost
            row[f"market_{hold}"] = direction * d_fair * ticks[s]
            row[f"residual_{hold}"] = direction * (d_beans - d_fair) * ticks[s]
        rows.append(row)
    trades = pd.DataFrame(rows)
    local = pd.DatetimeIndex(trades["signal_time"]).tz_convert("America/Chicago")
    trades["hour"], trades["weekday"], trades["month"] = local.hour, local.day_name().str[:3], local.month
    trades["window"] = np.select([trades["trading_date"] < pd.Timestamp("2023-01-01")], ["2021-22"], "2023-Nov24")
    return trades


def summary(trades: pd.DataFrame, keys: list[str], hold: int) -> pd.DataFrame:
    cols = [f"outright_{hold}", f"residual_{hold}", f"market_{hold}"]
    g = trades.groupby(keys + ["window"], observed=True)
    out = g[cols].mean().round(2)
    out["n"] = g.size()
    out["se_outright"] = g[f"outright_{hold}"].sem().round(2)
    return out.unstack("window")


if __name__ == "__main__":
    cfg = load_config(STRATEGY_DIR / "config.yaml", StudyConfig)
    bars = add_session_zscore(add_kalman_residual(build_bars(ROOT / cfg.data_path, 5), 5))
    bars = bars[bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start)]       # holdout stays locked
    trades = build_trades(bars)
    trades = trades[trades["trading_date"] >= pd.Timestamp(cfg.split.development_start)].reset_index(drop=True)
    for hold in HOLD_BARS:                                         # the split is exact: residual + market = gross outright
        assert np.allclose(trades[f"residual_{hold}"] + trades[f"market_{hold}"] - 1.0, trades[f"outright_{hold}"])
    choose = trades["window"] == "2021-22"
    for name in BUCKETED:
        edges = trades.loc[choose, name].quantile([1 / 3, 2 / 3]).values
        trades[f"{name}_bucket"] = pd.cut(trades[name], [-np.inf, *edges, np.inf], labels=["low", "mid", "high"])
    trades.to_csv(STRATEGY_DIR / "output" / "failure_trades.csv", index=False)
    print("saved", save_run(STRATEGY_DIR, "failure_study", cfg, {"trades": len(trades)}, tables={"failure_trades": trades}))
