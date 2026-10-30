"""
layer1_scorecard.py
-------------------
Layer 1, step 2: a scorecard for the CRUSH fair value. No thresholds, no trades.

Fair value of beans = alpha + b_meal log ZM + b_oil log ZL, from the crush Kalman filter, with the
hedge taken three ways (each uses only data before bar t):
    session : frozen at the previous session's close   (today's crush model)
    daily   : frozen at the previous trading day's close
    live    : the filter's estimate after the previous bar
gap(t) = log beans - log fair value. For every horizon h the hedge of bar t is HELD FIXED, so the
model cannot close a gap by re-fitting: every future move scored here is the market's.

Features at bar t (all known at t):
    gap_<hedge>       : the raw gap in ZS ticks                              (back to fair value)
    z20_session       : today's signal, the session gap as a 20-day z         (back to fair value)
    kalman_z_live     : the live gap (= the filter's prediction error) divided by the filter's own
                        predicted std sqrt(F): the Kalman's own normalization, no rolling window
    fair_move_1h      : the fair value's move over the last hour, session hedge, in ticks
                        (momentum angle: do beans follow a sharp product move?)
Targets over the next h, inside the session: gap move, beans move, fair-value move (ZS ticks).

Scorecard (by session Asia / Europe / US and half 2021-22 / 2023 - Nov 2024):
    1. explanatory power : R-squared of beans' move on the fair value's move
    2. gap behaviour     : size, tails, within-session half-life
    3. prediction        : rank IC with Newey-West t of every feature vs every target and horizon
    4. driver regime     : gap -> next gap move, split by the last hour's beans / fair correlation
Development only; the holdout stays locked.

    python -m strategies.bean_divergence.layer1_scorecard
"""
import numpy as np
import pandas as pd

from core_quant.config import ROOT, save_run
from core_quant.validation import nw_t_stat
from strategies.bean_divergence.failure_study import session_label
from strategies.bean_divergence.feature_study import STRATEGY_DIR, TICK_CENTS
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.features import add_session_zscore
from strategies.crush_strategy.kalman_filter import (DRIFT_TO_NOISE_PER_15_MINUTES, WARMUP_DAYS, design_matrix,
                                                     freeze_at_previous_session_close, run_kalman_filter)
from strategies.crush_strategy.strategy import STRATEGY_DIR as CRUSH_DIR, load

HORIZONS = {"5m": 1, "30m": 6, "1h": 12, "2h": 24, "4h": 48}
CORR_BARS = 12                       # driver regime: correlation over the last hour of 5-minute moves


def kalman_forecast_std(bars: pd.DataFrame, bar_minutes: int) -> np.ndarray:
    """The filter's own predicted std of each bar's prediction error, sqrt(F) = sqrt(x' P x + R).
    Same steps as kalman_filter.run_kalman_filter (hand-set settings), keeping F instead of the state."""
    y, X = design_matrix(bars)
    warm = bars["trading_date"].isin(np.sort(bars["trading_date"].unique())[:WARMUP_DAYS]).values
    state = np.linalg.lstsq(X[warm], y[warm], rcond=None)[0]
    observation_noise = np.var(y[warm] - X[warm] @ state)
    drift_to_noise = DRIFT_TO_NOISE_PER_15_MINUTES * bar_minutes / 15
    process_noise = drift_to_noise * observation_noise / np.mean(np.sum(X[warm] ** 2, axis=1)) * np.eye(3)
    covariance = np.eye(3)
    forecast_std = np.empty(len(bars))
    for t in range(len(bars)):
        covariance = covariance + process_noise
        x = X[t]
        forecast_variance = x @ covariance @ x + observation_noise
        forecast_std[t] = np.sqrt(forecast_variance)
        gain = covariance @ x / forecast_variance
        state = state + gain * (y[t] - x @ state)
        covariance = covariance - np.outer(gain, x) @ covariance
    return forecast_std


def hedge_versions(bars: pd.DataFrame, states: pd.DataFrame) -> dict[str, pd.DataFrame]:
    daily_last = states.groupby(bars["trading_date"].values).last().shift(1)     # previous day's last bar
    daily = daily_last.reindex(bars["trading_date"].values)
    daily.index = bars.index
    return {"session": freeze_at_previous_session_close(bars, states), "daily": daily, "live": states.shift(1)}


def log_fair(log_zm: pd.Series, log_zl: pd.Series, hedge: pd.DataFrame, ahead: int = 0) -> pd.Series:
    """Log fair value of bar t+ahead (negative = earlier) computed with the hedge of bar t."""
    return hedge["alpha"] + hedge["beta_meal"] * log_zm.shift(-ahead) + hedge["beta_oil"] * log_zl.shift(-ahead)


def half_life_minutes(gap: pd.Series, session_id: pd.Series, step: int = 6) -> float:
    """AR(1) of the gap sampled every `step` bars inside a session -> minutes for a gap to halve."""
    lagged = gap.shift(step).where(session_id.shift(step) == session_id)
    pair = pd.concat([gap, lagged], axis=1).dropna().iloc[::step]
    phi = np.polyfit(pair.iloc[:, 1], pair.iloc[:, 0], 1)[0]
    return np.log(0.5) / np.log(phi) * step * 5 if 0 < phi < 1 else np.inf


if __name__ == "__main__":
    cfg = load(CRUSH_DIR / "config.yaml")
    bars = build_bars(ROOT / cfg.data_path, 5)
    bars = bars[bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start)]        # holdout stays locked
    states = run_kalman_filter(bars, 5)
    hedges = hedge_versions(bars, states)

    log_zs, log_zm, log_zl = (np.log(bars[f"{leg}_adj_close"]) for leg in ["ZS", "ZM", "ZL"])
    ticks = bars["ZS_adj_close"] / TICK_CENTS                  # turns a log move into ZS ticks
    session_id = bars["session_id"]
    same_since = lambda k: session_id.shift(k) == session_id   # bar t-k is in the same session
    same_until = lambda k: session_id.shift(-k) == session_id  # bar t+k is in the same session
    session = pd.Series(session_label(bars.index, bars["session_type"]), index=bars.index)
    half = pd.Series(np.where(bars["trading_date"] < pd.Timestamp("2023-01-01"), "2021-22", "2023-Nov24"), index=bars.index)
    in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values

    # ---- features (all known at bar t)
    gaps_log = {name: log_zs - log_fair(log_zm, log_zl, h) for name, h in hedges.items()}
    features = pd.DataFrame(index=bars.index)
    for name in hedges:
        features[f"gap_{name}"] = gaps_log[name] * ticks
    features["z20_session"] = add_session_zscore(bars.assign(residual=gaps_log["session"]))["z"]
    forecast_std = kalman_forecast_std(bars, 5)
    features["kalman_z_live"] = gaps_log["live"] / forecast_std
    session_hedge = hedges["session"]
    features["fair_move_1h"] = ((log_fair(log_zm, log_zl, session_hedge) - log_fair(log_zm, log_zl, session_hedge, -CORR_BARS))
                                * ticks).where(same_since(CORR_BARS))
    # self-check: the live gap is exactly the filter's prediction error (state after bar t-1 applied to bar t)
    y, X = design_matrix(bars)
    innovation = y - np.sum(X * states.shift(1).values, axis=1)
    assert np.allclose(gaps_log["live"].values[1:], innovation[1:])

    # driver regime: last hour's correlation of beans and fair-value 5-minute moves (session hedge)
    beans_step = log_zs.diff().where(same_since(1))
    fair_step = (session_hedge["beta_meal"] * log_zm.diff() + session_hedge["beta_oil"] * log_zl.diff()).where(same_since(1))
    corr = beans_step.rolling(CORR_BARS).corr(fair_step).where(same_since(CORR_BARS - 1))

    fit_rows, ic_rows, regime_rows, size_rows = [], [], [], []
    for label, h in HORIZONS.items():
        for name, hedge in hedges.items():
            d_beans = ((log_zs.shift(-h) - log_zs) * ticks).where(same_until(h))
            d_fair = ((log_fair(log_zm, log_zl, hedge, h) - log_fair(log_zm, log_zl, hedge)) * ticks).where(same_until(h))
            targets = pd.DataFrame({"d_gap": d_beans - d_fair, "d_beans": d_beans, "d_fair": d_fair})
            # features scored against THIS hedge's targets: its own gap; the session-based ones against the session hedge
            own = [f"gap_{name}"] + (["z20_session", "fair_move_1h"] if name == "session" else []) \
                + (["kalman_z_live"] if name == "live" else [])
            frame = pd.concat([features[own], targets, corr.rename("corr"), session.rename("session"),
                               half.rename("half")], axis=1)[in_dev]
            for (sess, hf), f in frame.groupby(["session", "half"]):
                # 1. how much of beans' move the fair value explains (non-overlapping samples)
                sample = f.dropna(subset=["d_beans", "d_fair"]).iloc[::h]
                fit_rows.append({"hedge": name, "horizon": label, "session": sess, "half": hf,
                                 "r2_pct": sample["d_beans"].corr(sample["d_fair"]) ** 2 * 100})
                # 3. prediction: every feature vs every target
                for feat in own:
                    for target in ["d_gap", "d_beans", "d_fair"]:
                        pair = f[[feat, target]].dropna()
                        if len(pair) < 50:          # e.g. US: a 1 h history leaves no room for 4 h ahead
                            continue
                        ic_rows.append({"hedge": name, "feature": feat, "horizon": label, "session": sess, "half": hf,
                                        "target": target, "ic": pair[feat].corr(pair[target], method="spearman"),
                                        "t_nw": nw_t_stat(pair, [feat], target, lags=h).loc[feat, "t_nw"], "n": len(pair)})
                # 4. driver regime (1 h and 2 h): gap -> next gap / beans move, by correlation tercile
                if label in ("1h", "2h"):
                    r = f.dropna(subset=["corr", f"gap_{name}", "d_gap"])
                    edges = r["corr"].quantile([1 / 3, 2 / 3]).values
                    for regime, lo, hi in [("low corr: beans on their own", -1.01, edges[0]),
                                           ("middle", edges[0], edges[1]), ("high corr: moving together", edges[1], 1.01)]:
                        part = r[(r["corr"] > lo) & (r["corr"] <= hi)]
                        regime_rows.append({"hedge": name, "horizon": label, "session": sess, "half": hf, "regime": regime,
                                            "ic_gap_vs_next_gap": part[f"gap_{name}"].corr(part["d_gap"], method="spearman"),
                                            "ic_gap_vs_next_beans": part[f"gap_{name}"].corr(part["d_beans"], method="spearman"),
                                            "n": len(part)})
        print(f"{label}: done", flush=True)

    # 2. gap behaviour
    for name in hedges:
        g_all = pd.DataFrame({"gap": features[f"gap_{name}"], "session": session, "half": half, "sid": session_id})[in_dev]
        for (sess, hf), g in g_all.groupby(["session", "half"]):
            size_rows.append({"hedge": name, "session": sess, "half": hf, "gap_sd_ticks": g["gap"].std(),
                              "p05": g["gap"].quantile(0.05), "p95": g["gap"].quantile(0.95),
                              "kurtosis": g["gap"].kurt(), "half_life_min": half_life_minutes(g["gap"], g["sid"])})
    kalman_std_ticks = pd.Series(forecast_std * ticks.values)[in_dev]
    print(f"Kalman predicted std sqrt(F) in ZS ticks: median {kalman_std_ticks.median():.2f}, "
          f"5th-95th pct {kalman_std_ticks.quantile(0.05):.2f}-{kalman_std_ticks.quantile(0.95):.2f}")

    tables = {"fit": pd.DataFrame(fit_rows), "size": pd.DataFrame(size_rows), "ic": pd.DataFrame(ic_rows),
              "regime": pd.DataFrame(regime_rows)}
    for k, v in tables.items():
        v.to_csv(STRATEGY_DIR / "output" / f"layer1_scorecard_{k}.csv", index=False)
    print("saved", save_run(STRATEGY_DIR, "layer1_scorecard", cfg, {"hedges": list(hedges)}, tables=tables))
