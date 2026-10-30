"""
layer1_corn_wheat.py
--------------------
Layer 1, step 4: do corn and wheat improve the fair value of soybeans?

Data: 15-minute bars for all five legs (corn and wheat only exist at 15 minutes here).
    ZS / ZM / ZL : the crush pipeline's session-aligned bars, switch-free returns (opening gap dropped)
    ZC / ZW      : 15-minute files, stamped at the bar START. A corn bar is used only once it has closed
                   (start + 15 min <= our bar's end), so nothing from the future leaks in. Their rolls are NOT
                   adjusted in the files (corn jumps -22% / +21% around 24-28 Jun 2021), so the same rule is
                   applied: the opening bar's return is close / open, the gap into a session is dropped.
Every leg becomes a continuous log index = running sum of its switch-free returns. All legs lose their real
overnight gaps the same way, so the two fair values below are compared on equal terms.

    crush       : index_ZS = alpha + b_meal index_ZM + b_oil index_ZL
    crush+grain : the same plus b_corn index_ZC + b_wheat index_ZW
Both use the crush strategy's hand-set Kalman settings (drift per 15 minutes of clock time), generalized to
any number of regressors. Each hedge is held fixed from bar t (prior-close or live), as in the scorecard.

    (a) outright fair value: R-squared of beans' move on the fair value's move; IC of the residual vs the next
        soybean move (from the next bar's open)
    (b) the crush residual explained by grains: R-squared of the crush residual's move on corn + wheat moves,
        and the IC of the last hour's corn / wheat move vs the crush residual's next move
Development only; the holdout stays locked.

    python -m strategies.bean_divergence.layer1_corn_wheat
"""
import numpy as np
import pandas as pd

from core_quant.config import ROOT, save_run
from core_quant.validation import nw_t_stat
from strategies.bean_divergence.failure_study import session_label
from strategies.bean_divergence.feature_study import STRATEGY_DIR, TICK_CENTS
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.kalman_filter import (DRIFT_TO_NOISE_PER_15_MINUTES, WARMUP_DAYS,
                                                     freeze_at_previous_session_close, run_kalman_filter)
from strategies.crush_strategy.strategy import STRATEGY_DIR as CRUSH_DIR, load

BAR_MINUTES = 15
GRAIN_FILES = {"ZC": "CORN_futures_15m_2020-01-23_2026-10-30.parquet", "ZW": "WHEAT_futures_15m_2020-01-23_2026-10-30.parquet"}
HORIZONS = {"15m": 1, "1h": 4, "2h": 8, "4h": 16}


def kalman_states(y: np.ndarray, X: np.ndarray, dates: pd.Series, bar_minutes: int) -> np.ndarray:
    """The crush filter's hand-set settings for any number of regressors: state after each bar.
    Warm-up OLS on the first days gives the start state and R; Q = drift-to-noise x R / typical x'x."""
    warm = dates.isin(np.sort(dates.unique())[:WARMUP_DAYS]).values
    state = np.linalg.lstsq(X[warm], y[warm], rcond=None)[0]
    observation_noise = np.var(y[warm] - X[warm] @ state)
    drift_to_noise = DRIFT_TO_NOISE_PER_15_MINUTES * bar_minutes / 15
    k = X.shape[1]
    process_noise = drift_to_noise * observation_noise / np.mean(np.sum(X[warm] ** 2, axis=1)) * np.eye(k)
    covariance = np.eye(k)
    states = np.empty((len(y), k))
    for t in range(len(y)):
        covariance = covariance + process_noise
        x = X[t]
        forecast_variance = x @ covariance @ x + observation_noise
        gain = covariance @ x / forecast_variance
        state = state + gain * (y[t] - x @ state)
        covariance = covariance - np.outer(gain, x) @ covariance
        states[t] = state
    return states


def grain_returns(bars: pd.DataFrame, leg: str, file_name: str) -> pd.Series:
    """Switch-free 15-minute log returns of a grain leg, aligned to our bars (only CLOSED grain bars)."""
    raw = pd.read_parquet(ROOT / "data" / "raw" / file_name)
    raw["time"] = pd.to_datetime(raw["time"], utc=True).astype("datetime64[ns, UTC]")
    raw["known_at"] = raw["time"] + pd.Timedelta(minutes=15)            # stamped at the start: known at its end
    # our bars are stamped with their LAST minute, so their data is complete one minute later
    ours = pd.DataFrame({"known_at": (bars.index + pd.Timedelta(minutes=1)).astype("datetime64[ns, UTC]"),
                         "row": np.arange(len(bars))})
    joined = pd.merge_asof(ours.sort_values("known_at"), raw[["known_at", "open", "close"]].sort_values("known_at"),
                           on="known_at", direction="backward", tolerance=pd.Timedelta(minutes=14)).sort_values("row")
    close, open_ = joined["close"].values, joined["open"].values
    first_of_session = np.r_[True, bars["session_id"].values[1:] != bars["session_id"].values[:-1]]
    returns = np.empty(len(bars))
    for t in range(len(bars)):
        if np.isnan(close[t]):
            returns[t] = 0.0                                             # no grain bar: no move recorded
        elif first_of_session[t] or np.isnan(close[t - 1]):
            returns[t] = np.log(close[t] / open_[t])                     # opening bar: drop the gap (rolls live there)
        else:
            returns[t] = np.log(close[t] / close[t - 1])
    return pd.Series(returns, index=bars.index, name=f"{leg}_ret")


if __name__ == "__main__":
    cfg = load(CRUSH_DIR / "config.yaml")
    bars = build_bars(ROOT / cfg.data_path, BAR_MINUTES)
    bars = bars[bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start)].copy()      # holdout stays locked
    for leg, file_name in GRAIN_FILES.items():
        bars[f"{leg}_ret"] = grain_returns(bars, leg, file_name)
    index = {leg: bars[f"{leg}_ret"].cumsum() for leg in ["ZS", "ZM", "ZL", "ZC", "ZW"]}     # continuous log indices
    assert bars["ZC_ret"].abs().max() < 0.10, "a grain roll jump slipped through"

    # self-check: the generalized filter equals the crush filter when given the crush design
    check_bars = bars.iloc[:5000]
    X3 = np.column_stack([np.ones(5000), np.log(check_bars["ZM_adj_close"]), np.log(check_bars["ZL_adj_close"])])
    ours = kalman_states(np.log(check_bars["ZS_adj_close"].values), X3, check_bars["trading_date"], BAR_MINUTES)
    assert np.allclose(ours, run_kalman_filter(check_bars, BAR_MINUTES).values)

    y = index["ZS"].values
    designs = {"crush": ["ZM", "ZL"], "crush+grain": ["ZM", "ZL", "ZC", "ZW"]}
    ticks = bars["ZS_adj_close"] / TICK_CENTS
    session_id = bars["session_id"]
    session = pd.Series(session_label(bars.index, bars["session_type"]), index=bars.index)
    half = pd.Series(np.where(bars["trading_date"] < pd.Timestamp("2023-01-01"), "2021-22", "2023-Nov24"), index=bars.index)
    in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values

    rows, residuals = [], {}
    for model, legs in designs.items():
        X = np.column_stack([np.ones(len(bars))] + [index[leg].values for leg in legs])
        states = pd.DataFrame(kalman_states(y, X, bars["trading_date"], BAR_MINUTES), index=bars.index)
        hedges = {"prior close": freeze_at_previous_session_close(bars, states), "live": states.shift(1)}
        for hedge_name, hedge in hedges.items():
            def fair(ahead: int) -> pd.Series:      # fair value of bar t+ahead with the hedge of bar t
                value = hedge[0].copy()
                for j, leg in enumerate(legs, start=1):
                    value = value + hedge[j] * index[leg].shift(-ahead)
                return value
            gap = (index["ZS"] - fair(0)) * ticks
            residuals[(model, hedge_name)] = gap
            for label, h in HORIZONS.items():
                same = session_id.shift(-h) == session_id
                d_beans = ((index["ZS"].shift(-h) - index["ZS"]) * ticks).where(same)
                d_fair = ((fair(h) - fair(0)) * ticks).where(same)
                # outright target from the next bar's OPEN: the next bar's own move up to its open is excluded
                beans_from_open = ((index["ZS"].shift(-h) - (index["ZS"] + np.log(bars["ZS_adj_open"].shift(-1) / bars["ZS_adj_close"])))
                                   * ticks).where(same)
                frame = pd.DataFrame({"gap": gap, "d_beans": d_beans, "d_fair": d_fair, "fwd": beans_from_open,
                                      "session": session, "half": half})[in_dev]
                for (sess, hf), f in frame.groupby(["session", "half"]):
                    f = f.dropna()
                    if len(f) < 50:
                        continue
                    sample = f.iloc[::h]
                    rows.append({"model": model, "hedge": hedge_name, "horizon": label, "session": sess, "half": hf,
                                 "r2_pct": sample["d_beans"].corr(sample["d_fair"]) ** 2 * 100,
                                 "ic_outright": f["gap"].corr(f["fwd"], method="spearman"),
                                 "t_outright": nw_t_stat(f, ["gap"], "fwd", lags=h).loc["gap", "t_nw"]})
        print(f"{model}: done", flush=True)

    # (b) the crush residual explained by grains (prior-close crush hedge)
    crush_gap = residuals[("crush", "prior close")]
    grain_rows = []
    for label, h in {"1h": 4, "2h": 8}.items():
        same_back, same_ahead = session_id.shift(h) == session_id, session_id.shift(-h) == session_id
        d_gap_now = (crush_gap - crush_gap.shift(h)).where(same_back)
        d_corn_now = ((index["ZC"] - index["ZC"].shift(h)) * ticks).where(same_back)
        d_wheat_now = ((index["ZW"] - index["ZW"].shift(h)) * ticks).where(same_back)
        d_gap_next = (crush_gap.shift(-h) - crush_gap).where(same_ahead)
        frame = pd.DataFrame({"d_gap_now": d_gap_now, "corn": d_corn_now, "wheat": d_wheat_now, "d_gap_next": d_gap_next,
                              "session": session, "half": half})[in_dev]
        for (sess, hf), f in frame.groupby(["session", "half"]):
            f = f.dropna()
            sample = f.iloc[::h]
            fit = np.linalg.lstsq(np.column_stack([np.ones(len(sample)), sample["corn"], sample["wheat"]]), sample["d_gap_now"], rcond=None)
            explained = 1 - np.var(sample["d_gap_now"] - np.column_stack([np.ones(len(sample)), sample["corn"], sample["wheat"]]) @ fit[0]) / np.var(sample["d_gap_now"])
            row = {"horizon": label, "session": sess, "half": hf, "r2_same_time_pct": explained * 100}
            for g in ["corn", "wheat"]:
                row[f"ic_{g}_vs_next_crush_move"] = f[g].corr(f["d_gap_next"], method="spearman")
                row[f"t_{g}"] = nw_t_stat(f, [g], "d_gap_next", lags=h).loc[g, "t_nw"]
            grain_rows.append(row)

    tables = {"fair_value": pd.DataFrame(rows), "crush_vs_grain": pd.DataFrame(grain_rows)}
    for k, v in tables.items():
        v.to_csv(STRATEGY_DIR / "output" / f"layer1_corn_wheat_{k}.csv", index=False)
    print("saved", save_run(STRATEGY_DIR, "layer1_corn_wheat", cfg, {"designs": designs}, tables=tables))
