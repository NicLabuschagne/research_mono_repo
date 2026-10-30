"""
layer1_fx.py
------------
Layer 1, step 5: does USD/CNH (China's buying power) improve the fair value of soybeans?

5-minute bars (our base). USD/CNH is stamped at the bar START, so an FX bar is used only once it has closed;
between FX bars the last closed value is carried (FX is a level, a stale but known value is not look-ahead).
FX has no contract rolls.

    crush    : log ZS = alpha + b_meal log ZM + b_oil log ZL                 (back-adjusted crush legs, as before)
    crush+fx : the same plus b_fx log USDCNH
Same hand-set Kalman settings (layer1_corn_wheat.kalman_states), hedge held fixed from bar t (prior close / live).

    (a) outright fair value: R-squared of beans' move on the fair value's move; IC of the residual vs the next
        soybean move (from the next bar's open)
    (b) the crush residual explained by FX: same-time R-squared, and the IC of the last hour's FX move vs the
        crush residual's next move
Development only; the holdout stays locked.

    python -m strategies.bean_divergence.layer1_fx
"""
import numpy as np
import pandas as pd

from core_quant.config import ROOT, save_run
from core_quant.validation import nw_t_stat
from strategies.bean_divergence.failure_study import session_label
from strategies.bean_divergence.feature_study import STRATEGY_DIR, TICK_CENTS
from strategies.bean_divergence.layer1_corn_wheat import kalman_states
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.kalman_filter import freeze_at_previous_session_close, run_kalman_filter
from strategies.crush_strategy.strategy import STRATEGY_DIR as CRUSH_DIR, load

FX_FILE = "USD_CNH_forex_5m_2020-01-23_2026-10-30.parquet"
HORIZONS = {"5m": 1, "30m": 6, "1h": 12, "2h": 24, "4h": 48}


def fx_close(bars: pd.DataFrame) -> pd.Series:
    """Last CLOSED USD/CNH close known at each of our bars."""
    raw = pd.read_parquet(ROOT / "data" / "raw" / FX_FILE)
    raw["known_at"] = pd.to_datetime(raw["time"], utc=True).astype("datetime64[ns, UTC]") + pd.Timedelta(minutes=5)
    ours = pd.DataFrame({"known_at": (bars.index + pd.Timedelta(minutes=1)).astype("datetime64[ns, UTC]"),
                         "row": np.arange(len(bars))})
    joined = pd.merge_asof(ours.sort_values("known_at"), raw[["known_at", "close"]].sort_values("known_at"),
                           on="known_at", direction="backward").sort_values("row")
    return pd.Series(joined["close"].values, index=bars.index, name="USDCNH")


if __name__ == "__main__":
    cfg = load(CRUSH_DIR / "config.yaml")
    bars = build_bars(ROOT / cfg.data_path, 5)
    bars = bars[bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start)].copy()      # holdout stays locked
    bars["USDCNH"] = fx_close(bars)
    first_fx = bars["USDCNH"].first_valid_index()
    bars = bars.loc[first_fx:]                                          # FX file starts 22 Jan 2020
    assert bars["USDCNH"].notna().all()

    log = {"ZS": np.log(bars["ZS_adj_close"]), "ZM": np.log(bars["ZM_adj_close"]), "ZL": np.log(bars["ZL_adj_close"]),
           "FX": np.log(bars["USDCNH"])}
    y = log["ZS"].values
    designs = {"crush": ["ZM", "ZL"], "crush+fx": ["ZM", "ZL", "FX"]}
    # self-check: the general filter with the crush design equals the crush filter on these bars
    X3 = np.column_stack([np.ones(len(bars)), log["ZM"], log["ZL"]])
    assert np.allclose(kalman_states(y, X3, bars["trading_date"], 5)[-1000:], run_kalman_filter(bars, 5).values[-1000:])

    ticks = bars["ZS_adj_close"] / TICK_CENTS
    session_id = bars["session_id"]
    session = pd.Series(session_label(bars.index, bars["session_type"]), index=bars.index)
    half = pd.Series(np.where(bars["trading_date"] < pd.Timestamp("2023-01-01"), "2021-22", "2023-Nov24"), index=bars.index)
    in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
    from_next_open = np.log(bars["ZS_adj_open"].shift(-1) / bars["ZS_adj_close"])   # beans' move from t's close to t+1's open

    rows, residuals = [], {}
    for model, legs in designs.items():
        X = np.column_stack([np.ones(len(bars))] + [log[leg].values for leg in legs])
        states = pd.DataFrame(kalman_states(y, X, bars["trading_date"], 5), index=bars.index)
        for hedge_name, hedge in {"prior close": freeze_at_previous_session_close(bars, states), "live": states.shift(1)}.items():
            def fair(ahead: int) -> pd.Series:      # fair value of bar t+ahead with the hedge of bar t
                value = hedge[0].copy()
                for j, leg in enumerate(legs, start=1):
                    value = value + hedge[j] * log[leg].shift(-ahead)
                return value
            gap = (log["ZS"] - fair(0)) * ticks
            residuals[(model, hedge_name)] = gap
            for label, h in HORIZONS.items():
                same = session_id.shift(-h) == session_id
                d_beans = ((log["ZS"].shift(-h) - log["ZS"]) * ticks).where(same)
                d_fair = ((fair(h) - fair(0)) * ticks).where(same)
                fwd = ((log["ZS"].shift(-h) - log["ZS"] - from_next_open) * ticks).where(same)
                frame = pd.DataFrame({"gap": gap, "d_beans": d_beans, "d_fair": d_fair, "fwd": fwd,
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

    # (b) the crush residual explained by FX (prior-close crush hedge)
    crush_gap = residuals[("crush", "prior close")]
    fx_rows = []
    for label, h in {"1h": 12, "2h": 24}.items():
        same_back, same_ahead = session_id.shift(h) == session_id, session_id.shift(-h) == session_id
        frame = pd.DataFrame({"d_gap_now": (crush_gap - crush_gap.shift(h)).where(same_back),
                              "fx": ((log["FX"] - log["FX"].shift(h)) * 1e4).where(same_back),          # FX move in bp
                              "d_gap_next": (crush_gap.shift(-h) - crush_gap).where(same_ahead),
                              "session": session, "half": half})[in_dev]
        for (sess, hf), f in frame.groupby(["session", "half"]):
            f = f.dropna()
            sample = f.iloc[::h]
            fx_rows.append({"horizon": label, "session": sess, "half": hf,
                            "r2_same_time_pct": sample["d_gap_now"].corr(sample["fx"]) ** 2 * 100,
                            "ic_fx_vs_next_crush_move": f["fx"].corr(f["d_gap_next"], method="spearman"),
                            "t": nw_t_stat(f, ["fx"], "d_gap_next", lags=h).loc["fx", "t_nw"]})

    tables = {"fair_value": pd.DataFrame(rows), "crush_vs_fx": pd.DataFrame(fx_rows)}
    for k, v in tables.items():
        v.to_csv(STRATEGY_DIR / "output" / f"layer1_fx_{k}.csv", index=False)
    print("saved", save_run(STRATEGY_DIR, "layer1_fx", cfg, {"designs": designs, "fx_file": FX_FILE}, tables=tables))
