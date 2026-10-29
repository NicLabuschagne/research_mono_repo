"""
kalman_zscore.py
----------------
Step 1c: how should the residual be scaled into a z-score?

Every variant uses only PREVIOUS trading days of the same session type (today excluded),
so nothing about today leaks into today's z.

    roll20         : production. (residual - 20-day mean) / 20-day std
    roll20_nomean  : residual / 20-day RMS. With a responsive hedge the residual's mean is
                     ~0, and past days' residuals were measured against different hedges
    ewma5_nomean   : residual / EWMA RMS (half-life 5 days): reacts faster to vol regimes
    mad20          : (residual - 20-day median) / (1.4826 x 20-day MAD): robust to outliers
    ewma5_tod      : ewma5_nomean x time-of-session profile: 30-minute slots have their own
                     typical size (60-day RMS of the slot / 60-day RMS of the session)
    pct20          : percentile of |residual| among the previous 20 days, mapped to the
                     normal z with the same tail probability. Scale-free by construction

Scored on the new filter (MLE structure, speed 3e-3), both hedge modes, on 2021-22 and
2023 - Nov 2024: calibration (z spread, tail frequencies), reversion IC, and the move
after |z| >= 2 and after the top 5% of |z| (equal trade frequency for every variant).

    python -m strategies.crush_strategy.kalman_zscore
"""
import numpy as np
import pandas as pd

from core_quant.config import ROOT, latest_run, save_run
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.features import percentile_z
from strategies.crush_strategy.kalman_calibration import BAR_MINUTES, FIT_SPLIT, CrushKalman
from strategies.crush_strategy.kalman_reversion import reversion_targets, score
from strategies.crush_strategy.kalman_speed_sweep import HEDGE_MODES, residual_bars
from strategies.crush_strategy.strategy import STRATEGY_DIR, load

SPEED = 3e-3
EWMA_HALFLIFE_DAYS = 5
WINDOW_DAYS = 20
PROFILE_DAYS = 60
SLOT_BARS = 6                       # 30-minute slots
VARIANTS = ["roll20", "roll20_nomean", "ewma5_nomean", "mad20", "ewma5_tod", "pct20"]


def _past_daily(bars: pd.DataFrame, keys: list[str], stat) -> pd.Series:
    """stat(daily table) computed from previous days only, mapped back onto every bar."""
    daily = bars.groupby(keys + ["trading_date"])["residual"].agg(
        total="sum", total_sq=lambda r: (r ** 2).sum(), count="count")
    past = daily.groupby(level=keys, group_keys=False).apply(lambda d: stat(d).shift(1))
    return pd.Series(past.reindex(pd.MultiIndex.from_frame(bars[keys + ["trading_date"]])).values, index=bars.index)


def _rolling_mean_sq(d, days):
    return d["total_sq"].rolling(days).sum() / d["count"].rolling(days).sum()


def _ewma_mean_sq(d, halflife):
    return d["total_sq"].ewm(halflife=halflife).mean() / d["count"].ewm(halflife=halflife).mean()


def _past_mad_z(bars: pd.DataFrame) -> pd.Series:
    """Per session type, day by day: (residual - median) / (1.4826 x MAD) of the previous WINDOW_DAYS days."""
    out = pd.Series(np.nan, index=bars.index)
    for _, s in bars.groupby("session_type"):
        rows_by_day = s.groupby("trading_date").indices          # date -> row positions within s
        residual = s["residual"].values
        dates = sorted(rows_by_day)
        for i in range(WINDOW_DAYS, len(dates)):
            pool = np.concatenate([residual[rows_by_day[d]] for d in dates[i - WINDOW_DAYS:i]])
            pool = pool[~np.isnan(pool)]
            centre = np.median(pool)
            today = residual[rows_by_day[dates[i]]]
            out[s.index[rows_by_day[dates[i]]]] = (today - centre) / (1.4826 * np.median(np.abs(pool - centre)))
    return out


def add_z_variants(bars: pd.DataFrame) -> pd.DataFrame:
    bars = bars.copy()
    r = bars["residual"]
    key = ["session_type"]
    mean20 = _past_daily(bars, key, lambda d: d["total"].rolling(WINDOW_DAYS).sum() / d["count"].rolling(WINDOW_DAYS).sum())
    ms20 = _past_daily(bars, key, lambda d: _rolling_mean_sq(d, WINDOW_DAYS))
    bars["z_roll20"] = (r - mean20) / np.sqrt(ms20 - mean20 ** 2)
    bars["z_roll20_nomean"] = r / np.sqrt(ms20)
    ms_ewma = _past_daily(bars, key, lambda d: _ewma_mean_sq(d, EWMA_HALFLIFE_DAYS))
    bars["z_ewma5_nomean"] = r / np.sqrt(ms_ewma)
    bars["z_mad20"] = _past_mad_z(bars)
    bars["slot"] = bars["bar_number"] // SLOT_BARS
    slot_ms = _past_daily(bars, ["session_type", "slot"], lambda d: _rolling_mean_sq(d, PROFILE_DAYS))
    session_ms = _past_daily(bars, key, lambda d: _rolling_mean_sq(d, PROFILE_DAYS))
    bars["z_ewma5_tod"] = r / np.sqrt(ms_ewma * slot_ms / session_ms)
    bars["z_pct20"] = percentile_z(bars, WINDOW_DAYS)
    return bars


def score_variant(bars: pd.DataFrame, variant: str) -> pd.DataFrame:
    z = bars[f"z_{variant}"]
    table = score(bars.assign(z=z))              # IC and |z| >= 2 moves, per session and horizon
    extra = []
    sig = bars[bars["bar_number"] != 0].assign(z=z)
    for session, s in sig.groupby("session_type"):
        s = s.dropna(subset=["z"])
        top = s[s["z"].abs() >= s["z"].abs().quantile(0.95)]    # equal frequency for every variant
        row = {"session": session, "z_sd": s["z"].std(), "pct_abs_z_ge_2": (s["z"].abs() >= 2).mean() * 100,
               "pct_abs_z_ge_3": (s["z"].abs() >= 3).mean() * 100}
        for h in [12, 48]:
            d = reversion_targets(bars, h).loc[top.index]
            row[f"cents_top5_h{h}"] = (-np.sign(top["z"]) * d * top["ZS_adj_close"]).mean()
        extra.append(row)
    return table.merge(pd.DataFrame(extra), on="session")


if __name__ == "__main__":
    cfg = load()
    run_dir, metrics = latest_run(STRATEGY_DIR, "kalman_calibration")
    mle = metrics["params"]["mle_diag"]
    bars = build_bars(ROOT / cfg.data_path, BAR_MINUTES)
    dev = bars[bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start)]
    burn = int((dev["trading_date"] < pd.Timestamp(cfg.split.development_start)).sum())
    model = CrushKalman(dev, "mle_diag", burn)
    params = np.array([mle[name] for name in model.param_names])
    params[:3] *= SPEED
    windows = {"choose 2021-22": (cfg.split.development_start, FIT_SPLIT), "confirm 2023-Nov24": (FIT_SPLIT, cfg.split.holdout_start)}

    tables = []
    for mode in HEDGE_MODES:
        resid = add_z_variants(residual_bars(dev, model, params, mode))
        # self-check: production z (features.add_session_zscore) must equal the roll20 variant
        # (from development start: production counts the NaN first session in its July 2020 window)
        both = resid.loc[resid["trading_date"] >= pd.Timestamp(cfg.split.development_start), ["z", "z_roll20"]].dropna()
        assert np.allclose(both["z"], both["z_roll20"]), "roll20 != production z"
        for window, (a, b) in windows.items():
            w = resid[(resid["trading_date"] >= pd.Timestamp(a)) & (resid["trading_date"] < pd.Timestamp(b))]
            for variant in VARIANTS:
                tables.append(score_variant(w, variant).assign(mode=mode, variant=variant, window=window))
        print(f"{mode}: scored", flush=True)
    table = pd.concat(tables).set_index(["mode", "session", "variant", "window", "horizon_bars"]).sort_index()
    table.to_csv(STRATEGY_DIR / "output" / "zscore_variants.csv")
    print("saved", save_run(STRATEGY_DIR, "kalman_zscore", cfg, {"params_from": run_dir.name, "speed": SPEED,
                                                                   "variants": VARIANTS}, tables={"zscore_variants": table}))
