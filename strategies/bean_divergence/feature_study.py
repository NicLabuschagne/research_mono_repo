"""
feature_study.py
----------------
Gate 2 for the bean-divergence thesis (README): does relative strength vs products,
together with the outright bean move, predict the NEXT outright bean move?

Features at bar t (same session only, data up to t's close):
    zres_L   : change in the hedged residual over L bars (hedge frozen at the previous session
               close, from the crush Kalman filter), in units of its usual size
    zbean_L  : beans' own log return over L bars, same scaling
    zskew_L  : change in the TradingView skew line 6 ZS - 11 ZM - 9 ZL, same scaling (robustness)
    L = 30 min, 1 h, 2 h, and "open" (since the session's first bar)
    range_pos: where beans sit in today's range so far (0 = session low, 1 = session high)
Scaling: RMS of the same change over the previous 20 trading days, same session type.
Target  : beans' log return from the open of t+1 to the close of t+h, in ZS ticks; NaN if it
          would cross the session close (no holding through breaks).

1. IC of each feature vs the target (Newey-West t), per session and horizon
2. Event study: first bar of "residual up & beans down" (long) / mirror (short), vs a
   beans-only control (beans moved -> fade) and a residual-only rule
3. Regression: target ~ zres + zbean (+ interaction), HAC standard errors

Development only; choose on 2021-22, confirm on 2023 - Nov 2024. The holdout stays locked.

    python -m strategies.bean_divergence.feature_study
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from core_quant.config import ROOT, SplitConfig, load_config, save_run
from core_quant.validation import nw_t_stat
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.kalman_filter import add_kalman_residual

STRATEGY_DIR = Path(__file__).parent
TICK_CENTS = 0.25
WINDOWS = {"2021-22": ("2021-01-01", "2023-01-01"), "2023-Nov24": ("2023-01-01", "2024-12-01")}


@dataclass(frozen=True)
class ResearchConfig:
    bar_minutes: int = 5
    lookbacks_bars: tuple = (6, 12, 24)
    horizons_bars: tuple = (3, 6, 12, 24, 48)
    z_window_days: int = 20
    event_z: float = 1.0
    cost_ticks: float = 1.0


@dataclass(frozen=True)
class StudyConfig:
    data_path: str
    split: SplitConfig
    research: ResearchConfig


def same_session_lag(bars: pd.DataFrame, values: pd.Series, lag: int) -> pd.Series:
    lagged = values.shift(lag)
    return lagged.where(bars["session_id"].shift(lag) == bars["session_id"])


def past_rms(bars: pd.DataFrame, change: pd.Series, days: int) -> pd.Series:
    """RMS of `change` over the previous `days` trading days of the same session type (today excluded)."""
    frame = pd.DataFrame({"sq": change ** 2, "n": change.notna().astype(float),
                          "session_type": bars["session_type"], "date": bars["trading_date"]})
    daily = frame.groupby(["session_type", "date"])[["sq", "n"]].sum()
    rolling = daily.groupby(level="session_type", group_keys=False).apply(
        lambda d: (d["sq"].rolling(days).sum() / d["n"].rolling(days).sum()).shift(1))
    keys = pd.MultiIndex.from_arrays([bars["session_type"], bars["trading_date"]])
    return pd.Series(np.sqrt(rolling.reindex(keys).values), index=bars.index)


def build_features(bars: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    log_beans = np.log(bars["ZS_adj_close"])
    skew = 6 * bars["ZS_adj_close"] - 11 * bars["ZM_adj_close"] - 9 * bars["ZL_adj_close"]
    series = {"res": bars["residual"], "bean": log_beans, "skew": skew}
    f = pd.DataFrame(index=bars.index)
    session = bars["session_id"]
    for name, s in series.items():
        changes = {str(L): s - same_session_lag(bars, s, L) for L in cfg.lookbacks_bars}
        changes["open"] = s - s.groupby(session).transform("first")
        for label, change in changes.items():
            f[f"z{name}_{label}"] = change / past_rms(bars, change, cfg.z_window_days)
    low, high = log_beans.groupby(session).cummin(), log_beans.groupby(session).cummax()
    f["range_pos"] = ((log_beans - low) / (high - low)).where(high > low)
    hour = bars.index.tz_convert("America/Chicago").hour
    f["session"] = np.where(bars["session_type"] == "day", "us", np.where((hour >= 2) & (hour < 8), "europe", "asia"))
    for h in cfg.horizons_bars:
        future = np.log(bars["ZS_adj_close"].shift(-h)) - np.log(bars["ZS_adj_open"].shift(-1))
        ticks = future * bars["ZS_adj_close"] / TICK_CENTS
        f[f"fwd_{h}"] = ticks.where(session.shift(-h) == session)
    f["trading_date"] = bars["trading_date"]
    return f


def in_window(f: pd.DataFrame, window: str) -> pd.DataFrame:
    a, b = WINDOWS[window]
    return f[(f["trading_date"] >= a) & (f["trading_date"] < b)]


def ic_table(f: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    features = [c for c in f.columns if c.startswith("z") or c == "range_pos"]
    rows = []
    for window in WINDOWS:
        w = in_window(f, window)
        for session, s in w.groupby("session"):
            for h in cfg.horizons_bars:
                for k in features:
                    pair = s[[k, f"fwd_{h}"]].dropna()
                    if len(pair) < 50:            # e.g. US session: a 2 h lookback leaves no room for 4 h ahead
                        continue
                    nw = nw_t_stat(pair, [k], f"fwd_{h}", lags=h)
                    rows.append({"window": window, "session": session, "h": h, "feature": k,
                                 "ic": pair.corr(method="spearman").iloc[0, 1], "t_nw": nw.loc[k, "t_nw"], "n": len(pair)})
    return pd.DataFrame(rows)


def first_bar(condition: pd.Series, bars_session: pd.Series) -> pd.Series:
    """True on the first bar a condition holds (re-arms once it stops holding), within a session."""
    previous = condition.shift(1, fill_value=False) & (bars_session.shift(1) == bars_session)
    return condition & ~previous


def event_table(f: pd.DataFrame, bars: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    z, rows = cfg.event_z, []
    lookbacks = [str(L) for L in cfg.lookbacks_bars] + ["open"]
    for L in lookbacks:
        res, bean = f[f"zres_{L}"], f[f"zbean_{L}"]
        rules = {
            "divergence": (res >= z) & (bean <= -z), "divergence_short": (res <= -z) & (bean >= z),
            "divergence_at_extreme": (res >= z) & (bean <= -z) & (f["range_pos"] <= 0.2),
            "divergence_short_at_extreme": (res <= -z) & (bean >= z) & (f["range_pos"] >= 0.8),
            "beans_only_fade_long": bean <= -z, "beans_only_fade_short": bean >= z,
            "residual_only_long": res >= z, "residual_only_short": res <= -z,
        }
        for rule, condition in rules.items():
            sign = -1.0 if "short" in rule else 1.0
            events = first_bar(condition.fillna(False), bars["session_id"])
            for window in WINDOWS:
                e = in_window(f[events], window)
                for session, s in e.groupby("session"):
                    for h in cfg.horizons_bars:
                        gross = sign * s[f"fwd_{h}"].dropna()
                        rows.append({"lookback": L, "rule": rule, "window": window, "session": session, "h": h,
                                     "events": len(gross), "gross_ticks": gross.mean(),
                                     "net_ticks": gross.mean() - cfg.cost_ticks, "se": gross.sem()})
    return pd.DataFrame(rows)


def regression_table(f: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    rows = []
    for L in [str(L) for L in cfg.lookbacks_bars] + ["open"]:
        for window in WINDOWS:
            w = in_window(f, window)
            for session, s in w.groupby("session"):
                for h in cfg.horizons_bars:
                    d = s[[f"zres_{L}", f"zbean_{L}", f"fwd_{h}"]].dropna()
                    if len(d) < 50:
                        continue
                    X = d[[f"zres_{L}", f"zbean_{L}"]].clip(-5, 5)            # cap outliers in the features only
                    X = sm.add_constant(X.assign(interaction=X.iloc[:, 0] * X.iloc[:, 1]))
                    fit = sm.OLS(d[f"fwd_{h}"], X).fit(cov_type="HAC", cov_kwds={"maxlags": h})
                    rows.append({"lookback": L, "window": window, "session": session, "h": h, "n": len(d),
                                 "b_res": fit.params.iloc[1], "t_res": fit.tvalues.iloc[1],
                                 "b_bean": fit.params.iloc[2], "t_bean": fit.tvalues.iloc[2],
                                 "b_inter": fit.params.iloc[3], "t_inter": fit.tvalues.iloc[3],
                                 "r2_pct": fit.rsquared * 100})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = load_config(STRATEGY_DIR / "config.yaml", StudyConfig)
    r = cfg.research
    bars = add_kalman_residual(build_bars(ROOT / cfg.data_path, r.bar_minutes), r.bar_minutes)
    bars = bars[bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start)]          # holdout stays locked
    f = build_features(bars, r)
    assert f["trading_date"].max() < pd.Timestamp(cfg.split.holdout_start)
    tables = {"ic": ic_table(f, r), "events": event_table(f, bars, r), "regression": regression_table(f, r)}
    out = STRATEGY_DIR / "output"
    out.mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"feature_study_{name}.csv", index=False)
    print("saved", save_run(STRATEGY_DIR, "feature_study", cfg, {"rows": {k: len(v) for k, v in tables.items()}}, tables=tables))
