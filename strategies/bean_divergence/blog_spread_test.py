"""
blog_spread_test.py
-------------------
Quick test of the "Spread Alpha Atlas" blog method on the crush (ZS vs ZM / ZL):

    30-minute bars; Kalman hedge (the crush strategy's hand-set filter) frozen at the last bar of
    each trading day and held for the whole next day (the blog's "rebalance daily"); our
    session-frozen hedge as a comparison.
    Features on the spread S = log ZS - (alpha + b_meal log ZM + b_oil log ZL), lookbacks 5/10/20/50:
      zscore     : (S - lagged mean) / lagged std          -> position -sign
      distmean   : (S - rolling mean) / rolling std        -> position -sign
      bollinger  : %B > 0.95 short, < 0.05 long, else flat
      rsi        : 14-bar RSI of S, > 70 short, < 30 long   (lookback ignored)
      momentum   : sign(S - S lagged)                       -> follow (the control that should fail)
    Position at bar t (known at its close) earns the hedged return of bar t+1 with the hedge of t.
    Flat across session breaks. Cost: our real cost, 1 tick per leg round trip ~ 0.58 c/bu on the
    bean notional, half charged on each unit of position change.
    P&L split by the session the bar belongs to (Asia / Europe / US); development only.

    python -m strategies.bean_divergence.blog_spread_test
"""
from itertools import product

import numpy as np
import pandas as pd

from core_quant.config import ROOT, save_run
from strategies.bean_divergence.failure_study import session_label
from strategies.bean_divergence.feature_study import STRATEGY_DIR
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.kalman_filter import design_matrix, freeze_at_previous_session_close, run_kalman_filter
from strategies.crush_strategy.strategy import STRATEGY_DIR as CRUSH_DIR, TICK_COST_CENTS, load

BAR_MINUTES = 30
LOOKBACKS = [5, 10, 20, 50]
FEATURES = ["zscore", "distmean", "bollinger", "rsi", "momentum"]
WINDOWS = {"2021-22": ("2021-01-01", "2023-01-01"), "2023-Nov24": ("2023-01-01", "2024-12-01")}


def daily_frozen(bars: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    last = states.groupby(bars["trading_date"].values).last().shift(1)     # state at the previous day's last bar
    out = last.reindex(bars["trading_date"].values)
    out.index = bars.index
    return out


def positions(spread: pd.Series, feature: str, lb: int) -> pd.Series:
    if feature == "zscore":
        mean, std = spread.rolling(lb).mean().shift(1), spread.rolling(lb).std().shift(1)
        return -np.sign((spread - mean) / std)
    if feature == "distmean":
        return -np.sign((spread - spread.rolling(lb).mean()) / spread.rolling(lb).std())
    if feature == "bollinger":
        mean, std = spread.rolling(lb).mean(), spread.rolling(lb).std()
        pct_b = (spread - (mean - 2 * std)) / (4 * std)
        return pd.Series(np.select([pct_b > 0.95, pct_b < 0.05], [-1.0, 1.0], 0.0), index=spread.index).where(pct_b.notna())
    if feature == "rsi":
        change = spread.diff()
        gain, loss = change.clip(lower=0).rolling(14).mean(), (-change.clip(upper=0)).rolling(14).mean()
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        return pd.Series(np.select([rsi > 70, rsi < 30], [-1.0, 1.0], 0.0), index=spread.index).where(rsi.notna())
    if feature == "momentum":
        return np.sign(spread - spread.shift(lb))
    raise ValueError(feature)


if __name__ == "__main__":
    cfg = load(CRUSH_DIR / "config.yaml")
    bars = build_bars(ROOT / cfg.data_path, BAR_MINUTES)
    bars = bars[bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start)]      # holdout stays locked
    states = run_kalman_filter(bars, BAR_MINUTES)
    y, X = design_matrix(bars)
    log = {leg: np.log(bars[f"{leg}_adj_close"]) for leg in ["ZS", "ZM", "ZL"]}
    same_session_next = (bars["session_id"].shift(-1) == bars["session_id"]).values
    session = session_label(bars.index, bars["session_type"])
    cost_frac = TICK_COST_CENTS / bars["ZS_adj_close"]                               # round trip, fraction of bean notional
    rows = []
    for hedge_name, hedge in {"daily (blog)": daily_frozen(bars, states),
                              "session (ours)": freeze_at_previous_session_close(bars, states)}.items():
        spread = pd.Series(y - np.sum(X * hedge.values, axis=1), index=bars.index)
        # hedged return of bar t+1 with the hedge of bar t (the position held over that bar)
        next_ret = ((log["ZS"].shift(-1) - log["ZS"]) - hedge["beta_meal"] * (log["ZM"].shift(-1) - log["ZM"])
                    - hedge["beta_oil"] * (log["ZL"].shift(-1) - log["ZL"])).where(same_session_next)
        for feature, lb in product(FEATURES, LOOKBACKS):
            if feature == "rsi" and lb != LOOKBACKS[0]:
                continue
            pos = positions(spread, feature, lb).where(same_session_next, 0.0).fillna(0.0)   # flat into every break
            turnover = pos.diff().abs().fillna(pos.abs())
            gross = pos * next_ret.fillna(0.0)
            net = gross - turnover * cost_frac / 2
            frame = pd.DataFrame({"gross": gross, "net": net, "turn": turnover, "session": session,
                                  "date": bars["trading_date"]})
            for (w, (a, b)), sess in product(WINDOWS.items(), ["asia", "europe", "us", "all"]):
                f = frame[(frame["date"] >= a) & (frame["date"] < b)]
                if sess != "all":
                    f = f[f["session"] == sess]
                daily = f.groupby("date")[["gross", "net", "turn"]].sum()
                rows.append({"hedge": hedge_name, "feature": feature, "lookback": lb, "window": w, "session": sess,
                             "gross_sharpe": daily["gross"].mean() / daily["gross"].std() * np.sqrt(252),
                             "net_sharpe": daily["net"].mean() / daily["net"].std() * np.sqrt(252),
                             "gross_bps_per_turn": daily["gross"].sum() / max(daily["turn"].sum(), 1) * 1e4,
                             "changes_per_day": daily["turn"].mean()})
        print(f"{hedge_name}: done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(STRATEGY_DIR / "output" / "blog_spread_test.csv", index=False)
    print("saved", save_run(STRATEGY_DIR, "blog_spread_test", cfg, {"rows": len(table)}, tables={"blog_spread_test": table}))
