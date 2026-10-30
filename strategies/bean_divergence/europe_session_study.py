"""
europe_session_study.py
-----------------------
Trade only inside the European window of the evening session (02:00 CT to the 07:45 break
before the US session), where the descriptive work shows beans reverting toward fair value.
Data before the window may be used; every trade is closed by 07:45.

Two anchors for the gap:
    session     : hedge frozen at the previous session close (the crush strategy's model)
    europe_open : hedge frozen at the last bar before 02:00 CT; the gap then measures only the
                  dislocation built during Europe. Its z uses the previous 20 days of Europe bars.

Rules otherwise as the crush strategy: first crossing of |z| >= threshold (re-arm 0.5), entry at
the next bar's open, exits at 1 sigma profit / 4 sigma stop / the 07:45 close, one trade at a time,
an entry cutoff (min hours left to 07:45). Each trade scored as the hedged spread (2.3 ticks cost)
and as outright beans (1 tick). Compared with the current strategy and an Asia-only (19:00-02:00)
version. Development only: choose 2021-22, confirm 2023 - Nov 2024; the holdout stays locked.

    python -m strategies.bean_divergence.europe_session_study
"""
from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

from core_quant.config import ROOT, save_run
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.features import add_session_zscore, detect_signals
from strategies.crush_strategy.kalman_filter import design_matrix, freeze_at_previous_session_close, run_kalman_filter
from strategies.crush_strategy.strategy import STRATEGY_DIR as CRUSH_DIR, TICK_COST_CENTS, daily_returns, load, simulate_trades
from strategies.bean_divergence.feature_study import STRATEGY_DIR, TICK_CENTS

WINDOWS = {"2021-22": ("2021-01-01", "2023-01-01"), "2023-Nov24": ("2023-01-01", "2024-12-01")}
EUROPE_START_HOUR = 2
CUTOFFS = [1.0, 2.0, 3.0]
THRESHOLDS = [1.5, 2.0]


def chicago_hour(bars: pd.DataFrame) -> np.ndarray:
    return bars.index.tz_convert("America/Chicago").hour.values


def in_europe(bars: pd.DataFrame) -> np.ndarray:
    hour = chicago_hour(bars)
    return np.isin(bars["session_type"].values, ["evening", "europe"]) & (hour >= EUROPE_START_HOUR) & (hour < 8)


def anchored_bars(raw: pd.DataFrame, states: pd.DataFrame, anchor: str) -> pd.DataFrame:
    """Residual and z under one anchor. For europe_open, Europe bars get their own session type
    ("europe") so the z window and the signal detector use Europe bars only."""
    hedge = freeze_at_previous_session_close(raw, states)
    bars = raw.copy()
    if anchor == "europe_open":
        europe = in_europe(raw)
        start = europe & ~np.r_[False, europe[:-1]]                       # first Europe bar of each night
        state_before = states.shift(1)                                     # state after the previous bar
        anchor_state = pd.DataFrame(np.where(start[:, None], state_before.values, np.nan), index=raw.index, columns=states.columns)
        mask = np.repeat(europe[:, None], states.shape[1], axis=1)
        anchor_state = anchor_state.where(mask).groupby(raw["session_id"].values).ffill()
        hedge = hedge.where(~mask, anchor_state)
        bars["session_type"] = np.where(europe, "europe", bars["session_type"])
    y, X = design_matrix(raw)
    bars = bars.join(hedge.add_prefix("frozen_")).assign(residual=y - np.sum(X * hedge.values, axis=1))
    return add_session_zscore(bars)


def europe_signals(bars: pd.DataFrame, threshold: float, anchor: str) -> pd.DataFrame:
    session = "europe" if anchor == "europe_open" else "evening"
    s = detect_signals(bars, threshold, session=session)
    keep = in_europe(bars)[s["signal_position"].values]
    return s[keep].reset_index(drop=True)


def scores(bars: pd.DataFrame, trades: pd.DataFrame, cost_ticks: float) -> dict:
    out = {}
    if trades.empty:
        return out
    direction = np.where(trades["side"].str.startswith("long"), 1.0, -1.0)
    beans = direction * (np.log(bars["ZS_adj_close"].values[trades["exit_bar"]]) - np.log(bars["ZS_adj_open"].values[trades["entry_bar"]])) * trades["bean_price_cents"].values
    trades = trades.assign(spread_ticks=(trades["gross_cents"] - cost_ticks * TICK_COST_CENTS) / TICK_CENTS,
                           beans_ticks=beans / TICK_CENTS - 1.0)
    dates = pd.DatetimeIndex(sorted(bars["trading_date"].unique()))
    for w, (a, b) in WINDOWS.items():
        t = trades[(pd.to_datetime(trades["signal_date"]) >= a) & (pd.to_datetime(trades["signal_date"]) < b)]
        d = dates[(dates >= a) & (dates < b)]
        daily = daily_returns(t, d, cost_ticks)["ret"]
        out |= {f"{w}_trades": len(t), f"{w}_spread_ticks": t["spread_ticks"].mean(), f"{w}_spread_se": t["spread_ticks"].sem(),
                f"{w}_spread_sharpe": daily.mean() / daily.std() * np.sqrt(252),
                f"{w}_beans_ticks": t["beans_ticks"].mean(), f"{w}_beans_se": t["beans_ticks"].sem(),
                f"{w}_hit_pct": (t["spread_ticks"] > 0).mean() * 100}
    return out


if __name__ == "__main__":
    cfg = load(CRUSH_DIR / "config.yaml")
    raw = build_bars(ROOT / cfg.data_path, cfg.strategy.bar_minutes)
    raw = raw[raw["trading_date"] < pd.Timestamp(cfg.split.holdout_start)]         # holdout stays locked
    states = run_kalman_filter(raw, cfg.strategy.bar_minutes)
    by_anchor = {a: anchored_bars(raw, states, a) for a in ["session", "europe_open"]}

    # self-check: the Europe-open hedge is constant within a night's Europe window and equals the state before 02:00
    eb = by_anchor["europe_open"]
    win = eb[eb["session_type"] == "europe"]
    assert (win.groupby("session_id")["frozen_beta_meal"].nunique(dropna=False) <= 1).all()
    first = win.groupby("session_id").head(1)
    pos = raw.index.get_indexer(first.index)
    assert np.allclose(first["frozen_beta_meal"].values, states["beta_meal"].values[pos - 1])

    rows = []
    base = by_anchor["session"]
    baseline = simulate_trades(base, detect_signals(base, cfg.strategy.threshold), cfg.strategy)
    rows.append({"version": "current strategy (all evening, 3 h cutoff)"} | scores(base, baseline, cfg.strategy.cost_ticks))
    asia = detect_signals(base, cfg.strategy.threshold)
    asia = asia[~in_europe(base)[asia["signal_position"].values]]
    rows.append({"version": "Asia only (19:00-02:00 entries)"} | scores(base, simulate_trades(base, asia, cfg.strategy), cfg.strategy.cost_ticks))
    for anchor, cutoff, threshold in product(["session", "europe_open"], CUTOFFS, THRESHOLDS):
        bars = by_anchor[anchor]
        config = replace(cfg.strategy, threshold=threshold, min_hours_left=cutoff)
        trades = simulate_trades(bars, europe_signals(bars, threshold, anchor), config)
        rows.append({"version": f"Europe only, anchor {anchor}, cutoff {cutoff:g} h, |z| >= {threshold:g}"}
                    | scores(bars, trades, cfg.strategy.cost_ticks))
    table = pd.DataFrame(rows)
    table.to_csv(STRATEGY_DIR / "output" / "europe_session_study.csv", index=False)
    print("saved", save_run(STRATEGY_DIR, "europe_session_study", cfg, {"versions": len(table)}, tables={"europe_session": table}))
