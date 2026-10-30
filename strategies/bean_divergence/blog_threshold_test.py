"""
blog_threshold_test.py
----------------------
The blog's spread construction, traded DISCRETELY instead of holding a position every bar:

    30-minute bars, Kalman hedge frozen at the previous trading day's last bar (constant through
    the whole evening session), rolling z of the spread over `lb` bars with lagged mean / std.
    The window's past spread is recomputed with TODAY's hedge, so a hedge change does not put a
    jump into the statistics (nothing after the signal bar is used).
    Entry : first bar with |z| >= k, Asia or Europe only (no US trades), next bar's open,
            hedge held fixed, one trade at a time
    Exit  : on a bar close once |z| <= 0.5, or at the evening session's close (07:45 CT)
    Cost  : 1 tick per leg round trip (0.58 c/bu = 2.3 ZS ticks)
Reported by entry session (Asia 19:00-02:00, Europe 02:00-07:45) and half; development only.

    python -m strategies.bean_divergence.blog_threshold_test
"""
from itertools import product

import numpy as np
import pandas as pd

from core_quant.config import ROOT, save_run
from strategies.bean_divergence.failure_study import session_label
from strategies.bean_divergence.feature_study import STRATEGY_DIR, TICK_CENTS
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.kalman_filter import design_matrix, run_kalman_filter
from strategies.crush_strategy.strategy import STRATEGY_DIR as CRUSH_DIR, TICK_COST_CENTS, load

LOOKBACKS = [10, 20, 50]
ENTRY_Z = [2.0, 2.5]
EXIT_Z = 0.5
WINDOWS = {"2021-22": ("2021-01-01", "2023-01-01"), "2023-Nov24": ("2023-01-01", "2024-12-01")}


def rolling_z_with_todays_hedge(bars: pd.DataFrame, states: pd.DataFrame, lb: int) -> tuple[pd.Series, pd.DataFrame]:
    """z for every bar of day d, from the spread of the last `lb` bars recomputed with day d's hedge."""
    y, X = design_matrix(bars)
    dates = bars["trading_date"].values
    day_last = states.groupby(dates).last().shift(1)                 # hedge for day d = state at day d-1's last bar
    hedge = day_last.reindex(dates)
    hedge.index = bars.index
    z = np.full(len(bars), np.nan)
    starts = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    ends = np.r_[starts[1:], len(bars)]
    for a, b in zip(starts, ends):
        h = hedge.values[a]
        if np.isnan(h).any():
            continue
        lo = max(0, a - lb - 1)
        s = y[lo:b] - X[lo:b] @ h                                    # spread history under today's hedge
        s = pd.Series(s)
        mean, std = s.rolling(lb).mean().shift(1), s.rolling(lb).std().shift(1)
        z[a:b] = ((s - mean) / std).values[a - lo:]
    return pd.Series(z, index=bars.index), hedge


def trades_for(bars: pd.DataFrame, z: pd.Series, hedge: pd.DataFrame, k: float, entry_sessions: set) -> pd.DataFrame:
    lo_, lc = {l: np.log(bars[f"{l}_adj_open"].values) for l in ["ZS", "ZM", "ZL"]}, {l: np.log(bars[f"{l}_adj_close"].values) for l in ["ZS", "ZM", "ZL"]}
    session = bars["session_id"].values
    last = pd.Series(np.arange(len(bars))).groupby(session).transform("max").values
    label = session_label(bars.index, bars["session_type"])
    zv, rows, t = z.values, [], 0
    while t < len(bars) - 1:
        if label[t] in entry_sessions and abs(zv[t]) >= k and t + 1 <= last[t] and session[t + 1] == session[t]:
            direction, e = -np.sign(zv[t]), t + 1
            a, bm, bo = hedge.values[t]
            x = next((i for i in range(e, last[t] + 1) if abs(zv[i]) <= EXIT_Z), last[t])
            resid = lambda P, i: P["ZS"][i] - (a + bm * P["ZM"][i] + bo * P["ZL"][i])
            gross = direction * (resid(lc, x) - resid(lo_, e)) * bars["ZS_adj_close"].values[t]
            rows.append({"signal_date": bars["trading_date"].values[t], "entry_session": label[t],
                         "net_ticks": (gross - TICK_COST_CENTS) / TICK_CENTS, "bars_held": x - e + 1,
                         "exit": "z" if abs(zv[x]) <= EXIT_Z else "close"})
            t = x + 1
        else:
            t += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg = load(CRUSH_DIR / "config.yaml")
    bars = build_bars(ROOT / cfg.data_path, 30)
    bars = bars[bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start)]        # holdout stays locked
    states = run_kalman_filter(bars, 30)
    rows = []
    for lb in LOOKBACKS:
        z, hedge = rolling_z_with_todays_hedge(bars, states, lb)
        if lb == LOOKBACKS[0]:          # self-check: z of bar i does not change when later bars change
            i = len(bars) // 2
            alt = bars.copy(); alt.iloc[i + 1:, alt.columns.get_loc("ZS_adj_close")] *= 1.05
            z2, _ = rolling_z_with_todays_hedge(alt, run_kalman_filter(alt, 30), lb)
            assert np.allclose(z.iloc[: i + 1].values, z2.iloc[: i + 1].values, equal_nan=True)
        for k, sessions in product(ENTRY_Z, [{"asia"}, {"europe"}, {"asia", "europe"}]):
            tr = trades_for(bars, z, hedge, k, sessions)
            for w, (a, b) in WINDOWS.items():
                t = tr[(pd.to_datetime(tr["signal_date"]) >= a) & (pd.to_datetime(tr["signal_date"]) < b)]
                rows.append({"lookback_bars": lb, "entry_z": k, "entry_sessions": "+".join(sorted(sessions)), "window": w,
                             "trades": len(t), "net_ticks": t["net_ticks"].mean(), "se": t["net_ticks"].sem(),
                             "hit_pct": (t["net_ticks"] > 0).mean() * 100, "median_hold_h": t["bars_held"].median() / 2,
                             "pct_exit_at_close": (t["exit"] == "close").mean() * 100})
    table = pd.DataFrame(rows)
    table.to_csv(STRATEGY_DIR / "output" / "blog_threshold_test.csv", index=False)
    print("saved", save_run(STRATEGY_DIR, "blog_threshold_test", cfg, {"rows": len(table)}, tables={"blog_threshold": table}))
