"""Gross edge of the 5-bar signals by time of day of the TRADE bar (t+1), in Chicago time."""
import numpy as np
import pandas as pd

from backtest import run, TICK

FEATURES = ("keltner_rev_5", "skew_fade_5")
SPLIT = "2024-03-01"  # blog's IS/OOS split

df = pd.read_parquet("soybean_30m_features.parquet").loc["2022-03-01":"2026-02-28 23:59"]
# CBOT sessions are fixed in Chicago time, so UTC slots would drift with daylight saving
trade_bar_ct = df.index.to_series().shift(-1).dt.tz_convert("America/Chicago").dt.strftime("%H:%M")
move = (np.log(df.close) - np.log(df.open)).shift(-1).abs() * 1e4
tick_bp = TICK / df.open.shift(-1) * 1e4

cols = {"n": move.groupby(trade_bar_ct).size(), "avg_move_bp": move.groupby(trade_bar_ct).mean(),
        "tick_bp": tick_bp.groupby(trade_bar_ct).mean()}
for f in FEATURES:
    g = run(df, df[f]) * 1e4
    slot = trade_bar_ct.loc[g.index]
    grp = g.groupby(slot)
    cols[f"{f}_bp"] = grp.mean()
    cols[f"{f}_ticks"] = (g / tick_bp.loc[g.index]).groupby(slot).mean()  # per-trade ticks, then averaged
    cols[f"{f}_t"] = grp.mean() / grp.std() * np.sqrt(grp.size())  # 1-bar trades don't overlap
    gt = g / tick_bp.loc[g.index]
    cols[f"{f}_IS"] = gt[g.index < SPLIT].groupby(slot[g.index < SPLIT]).mean()  # in ticks
    cols[f"{f}_OOS"] = gt[g.index >= SPLIT].groupby(slot[g.index >= SPLIT]).mean()
res = pd.DataFrame(cols)
res = res[res.n >= 200]  # drop rare slots (e.g. odd session edges)
res = res.loc[sorted(res.index, key=lambda h: (h < "17:00", h))]  # session order: evening open first
res.round(3).to_csv("soybean_time_of_day.csv")

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    show = ["n", "avg_move_bp"] + [f"{f}_{k}" for f in FEATURES for k in ("bp", "ticks", "t", "IS", "OOS")]
    print(res[show].round(2).to_string())
