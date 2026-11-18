"""AUDIT, second look (statistics only): the user's discretionary read - beans stretched away from BOTH the live
fair value and the session fair value - and a chart of the two fair values on 5-minute candles.
Fixed before the run: live tail = |live z| >= 1.645 (the 4-hour z in use). Session z = session gap / its 20-day RMS.
  aligned  : session z on the same side as the live z and |session z| >= 1
  neutral  : same side, |session z| < 1
  opposed  : session z on the other side"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import ROOT
from strategies.bean_divergence.fair_value import (STRATEGY_DIR, TICK_CENTS, adaptive_scale, crush_gap, crush_hedge,
                                                   half_label, load, load_bars, session_label)

pd.set_option("display.width", 250)
cfg = load()
bars = load_bars(cfg)
in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
session = session_label(bars.index, bars["session_type"])
half = half_label(bars["trading_date"])
dates = bars["trading_date"]
log_s = np.log(bars["ZS_adj_close"])
to_ticks = bars["ZS_adj_close"] / TICK_CENTS
window = 20 * 211

live_gap = crush_gap(bars, crush_hedge(bars, True, 5))
session_gap = crush_gap(bars, crush_hedge(bars, False, 5))
live_z = live_gap / adaptive_scale(live_gap, cfg.research.z_half_life_bars)
session_z = session_gap / np.sqrt((session_gap ** 2).rolling(window, min_periods=window // 2).mean()).shift(1)

tail = (live_z.abs() >= 1.645).values & in_dev & session_z.notna().values
same_side = (np.sign(live_z) == np.sign(session_z)).values
groups = {"every live tail": tail, "aligned (session z >= 1 the same way)": tail & same_side & (session_z.abs() >= 1).values,
          "neutral (same side, session z < 1)": tail & same_side & (session_z.abs() < 1).values,
          "opposed (session z the other way)": tail & ~same_side}


def fade(mask: np.ndarray, bars_ahead: int) -> str:
    """Mean beans move in the fade direction (ticks, before costs), t from daily sums."""
    move = (log_s.shift(-bars_ahead) - log_s) * to_ticks
    chosen = mask & move.notna().values
    signed = (-np.sign(live_z) * move)[chosen]
    daily = signed.groupby(dates[chosen]).sum()
    return f"{signed.mean():+.2f} ({daily.mean() / (daily.std() / np.sqrt(len(daily))):+.1f})"


rows = []
for name, mask in groups.items():
    row = {"bars": name, "share of tails": mask.sum() / tail.sum()}
    row.update({f"{h * 5}m": fade(mask, h) for h in [3, 6, 12, 24, 48]})
    row.update({label: fade(mask & extra, 12) for label, extra in [
        ("60m 2021-22", half == "2021-22"), ("60m 2023-Nov24", half == "2023-Nov24"),
        ("60m asia", session == "asia"), ("60m europe", session == "europe"), ("60m us", session == "us")]})
    rows.append(row)
print("== live tails split by the session fair value: beans move in the FADE direction, ticks before costs (t)")
print(pd.DataFrame(rows).round(2).T.to_string(header=False))

# ---------------------------------------------------------------- chart: three trading days, 5-minute candles
days = sorted(bars["trading_date"].unique())[-6:-3]                       # before the Thanksgiving week's holiday
shown = bars["trading_date"].isin(days).values
part = bars[shown]
minutes = pd.read_parquet(ROOT / cfg.data_path, columns=["ZS_high", "ZS_low", "ZS_close"])
minutes = minutes[(minutes.index > bars.index[np.flatnonzero(shown)[0] - 1]) & (minutes.index <= part.index[-1])]
bar_of_minute = np.searchsorted(part.index, minutes.index, side="left")   # a bar is stamped with its last minute
factor = (part["ZS_adj_close"] / minutes["ZS_close"].reindex(part.index)).values      # roll adjustment of that bar
high = minutes["ZS_high"].groupby(bar_of_minute).max().values * factor
low = minutes["ZS_low"].groupby(bar_of_minute).min().values * factor
open_, close = part["ZS_adj_open"].values, part["ZS_adj_close"].values
x = np.arange(len(part))
colours = np.where(close >= open_, "#2ca02c", "#d62728")

figure, (top, bottom) = plt.subplots(2, 1, figsize=(22, 12), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
top.vlines(x, low, high, color=colours, lw=0.7, alpha=0.7)
top.bar(x, np.maximum(np.abs(close - open_), 0.05), bottom=np.minimum(open_, close), width=0.7, color=colours, alpha=0.7)
top.plot(x, close - live_gap[shown].values * TICK_CENTS, color="black", lw=1.3, label="live fair value (Kalman, hedge from the bar before)")
top.plot(x, close - session_gap[shown].values * TICK_CENTS, color="darkorange", lw=1.3, label="session fair value (hedge frozen at the prior session close)")
top.set_ylabel("soybeans, cents per bushel (roll-adjusted)")
top.set_title(f"{days[0]:%d %b %Y} to {days[-1]:%d %b %Y}, 5-minute bars: beans against the two fair values", loc="left")
top.legend(loc="upper left")
bottom.plot(x, live_z[shown].values, color="black", lw=1.0, label="live z (gap / its 4-hour size)")
bottom.plot(x, session_z[shown].values, color="darkorange", lw=1.0, label="session z (gap / its 20-day size)")
for level in [1.645, -1.645]:
    bottom.axhline(level, color="grey", ls="--", lw=1)
bottom.axhline(0, color="grey", lw=0.8)
bottom.set_ylim(-5, 5)
bottom.set_ylabel("z: + = beans above fair value (rich)")
bottom.legend(loc="upper left")
session_ids = part["session_id"].values
starts = np.flatnonzero(np.r_[True, session_ids[1:] != session_ids[:-1]])
local = part.index.tz_convert("Europe/London")
bottom.set_xticks(starts, [f"{local[i]:%d %b %H:%M}" for i in starts])
for axis in (top, bottom):
    for session_start in starts:
        axis.axvline(session_start - 0.5, color="grey", lw=0.5, alpha=0.6)
    axis.grid(axis="y", alpha=0.3)
top.set_xlim(-1, len(x) + 1)
figure.tight_layout()
out_dir = STRATEGY_DIR / "output" / "audit_20261116"
out_dir.mkdir(exist_ok=True)
figure.savefig(out_dir / "fair_values_5min.png", dpi=110)
print(f"\nchart: {out_dir / 'fair_values_5min.png'}")
