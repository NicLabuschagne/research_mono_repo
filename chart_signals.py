"""Plot 30m soybean bars with keltner_rev_5 / skew_fade_5 and the trades they trigger.
usage: chart_signals.py [start_date] [end_date]   (default: last 3 trading days)"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import _atr

LONG, SHORT = "#2a78d6", "#eb6834"  # diverging pair: blue = long, orange = short
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FEATS = {"keltner_rev_5": "Keltner reversion (5)", "skew_fade_5": "Skew fade (5)"}

df = pd.read_parquet("soybean_30m_features.parquet")
df["ma5"] = df.close.rolling(5).mean()
df["atr5"] = _atr(df.high, df.low, df.close, 5)
# thresholds for "extreme" = full-sample 10th/90th percentile (display only, not a trading rule)
thr = {f: df[f].quantile([0.1, 0.9]).values for f in FEATS}
nxt_ret = (df.close - df.open).shift(-1)  # P&L of the trade a bar-t signal triggers, in price units

days = sorted(set(df.index.date))
a = pd.Timestamp(sys.argv[1], tz="UTC") if len(sys.argv) > 1 else pd.Timestamp(days[-4], tz="UTC")
b = pd.Timestamp(sys.argv[2], tz="UTC") + pd.Timedelta("1D") if len(sys.argv) > 2 else pd.Timestamp(days[-1], tz="UTC")
w = df[(df.index >= a) & (df.index < b)]
x = np.arange(len(w))  # bar index, so overnight gaps don't leave blank space

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK})
fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True, facecolor=SURF,
                         gridspec_kw={"height_ratios": [3, 1.2, 3, 1.2], "hspace": 0.22})


def candles(ax):
    up = w.close >= w.open
    ax.vlines(x, w.low, w.high, color=INK2, lw=0.8, zorder=2)
    ax.bar(x, (w.close - w.open).abs().clip(lower=1e-4), bottom=np.minimum(w.open, w.close),
           width=0.6, color=np.where(up, "#ffffff", INK2), edgecolor=INK2, lw=0.8, zorder=3)
    ax.plot(x, w.ma5, color=INK, lw=1.2, label="MA(5)", zorder=1)
    ax.fill_between(x, w.ma5 - w.atr5, w.ma5 + w.atr5, color=INK2, alpha=0.08, lw=0, label="MA(5) ± ATR(5)")


for i, (f, name) in enumerate(FEATS.items()):
    pax, sax = axes[2 * i], axes[2 * i + 1]
    candles(pax)
    lo, hi = thr[f]
    # the trade happens on the bar AFTER the signal: enter at its open, exit at its close
    for j in range(len(w) - 1):
        s = w[f].iloc[j]
        if not (s <= lo or s >= hi):
            continue
        long_ = s > 0
        win = nxt_ret.loc[w.index[j]] * (1 if long_ else -1) > 0
        y = w.low.iloc[j + 1] - 0.4 * w.atr5.iloc[j + 1] if long_ else w.high.iloc[j + 1] + 0.4 * w.atr5.iloc[j + 1]
        c = LONG if long_ else SHORT
        pax.scatter(x[j + 1], y, marker="^" if long_ else "v", s=70, zorder=4,
                    facecolor=c if win else SURF, edgecolor=c, lw=1.5)
    pax.set_title(f"{name}: extreme-decile signals → trade on the next bar", loc="left", fontsize=10)
    pax.set_ylabel("price ($/bu)")
    # legend proxies
    for m, c, lab in [("^", LONG, "long (signal ≥ 90th pct)"), ("v", SHORT, "short (signal ≤ 10th pct)")]:
        pax.scatter([], [], marker=m, s=70, facecolor=c, edgecolor=c, label=lab + ", filled = win")
    pax.scatter([], [], marker="^", s=70, facecolor=SURF, edgecolor=INK2, label="hollow = loss")
    pax.legend(loc="upper left", ncol=3, frameon=False, fontsize=8)

    sax.bar(x, w[f], width=0.7, color=np.where(w[f] >= 0, LONG, SHORT), zorder=2)
    for t in (lo, hi):
        sax.axhline(t, color=INK2, lw=0.8, ls="--", zorder=1)
    sax.axhline(0, color=INK2, lw=0.6)
    sax.set_ylabel("signal")
    sax.text(1.0, 1.02, "sign rule trades every bar: blue = long next bar, orange = short · dashed = 10th/90th pct",
             transform=sax.transAxes, ha="right", fontsize=8, color=INK2)

for ax in axes:
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
ticks = x[::6]
axes[-1].set_xticks(ticks, [w.index[k].strftime("%b %d\n%H:%M") for k in ticks])
axes[-1].set_xlabel("UTC (overnight and weekend gaps removed)")
fig.suptitle(f"Soybeans 30m, {w.index[0]:%d %b %Y} to {w.index[-1]:%d %b %Y}", x=0.125, ha="left", fontsize=12, color=INK)
fig.savefig("soybean_30m_signals.png", dpi=130, bbox_inches="tight", facecolor=SURF)
print(len(w), "bars,", {f: int(((w[f] <= thr[f][0]) | (w[f] >= thr[f][1])).sum()) for f in FEATS}, "extreme signals")
