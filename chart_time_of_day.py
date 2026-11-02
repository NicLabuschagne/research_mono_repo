"""Chart: volatility and gross edge (ticks/trade) by time of day of the trade bar, Chicago time."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from time_of_day import res, FEATURES

BAR, IS_C, OOS_C = "#2a78d6", "#52514e", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NAMES = {"keltner_rev_5": "Keltner reversion (5)", "skew_fade_5": "Skew fade (5)"}

x = np.arange(len(res))
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK2, "axes.titlecolor": INK})
fig, axes = plt.subplots(3, 1, figsize=(15, 9.5), sharex=True, facecolor=SURF,
                         gridspec_kw={"height_ratios": [1.3, 2, 2], "hspace": 0.3})

axes[0].bar(x, res.avg_move_bp, width=0.7, color=INK2)
axes[0].set_title("Average absolute move of the trade bar (bp)", loc="left")
axes[0].set_ylabel("bp")

for ax, f in zip(axes[1:], FEATURES):
    ax.bar(x, res[f"{f}_ticks"], width=0.7, color=BAR, label="full window (Mar 22 – Feb 26)", zorder=2)
    ax.scatter(x - 0.18, res[f"{f}_IS"], s=22, color=IS_C, zorder=3, label="in-sample (Mar 22 – Feb 24)")
    ax.scatter(x + 0.18, res[f"{f}_OOS"], s=22, color=OOS_C, zorder=3, label="out-of-sample (Mar 24 – Feb 26)")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axhline(1, color=INK, lw=1, ls="--")
    ax.text(len(x) - 0.5, 1.05, "1 tick = round-trip cost of crossing the spread", ha="right", va="bottom", fontsize=8, color=INK)
    ax.set_ylim(-1.8, 1.8)
    ax.set_title(f"{NAMES[f]}: gross edge per trade (ticks)", loc="left")
    ax.set_ylabel("ticks / trade")
    ax.legend(loc="lower left", ncol=3, frameon=False, fontsize=8)

# shade the day session (08:30-13:00 CT)
day = [i for i, h in enumerate(res.index) if "08:30" <= h <= "13:00"]
for ax in axes:
    ax.axvspan(day[0] - 0.5, day[-1] + 0.5, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.set_facecolor(SURF)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].text(np.mean(day), axes[0].get_ylim()[1] * 0.95, "day session", ha="center", va="top", fontsize=8, color=INK2)
axes[-1].set_xticks(x, res.index, rotation=90)
axes[-1].set_xlabel("start of trade bar, Chicago time (evening session first)")
fig.suptitle("Soybeans 30m: when does the 5-bar edge exist? (gross, no costs)", x=0.125, ha="left", fontsize=12, color=INK)
fig.savefig("soybean_time_of_day.png", dpi=130, bbox_inches="tight", facecolor=SURF)
