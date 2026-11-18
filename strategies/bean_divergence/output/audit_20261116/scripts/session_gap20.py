"""How often is beans 20 ticks or more from its session fair value, and what happens next? (development data only)
  gap      : beans minus the session fair value (hedge frozen at the previous session close), ZS ticks
  episode  : starts at the first 5-minute close with |gap| >= 20 on one side; that side can start a new episode only
             after the gap has been back inside 10 ticks; a new session starts fresh
  outcome  : the beans move TOWARD fair value from the next open, inside the same session, before costs
Descriptive only: no pass mark, nothing is chosen from it."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategies.bean_divergence.fair_value import (STRATEGY_DIR, TICK_CENTS, crush_gap, crush_hedge, half_label, load,
                                                   load_bars, session_label)
from strategies.bean_divergence.triggers import find_triggers

LEVEL, REARM = 20, 10
pd.set_option("display.width", 250)
cfg = load()
bars = load_bars(cfg)
in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
session_id, dates = bars["session_id"].values, bars["trading_date"].values
session, half = session_label(bars.index, bars["session_type"]), half_label(bars["trading_date"])
years = bars["trading_date"].dt.year.values
open_s, close_s = bars["ZS_adj_open"].values, bars["ZS_adj_close"].values
gap = crush_gap(bars, crush_hedge(bars, False, 5)).values
n = len(bars)
session_last = np.searchsorted(session_id, session_id, side="right") - 1
weeks = bars.loc[in_dev, "trading_date"].nunique() / 5
ok = in_dev & ~np.isnan(gap)

# ---------------------------------------------------------------- how big is the gap
size = np.abs(gap)
print("== 1. Size of the gap to the session fair value, all 5-minute bars (ticks)")
rows = []
for name, chosen in [("all", ok)] + [(str(y), ok & (years == y)) for y in [2021, 2022, 2023, 2024]] + [(s, ok & (session == s)) for s in ["asia", "europe", "us"]]:
    row = {"": name, "median": np.median(size[chosen]), "75%": np.percentile(size[chosen], 75), "90%": np.percentile(size[chosen], 90),
           "95%": np.percentile(size[chosen], 95), "99%": np.percentile(size[chosen], 99)}
    for level in [10, 20, 30, 40]:
        row[f"bars at {level}+"] = f"{(size[chosen] >= level).mean():.0%}"
    rows.append(row)
print(pd.DataFrame(rows).round(1).to_string(index=False))
peak = pd.Series(size[ok]).groupby(session_id[ok]).max()
kind = pd.Series(np.where(session[ok] == "us", "day session", "overnight session")).groupby(session_id[ok]).first()
print("\nlargest gap reached in a session (ticks):")
print(pd.DataFrame({name: {"sessions": len(part), "median": part.median(), "75%": part.quantile(0.75), "90%": part.quantile(0.9),
                           "reach 20+": f"{(part >= 20).mean():.0%}", "reach 30+": f"{(part >= 30).mean():.0%}", "reach 40+": f"{(part >= 40).mean():.0%}"}
                    for name, part in peak.groupby(kind)}).round(1).to_string())

# ---------------------------------------------------------------- episodes
rows = []
for start, rich in find_triggers(np.nan_to_num(gap), session_id, LEVEL, REARM):
    entry = start + 1
    if not in_dev[start] or entry > session_last[start]:
        continue
    end = start
    while end + 1 <= session_last[start] and rich * gap[end + 1] >= REARM:        # the episode lasts until back inside 10
        end += 1
    row = {"date": dates[start], "year": years[start], "session": session[start], "half": half[start],
           "minutes over 10": 5 * (end - start + 1), "peak gap": np.abs(gap[start:end + 1]).max(), "gap at start": abs(gap[start]),
           "back to fair value": bool((rich * gap[start:session_last[start] + 1] <= 0).any())}
    for name, bars_ahead in [("1 hour", 12), ("4 hours", 48), ("session end", 10 ** 6)]:
        stop = min(start + bars_ahead, session_last[start])
        row[name] = -rich * (close_s[stop] - open_s[entry]) / TICK_CENTS
    path = -rich * (close_s[entry:session_last[start] + 1] - open_s[entry]) / TICK_CENTS
    row["best"], row["worst"] = max(path.max(), 0), min(path.min(), 0)
    rows.append(row)
episodes = pd.DataFrame(rows)
print(f"\n== 2. Episodes of 20+ ticks: {len(episodes)} in development, {len(episodes) / weeks:.1f} a week")
print("a week by session: " + ", ".join(f"{name} {len(part) / weeks:.1f}" for name, part in episodes.groupby("session"))
      + " | by year: " + ", ".join(f"{year} {len(part) / (weeks / 3.92):.1f}" for year, part in episodes.groupby("year")))
print(f"minutes spent beyond 10 ticks: median {episodes['minutes over 10'].median():.0f}, 75% {episodes['minutes over 10'].quantile(0.75):.0f}; "
      f"peak gap: median {episodes['peak gap'].median():.0f}, 90% {episodes['peak gap'].quantile(0.9):.0f} ticks; "
      f"back at fair value before the session ends: {episodes['back to fair value'].mean():.0%}")

print("\n== 3. What beans did after the FIRST close at 20+ ticks (move toward fair value, ticks before costs)")
columns = ["1 hour", "4 hours", "session end", "best", "worst"]
summary = episodes[columns].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).drop(index=["count", "min", "max"])
summary.loc["moved our way"] = [(episodes[c] > 0).mean() for c in columns]
summary.loc["winner (mean)"] = [episodes.loc[episodes[c] > 0, c].mean() for c in columns]
summary.loc["loser (mean)"] = [episodes.loc[episodes[c] <= 0, c].mean() for c in columns]
daily = lambda c: episodes.groupby("date")[c].sum()
summary.loc["t (daily sums)"] = [daily(c).mean() / (daily(c).std() / np.sqrt(len(daily(c)))) for c in columns]
print(summary.round(2).to_string())
print("\nmean at 4 hours by session and half:")
print(episodes.groupby("session")["4 hours"].agg(["size", "mean", "median"]).round(2).T.to_string())
print(episodes.groupby("half")["4 hours"].agg(["size", "mean", "median"]).round(2).T.to_string())

# ---------------------------------------------------------------- chart
figure, axes = plt.subplots(1, 3, figsize=(24, 7))
axes[0].hist(np.clip(gap[ok], -60, 60), bins=120, color="darkorange", alpha=0.8)
for level in [-20, 20]:
    axes[0].axvline(level, color="black", ls="--")
axes[0].set_title(f"Gap to the session fair value, every 5-minute bar (ticks, clipped at 60). {(size[ok] >= 20).mean():.0%} of bars are beyond 20", loc="left")
axes[0].set_xlabel("ticks: + = beans above fair value")
axes[1].hist(np.clip(peak, 0, 100), bins=50, color="#3182bd", alpha=0.8)
axes[1].axvline(20, color="black", ls="--")
axes[1].set_title(f"Largest gap reached in each session (ticks). {(peak >= 20).mean():.0%} of sessions reach 20", loc="left")
axes[1].set_xlabel("ticks")
for column, colour in [("1 hour", "#9ecae1"), ("4 hours", "#08306b")]:
    axes[2].hist(np.clip(episodes[column], -80, 80), bins=60, color=colour, alpha=0.6, label=f"{column}: mean {episodes[column].mean():+.1f}, median {episodes[column].median():+.1f}")
axes[2].axvline(0, color="black", lw=0.8)
axes[2].set_title("Beans move toward fair value after the first close at 20+ ticks", loc="left")
axes[2].set_xlabel("ticks before costs (clipped at 80)")
axes[2].legend()
for axis in axes:
    axis.grid(axis="y", alpha=0.3)
figure.tight_layout()
path = STRATEGY_DIR / "output" / "audit_20261116" / "session_gap20.png"
figure.savefig(path, dpi=100)
print(path)

# ---------------------------------------------------------------- 4: is the pull stronger further out? every bar, by gap size
print("\n== 4. Every bar by gap size: beans move toward fair value from the next open (mean by day, t), and the share of the gap that closed")
t_index = np.arange(n)
rows = []
for low, high in [(10, 20), (20, 30), (30, 40), (40, 1e9)]:
    chosen = ok & (size >= low) & (size < high) & (t_index + 1 <= session_last)
    row = {"gap (ticks)": f"{low}-{high:g}" if high < 1e9 else f"{low}+", "share of bars": f"{chosen.sum() / ok.sum():.1%}", "average gap": round(size[chosen].mean(), 1)}
    for name, bars_ahead in [("1 hour", 12), ("4 hours", 48), ("session end", 10 ** 6)]:
        stop = np.minimum(t_index + bars_ahead, session_last)
        move = -np.sign(gap) * (close_s[stop] - open_s[np.minimum(t_index + 1, n - 1)]) / TICK_CENTS
        by_day = pd.Series(move[chosen]).groupby(dates[chosen]).mean()
        row[name] = f"{by_day.mean():+.2f} ({by_day.mean() / (by_day.std() / np.sqrt(len(by_day))):+.1f})"
        row[f"{name}: our way"] = f"{(move[chosen] > 0).mean():.0%}"
        if name == "4 hours":
            row["gap closed by beans in 4 hours"] = f"{by_day.mean() / size[chosen].mean():.0%}"
            for key in ["2021-22", "2023-Nov24"]:
                part = pd.Series(move[chosen & (half == key)]).groupby(dates[chosen & (half == key)]).mean()
                row[f"4 hours {key}"] = round(part.mean(), 2)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
