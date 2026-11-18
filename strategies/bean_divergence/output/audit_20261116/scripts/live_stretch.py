"""A SIMPLE live-gap signal, standardised for volatility (development data only). No turn condition, no tick floor.
  score    : live gap (10x slower Kalman) / the expected 1-hour beans move. 1.0 = beans are one typical hour's move
             away from the live fair value, whatever the year or session.
  stretch  : first 5-minute close with |score| >= level; the same side again only after |score| < half the level;
             a new session starts fresh.
  levels   : chosen by FREQUENCY only, before looking at outcomes: the ones giving about 5 and about 10 a week.
  session  : at the stretch bar the session gap is 'against' (other side), 'same side, small' (under 0.75 x the
             1-hour move) or 'same side, large' (0.75 or more). Shown alongside, not used to pick signals.
  outcome  : beans move in the fade direction from the next open, inside the session, ticks before costs.
Descriptive: no pass mark; the question is where the signals fall and whether the session gap tells them apart."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import strategies.crush_strategy.kalman_filter as kalman_filter
from core_quant.config import ROOT
from strategies.bean_divergence.fair_value import (STRATEGY_DIR, TICK_CENTS, adaptive_scale, crush_gap, crush_hedge,
                                                   half_label, load, load_bars, session_label)
from strategies.bean_divergence.triggers import find_triggers

pd.set_option("display.width", 250)
cfg = load()
bars = load_bars(cfg)
in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
session_id, dates = bars["session_id"].values, bars["trading_date"].values
session, half = session_label(bars.index, bars["session_type"]), half_label(bars["trading_date"])
years = bars["trading_date"].dt.year.values
hour_ct = bars.index.tz_convert("America/Chicago").hour.values
open_s, close_s = bars["ZS_adj_open"].values, bars["ZS_adj_close"].values
weeks = bars.loc[in_dev, "trading_date"].nunique() / 5
n = len(bars)
session_last = np.searchsorted(session_id, session_id, side="right") - 1

kalman_filter.DRIFT_TO_NOISE_PER_15_MINUTES = 0.01
live_gap = crush_gap(bars, crush_hedge(bars, True, 5)).values
kalman_filter.DRIFT_TO_NOISE_PER_15_MINUTES = 0.1
session_gap = crush_gap(bars, crush_hedge(bars, False, 5)).values
yard = (adaptive_scale(bars["ZS_ret"] * bars["ZS_adj_close"] / TICK_CENTS, 48) * np.sqrt(12)).values   # expected 1-hour move
score, session_score = np.nan_to_num(live_gap / yard), session_gap / yard


def stretches(level: float) -> list[tuple[int, int]]:
    return [(t, side) for t, side in find_triggers(score, session_id, level, level / 2) if in_dev[t] and t + 1 <= session_last[t]]


grid = {level: len(stretches(level)) / weeks for level in np.round(np.arange(0.4, 1.81, 0.05), 2)}
print("== stretches a week by level (x the expected 1-hour move): " + ", ".join(f"{level:.2f}: {count:.1f}" for level, count in grid.items()))
LEVELS = {name: min(grid, key=lambda level: abs(grid[level] - target)) for name, target in [("about 5 a week", 5), ("about 10 a week", 10)]}

books = {}
for name, level in LEVELS.items():
    rows = []
    for t, side in stretches(level):
        entry, agree = t + 1, side * session_score[t]
        row = {"bar": t, "side": side, "date": dates[t], "year": years[t], "session": session[t], "half": half[t], "hour": hour_ct[t],
               "gap ticks": abs(live_gap[t]), "session gap ticks": side * session_gap[t],
               "session gap": "against" if agree < 0 else "same side, small" if agree < 0.75 else "same side, large"}
        for label, bars_ahead in [("30 min", 6), ("1 hour", 12), ("2 hours", 24), ("session end", 10 ** 6)]:
            row[label] = -side * (close_s[min(t + bars_ahead, session_last[t])] - open_s[entry]) / TICK_CENTS
        path = -side * (close_s[entry:min(entry + 12, session_last[t] + 1)] - open_s[entry]) / TICK_CENTS
        row["best 1h"], row["worst 1h"] = max(path.max(), 0), min(path.min(), 0)
        rows.append(row)
    books[name] = pd.DataFrame(rows)


def line(part: pd.DataFrame) -> dict:
    def mean_t(column):
        daily = part.groupby("date")[column].sum()
        return f"{part[column].mean():+.2f} ({daily.mean() / (daily.std() / np.sqrt(len(daily))):+.1f})" if len(daily) > 30 else f"n={len(part)}"
    one = part["1 hour"]
    return {"a week": round(len(part) / weeks, 1), "live gap, ticks (median)": round(part["gap ticks"].median(), 1),
            "30 min": mean_t("30 min"), "1 hour": mean_t("1 hour"), "2 hours": mean_t("2 hours"), "session end": mean_t("session end"),
            "our way, 1 hour": f"{(one > 0).mean():.0%}", "winner / loser, 1 hour": f"{one[one > 0].mean():+.1f} / {one[one <= 0].mean():+.1f}",
            "best / worst in the hour (median)": f"{part['best 1h'].median():.1f} / {part['worst 1h'].median():.1f}"}


for name, book in books.items():
    print(f"\n== {name}: live gap at {LEVELS[name]:.2f} x the expected 1-hour move. Beans in the fade direction, ticks before costs, mean (t)")
    table = {"all": line(book)}
    for column in ["session", "session gap", "half"]:
        for key, part in book.groupby(column):
            table[key] = line(part)
    print(pd.DataFrame(table).to_string())
    print("-- live gap in ticks at the signal (25% / median / 75%), by session and by year")
    print(book.groupby("session")["gap ticks"].quantile([0.25, 0.5, 0.75]).unstack().round(1).T.to_string())
    print(book.groupby("year")["gap ticks"].quantile([0.25, 0.5, 0.75]).unstack().round(1).T.to_string())
    print("-- signals a week by hour (Chicago time)")
    print((book.groupby("hour").size() / weeks).round(2).to_frame("a week").T.to_string())
    print("-- session-gap state by session: signals a week, and beans at 1 hour")
    cross = book.groupby(["session", "session gap"])["1 hour"].agg(["size", "mean"])
    cross["a week"] = (cross.pop("size") / weeks).round(2)
    print(cross.round(2).unstack().to_string())

# ---------------------------------------------------------------- charts: the ~5 a week signal, coloured by the session gap
book = books["about 5 a week"]
loose = books["about 10 a week"]
minutes = pd.read_parquet(ROOT / cfg.data_path, columns=["ZS_high", "ZS_low", "ZS_close"])
minutes = minutes[minutes.index <= bars.index[-1]]
bar_of_minute = np.searchsorted(bars.index, minutes.index, side="left")
factor = (bars["ZS_adj_close"] / minutes["ZS_close"].reindex(bars.index)).values
high_s = minutes["ZS_high"].groupby(bar_of_minute).max().reindex(range(n)).values * factor
low_s = minutes["ZS_low"].groupby(bar_of_minute).min().reindex(range(n)).values * factor
COLOURS = {"same side, large": "#1f77b4", "same side, small": "white", "against": "#d62728"}
trading_dates = np.array(sorted(bars["trading_date"].unique()))
out_dir = STRATEGY_DIR / "output" / "audit_20261116"
for window_start in ["2021-03-15", "2022-09-12", "2023-06-12", "2024-02-12", "2024-11-21"]:
    days = trading_dates[trading_dates >= pd.Timestamp(window_start)][:3]
    shown = np.flatnonzero(bars["trading_date"].isin(days).values)
    first, last = shown[0], shown[-1] + 1
    x = np.arange(last - first)
    open_, close, high, low = open_s[first:last], close_s[first:last], high_s[first:last], low_s[first:last]
    colours = np.where(close >= open_, "#2ca02c", "#d62728")
    reach = (high.max() - low.min()) * 0.03
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(24, 13), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    top.vlines(x, low, high, color=colours, lw=0.7, alpha=0.6)
    top.bar(x, np.maximum(np.abs(close - open_), 0.05), bottom=np.minimum(open_, close), width=0.7, color=colours, alpha=0.6)
    top.plot(x, close - live_gap[first:last] * TICK_CENTS, color="black", lw=1.3, label="live fair value (10x slower)")
    top.plot(x, close - session_gap[first:last] * TICK_CENTS, color="darkorange", lw=1.3, label="session fair value")
    inside = loose[(loose["bar"] >= first) & (loose["bar"] < last) & ~loose["bar"].isin(book["bar"])]
    for sign, marker in [(1, "v"), (-1, "^")]:
        xs = inside.loc[inside["side"] == sign, "bar"].values - first
        ys = high[xs] + reach if sign == 1 else low[xs] - reach
        top.scatter(xs, ys, marker=marker, s=70, color="#bdbdbd", edgecolor="grey", zorder=4,
                    label=f"looser level ({LEVELS['about 10 a week']:.2f}) only: {len(inside)}" if sign == 1 else None)
    inside = book[(book["bar"] >= first) & (book["bar"] < last)]
    for state, colour in COLOURS.items():
        part = inside[inside["session gap"] == state]
        for sign, marker in [(1, "v"), (-1, "^")]:
            xs = part.loc[part["side"] == sign, "bar"].values - first
            ys = high[xs] + reach * 2.2 if sign == 1 else low[xs] - reach * 2.2
            top.scatter(xs, ys, marker=marker, s=260, color=colour, edgecolor="black", linewidth=1.5, zorder=5,
                        label=f"live stretch, session gap {state}: {len(part)}" if sign == 1 else None)
    top.set_ylabel("soybeans, cents per bushel")
    top.set_title(f"{pd.Timestamp(days[0]):%d %b %Y} to {pd.Timestamp(days[-1]):%d %b %Y}, 5-minute bars. Large triangle = live gap at "
                  f"{LEVELS['about 5 a week']:.2f} x the expected 1-hour move, coloured by the session gap. Down = beans rich (sell), up = cheap (buy)", loc="left")
    top.legend(loc="upper left", ncol=3)
    bottom.bar(x, score[first:last], width=0.8, color="grey", alpha=0.6, label="live gap / expected 1-hour move")
    bottom.plot(x, session_score[first:last], color="darkorange", lw=1.3, label="session gap / expected 1-hour move")
    for name, style in [("about 5 a week", "--"), ("about 10 a week", ":")]:
        for sign in [1, -1]:
            bottom.axhline(sign * LEVELS[name], color="black", ls=style, lw=1, label=f"level, {name}" if sign == 1 else None)
    bottom.axhline(0, color="grey", lw=0.8)
    bottom.set_ylim(-4, 4)
    bottom.set_ylabel("gap in units of the expected 1-hour move")
    bottom.legend(loc="upper left", ncol=4)
    ids = session_id[first:last]
    local = bars.index[first:last].tz_convert("Europe/London")
    ticks = np.arange(0, len(x), 24)
    bottom.set_xticks(ticks, [f"{local[i]:%d %b\n%H:%M}" for i in ticks], fontsize=8)
    for axis in (top, bottom):
        for session_start in np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]):
            axis.axvline(session_start - 0.5, color="grey", lw=0.8, alpha=0.7)
        axis.grid(axis="y", alpha=0.3)
    top.set_xlim(-1, len(x) + 1)
    figure.tight_layout()
    path = out_dir / f"live_stretch_{window_start}.png"
    figure.savefig(path, dpi=100)
    plt.close(figure)
    print(path)
