"""The standardised live stretch (live_stretch.py) WITH the turn condition back, split by the session gap
(development data only).
  stretch  : live gap / expected 1-hour beans move reaches the level (0.95 = about 5 a week, 0.75 = about 10 a week)
  turn     : the first close within 30 minutes of the stretch where the live gap is no wider than the bar before AND
             the fair value has not moved against the trade over the last 15 minutes. Dropped if the gap falls back
             inside half the level first. This is the signal.
  session  : at the turn bar the session gap is 'against', 'same side, small' (under 0.75 x the 1-hour move),
             'same side, large' (0.75 to 1.5) or 'same side, extreme' (1.5 or more, its own largest ~10%).
  outcome  : beans move in the fade direction from the next open, inside the session, ticks before costs.
Reading rule, fixed before the run: a group is a lead only with t >= 2 at one hour AND positive in both halves."""
import contextlib, io, runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategies.bean_divergence.fair_value import crush_hedge

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(__file__.replace("live_stretch_turn.py", "live_stretch.py"))
bars, stretches, score, session_score, live_gap, session_gap = (g[k] for k in ["bars", "stretches", "score", "session_score", "live_gap", "session_gap"])
session_id, session_last, dates, session, half, hour_ct, weeks = (g[k] for k in ["session_id", "session_last", "dates", "session", "half", "hour_ct", "weeks"])
open_s, close_s, high_s, low_s, TICK, LEVELS, out_dir = (g[k] for k in ["open_s", "close_s", "high_s", "low_s", "TICK_CENTS", "LEVELS", "out_dir"])
pd.set_option("display.width", 250)
hedge = crush_hedge(bars, True, 5)
fair_move = ((hedge["beta_meal"] * bars["ZM_ret"].rolling(3).sum() + hedge["beta_oil"] * bars["ZL_ret"].rolling(3).sum())
             * bars["ZS_adj_close"] / TICK).values                                 # fair value's own 15-minute move, ticks
STATES = ["against", "same side, small", "same side, large", "same side, extreme"]


def state(agree: float) -> str:
    return STATES[0] if agree < 0 else STATES[1] if agree < 0.75 else STATES[2] if agree < 1.5 else STATES[3]


def turn_book(level: float) -> pd.DataFrame:
    rows = []
    for start, side in stretches(level):
        for t in range(start, min(start + 7, session_last[start])):               # leaves room for the next open
            if side * score[t] < level / 2:
                break
            if abs(live_gap[t]) <= abs(live_gap[t - 1]) and side * fair_move[t] <= 0:
                entry = t + 1
                row = {"bar": t, "side": side, "date": dates[t], "session": session[t], "half": half[t], "hour": hour_ct[t],
                       "waited": 5 * (t - start), "gap ticks": abs(live_gap[t]), "session gap ticks": side * session_gap[t],
                       "session gap": state(side * session_score[t])}
                for label, bars_ahead in [("30 min", 6), ("1 hour", 12), ("2 hours", 24), ("session end", 10 ** 6)]:
                    row[label] = -side * (close_s[min(t + bars_ahead, session_last[t])] - open_s[entry]) / TICK
                path = -side * (close_s[entry:min(entry + 12, session_last[t] + 1)] - open_s[entry]) / TICK
                row["best 1h"], row["worst 1h"] = max(path.max(), 0), min(path.min(), 0)
                rows.append(row)
                break
    return pd.DataFrame(rows)


def line(part: pd.DataFrame) -> dict:
    def mean_t(column, frame=part):
        daily = frame.groupby("date")[column].sum()
        return f"{frame[column].mean():+.2f} ({daily.mean() / (daily.std() / np.sqrt(len(daily))):+.1f})" if len(daily) > 30 else f"n={len(frame)}"
    one = part["1 hour"]
    return {"a week": round(len(part) / weeks, 1), "live gap / session gap, ticks (median)": f"{part['gap ticks'].median():.0f} / {part['session gap ticks'].median():.0f}",
            "30 min": mean_t("30 min"), "1 hour": mean_t("1 hour"), "2 hours": mean_t("2 hours"), "session end": mean_t("session end"),
            "our way, 1 hour": f"{(one > 0).mean():.0%}", "winner / loser, 1 hour": f"{one[one > 0].mean():+.1f} / {one[one <= 0].mean():+.1f}",
            "best / worst in the hour (median)": f"{part['best 1h'].median():.1f} / {part['worst 1h'].median():.1f}",
            "1 hour, 2021-22": mean_t("1 hour", part[part["half"] == "2021-22"]), "1 hour, 2023-24": mean_t("1 hour", part[part["half"] == "2023-Nov24"])}


books = {name: turn_book(level) for name, level in LEVELS.items()}
for name, book in books.items():
    print(f"\n== live stretch at {LEVELS[name]:.2f} ({name} before the turn) + turn: {len(book) / weeks:.1f} a week, "
          f"median wait {book['waited'].median():.0f} minutes. Beans in the fade direction, ticks before costs, mean (t)")
    table = {"all": line(book)}
    for key in STATES:
        table[key] = line(book[book["session gap"] == key])
    for key in ["asia", "europe", "us"]:
        table[key] = line(book[book["session"] == key])
    print(pd.DataFrame(table).to_string())
    cross = book.groupby(["session", "session gap"])["1 hour"].agg(["size", "mean"])
    cross["a week"] = (cross.pop("size") / weeks).round(2)
    print("-- by session and session-gap state: beans at 1 hour (mean) and signals a week")
    print(cross.round(2).unstack().reindex(columns=STATES, level=1).to_string())

# ---------------------------------------------------------------- charts: turns at the looser level, coloured by the session gap
book = books["about 10 a week"]
strict = set(books["about 5 a week"]["bar"])
COLOURS = {"same side, extreme": "#08306b", "same side, large": "#6baed6", "same side, small": "white", "against": "#d62728"}
trading_dates = np.array(sorted(bars["trading_date"].unique()))
n = len(bars)
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
    top.plot(x, close - live_gap[first:last] * TICK, color="black", lw=1.3, label="live fair value (10x slower)")
    top.plot(x, close - session_gap[first:last] * TICK, color="darkorange", lw=1.3, label="session fair value")
    inside = book[(book["bar"] >= first) & (book["bar"] < last)]
    for key, colour in COLOURS.items():
        part = inside[inside["session gap"] == key]
        for sign, marker in [(1, "v"), (-1, "^")]:
            chosen = part[part["side"] == sign]
            xs = chosen["bar"].values - first
            ys = high[xs] + reach * 1.5 if sign == 1 else low[xs] - reach * 1.5
            sizes = np.where(chosen["bar"].isin(strict), 330, 150)                 # large = also a signal at the stricter level
            top.scatter(xs, ys, marker=marker, s=sizes, color=colour, edgecolor="black", linewidth=1.5, zorder=5,
                        label=f"turn, session gap {key}: {len(part)}" if sign == 1 else None)
    top.set_ylabel("soybeans, cents per bushel")
    top.set_title(f"{pd.Timestamp(days[0]):%d %b %Y} to {pd.Timestamp(days[-1]):%d %b %Y}, 5-minute bars. Live stretch ({LEVELS['about 10 a week']:.2f} x the 1-hour move) "
                  f"then the turn, coloured by the session gap; large marker = also stretched at {LEVELS['about 5 a week']:.2f}. Down = sell, up = buy", loc="left")
    top.legend(loc="upper left", ncol=3)
    bottom.bar(x, score[first:last], width=0.8, color="grey", alpha=0.6, label="live gap / expected 1-hour move")
    bottom.plot(x, session_score[first:last], color="darkorange", lw=1.3, label="session gap / expected 1-hour move")
    for level, style, label in [(LEVELS["about 5 a week"], "--", "live levels (0.95 dashed, 0.75 dotted)"), (LEVELS["about 10 a week"], ":", None), (1.5, "-.", "session extreme (1.5)")]:
        for sign in [1, -1]:
            bottom.axhline(sign * level, color="darkorange" if level == 1.5 else "black", ls=style, lw=1, label=label if sign == 1 else None)
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
    path = out_dir / f"live_turn_{window_start}.png"
    figure.savefig(path, dpi=100)
    plt.close(figure)
    print(path)
