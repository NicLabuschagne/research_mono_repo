"""The stretch + turn signals (live_stretch_turn.py, both levels) with the exit AT THE LIVE FAIR VALUE, no time stop
(development data only). Fixed before the run:
  entry  : the open of the bar after the turn (market), beans outright
  exit   : at every close a limit rests at that bar's live fair value (10x slower Kalman) for the next bar. Filled at
           the limit when the next bar trades through it by a tick, or at the open if it opens beyond. The trade is
           held for as long as it takes, through session closes and reopens.
  cost   : 1 tick a trade
  pass   : net > 0 with t >= 2 (daily sums) and positive in both halves.
'Worst point' is how far the trade went against us before the exit (from 5-minute highs and lows)."""
import contextlib, io, runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(__file__.replace("live_exit.py", "live_stretch_turn.py"))
bars, books, live_gap, session_gap, STATES, LEVELS = (g[k] for k in ["bars", "books", "live_gap", "session_gap", "STATES", "LEVELS"])
open_s, close_s, high_s, low_s, TICK, session_id, weeks, out_dir = (g[k] for k in ["open_s", "close_s", "high_s", "low_s", "TICK", "session_id", "weeks", "out_dir"])
n, COST = len(bars), 1.0
fair = np.round((close_s - live_gap * TICK) / TICK) * TICK                         # live fair value on the tick grid
pd.set_option("display.width", 250)


def trade(t: int, side: int) -> dict:
    """side +1 = beans rich -> short. Walk forward from the signal bar until the resting limit at fair value fills."""
    position, entry = -side, t + 1
    bar = t
    while bar + 1 < n:
        target, nxt = fair[bar], bar + 1
        if nxt > entry and position * (open_s[nxt] - target) >= 0:                 # opened at or beyond fair value
            price = open_s[nxt]
            break
        if high_s[nxt] >= target + TICK if position > 0 else low_s[nxt] <= target - TICK:
            price = target
            break
        bar += 1
    else:
        return {}
    against = (low_s[entry:nxt + 1] - open_s[entry]).min() if position > 0 else (open_s[entry] - high_s[entry:nxt + 1]).min()
    return {"exit_bar": nxt, "exit_price": price, "gross": position * (price - open_s[entry]) / TICK, "worst point": min(against / TICK, 0),
            "hours": (nxt - entry + 1) / 12, "same session": session_id[nxt] == session_id[entry]}


def line(part: pd.DataFrame) -> dict:
    net = part["gross"] - COST
    daily = net.groupby(part["date"]).sum()
    by_half = net.groupby(part["half"]).mean()
    return {"a week": round(len(part) / weeks, 1), "before cost": round(part["gross"].mean(), 2),
            "net (t)": f"{net.mean():+.2f} ({daily.mean() / (daily.std() / np.sqrt(len(daily))):+.1f})" if len(daily) > 30 else f"{net.mean():+.2f} (n={len(part)})",
            "won": f"{(net > 0).mean():.0%}", "winner / loser": f"{net[net > 0].mean():+.1f} / {net[net <= 0].mean():+.1f}",
            "worst trade": round(net.min()), "worst 5% of trades": round(net.quantile(0.05), 1),
            "worst point (median / worst 10%)": f"{part['worst point'].median():.0f} / {part['worst point'].quantile(0.1):.0f}",
            "hours held (median / 90% / longest)": f"{part['hours'].median():.1f} / {part['hours'].quantile(0.9):.1f} / {part['hours'].max():.0f}",
            "out in the same session": f"{part['same session'].mean():.0%}",
            "net 2021-22 / 2023-24": f"{by_half.get('2021-22', np.nan):+.2f} / {by_half.get('2023-Nov24', np.nan):+.2f}"}


results = {}
for name, book in books.items():
    outcome = pd.DataFrame([trade(t, side) for t, side in zip(book["bar"], book["side"])], index=book.index)
    book = pd.concat([book, outcome], axis=1).dropna(subset=["gross"])
    results[name] = book
    table = {"all": line(book)}
    for key in STATES:
        table[key] = line(book[book["session gap"] == key])
    for key in ["asia", "europe", "us"]:
        table[key] = line(book[book["session"] == key])
    print(f"\n== live stretch at {LEVELS[name]:.2f} + turn, exit at the LIVE fair value with no time stop. ZS ticks, net of {COST:g} tick")
    print(pd.DataFrame(table).to_string())
    held = pd.cut(book["hours"], [0, 0.5, 1, 2, 4, 8, 1e9], labels=["under 30 min", "30-60 min", "1-2 hours", "2-4 hours", "4-8 hours", "over 8 hours"])
    by_time = (book["gross"] - COST).groupby(held, observed=True).agg(["size", "mean"])
    by_time["share of trades"] = (by_time.pop("size") / len(book)).map("{:.0%}".format)
    print("-- net by how long the trade took to reach fair value")
    print(by_time.round(2).T.to_string())

# ---------------------------------------------------------------- charts: the looser level
book = results["about 10 a week"]
trading_dates = np.array(sorted(bars["trading_date"].unique()))
for window_start in ["2021-03-15", "2022-09-12", "2023-06-12", "2024-02-12", "2024-11-21"]:
    days = trading_dates[trading_dates >= pd.Timestamp(window_start)][:3]
    shown = np.flatnonzero(bars["trading_date"].isin(days).values)
    first, last = shown[0], shown[-1] + 1
    x = np.arange(last - first)
    open_, close, high, low = open_s[first:last], close_s[first:last], high_s[first:last], low_s[first:last]
    colours = np.where(close >= open_, "#2ca02c", "#d62728")
    figure, axis = plt.subplots(figsize=(24, 11))
    axis.vlines(x, low, high, color=colours, lw=0.7, alpha=0.5)
    axis.bar(x, np.maximum(np.abs(close - open_), 0.05), bottom=np.minimum(open_, close), width=0.7, color=colours, alpha=0.5)
    axis.plot(x, fair[first:last], color="black", lw=1.3, label="live fair value (10x slower)")
    axis.plot(x, close - session_gap[first:last] * TICK, color="darkorange", lw=1.2, label="session fair value")
    total, count = 0.0, 0
    for row in book[(book["bar"] >= first) & (book["bar"] < last)].itertuples():
        net = row.gross - COST
        total, count = total + net, count + 1
        colour = "#1f77b4" if net > 0 else "#7f0000"
        entry_x, exit_x = row.bar + 1 - first, min(int(row.exit_bar) - first, len(x) - 1)
        axis.plot([entry_x, exit_x], [open_s[row.bar + 1], row.exit_price], color=colour, lw=1.6, ls="--", zorder=4)
        axis.scatter(entry_x, open_s[row.bar + 1], marker="v" if row.side > 0 else "^", s=220, color=colour, edgecolor="black", zorder=5)
        axis.scatter(exit_x, row.exit_price, marker="*" if row.exit_bar < last else ">", s=220, color=colour, edgecolor="black", zorder=5)
        axis.annotate(f"{net:+.1f}", (exit_x, row.exit_price), textcoords="offset points", xytext=(6, 8), fontsize=11)
    ids = session_id[first:last]
    for session_start in np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]):
        axis.axvline(session_start - 0.5, color="grey", lw=0.8, alpha=0.7)
    local = bars.index[first:last].tz_convert("Europe/London")
    ticks = np.arange(0, len(x), 24)
    axis.set_xticks(ticks, [f"{local[i]:%d %b\n%H:%M}" for i in ticks], fontsize=8)
    axis.set_xlim(-1, len(x) + 1)
    axis.set_ylabel("soybeans, cents per bushel")
    axis.grid(axis="y", alpha=0.3)
    axis.set_title(f"{pd.Timestamp(days[0]):%d %b %Y} to {pd.Timestamp(days[-1]):%d %b %Y}: stretch ({LEVELS['about 10 a week']:.2f}) + turn, out at the live fair value, no time stop. "
                   f"{count} trades, net {total:+.1f} ticks. Triangle = entry (down = sell), star = exit, arrow = exit after this window, number = net ticks", loc="left")
    axis.legend(loc="upper left")
    figure.tight_layout()
    path = out_dir / f"live_exit_{window_start}.png"
    figure.savefig(path, dpi=100)
    plt.close(figure)
    print(path)
