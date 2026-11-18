"""The spread traded BACK TO ITS FAIR VALUE (development data only). Fixed before the run:
  fair value : the spread's own average, memory 30 minutes / 2 hours / 8 hours
  entry      : first 5-minute close with |z| >= 1.645 (z = gap / its 20-day size), the same side again only after
               |z| < 0.6745; all three legs at the next bar's open; hedge ratios of the signal bar held to the exit
  exit       : the first close at or through the fair value of that moment -> next bar's open; else after one
               trading day (211 bars) at that close. One position at a time.
  cost       : shown three ways: none (every leg filled passively, the autospreader ideal), 1.1 ticks (meal and oil
               crossed, beans passive), 2.1 ticks (every leg crossed)
  pass       : net at 1.1 ticks > 0 with t >= 2 (daily sums) and positive in both halves."""
import contextlib, io, runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategies.bean_divergence.triggers import find_triggers

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(__file__.replace("spread_trades.py", "spread_memory.py"))
bars, spread, hedge, in_dev, half, TICK, out_dir = (g[k] for k in ["bars", "spread", "hedge", "in_dev", "half", "TICK_CENTS", "path"])
out_dir = out_dir.parent
session_id, dates = bars["session_id"].values, bars["trading_date"].values
weeks = bars.loc[in_dev, "trading_date"].nunique() / 5
log_open = {leg: np.log(bars[f"{leg}_adj_open"].values) for leg in ["ZS", "ZM", "ZL"]}
log_close = {leg: np.log(bars[f"{leg}_adj_close"].values) for leg in ["ZS", "ZM", "ZL"]}
beta_meal, beta_oil, price_ticks = hedge["beta_meal"].values, hedge["beta_oil"].values, bars["ZS_adj_close"].values / TICK
MAX_HOLD_BARS, TRIGGER_Z, REARM_Z = 211, 1.645, 0.6745
pd.set_option("display.width", 250)


def trade_book(memory_bars: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    fair = spread.ewm(alpha=1.0 / memory_bars, adjust=False).mean().shift(1)
    gap = (spread - fair).values
    z = gap / np.sqrt((pd.Series(gap) ** 2).rolling(20 * 211, min_periods=10 * 211).mean()).shift(1).values
    rows, busy_until = [], 0
    for t, rich in find_triggers(z, session_id, TRIGGER_Z, REARM_Z):
        entry = t + 1
        if not in_dev[t] or entry < busy_until or entry + MAX_HOLD_BARS + 1 >= len(bars):
            continue
        held = lambda log, i: log["ZS"][i] - beta_meal[t] * log["ZM"][i] - beta_oil[t] * log["ZL"][i]   # hedge of bar t
        last, reason = entry, "time"
        while last < entry + MAX_HOLD_BARS:
            if rich * gap[last] <= 0:
                reason = "fair value"
                break
            last += 1
        exit_level = held(log_open, last + 1) if reason == "fair value" else held(log_close, last)
        busy_until = last + 1
        rows.append({"signal_bar": t, "entry_bar": entry, "exit_bar": last + (reason == "fair value"), "rich": rich, "reason": reason,
                     "gap_at_entry": abs(gap[t]), "gross": -rich * (exit_level - held(log_open, entry)) * price_ticks[t],
                     "date": dates[t], "half": half[t]})
    return pd.DataFrame(rows), fair.values, z


def mean_t(frame: pd.DataFrame, cost: float) -> str:
    net = frame["gross"] - cost
    daily = net.groupby(frame["date"]).sum()
    return f"{net.mean():+.2f} ({daily.mean() / (daily.std() / np.sqrt(len(daily))):+.1f})"


books, table = {}, {}
for name, memory_bars in [("30 min", 6), ("2 hours", 24), ("8 hours", 96)]:
    trades, fair, z = trade_book(memory_bars)
    books[name] = (trades, fair, z)
    net = trades["gross"] - 1.1
    by_half = net.groupby(trades["half"]).mean()
    by_reason = trades.groupby("reason")["gross"].agg(["size", "mean"])
    table[f"memory {name}"] = {
        "trades a week": round(len(trades) / weeks, 1), "stretch at entry (ticks)": round(trades["gap_at_entry"].mean(), 1),
        "gross, no cost (t)": mean_t(trades, 0), "net at 1.1 (t)": mean_t(trades, 1.1), "net at 2.1 (t)": mean_t(trades, 2.1),
        "won at 1.1": f"{(net > 0).mean():.0%}", "winner / loser at 1.1": f"{net[net > 0].mean():+.1f} / {net[net <= 0].mean():+.1f}",
        "worst trade": round(net.min(), 1), "out at fair value": f"{(trades['reason'] == 'fair value').mean():.0%}",
        "hours held (median / mean)": f"{(trades['exit_bar'] - trades['entry_bar']).median() / 12:.1f} / {(trades['exit_bar'] - trades['entry_bar']).mean() / 12:.1f}",
        "gross when out at fair value": round(by_reason.loc["fair value", "mean"], 2),
        "gross when timed out": f"{by_reason.loc['time', 'mean']:+.2f} (n={by_reason.loc['time', 'size']})" if "time" in by_reason.index else "none",
        "net at 1.1: 2021-22 / 2023-24": f"{by_half['2021-22']:+.2f} / {by_half['2023-Nov24']:+.2f}"}
print("== The spread faded back to its own fair value; ZS ticks per trade; t from daily sums")
print(pd.DataFrame(table).to_string())

# ---------------------------------------------------------------- charts: the spread, its fair value, the trades
trades, fair, z = books["2 hours"]
spread_v = spread.values
trading_dates = np.array(sorted(bars["trading_date"].unique()))
for window_start in ["2021-03-15", "2022-09-12", "2023-06-12", "2024-02-12", "2024-11-21"]:
    days = trading_dates[trading_dates >= pd.Timestamp(window_start)][:3]
    shown = np.flatnonzero(bars["trading_date"].isin(days).values)
    first, last = shown[0], shown[-1] + 1
    x = np.arange(last - first)
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(24, 12), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})
    top.plot(x, spread_v[first:last], color="black", lw=1.0, label="spread: beans minus the meal and oil hedge (ZS ticks)")
    top.plot(x, fair[first:last], color="#1f77b4", lw=1.5, label="its fair value: own average, memory 2 hours")
    total = 0.0
    for trade in trades[(trades["entry_bar"] >= first) & (trades["exit_bar"] < last)].itertuples():
        net = trade.gross - 1.1
        total += net
        colour = "#2ca02c" if net > 0 else "#d62728"
        xs = [trade.entry_bar - first, trade.exit_bar - first]
        ys = [spread_v[trade.entry_bar - 1], spread_v[trade.exit_bar - 1]]          # an open = the bar before's close
        top.plot(xs, ys, color=colour, lw=1.6, ls="--", zorder=4)
        top.scatter(xs[0], ys[0], marker="v" if trade.rich > 0 else "^", s=220, color=colour, edgecolor="black", zorder=5)
        top.scatter(xs[1], ys[1], marker="*" if trade.reason == "fair value" else "X", s=220, color=colour, edgecolor="black", zorder=5)
        top.annotate(f"{net:+.1f}", (xs[1], ys[1]), textcoords="offset points", xytext=(6, 8), fontsize=11)
    top.set_ylabel("spread, ZS ticks")
    top.set_title(f"{pd.Timestamp(days[0]):%d %b %Y} to {pd.Timestamp(days[-1]):%d %b %Y}: the spread and its trades, net {total:+.1f} ticks after a 1.1-tick cost. "
                  f"Triangle down = sell the spread, up = buy; star = out at fair value, X = timed out; number = net ticks", loc="left")
    top.legend(loc="upper left")
    bottom.plot(x, z[first:last], color="black", lw=0.9, label="z: gap / its 20-day size")
    for level in [TRIGGER_Z, -TRIGGER_Z]:
        bottom.axhline(level, color="grey", ls="--", lw=1)
    bottom.axhline(0, color="grey", lw=0.8)
    bottom.set_ylabel("z: + = spread above fair value")
    bottom.legend(loc="upper left")
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
    path = out_dir / f"spread_trades_{window_start}.png"
    figure.savefig(path, dpi=100)
    plt.close(figure)
    print(path)
