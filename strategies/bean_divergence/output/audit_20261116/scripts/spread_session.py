"""The spread against a SESSION-ANCHORED fair value (development data only). Fixed before the run:
  fair value : the hedge frozen at the previous session's close (the orange line: fair_value.crush_hedge live=False).
               The gap is then the spread itself, measured from where the last session left it.
  entry      : first 5-minute close with |z| >= 1.645 (z = gap / its 20-day size), the same side again only after
               |z| < 0.6745, a new session starts fresh; all three legs at the next bar's open
  exit       : the first close at or through fair value -> next bar's open; else the session's last close
  cost       : none (every leg passive) / 1.1 ticks (beans passive) / 2.1 ticks (all crossed)
  pass       : net at 1.1 ticks > 0 with t >= 2 (daily sums) and positive in both halves."""
import contextlib, io, runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategies.bean_divergence.fair_value import crush_gap, crush_hedge, session_label

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(__file__.replace("spread_session.py", "spread_trades.py"))
bars, in_dev, half, weeks, find_triggers, out_dir = (g[k] for k in ["bars", "in_dev", "half", "weeks", "find_triggers", "out_dir"])
session_id, dates, log_open, log_close, price_ticks = (g[k] for k in ["session_id", "dates", "log_open", "log_close", "price_ticks"])
session = session_label(bars.index, bars["session_type"])
pd.set_option("display.width", 250)

hedge = crush_hedge(bars, False, 5)                                                # frozen at the previous session close
gap_series = crush_gap(bars, hedge)                                                # ZS ticks, + = spread above fair value
gap = gap_series.values
z = (gap_series / np.sqrt((gap_series ** 2).rolling(20 * 211, min_periods=10 * 211).mean()).shift(1)).values
beta_meal, beta_oil = hedge["beta_meal"].values, hedge["beta_oil"].values

rows, busy_until = [], 0
for t, rich in find_triggers(z, session_id, 1.645, 0.6745):
    entry = t + 1
    if not in_dev[t] or entry < busy_until or entry >= len(bars) or session_id[entry] != session_id[t]:
        continue
    held = lambda log, i: log["ZS"][i] - beta_meal[t] * log["ZM"][i] - beta_oil[t] * log["ZL"][i]
    last, reason = entry, "session end"
    while True:
        if rich * gap[last] <= 0:
            reason = "fair value"
            break
        if last + 1 >= len(bars) or session_id[last + 1] != session_id[entry]:
            break
        last += 1
    next_in_session = last + 1 < len(bars) and session_id[last + 1] == session_id[entry]
    at_open = reason == "fair value" and next_in_session
    exit_level = held(log_open, last + 1) if at_open else held(log_close, last)
    busy_until = last + 1
    rows.append({"signal_bar": t, "entry_bar": entry, "exit_bar": last + at_open, "rich": rich, "reason": reason,
                 "gap_at_entry": abs(gap[t]), "gross": -rich * (exit_level - held(log_open, entry)) * price_ticks[t],
                 "date": dates[t], "half": half[t], "session": session[t]})
trades = pd.DataFrame(rows)


def describe(part: pd.DataFrame) -> dict:
    def mean_t(cost):
        net = part["gross"] - cost
        daily = net.groupby(part["date"]).sum()
        return f"{net.mean():+.2f} ({daily.mean() / (daily.std() / np.sqrt(len(daily))):+.1f})"
    net = part["gross"] - 1.1
    by_half = net.groupby(part["half"]).mean()
    timed = part[part["reason"] == "session end"]["gross"]
    return {"trades a week": round(len(part) / weeks, 1), "stretch at entry (ticks)": round(part["gap_at_entry"].mean(), 1),
            "gross, no cost (t)": mean_t(0), "net at 1.1 (t)": mean_t(1.1), "net at 2.1 (t)": mean_t(2.1),
            "won at 1.1": f"{(net > 0).mean():.0%}", "winner / loser at 1.1": f"{net[net > 0].mean():+.1f} / {net[net <= 0].mean():+.1f}",
            "worst trade": round(net.min(), 1), "out at fair value": f"{(part['reason'] == 'fair value').mean():.0%}",
            "gross at fair value / at session end": f"{part[part['reason'] == 'fair value']['gross'].mean():+.1f} / {timed.mean():+.1f}",
            "hours held (median)": round((part["exit_bar"] - part["entry_bar"]).median() / 12, 1),
            "net at 1.1: 2021-22 / 2023-24": f"{by_half.get('2021-22', np.nan):+.2f} / {by_half.get('2023-Nov24', np.nan):+.2f}"}


table = {"all sessions": describe(trades)}
for name, part in trades.groupby("session"):
    table[name] = describe(part)
print("== The spread faded back to the SESSION fair value; ZS ticks per trade; t from daily sums")
print(pd.DataFrame(table).to_string())

# ---------------------------------------------------------------- charts
trading_dates = np.array(sorted(bars["trading_date"].unique()))
for window_start in ["2021-03-15", "2022-09-12", "2023-06-12", "2024-02-12", "2024-11-21"]:
    days = trading_dates[trading_dates >= pd.Timestamp(window_start)][:3]
    shown = np.flatnonzero(bars["trading_date"].isin(days).values)
    first, last = shown[0], shown[-1] + 1
    x = np.arange(last - first)
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(24, 12), sharex=True, gridspec_kw={"height_ratios": [2.4, 1]})
    top.plot(x, gap[first:last], color="black", lw=1.0, label="spread minus its session fair value (ZS ticks)")
    top.axhline(0, color="darkorange", lw=1.6, label="session fair value (hedge frozen at the previous session close)")
    total = 0.0
    for trade in trades[(trades["entry_bar"] >= first) & (trades["exit_bar"] < last)].itertuples():
        net = trade.gross - 1.1
        total += net
        colour = "#2ca02c" if net > 0 else "#d62728"
        xs = [trade.signal_bar - first, trade.exit_bar - 1 - first if trade.reason == "fair value" else trade.exit_bar - first]
        ys = [gap[trade.signal_bar], gap[xs[1] + first]]
        top.plot(xs, ys, color=colour, lw=1.6, ls="--", zorder=4)
        top.scatter(xs[0], ys[0], marker="v" if trade.rich > 0 else "^", s=220, color=colour, edgecolor="black", zorder=5)
        top.scatter(xs[1], ys[1], marker="*" if trade.reason == "fair value" else "X", s=220, color=colour, edgecolor="black", zorder=5)
        top.annotate(f"{net:+.1f}", (xs[1], ys[1]), textcoords="offset points", xytext=(6, 8), fontsize=11)
    top.set_ylabel("ZS ticks: + = spread above fair value")
    top.set_title(f"{pd.Timestamp(days[0]):%d %b %Y} to {pd.Timestamp(days[-1]):%d %b %Y}: the spread against its session fair value, net {total:+.1f} ticks after "
                  f"a 1.1-tick cost. Triangle down = sell the spread, up = buy; star = out at fair value, X = session end; number = net ticks", loc="left")
    top.legend(loc="upper left")
    bottom.plot(x, z[first:last], color="black", lw=0.9, label="z: gap / its 20-day size")
    for level in [1.645, -1.645]:
        bottom.axhline(level, color="grey", ls="--", lw=1)
    bottom.axhline(0, color="grey", lw=0.8)
    bottom.set_ylabel("z")
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
    path = out_dir / f"spread_session_{window_start}.png"
    figure.savefig(path, dpi=100)
    plt.close(figure)
    print(path)

# ---------------------------------------------------------------- check: were all three legs actually trading at the entry?
volume = {leg: bars[f"{leg}_volume"].values for leg in ["ZS", "ZM", "ZL"]}
trades["thinnest_leg_lots"] = [min(volume[leg][bar - 1:bar + 1].sum() for leg in volume) for bar in trades["entry_bar"]]   # signal + entry bar
trades["hour_ct"] = bars.index[trades["signal_bar"].values].tz_convert("America/Chicago").hour
trades["net"] = trades["gross"] - 1.1
print("\n-- net at 1.1 by the lots traded in the thinnest leg over the signal and entry bars (10 minutes)")
bands = pd.cut(trades["thinnest_leg_lots"], [-1, 0, 20, 100, 500, 1e9], labels=["none", "1-20", "21-100", "101-500", "over 500"])
print(trades.groupby(["session", bands], observed=True)["net"].agg(["size", "mean"]).round(2).to_string())
print("\n-- asia trades by hour of the signal (Chicago time)")
print(trades[trades["session"] == "asia"].groupby("hour_ct")["net"].agg(["size", "mean"]).round(2).to_string())
