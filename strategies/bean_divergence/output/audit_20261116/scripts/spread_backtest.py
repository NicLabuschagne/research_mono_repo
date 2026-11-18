"""BACKTEST of the raw session-anchored spread signal, no filters (development data, Jan 2021 - Nov 2024), and how
it relates to the outright beans alert.
  spread rule : fair value = hedge frozen at the previous session close. Enter at the first close with |z| >= 1.5
                (z = gap / its 20-day size; same side again only after |z| < 0.615; new session starts fresh), all
                legs at the next open. Exit at the first close back at fair value (next open) or the session's last
                close. One position at a time, one beans contract hedged with meal and oil.
  costs       : 1.1 ticks a trade (beans passive) is the main line; 0 and 2.1 alongside.
  outright    : the beans alert as drawn (bean_spread.py: live 10x slower, stretched 8% of bars, session agreeing,
                full turn), scored as the beans move 60 minutes after the next open less 1 tick.
P&L is booked on the trading date of the entry (every trade ends inside its session)."""
import contextlib, io, runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategies.bean_divergence.fair_value import STRATEGY_DIR, crush_gap, crush_hedge

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(__file__.replace("spread_backtest.py", "bean_spread.py"))
bars, in_dev, session_id, half, dates, weeks, session = (g[k] for k in ["bars", "in_dev", "session_id", "half", "dates", "weeks", "session"])
open_s, close_s, TICK, find_triggers = g["open_s"], g["close_s"], g["TICK"], g["find_triggers"]
outright_events = g["found"]["beans yardstick, gap + fair value turn (as drawn)"][0]
Z_LEVEL, REARM, COST = 1.5, 0.615, 1.1
pd.set_option("display.width", 250)

hedge = crush_hedge(bars, False, 5)
gap_series = crush_gap(bars, hedge)
gap = gap_series.values
z = (gap_series / np.sqrt((gap_series ** 2).rolling(20 * 211, min_periods=10 * 211).mean()).shift(1)).values
beta_meal, beta_oil = hedge["beta_meal"].values, hedge["beta_oil"].values
log_open = {leg: np.log(bars[f"{leg}_adj_open"].values) for leg in ["ZS", "ZM", "ZL"]}
log_close = {leg: np.log(bars[f"{leg}_adj_close"].values) for leg in ["ZS", "ZM", "ZL"]}

rows, busy_until = [], 0
for t, rich in find_triggers(z, session_id, Z_LEVEL, REARM):
    entry = t + 1
    if not in_dev[t] or entry < busy_until or entry >= len(bars) or session_id[entry] != session_id[t]:
        continue
    held = lambda log, i: log["ZS"][i] - beta_meal[t] * log["ZM"][i] - beta_oil[t] * log["ZL"][i]
    last = entry
    while rich * gap[last] > 0 and last + 1 < len(bars) and session_id[last + 1] == session_id[entry]:
        last += 1
    reached = rich * gap[last] <= 0 and last + 1 < len(bars) and session_id[last + 1] == session_id[entry]
    exit_level = held(log_open, last + 1) if reached else held(log_close, last)
    busy_until = last + 1
    rows.append({"bar": t, "rich": rich, "date": pd.Timestamp(dates[t]), "session": session[t], "session_id": session_id[t],
                 "gross": -rich * (exit_level - held(log_open, entry)) * close_s[t] / TICK, "reached": reached})
trades = pd.DataFrame(rows)
trades["net"] = trades["gross"] - COST
all_days = pd.DatetimeIndex(sorted(bars.loc[in_dev, "trading_date"].unique()))


def daily(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("date")[column].sum().reindex(all_days, fill_value=0.0)


def performance(frame: pd.DataFrame, column: str = "net") -> dict:
    pnl = daily(frame, column)
    curve = pnl.cumsum()
    values = frame[column]
    return {"trades": len(frame), "a week": round(len(frame) / (len(all_days) / 5), 1) if len(frame) > 200 else "",
            "net per trade": round(values.mean(), 2), "total ticks": round(values.sum()), "won": f"{(values > 0).mean():.0%}",
            "winner / loser": f"{values[values > 0].mean():+.1f} / {values[values <= 0].mean():+.1f}",
            "Sharpe (yearly)": round(pnl.mean() / pnl.std() * np.sqrt(252), 2),
            "worst drawdown (ticks)": round((curve - curve.cummax()).min()), "worst day": round(pnl.min()), "best day": round(pnl.max())}


table = {"all, cost 1.1": performance(trades), "all, no cost": performance(trades, "gross"),
         "all, cost 2.1": performance(trades.assign(net=trades["gross"] - 2.1))}
for name, part in trades.groupby("session"):
    table[name] = performance(part)
print("== Backtest, session-anchored spread, z 1.5, one contract, ZS ticks (1 tick = $12.50 a contract). Cost 1.1 unless stated")
print(pd.DataFrame(table).to_string())
by_year = {}
for year, part in trades.groupby(trades["date"].dt.year):
    pnl = daily(part, "net")[all_days.year == year]
    by_year[year] = {"trades": len(part), "net per trade": round(part["net"].mean(), 2), "total ticks": round(part["net"].sum()),
                     "won": f"{(part['net'] > 0).mean():.0%}", "Sharpe": round(pnl.mean() / pnl.std() * np.sqrt(252), 2),
                     "overnight total": round(part.loc[part["session"] != "us", "net"].sum()), "us total": round(part.loc[part["session"] == "us", "net"].sum())}
print("\n-- by year, cost 1.1")
print(pd.DataFrame(by_year).to_string())
monthly = daily(trades, "net").groupby(all_days.to_period("M")).sum()
print(f"\nmonths: {len(monthly)}, positive {(monthly > 0).mean():.0%}, best {monthly.max():+.0f}, worst {monthly.min():+.0f} ticks")

# ---------------------------------------------------------------- the outright alert against the spread signal
rows = []
for t, rich in outright_events:
    entry = t + 1
    if not in_dev[t] or entry >= len(bars) or session_id[entry] != session_id[t]:
        continue
    end = min(entry + 11, np.searchsorted(session_id, session_id[entry], side="right") - 1)
    rows.append({"bar": t, "rich": rich, "date": pd.Timestamp(dates[t]), "session": session[t], "session_id": session_id[t],
                 "net": -rich * (close_s[end] - open_s[entry]) / TICK - 1.0})
outright = pd.DataFrame(rows)


def match(own: pd.DataFrame, other: pd.DataFrame) -> pd.Series:
    """For each row of `own`: 'same side', 'opposite side' or 'none' - did `other` signal in the same session?"""
    sides = other.groupby("session_id")["rich"].agg(set)
    result = []
    for row in own.itertuples():
        seen = sides.get(row.session_id, set())
        result.append("same side" if row.rich in seen else "opposite side" if seen else "none")
    return pd.Series(result, index=own.index)


trades["outright"], outright["spread"] = match(trades, outright), match(outright, trades)
print(f"\n== Spread signal ({len(trades) / weeks:.1f} a week) against the outright beans alert ({len(outright) / weeks:.1f} a week)")
print("-- spread trades, by whether the outright alert fired in the same session (net at 1.1)")
print(trades.groupby("outright")["net"].agg(trades="size", net="mean", won=lambda v: (v > 0).mean()).round(2).to_string())
print("-- outright alerts, by whether the spread signal fired in the same session (beans, 60 minutes, net of 1 tick)")
print(outright.groupby("spread")["net"].agg(alerts="size", net="mean", won=lambda v: (v > 0).mean()).round(2).to_string())
both = trades.merge(outright, on=["session_id", "rich"], suffixes=("_spread", "_outright"))
lag = (both["bar_outright"] - both["bar_spread"]) * 5
print(f"when both fire the same way in a session ({len(both)} pairs): the outright alert comes a median {lag.median():.0f} minutes after the spread signal "
      f"(spread first in {(lag > 0).mean():.0%})")
spread_daily, outright_daily = daily(trades, "net"), daily(outright, "net")
active = (spread_daily != 0) & (outright_daily != 0)
weekly = pd.DataFrame({"spread": spread_daily, "outright": outright_daily}).groupby(all_days.to_period("W")).sum()
print(f"correlation of daily P&L: all days {spread_daily.corr(outright_daily):+.2f}; days both traded ({active.sum()}) "
      f"{spread_daily[active].corr(outright_daily[active]):+.2f}; weekly {weekly['spread'].corr(weekly['outright']):+.2f}")
live_gap = g["gap"].values
ok = in_dev & ~np.isnan(z) & ~np.isnan(live_gap)
print(f"correlation of the gaps themselves, all bars: session gap with the live beans gap {np.corrcoef(gap[ok], live_gap[ok])[0, 1]:+.2f}")

# ---------------------------------------------------------------- chart
figure, axes = plt.subplots(2, 2, figsize=(24, 13))
(curve_axis, session_axis), (drawdown_axis, month_axis) = axes
for label, column_cost, colour in [("no cost (every leg passive)", 0.0, "#9ecae1"), ("cost 1.1 (beans passive)", 1.1, "black"), ("cost 2.1 (all crossed)", 2.1, "#d62728")]:
    curve_axis.plot(all_days, daily(trades.assign(net=trades["gross"] - column_cost), "net").cumsum(), color=colour, lw=2 if column_cost == 1.1 else 1.2, label=label)
curve_axis.plot(all_days, outright_daily.cumsum(), color="#2ca02c", lw=1.2, ls="--", label="outright beans alert, 60-minute hold, 1 tick")
curve_axis.axvline(pd.Timestamp("2023-01-01"), color="grey", ls=":")
curve_axis.set_title("Cumulative P&L, ticks, one contract", loc="left")
curve_axis.legend()
for name, colour in [("asia", "#1f77b4"), ("europe", "#2ca02c"), ("us", "#d62728")]:
    session_axis.plot(all_days, daily(trades[trades["session"] == name], "net").cumsum(), color=colour, lw=1.6, label=f"signal in {name}")
session_axis.set_title("Cumulative P&L by session of the signal, cost 1.1", loc="left")
session_axis.legend()
curve = spread_daily.cumsum()
drawdown_axis.fill_between(all_days, curve - curve.cummax(), 0, color="#d62728", alpha=0.5)
drawdown_axis.set_title("Drawdown from the high, ticks, cost 1.1", loc="left")
month_axis.bar(monthly.index.to_timestamp(), monthly.values, width=20, color=np.where(monthly.values > 0, "#2ca02c", "#d62728"))
month_axis.set_title("P&L by month, ticks, cost 1.1", loc="left")
for axis in axes.ravel():
    axis.axhline(0, color="grey", lw=0.8)
    axis.grid(axis="y", alpha=0.3)
figure.tight_layout()
path = STRATEGY_DIR / "output" / "audit_20261116" / "spread_session_backtest.png"
figure.savefig(path, dpi=100)
print(path)
