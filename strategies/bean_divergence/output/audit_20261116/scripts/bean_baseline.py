"""BASELINE (development data only): fade a divergence of beans from a fair value, exit back at that value.
Fixed before the run:
  entry   one trigger per move (triggers.find_triggers): first bar with |z| >= 1.645, same side again only after
          |z| < 0.6745, a new session starts fresh. Signal at a 5-minute close, fill at the next bar's open.
  exit    the first bar that CLOSES at or through the exit fair value -> next bar's open; else the session's last close.
  book    outright beans, one position at a time, 1 tick round trip.
  z       gap / its typical size over the previous 20 trading days (live: the 4-hour z in use).
  aligned a live trigger counts only when the slow z is on the same side and >= 1.
  pass    net per trade > 0 with t >= 2, positive in both halves, at least 2 trades a week."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import ROOT
from strategies.bean_divergence.fair_value import (STRATEGY_DIR, TICK_CENTS, adaptive_scale, crush_gap, crush_hedge,
                                                   half_label, load, load_bars, session_label)
from strategies.bean_divergence.triggers import find_triggers

pd.set_option("display.width", 250)
cfg = load()
r = cfg.research
bars = load_bars(cfg)
in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
bars_per_day = 211
window = 20 * bars_per_day
session = session_label(bars.index, bars["session_type"])
half = half_label(bars["trading_date"])
session_id = bars["session_id"].values
open_s, close_s = bars["ZS_adj_open"].values, bars["ZS_adj_close"].values
log_s, log_m, log_o = (np.log(bars[f"{leg}_adj_close"]) for leg in ["ZS", "ZM", "ZL"])
weeks = bars.loc[in_dev, "trading_date"].nunique() / 5


def slow_z(gap: pd.Series) -> pd.Series:
    return gap / np.sqrt((gap ** 2).rolling(window, min_periods=window // 2).mean()).shift(1)


# the fair values
live_gap = crush_gap(bars, crush_hedge(bars, True, 5))
session_gap = crush_gap(bars, crush_hedge(bars, False, 5))
r_s, r_m, r_o = bars["ZS_ret"], bars["ZM_ret"], bars["ZL_ret"]
cov_mm, cov_oo, cov_mo = r_m.rolling(window).var(), r_o.rolling(window).var(), r_m.rolling(window).cov(r_o)
cov_sm, cov_so = r_s.rolling(window).cov(r_m), r_s.rolling(window).cov(r_o)
determinant = cov_mm * cov_oo - cov_mo ** 2
beta_meal = ((cov_sm * cov_oo - cov_so * cov_mo) / determinant).shift(1)
beta_oil = ((cov_so * cov_mm - cov_sm * cov_mo) / determinant).shift(1)
own = (log_s.diff() - beta_meal * log_m.diff() - beta_oil * log_o.diff()).fillna(0).cumsum()


def anchored_gap(memory_bars: int) -> pd.Series:
    gap_log = own - own.ewm(alpha=1.0 / memory_bars, adjust=False).mean().shift(1)
    return (gap_log * bars["ZS_adj_close"] / TICK_CENTS).where(beta_meal.notna())


gaps = {"live": live_gap, "anchored 2h": anchored_gap(24), "anchored 8h": anchored_gap(96), "session": session_gap}
zs = {name: slow_z(gap) for name, gap in gaps.items()}
zs["live"] = live_gap / adaptive_scale(live_gap, r.z_half_life_bars)


def triggers_of(name: str, aligned_with: str | None = None) -> list[tuple[int, int]]:
    events = find_triggers(zs[name].values, session_id, r.trigger_z, r.rearm_z)
    if aligned_with is None:
        return events
    slow = zs[aligned_with].values
    return [(t, side) for t, side in events if side * slow[t] >= 1]           # same side and at least 1


def simulate(events: list[tuple[int, int]], exit_gap: np.ndarray) -> pd.DataFrame:
    """One row per trade taken. side +1 = beans rich -> we SELL. move_ticks is before costs."""
    rows, busy_until = [], 0
    for t, rich in events:
        entry = t + 1
        if entry >= len(bars) or session_id[entry] != session_id[t] or entry < busy_until or not in_dev[t]:
            continue
        last, reason = entry, "session end"
        while True:
            if rich * exit_gap[last] <= 0:                                    # closed at or through fair value
                reason = "fair value"
                break
            if last + 1 >= len(bars) or session_id[last + 1] != session_id[entry]:
                break
            last += 1
        next_in_session = last + 1 < len(bars) and session_id[last + 1] == session_id[entry]
        fill = open_s[last + 1] if reason == "fair value" and next_in_session else close_s[last]
        busy_until = last + 1
        rows.append({"signal_bar": t, "entry_bar": entry, "exit_bar": last, "rich": rich, "exit_reason": reason,
                     "move_ticks": -rich * (fill - open_s[entry]) / TICK_CENTS, "exit_price": fill,
                     "session": session[t], "half": half[t]})
    return pd.DataFrame(rows)


def summary(trades: pd.DataFrame) -> dict:
    net = trades["move_ticks"] - r.cost_ticks
    return {"trades": len(net), "per_week": len(net) / weeks, "net_per_trade": net.mean(),
            "t": net.mean() / (net.std() / np.sqrt(len(net))), "win_rate": (net > 0).mean(),
            "avg_winner": net[net > 0].mean(), "avg_loser": net[net <= 0].mean(), "worst": net.min(),
            "minutes_held": 5 * (trades["exit_bar"] - trades["entry_bar"] + 1).mean(),
            "at_fair_value": (trades["exit_reason"] == "fair value").mean(),
            "gap_at_entry": np.nan}


RULES = {  # name: (trigger fair value, aligned with, exit fair value)
    "1 live": ("live", None, "live"),
    "2 anchored 2h": ("anchored 2h", None, "anchored 2h"),
    "3 anchored 8h": ("anchored 8h", None, "anchored 8h"),
    "4 session": ("session", None, "session"),
    "5 live + session aligned, exit live": ("live", "session", "live"),
    "6 live + session aligned, exit session": ("live", "session", "session"),
    "7 live + anchored 8h aligned, exit 8h": ("live", "anchored 8h", "anchored 8h"),
}
books, rows, split_rows = {}, [], []
for name, (trigger, aligned_with, exit_at) in RULES.items():
    trades = simulate(triggers_of(trigger, aligned_with), gaps[exit_at].values)
    books[name] = trades
    row = {"rule": name, **summary(trades)}
    row["gap_at_entry"] = np.abs(gaps[exit_at].values[trades["signal_bar"].values]).mean()
    rows.append(row)
    split = {"rule": name}
    for column in ["half", "session"]:
        for key, part in trades.groupby(column):
            s = summary(part)
            split[key] = f"{s['net_per_trade']:+.2f} ({s['t']:+.1f}, {s['per_week']:.1f}/wk)"
    split_rows.append(split)

print(f"== development {bars.index[in_dev][0]:%Y-%m} to {bars.index[-1]:%Y-%m}, ZS ticks AFTER a {r.cost_ticks:g}-tick cost "
      f"(net per trade = all trades; average winner / loser = winners / losers only)")
print(pd.DataFrame(rows).set_index("rule").round(2).T.to_string())
print("\n-- net per trade (t, trades a week) by half and by entry session")
print(pd.DataFrame(split_rows).set_index("rule").to_string())
for name in ["4 session", "6 live + session aligned, exit session"]:
    by_exit = books[name].groupby(["session", "exit_reason"])["move_ticks"].agg(["size", "mean"])
    by_exit["mean"] -= r.cost_ticks
    print(f"\n-- {name}: by entry session and exit (trades, net per trade)")
    print(by_exit.round(2).to_string())

# ---------------------------------------------------------------- chart: rule 6 on the window the user marked up
trades = books["6 live + session aligned, exit session"]
days = sorted(bars["trading_date"].unique())[-6:-3]
shown = np.flatnonzero(bars["trading_date"].isin(days).values)
first, last = shown[0], shown[-1] + 1
part = bars.iloc[first:last]
minutes = pd.read_parquet(ROOT / cfg.data_path, columns=["ZS_high", "ZS_low", "ZS_close"])
minutes = minutes[(minutes.index > bars.index[first - 1]) & (minutes.index <= part.index[-1])]
bar_of_minute = np.searchsorted(part.index, minutes.index, side="left")
factor = (part["ZS_adj_close"] / minutes["ZS_close"].reindex(part.index)).values
high = minutes["ZS_high"].groupby(bar_of_minute).max().values * factor
low = minutes["ZS_low"].groupby(bar_of_minute).min().values * factor
open_, close = open_s[first:last], close_s[first:last]
x = np.arange(last - first)
colours = np.where(close >= open_, "#2ca02c", "#d62728")
figure, (top, bottom) = plt.subplots(2, 1, figsize=(22, 12), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
top.vlines(x, low, high, color=colours, lw=0.7, alpha=0.6)
top.bar(x, np.maximum(np.abs(close - open_), 0.05), bottom=np.minimum(open_, close), width=0.7, color=colours, alpha=0.6)
top.plot(x, close - live_gap.values[first:last] * TICK_CENTS, color="black", lw=1.2, label="live fair value")
top.plot(x, close - session_gap.values[first:last] * TICK_CENTS, color="darkorange", lw=1.2, label="session fair value")
net_total = 0.0
for trade in trades[(trades["entry_bar"] >= first) & (trades["entry_bar"] < last)].itertuples():
    net = trade.move_ticks - r.cost_ticks
    net_total += net
    colour = "#1f77b4" if net > 0 else "#7f0000"
    entry_x, exit_x = trade.entry_bar - first, trade.exit_bar - first
    top.plot([entry_x, exit_x], [open_s[trade.entry_bar], trade.exit_price], color=colour, lw=1.4, ls="--", zorder=4)
    top.scatter(entry_x, open_s[trade.entry_bar], marker="v" if trade.rich > 0 else "^", s=170, color=colour, edgecolor="black", zorder=5)
    top.scatter(exit_x, trade.exit_price, marker="*" if trade.exit_reason == "fair value" else "X", s=150, color=colour, edgecolor="black", zorder=5)
    top.annotate(f"{net:+.1f}", (exit_x, trade.exit_price), textcoords="offset points", xytext=(5, 6), fontsize=10)
top.set_ylabel("soybeans, cents per bushel (roll-adjusted)")
top.set_title(f"Rule 6 (live trigger with the session z aligned, exit at the session fair value), {days[0]:%d %b} to {days[-1]:%d %b %Y}: "
              f"net {net_total:+.1f} ticks. Triangle = entry (down = sell), star = exit at fair value, X = session end, number = net ticks", loc="left")
top.legend(loc="upper left")
bottom.plot(x, zs["live"].values[first:last], color="black", lw=1.0, label="live z")
bottom.plot(x, zs["session"].values[first:last], color="darkorange", lw=1.2, label="session z")
for level in [1.645, -1.645, 1, -1]:
    bottom.axhline(level, color="grey", ls="--" if abs(level) > 1 else ":", lw=1)
bottom.axhline(0, color="grey", lw=0.8)
bottom.set_ylim(-5, 5)
bottom.set_ylabel("z: + = beans above fair value (rich)")
bottom.legend(loc="upper left")
ids = session_id[first:last]
starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
local = part.index.tz_convert("Europe/London")
bottom.set_xticks(starts, [f"{local[i]:%d %b %H:%M}" for i in starts])
for axis in (top, bottom):
    for session_start in starts:
        axis.axvline(session_start - 0.5, color="grey", lw=0.5, alpha=0.6)
    axis.grid(axis="y", alpha=0.3)
top.set_xlim(-1, len(x) + 1)
figure.tight_layout()
path = STRATEGY_DIR / "output" / "audit_20261116" / "baseline_rule6_trades.png"
figure.savefig(path, dpi=110)
print(f"\nchart: {path}")
