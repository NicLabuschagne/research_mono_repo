"""
harvest.py
----------
Layer 2: are we harvesting the base signal's edge, before barriers and a meta model? Wheat outright, 1-session hold
(trader's choice), the base rule of kalman_rule.py otherwise. Development data, IN SAMPLE. Saved to runs/<time>_harvest/.

    python -m strategies.wheat_corn_lag.harvest

A  threshold window: the tail cut-offs come from the past W of s. W = 3 / 6 / 12 / 24 months / all past (expanding),
   at 10% and 20% tails, one position at a time. Rule fixed before the run: keep 12 months unless another window
   beats its weekly t by more than 0.5 at BOTH tails (smaller differences are noise).
B  several triggers in one session: every trigger as its own trade (no position limit), numbered within its session
   (1st, 2nd, 3rd+). Is the 2nd trigger as good as the 1st?
C  harvest grid on the chosen window: tails 10 / 20 / 30%; re-arm each session only, or also 36 bars (half a
   session) / 18 bars (4.5 hours) after the last trigger while s stays in the tail; positions open at once
   1 / 2 / 3 / no limit. Rule fixed before the run: the most trades per week with net per trade >= 1.5 c, weekly
   t >= 2.5 and >= 5 of 6 years positive (ties: more cents per week).
   Amended after the first run (2026-11-02): the choice must hold at most 3 positions at once. The first run
   picked "no limit", up to 10 contracts open together (worst week -165 c): not one account trading one contract.
   A risk limit, not a performance choice; the performance criteria are unchanged and "no limit" stays in the grid.
D  the chosen harvest in detail: trades per week, win rate, expected value per trade, by trading session (time of
   day), weekday, side and year, next to the current rule.

Units: cents (1 tick = 0.25 c; 1 c = $50 per ZW contract). net per trade = all trades after costs = the expected
value per trade; average winner / loser = the winning / losing trades only. Weekly P&L = net cents of the trades
that EXIT that week (shortcut: booked at exit, not marked to market; fine for 1-session trades), so trades that
overlap are counted correctly. Weekly t = mean / (sd / sqrt(weeks)), weeks with no trade count as 0.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import save_run
from core_quant.theme import set_theme
from strategies.wheat_corn_lag.features import build_features
from strategies.wheat_corn_lag.kalman_rule import TICK_CENTS, cap_open, cents_row, kalman_signal, tail_triggers, trades_for
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load, load_bars

SESSION_BARS = 71
WINDOWS = {"3 months": 63 * SESSION_BARS, "6 months": 126 * SESSION_BARS, "12 months": 252 * SESSION_BARS,
           "24 months": 504 * SESSION_BARS, "all past": None}
TAILS = [0.10, 0.20, 0.30]
REARMS = {"each session": None, "+ every half session": 36, "+ every 4.5 h": 18}
MAX_OPEN = {"1": 1, "2": 2, "3": 3, "no limit": None}
CURRENT = {"window": "12 months", "tail": 0.10, "rearm": "each session", "max_open": "1"}
TIME_OF_DAY = [(19 * 60, 24 * 60, "1 evening 19:00-24:00 CT (Asia)"),
               (0, 7 * 60 + 45, "2 overnight 00:00-07:45 CT (Europe)"),
               (8 * 60 + 30, 10 * 60 + 30, "3 day open 08:30-10:30 CT"),
               (10 * 60 + 30, 13 * 60 + 20, "4 day 10:30-13:20 CT (to the close)")]


def thresholds(s: pd.Series, window_bars: int | None, tail: float, min_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """Upper and lower tail cut-offs from past bars only (shifted one bar)."""
    if window_bars is None:
        past = s.expanding(min_periods=min_bars)
    else:
        past = s.rolling(window_bars, min_periods=min_bars)
    return past.quantile(1 - tail).shift(1).values, past.quantile(tail).shift(1).values


def weekly_pnl(trades: pd.DataFrame, bars: pd.DataFrame, cost_cents: float) -> pd.Series:
    """Net cents per calendar week (Friday week end), by exit time; weeks with no trade are 0."""
    all_weeks = pd.Series(0.0, index=bars.index).resample("W-FRI").sum().index
    if trades.empty:
        return pd.Series(0.0, index=all_weeks)
    net = pd.Series((trades["move_c"] - cost_cents).values, index=bars.index[trades["exit"].values])
    return net.resample("W-FRI").sum().reindex(all_weeks, fill_value=0.0)


def most_open_at_once(trades: pd.DataFrame, bar_count: int) -> int:
    change = np.zeros(bar_count + 1)
    np.add.at(change, trades["entry"].values, 1)
    np.add.at(change, trades["exit"].values + 1, -1)
    return int(np.cumsum(change).max()) if len(trades) else 0


def harvest_stats(trades: pd.DataFrame, bars: pd.DataFrame, cost_cents: float) -> dict:
    weekly = weekly_pnl(trades, bars, cost_cents)
    equity = weekly.cumsum()
    yearly = (trades["move_c"] - cost_cents).groupby(bars.index[trades["exit"].values].year).sum()
    return {**cents_row(trades, cost_cents), "per_week": len(trades) / len(weekly),
            "cents_per_week": weekly.mean(), "weekly_t": weekly.mean() / (weekly.std() / np.sqrt(len(weekly))),
            "sharpe": weekly.mean() / weekly.std() * np.sqrt(52), "years_positive": int((yearly > 0).sum()),
            "worst_week_c": weekly.min(), "max_drawdown_c": (equity - equity.cummax()).min(),
            "most_open": most_open_at_once(trades, len(bars))}


def time_of_day(times: pd.Series) -> pd.Series:
    """Trading-session bucket of each entry (the entry bar's OPEN time, Chicago)."""
    chicago = pd.DatetimeIndex(times).tz_convert("America/Chicago")
    minutes = chicago.hour * 60 + chicago.minute
    labels = np.full(len(minutes), "5 other", dtype=object)
    for start, end, label in TIME_OF_DAY:
        labels[(minutes >= start) & (minutes < end)] = label
    return pd.Series(labels, index=times.index)


def breakdown(trades: pd.DataFrame, by: pd.Series, weeks: int, cost_cents: float) -> pd.DataFrame:
    rows = []
    for key, part in trades.groupby(by):
        rows.append({"group": key, "per_week": len(part) / weeks, **cents_row(part, cost_cents)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ charts
def window_chart(table: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for axis, column, title in [(axes[0], "weekly_t", "weekly t of net P&L"),
                                (axes[1], "net_per_trade_c", "net per trade, cents")]:
        table.pivot(index="window", columns="tail", values=column).loc[list(WINDOWS)].plot.bar(ax=axis, rot=0)
        axis.set_title(title, loc="left")
        axis.set_xlabel("threshold window")
        axis.axhline(0, color="grey", lw=0.8)
    figure.tight_layout()
    return figure


def rank_chart(table: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10, 4.2))
    labels = [f"{row.design}\n{row.trigger} (n={row.trades})" for row in table.itertuples()]
    x = np.arange(len(table))
    axis.bar(x, table["net_per_trade_c"], label="net per trade")
    axis.scatter(x, table["avg_winner_c"], marker="^", color="#2ca02c", zorder=3, label="average winner")
    axis.scatter(x, table["avg_loser_c"], marker="v", color="#d62728", zorder=3, label="average loser")
    axis.axhline(0, color="grey", lw=0.8)
    axis.set_xticks(x, labels, fontsize=7)
    axis.set_ylabel("cents per trade")
    axis.set_title("1st vs later triggers in the same session (every trigger traded)", loc="left")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def grid_chart(grid: pd.DataFrame, chosen: dict) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for max_open, part in grid.groupby("max_open"):
        axis.scatter(part["per_week"], part["net_per_trade_c"], s=30 + part["tail"] * 300, alpha=0.75,
                     label=f"max open {max_open}")
    pick = grid[(grid["tail"] == chosen["tail"]) & (grid["rearm"] == chosen["rearm"]) &
                (grid["max_open"] == chosen["max_open"])]
    axis.scatter(pick["per_week"], pick["net_per_trade_c"], s=260, facecolors="none", edgecolors="black", lw=1.5,
                 label="chosen (rule fixed before the run)")
    axis.axhline(1.5, color="#d62728", ls=":", lw=0.8)
    axis.axhline(0, color="grey", lw=0.8)
    axis.set_xlabel("trades per week")
    axis.set_ylabel("net per trade, cents")
    axis.set_title("Harvest grid: frequency vs edge per trade (bubble size = tail share)", loc="left")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def session_chart(table: pd.DataFrame, current: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for axis, column, title in [(axes[0], "per_week", "trades per week"),
                                (axes[1], "net_per_trade_c", "net per trade, cents")]:
        both = pd.DataFrame({"chosen": table.set_index("group")[column],
                             "current rule": current.set_index("group")[column]})
        both.index = [label[2:].replace(" CT", "\nCT") for label in both.index]
        both.plot.bar(ax=axis, rot=0)
        axis.axhline(0, color="grey", lw=0.8)
        axis.set_title(f"By entry time: {title}", loc="left")
        axis.tick_params(axis="x", labelsize=7)
    figure.tight_layout()
    return figure


def equity_chart(curves: dict) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(11, 4.4))
    for label, weekly in curves.items():
        axis.plot(weekly.index, weekly.cumsum(), lw=1.6, label=label)
    axis.axhline(0, color="grey", lw=0.8)
    axis.set_ylabel("cumulative net cents, 1 contract per trade")
    axis.set_title("Development equity, in sample", loc="left")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    rule = cfg.kalman_rule
    bars = load_bars(cfg)                                        # development data only
    features = build_features(bars, cfg.features)
    s = kalman_signal(features, rule)
    cost_cents = rule.cost_ticks * TICK_CENTS
    session_id = bars["session"].values
    cache = {}

    def run(window: str, tail: float, rearm: str, max_open: str) -> pd.DataFrame:
        if (window, tail) not in cache:
            cache[(window, tail)] = thresholds(s, WINDOWS[window], tail, rule.quantile_min_bars)
        upper, lower = cache[(window, tail)]
        events = tail_triggers(s.values, upper, lower, session_id, REARMS[rearm])
        return cap_open(trades_for(bars, events, rule.hold_bars), MAX_OPEN[max_open])

    current = run(**CURRENT)
    weeks = len(weekly_pnl(current, bars, cost_cents))
    assert len(current) == 195, "the current rule no longer reproduces kalman_rule.py's 195 development trades"

    # A: threshold window
    rows = []
    for window in WINDOWS:
        for tail in TAILS[:2]:
            rows.append({"window": window, "tail": tail,
                         **harvest_stats(run(window, tail, "each session", "1"), bars, cost_cents)})
    windows = pd.DataFrame(rows)
    t_of = windows.pivot(index="window", columns="tail", values="weekly_t")
    beats = (t_of.sub(t_of.loc["12 months"], axis=1) > 0.5).all(axis=1)
    window = t_of[beats].mean(axis=1).idxmax() if beats.any() else "12 months"

    # B: 1st vs later triggers in one session, every trigger traded
    rows = []
    for tail in TAILS[:2]:
        for rearm in ["each session", "+ every 4.5 h"]:
            every = run(window, tail, rearm, "no limit")
            session_of = pd.Series(session_id[every["entry"].values - 1], index=every.index)
            number = every.groupby(session_of).cumcount() + 1
            for label, part in [("1st", every[number == 1]), ("2nd", every[number == 2]), ("3rd+", every[number >= 3])]:
                if len(part):
                    rows.append({"design": f"{tail:.0%} {rearm}", "trigger": label, **cents_row(part, cost_cents)})
    ranks = pd.DataFrame(rows)

    # C: harvest grid
    rows = []
    for tail in TAILS:
        for rearm in REARMS:
            for max_open in MAX_OPEN:
                rows.append({"tail": tail, "rearm": rearm, "max_open": max_open,
                             **harvest_stats(run(window, tail, rearm, max_open), bars, cost_cents)})
    grid = pd.DataFrame(rows)
    allowed = grid[(grid["net_per_trade_c"] >= 1.5) & (grid["weekly_t"] >= 2.5) & (grid["years_positive"] >= 5)
                   & (grid["max_open"] != "no limit")]
    if allowed.empty:
        chosen = {key: CURRENT[key] for key in ["tail", "rearm", "max_open"]}
    else:
        best = allowed.sort_values(["per_week", "cents_per_week"], ascending=False).iloc[0]
        chosen = {"tail": best["tail"], "rearm": best["rearm"], "max_open": best["max_open"]}

    # D: the chosen harvest in detail, next to the current rule
    trades = run(window, **chosen)
    entry_times = pd.Series(bars.index[trades["entry"].values - 1], index=trades.index)   # = the entry bar's open
    current_times = pd.Series(bars.index[current["entry"].values - 1], index=current.index)
    by_session = breakdown(trades, time_of_day(entry_times), weeks, cost_cents)
    current_by_session = breakdown(current, time_of_day(current_times), weeks, cost_cents)
    weekday = pd.DatetimeIndex(entry_times).tz_convert("America/Chicago").day_name().str[:3]
    by_weekday = breakdown(trades, pd.Series(weekday, index=trades.index), weeks, cost_cents)
    by_side = breakdown(trades, trades["side"].map({1: "long", -1: "short"}), weeks, cost_cents)
    by_year = breakdown(trades, trades["year"], weeks, cost_cents)
    weeks_in_year = weekly_pnl(trades, bars, cost_cents).groupby(lambda week: week.year).size()
    by_year["per_week"] = by_year["trades"] / weeks_in_year.reindex(by_year["group"]).values   # that year's weeks
    summary = pd.DataFrame([{"rule": "current", **harvest_stats(current, bars, cost_cents)},
                            {"rule": f"chosen: {window}, {chosen['tail']:.0%}, {chosen['rearm']}, max open "
                                     f"{chosen['max_open']}", **harvest_stats(trades, bars, cost_cents)}])

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    money = ["trades", "per_week", "net_per_trade_c", "t", "hit_rate", "avg_winner_c", "avg_loser_c",
             "cents_per_week", "weekly_t", "sharpe", "years_positive", "worst_week_c", "max_drawdown_c", "most_open"]
    print(f"cost {cost_cents:.2f} c round trip; {weeks} weeks. net per trade = all trades after costs (expected value);"
          f" average winner / loser = winning / losing trades only; hit_rate = share of trades > 0 after costs")
    print("\n== A: threshold window (each session re-arm, one position)")
    print(windows[["window", "tail"] + money].round(2).to_string(index=False))
    print(f"window choice (rule fixed before the run): {window}")
    print("\n== B: 1st vs later triggers in the same session (every trigger traded, no position limit)")
    print(ranks.round(2).to_string(index=False))
    print("\n== C: harvest grid (sorted by trades per week)")
    print(grid.sort_values("per_week", ascending=False)[["tail", "rearm", "max_open"] + money].round(2)
          .to_string(index=False))
    print(f"choice (rule fixed before the run): {chosen}")
    print("\n== D: current vs chosen")
    print(summary[["rule"] + money].round(2).to_string(index=False))
    for title, table in [("by entry time (trading session)", by_session), ("by weekday (Chicago)", by_weekday),
                         ("by side", by_side), ("by year", by_year)]:
        print(f"\n-- chosen, {title}")
        print(table.round(2).to_string(index=False))
    print("\n-- current rule, by entry time")
    print(current_by_session.round(2).to_string(index=False))

    figures = {"A_window": window_chart(windows), "B_trigger_number": rank_chart(ranks),
               "C_grid": grid_chart(grid, chosen), "D_by_session": session_chart(by_session, current_by_session),
               "D_equity": equity_chart({"current rule": weekly_pnl(current, bars, cost_cents),
                                         "chosen harvest": weekly_pnl(trades, bars, cost_cents)})}
    run_dir = save_run(STRATEGY_DIR, "harvest", cfg, {"window": window, "chosen": chosen},
                       {"A_windows": windows, "B_trigger_number": ranks, "C_grid": grid, "D_summary": summary,
                        "D_by_session": by_session, "D_by_weekday": by_weekday, "D_by_side": by_side,
                        "D_by_year": by_year}, figures)
    print(f"\nsaved to {run_dir}")
