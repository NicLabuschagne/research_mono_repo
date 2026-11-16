"""
kalman_rule.py
--------------
The adopted base rule (2026-11-02) and its walk-forward. Params: kalman_rule in config.yaml, grid: kalman_grid.

    python -m strategies.wheat_corn_lag.kalman_rule      # walk-forward on the development period, gates, cents report
    python -m strategies.wheat_corn_lag.kalman_rule --holdout   # holdout A, ONCE, after a passing walk-forward

Signal    s = -kzr_{M}s: the Kalman level gap (wheat vs alpha + beta x corn, live) / its 20-session RMS around zero.
          + = wheat cheap vs corn -> buy wheat. Layer 1: IC 0.13 at 1 session, deciles in order.
Tradable  only the bars Layer 1 scored: model A's three lookbacks must be warm too. A warms up again after every
          possible roll, so this leaves out the ~6 sessions after a roll window, where the Kalman IC was 0.07.
Triggers  long when s >= the (1 - tail) quantile of the past quantile_window_bars of s, short when s <= the tail
          quantile (both shifted one bar: the thresholds use bars before t only). Re-arm on leaving the tail,
          at every new session when per_session, and rearm_bars after the last trigger while s stays in the tail.
Book      at most max_open positions at once: a trigger while that many trades are open is skipped.
Fill      signal at bar t's close, entry at bar t+1's open; exit at the close hold_bars later, or at the last bar
          before a possible roll. Costs cost_ticks x 0.25 c round trip.
Exits     optional target / safety stop in multiples of the Kalman gap at entry (target_gap, stop_gap; 0 = off):
          target = limit at the level (or a better open), stop = at the level or a gap-through open, 1 tick worse;
          both touched in one 15-minute bar -> the stop. An early exit frees its slot for the next trigger.
Minimum   min_gap_cents: skip a trigger whose Kalman gap at entry (cents at the entry price) is smaller; it never
          takes a slot.
P&L       log return of wheat notional, marked to market bar by bar, summed per session (session_returns):
          the unit the walk-forward scores. Next to it, in CENTS: net per trade (all trades, after costs) and
          average winner / average loser (the size of the winning / losing trades only).
"""
import sys
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.walk_forward import run_holdout, run_validation, score
from strategies.wheat_corn_lag.features import build_features
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, KalmanRuleConfig, load, load_bars

A_LOOKBACKS = (106, 142, 213)
TICK_CENTS = 0.25
TAILS_CENTS = [5, 10]                       # win / loss tails reported next to the net (cents)
STOP_SLIPPAGE_CENTS = TICK_CENTS


def tail_triggers(s: np.ndarray, upper: np.ndarray, lower: np.ndarray, reset_id: np.ndarray,
                  rearm_bars: int | None = None) -> list[tuple[int, int]]:
    """(bar, side): +1 when s enters the upper tail, -1 when it enters the lower tail; re-arm on leaving the tail,
    when reset_id changes (a new session in the per-session mode), or rearm_bars after the last trigger on that
    side while s is still in the tail."""
    events, armed_long, armed_short = [], True, True
    last_long = last_short = -len(s)
    for t in range(len(s)):
        if t > 0 and reset_id[t] != reset_id[t - 1]:
            armed_long = armed_short = True
        if rearm_bars:                                              # None or 0 = no timed re-arm
            if t - last_long >= rearm_bars:
                armed_long = True
            if t - last_short >= rearm_bars:
                armed_short = True
        if np.isnan(s[t]) or np.isnan(upper[t]):
            continue
        if s[t] >= upper[t]:
            if armed_long:
                events.append((t, 1))
                armed_long, last_long = False, t
        else:
            armed_long = True
        if s[t] <= lower[t]:
            if armed_short:
                events.append((t, -1))
                armed_short, last_short = False, t
        else:
            armed_short = True
    return events


def trades_for(bars, events: list[tuple[int, int]], hold: int) -> pd.DataFrame:
    """One row per trigger: entry/exit bar, side and the result in cents before costs."""
    open_w, close_w, segment = bars["W_open"].values, bars["W"].values, bars["segment"].values
    rows = []
    for t, side in events:
        entry = t + 1
        if entry >= len(bars) or segment[entry] != segment[t]:
            continue                                                  # the fill would be on the other side of a roll
        exit_bar = min(entry + hold, len(bars) - 1)
        while segment[exit_bar] != segment[entry]:                    # walk back to the last bar before the cut
            exit_bar -= 1
        rows.append({"time": bars.index[t], "entry": entry, "exit": exit_bar, "side": side,
                     "year": bars.index[t].year, "move_c": side * (close_w[exit_bar] - open_w[entry])})
    return pd.DataFrame(rows)


def cap_open(trades: pd.DataFrame, max_open: int | None) -> pd.DataFrame:
    """Keep a trade only if fewer than max_open kept trades are still open at its entry (None = no limit)."""
    if max_open is None:
        return trades
    keep, open_exits = [], []
    for _, trade in trades.iterrows():
        open_exits = [exit_bar for exit_bar in open_exits if exit_bar >= trade["entry"]]
        if len(open_exits) < max_open:
            keep.append(True)
            open_exits.append(trade["exit"])
        else:
            keep.append(False)
    return trades[keep]


def flat_only(trades: pd.DataFrame) -> pd.DataFrame:
    """Keep a trade only if the previous kept trade has already exited."""
    return cap_open(trades, 1)


def trade_paths(bars: pd.DataFrame, entry: np.ndarray, side: np.ndarray, look: int) -> dict[str, np.ndarray]:
    """For every trade (rows) and every bar from entry to entry + look (columns), in cents for us:
    close, the best and worst point inside the bar, and the bar's open. NaN after a possible roll."""
    side = side[:, None]
    index = entry[:, None] + np.arange(look + 1)[None, :]
    inside = index < len(bars)
    index = np.minimum(index, len(bars) - 1)
    segment = bars["segment"].values
    inside &= segment[index] == segment[entry][:, None]                 # segments only go up: once out, out
    price = bars["W_open"].values[entry][:, None]
    high, low = bars["W_high"].values[index], bars["W_low"].values[index]
    paths = {"close": side * (bars["W"].values[index] - price),
             "best": np.where(side > 0, high - price, price - low),
             "worst": np.where(side > 0, low - price, price - high),
             "open": side * (bars["W_open"].values[index] - price)}
    for name in paths:
        paths[name] = np.where(inside, paths[name], np.nan)
    return paths


def barrier_exit(paths: dict, target: np.ndarray, stop: np.ndarray, hold: int, cost: float) -> pd.DataFrame:
    """Net cents, exit reason and bars held for every trade, with a target / stop in cents per trade (inf = none)."""
    close, best, worst, bar_open = (paths[name][:, :hold + 1] for name in ["close", "best", "worst", "open"])
    valid = ~np.isnan(close)
    last = valid.sum(axis=1) - 1                                          # last bar before the hold ends or a roll
    stop_touch = np.nan_to_num(worst, nan=0.0) <= -stop[:, None]
    target_touch = np.nan_to_num(best, nan=0.0) >= target[:, None]
    never = close.shape[1] + 10
    first_stop = np.where(stop_touch.any(axis=1), stop_touch.argmax(axis=1), never)
    first_target = np.where(target_touch.any(axis=1), target_touch.argmax(axis=1), never)

    rows = np.arange(len(close))
    stopped = (first_stop <= first_target) & (first_stop <= last)          # same bar: the stop (conservative)
    hit = ~stopped & (first_target <= last)
    gross = close[rows, last]
    bars_held = last.copy()
    open_at_stop = bar_open[rows, np.minimum(first_stop, hold)]
    gross = np.where(stopped, np.minimum(-stop, open_at_stop) - STOP_SLIPPAGE_CENTS, gross)
    open_at_target = bar_open[rows, np.minimum(first_target, hold)]
    gross = np.where(hit, np.maximum(target, open_at_target), gross)
    bars_held = np.where(stopped, first_stop, np.where(hit, first_target, bars_held))
    reason = np.where(stopped, "stop", np.where(hit, "target", "time"))
    return pd.DataFrame({"net_c": gross - cost, "reason": reason, "bars_held": bars_held})


def gap_barriers(bars: pd.DataFrame, trades: pd.DataFrame, gap: pd.Series, rule: KalmanRuleConfig) -> pd.DataFrame:
    """Target and safety stop as multiples of the Kalman gap at the signal bar (cents at the entry price); the time
    exit stays. Applied BEFORE the position limit, so a trade that exits early frees its slot."""
    entry = trades["entry"].values
    gap_cents = np.abs(gap.values[entry - 1]) * bars["W_open"].values[entry]
    target = rule.target_gap * gap_cents if rule.target_gap else np.full(len(trades), np.inf)
    stop = rule.stop_gap * gap_cents if rule.stop_gap else np.full(len(trades), np.inf)
    result = barrier_exit(trade_paths(bars, entry, trades["side"].values, rule.hold_bars), target, stop,
                          rule.hold_bars, cost=0.0)
    trades = trades.copy()
    trades["exit"] = entry + result["bars_held"].values
    trades["move_c"] = result["net_c"].values                    # cents before costs (stop slippage included)
    trades["exit_reason"] = result["reason"].values
    return trades


def tail_table(moves_cents: pd.Series, cost_cents: float) -> dict:
    net = moves_cents.dropna() - cost_cents
    row = {"trades": len(net), "mean_c": net.mean(), "median_c": net.median(), "hit_rate": (net > 0).mean(),
           "avg_win_c": net[net > 0].mean(), "avg_loss_c": net[net <= 0].mean()}
    for tail in TAILS_CENTS:
        row[f"share_>=+{tail}c"] = (net >= tail).mean()
        row[f"share_<=-{tail}c"] = (net <= -tail).mean()
    return row


def session_returns(bars: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Per trading date: P&L marked to market bar by bar (log return of notional), costs at entry, trades closed."""
    wheat = np.zeros(len(bars))           # wheat held over each bar, summed over open trades (they may overlap)
    corn = np.zeros(len(bars))            # corn held (the spread hedge), summed the same way
    extra = np.zeros(len(bars))           # the entry bar: open-to-close move, minus costs
    log_w, log_c = bars["log_w"].values, bars["log_c"].values
    for trade in trades.itertuples():
        wheat[trade.entry_bar + 1:trade.exit_bar + 1] += trade.side          # full bars: entry+1 .. exit
        corn[trade.entry_bar + 1:trade.exit_bar + 1] -= trade.side * trade.hedge
        entry_bar_move = (log_w[trade.entry_bar] - trade.entry_open_w) - trade.hedge * (log_c[trade.entry_bar] - trade.entry_open_c)
        extra[trade.entry_bar] += trade.side * entry_bar_move - trade.cost
        fill = getattr(trade, "exit_fill_log", np.nan)
        if not np.isnan(fill):                        # exit at a limit / stop price instead of the bar's close
            extra[trade.exit_bar] += trade.side * (fill - log_w[trade.exit_bar])
    bar_pnl = wheat * np.nan_to_num(bars["ret_w"].values) + corn * np.nan_to_num(bars["ret_c"].values) + extra
    dates = bars.index.normalize().tz_localize(None)
    daily = pd.DataFrame({"ret": pd.Series(bar_pnl, index=bars.index).groupby(dates).sum()})
    closed = pd.Series(1, index=trades["exit_time"].dt.normalize().dt.tz_localize(None)) if len(trades) else pd.Series(dtype=int)
    daily["trades"] = closed.groupby(level=0).sum().reindex(daily.index, fill_value=0)
    return daily


def kalman_signal(features: pd.DataFrame, rule: KalmanRuleConfig) -> pd.Series:
    """s on the tradable bars (the bars Layer 1 scored), NaN elsewhere."""
    warm = features[[f"own_ze_{lookback}" for lookback in A_LOOKBACKS]].notna().all(axis=1)
    return (-features[f"kzr_{rule.memory_sessions}s"]).where(warm)


def book(bars: pd.DataFrame, s: pd.Series, rule: KalmanRuleConfig, gap: pd.Series | None = None) -> pd.DataFrame:
    """The trades one account takes, with the columns session_returns needs and the result in cents.
    gap (the Kalman gap, log units) is needed only when the rule has a gap target, stop or minimum."""
    rolling = s.rolling(rule.quantile_window_bars, min_periods=rule.quantile_min_bars)
    upper = rolling.quantile(1 - rule.tail).shift(1).values
    lower = rolling.quantile(rule.tail).shift(1).values
    reset_id = bars["session"].values if rule.per_session else bars["segment"].values
    events = tail_triggers(s.values, upper, lower, reset_id, rule.rearm_bars)
    trades = trades_for(bars, events, rule.hold_bars)
    if rule.min_gap_cents and len(trades):
        entry = trades["entry"].values
        gap_cents = np.abs(gap.values[entry - 1]) * bars["W_open"].values[entry]
        trades = trades[gap_cents >= rule.min_gap_cents]        # too small to pay costs and stop risk: skipped,
    if rule.target_gap or rule.stop_gap:                        # before the position limit, so no slot is used
        trades = gap_barriers(bars, trades, gap, rule)
    trades = cap_open(trades, rule.max_open)
    if trades.empty:
        return trades
    return finish_trades(bars, trades, rule.cost_ticks)


def finish_trades(bars: pd.DataFrame, trades: pd.DataFrame, cost_ticks: float) -> pd.DataFrame:
    """Add the columns session_returns needs to a table of wheat-outright trades (entry, exit, side, move_c)."""
    trades = trades.rename(columns={"entry": "entry_bar", "exit": "exit_bar"}).reset_index(drop=True)
    open_price = bars["W_open"].values[trades["entry_bar"]]
    trades["entry_open_w"] = np.log(open_price)
    trades["entry_open_c"] = 0.0
    trades["hedge"] = 0.0
    trades["cost"] = cost_ticks * TICK_CENTS / open_price                       # log units
    trades["exit_time"] = bars.index[trades["exit_bar"]]
    trades["exit_fill_log"] = np.log(open_price + trades["side"] * trades["move_c"])   # the close, or a limit / stop
    trades["ret"] = trades["side"] * (trades["exit_fill_log"] - trades["entry_open_w"]) - trades["cost"]
    return trades


def make_kalman_evaluator(cfg, include_holdout: bool = False, end: pd.Timestamp | None = None):
    """Load and build features once; return evaluate(params, start, end), every trading date, bars and features.
    end: bars after it are never loaded (holdout A stops at split.holdout_end, so holdout B stays unseen)."""
    bars = load_bars(cfg, include_holdout=include_holdout, end=end)
    features = build_features(bars, cfg.features)
    trading_dates = pd.DatetimeIndex(bars.index.normalize().tz_localize(None).unique())

    def evaluate(params: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        rule = replace(cfg.kalman_rule, **params)
        gap = features[f"kgap_{rule.memory_sessions}s"]
        trades = book(bars, kalman_signal(features, rule), rule, gap)   # on ALL bars so thresholds are warm, then cut
        daily = session_returns(bars, trades)
        daily = daily[(daily.index >= start) & (daily.index < end)]
        daily = daily.assign(ret=daily["ret"] * 100)         # percent of one contract's value: the tear sheet's unit
        if len(trades):
            exit_dates = trades["exit_time"].dt.normalize().dt.tz_localize(None)
            kept = trades[(exit_dates >= start) & (exit_dates < end)].copy()
            kept["exit_date"] = exit_dates[kept.index]
            kept["ret"] *= 100
            daily.attrs["trades"] = kept
        return daily

    return evaluate, trading_dates, bars, features


def cents_row(trades: pd.DataFrame, cost_cents: float) -> dict:
    """Net per trade and average winner / loser, in cents (tail_table's numbers, relabelled)."""
    row = tail_table(trades["move_c"], cost_cents)
    net = trades["move_c"] - cost_cents
    row["t"] = net.mean() / (net.std() / np.sqrt(len(net))) if len(net) > 1 else np.nan
    return {"trades": row["trades"], "net_per_trade_c": row["mean_c"], "t": row["t"], "hit_rate": row["hit_rate"],
            "avg_winner_c": row["avg_win_c"], "avg_loser_c": row["avg_loss_c"],
            "share_>=+10c": row["share_>=+10c"], "share_<=-10c": row["share_<=-10c"]}


def cents_chart(table: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(11, 4.6))
    labels = [f"{row.period}\n{row.what}\nn={row.trades}" for row in table.itertuples()]
    x = np.arange(len(table))
    axis.bar(x, table["net_per_trade_c"], color="#2b7bba", label="net per trade (all trades, after costs)")
    axis.scatter(x, table["avg_winner_c"], marker="^", color="#2ca02c", zorder=3, label="average winner")
    axis.scatter(x, table["avg_loser_c"], marker="v", color="#d62728", zorder=3, label="average loser")
    axis.axhline(0, color="grey", lw=0.8)
    axis.set_xticks(x, labels, fontsize=7)
    axis.set_ylabel("cents per trade")
    axis.set_title("Walk-forward out of sample: each fold traded with the pick made from earlier folds only", loc="left")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def engine_cfg(cfg):
    """The engine reads the grid from validation.grid and the selected params from cfg.strategy. Validation and
    holdout both use this, so their config hashes match."""
    return replace(cfg, strategy=cfg.kalman_rule,
                   validation=replace(cfg.validation, grid=cfg.kalman_grid, heatmaps=[]))


def holdout_a(cfg) -> None:
    """Holdout A, once: the locked rule from split.holdout_start to split.holdout_end. Bars after holdout_end are
    never loaded, so holdout B stays unseen. The engine refuses unless the latest validation has this exact config
    and passed, and refuses a second holdout run."""
    from strategies.wheat_corn_lag.harvest import time_of_day          # harvest imports this module
    end = pd.Timestamp(cfg.split.holdout_end, tz="UTC")
    evaluate, _, bars, _ = make_kalman_evaluator(cfg, include_holdout=True, end=end)
    result = run_holdout(engine_cfg(cfg), evaluate, STRATEGY_DIR)
    trades = result["daily"].attrs["trades"]
    cost = cfg.kalman_rule.cost_ticks * TICK_CENTS
    weeks = (pd.Timestamp(cfg.split.holdout_end) - pd.Timestamp(cfg.split.holdout_start)).days / 7
    net = trades["move_c"] - cost
    summary = pd.Series({**cents_row(trades, cost), "per_week": len(trades) / weeks,
                         "share_target": (trades["exit_reason"] == "target").mean(),
                         "share_stop": (trades["exit_reason"] == "stop").mean(),
                         "hours_held": (trades["exit_bar"] - trades["entry_bar"]).mean() / 4,
                         "longs": int((trades["side"] > 0).sum()), "shorts": int((trades["side"] < 0).sum()),
                         "net_cents_total": net.sum(), "dollars_per_contract": net.sum() * 50})

    def by(groups) -> pd.DataFrame:
        return pd.DataFrame([{"group": key, **cents_row(part, cost)} for key, part in trades.groupby(groups)])

    by_month = by(trades["exit_time"].dt.strftime("%Y-%m"))
    entry_times = pd.Series(bars.index[trades["entry_bar"].values - 1], index=trades.index)
    by_session = by(time_of_day(entry_times))
    by_side = by(trades["side"].map({1: "long", -1: "short"}))
    by_exit = by(trades["exit_reason"])

    checks = [("trades 35-90", 35 <= len(trades) <= 90, len(trades)),
              ("win rate 60-67% (descriptive)", 0.60 <= summary["hit_rate"] <= 0.67, summary["hit_rate"]),
              ("about 40% exit at the target (descriptive)", 0.30 <= summary["share_target"] <= 0.50,
               summary["share_target"]),
              ("net per trade not below -1 c (warning line)", summary["net_per_trade_c"] >= -1.0,
               summary["net_per_trade_c"]),
              ("net per trade in the consistent band -0.7..+3.7 c", -0.7 <= summary["net_per_trade_c"] <= 3.7,
               summary["net_per_trade_c"])]
    pd.set_option("display.width", 220)
    print(f"\n== HOLDOUT A {cfg.split.holdout_start:%Y-%m-%d} to {cfg.split.holdout_end:%Y-%m-%d}, locked rule, cents"
          f" (net per trade = all trades after costs)")
    print(summary.round(2).to_string())
    print("\n== against the expectations written before the run")
    for name, ok, value in checks:
        print(f"  {'OK  ' if ok else 'NOT '} {name}: {value:.2f}")
    for title, table in [("by month (exit)", by_month), ("by entry time (Chicago session)", by_session),
                         ("by side", by_side), ("by exit", by_exit)]:
        print(f"\n-- {title}")
        print(table.round(2).to_string(index=False))

    run_dir = result["run_dir"]
    summary.to_frame("holdout_A").to_csv(run_dir / "cents_summary.csv")
    trades.to_csv(run_dir / "trades.csv")
    for name, table in [("by_month", by_month), ("by_session", by_session), ("by_side", by_side), ("by_exit", by_exit)]:
        table.to_csv(run_dir / f"cents_{name}.csv", index=False)
    figure, axis = plt.subplots(figsize=(11, 4.4))
    axis.plot(trades["exit_time"], net.cumsum().values, marker="o", ms=3, lw=1.5)
    axis.axhline(0, color="grey", lw=0.8)
    axis.set_ylabel("cumulative net cents, 1 contract per trade")
    axis.set_title(f"Holdout A, locked rule: {len(trades)} trades, net per trade {summary['net_per_trade_c']:+.2f} c",
                   loc="left")
    figure.tight_layout()
    figure.savefig(run_dir / "cents_equity.png", dpi=130)
    print(f"\ncents tables, trades and chart added to {run_dir}")


if __name__ == "__main__":
    cfg = load()
    if "--holdout" in sys.argv:
        holdout_a(cfg)
        sys.exit()
    evaluate, trading_dates, bars, features = make_kalman_evaluator(cfg)          # development data only

    # self-check: the tradable bars are exactly the bars Layer 1 scored like for like
    from strategies.wheat_corn_lag.layer1 import SPEEDS, build_signals
    signals = build_signals(features, cfg.features.level_windows_bars, cfg.features.kalman_memory_sessions)
    scored = signals[[name for _, name in SPEEDS]].notna().all(axis=1)
    assert (kalman_signal(features, cfg.kalman_rule).notna() == scored).all(), "tradable bars != Layer 1 sample"

    # self-check: overlapping trades now stack in session_returns; one position at a time must give the
    # development Sharpe of the first walk-forward (runs/20261102-143002_validation)
    old_rule = {"tail": 0.10, "rearm_bars": 0, "max_open": 1, "target_gap": 0.0, "stop_gap": 0.0,
                "min_gap_cents": 0.0}
    old_sharpe = score(evaluate(old_rule, pd.Timestamp(cfg.split.development_start),
                                pd.Timestamp(cfg.split.holdout_start)))["sharpe"]
    assert abs(old_sharpe - 1.1927446146071805) < 1e-9, old_sharpe

    # self-check: with limit / stop fills, the marked-to-market session P&L must add up to the trades' own returns
    marked = evaluate({}, pd.Timestamp(cfg.split.development_start), pd.Timestamp(cfg.split.holdout_start))
    assert abs(marked["ret"].sum() - marked.attrs["trades"]["ret"].sum()) < 1e-9, "session P&L != trade returns"

    kalman_cfg = engine_cfg(cfg)
    result = run_validation(kalman_cfg, evaluate, trading_dates, STRATEGY_DIR)

    # the same result in cents: every fold's pick (made from earlier folds only) next to the selected params
    cost_cents = cfg.kalman_rule.cost_ticks * TICK_CENTS
    starts = [pd.Timestamp(first) for first, _ in result["metrics"]["folds"]] + [pd.Timestamp(cfg.split.holdout_start)]
    params_of = result["results"].drop_duplicates("combo").set_index("combo")[list(cfg.kalman_grid)]
    selected = {name: getattr(cfg.kalman_rule, name) for name in cfg.kalman_grid}
    rows, oos_trades = [], []
    for pick in result["picks"].itertuples():
        params = params_of.loc[pick.combo].to_dict()
        start, end = starts[pick.fold], starts[pick.fold + 1]
        period = f"{start:%Y-%m}..{end - pd.Timedelta(days=1):%Y-%m}"
        picked = evaluate(params, start, end).attrs["trades"]
        oos_trades.append(picked)
        what = "pick " + ", ".join(f"{name} {value:g}" for name, value in params.items())
        rows.append({"period": period, "what": what, **cents_row(picked, cost_cents)})
        fixed = evaluate(selected, start, end).attrs["trades"]
        rows.append({"period": period, "what": "selected", **cents_row(fixed, cost_cents)})
    rows.append({"period": "all OOS folds", "what": "walk-forward picks", **cents_row(pd.concat(oos_trades), cost_cents)})
    development = evaluate(selected, pd.Timestamp(cfg.split.development_start), starts[-1]).attrs["trades"]
    rows.append({"period": "whole development", "what": "selected (in sample)", **cents_row(development, cost_cents)})
    table = pd.DataFrame(rows)

    # every grid combo over the whole development period: does an exit rule buy more entries?
    weeks = (starts[-1] - pd.Timestamp(cfg.split.development_start)).days / 7
    mean_sharpe = result["summary"].set_index("combo")["mean_sharpe"]
    combos = []
    for combo, params in params_of.iterrows():
        traded = evaluate(params.to_dict(), pd.Timestamp(cfg.split.development_start), starts[-1]).attrs["trades"]
        reasons = traded.get("exit_reason", pd.Series("time", index=traded.index)).value_counts(normalize=True)
        combos.append({**params.to_dict(), "per_week": len(traded) / weeks, **cents_row(traded, cost_cents),
                       "cents_per_week": (traded["move_c"] - cost_cents).sum() / weeks,
                       "hours_held": (traded["exit_bar"] - traded["entry_bar"]).mean() / 4,
                       "share_target": reasons.get("target", 0.0), "share_stop": reasons.get("stop", 0.0),
                       "mean_fold_sharpe": mean_sharpe[combo]})
    combos = pd.DataFrame(combos)

    pd.set_option("display.width", 220)
    print(f"\n== in CENTS (cost {cost_cents:.2f} c round trip; 1 tick = 0.25 c). net per trade = all trades after costs;"
          f" average winner / loser = the winning / losing trades only")
    print(table.round(2).to_string(index=False))
    print("\n== every combo, whole development period (in sample)")
    print(combos.round(2).to_string(index=False))
    table.to_csv(result["run_dir"] / "cents_by_fold.csv", index=False)
    combos.to_csv(result["run_dir"] / "combos_in_cents.csv", index=False)
    picks_only = table[table["what"] != "selected"].iloc[:-1]
    cents_chart(picks_only).savefig(result["run_dir"] / "cents_by_fold.png", dpi=130)
    print(f"\ncents table and chart added to {result['run_dir']}")
