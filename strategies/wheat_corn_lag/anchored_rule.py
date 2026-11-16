"""
anchored_rule.py
----------------
The short-memory rule (2026-11-16). Params: anchored_rule in config.yaml, grid: anchored_grid.

    python -m strategies.wheat_corn_lag.anchored_rule             # walk-forward on development, gates, cents, charts
    python -m strategies.wheat_corn_lag.anchored_rule --forward   # chart only: the selected rule from 2026-11-02 on
                                                                  # the live files (a look, NOT evidence)

Fair value  moves with corn by the returns hedge ratio (15-minute returns of the beta_window_sessions before the bar)
            and is pulled back toward wheat at 1 / memory. In one line: own = running total of (wheat return - beta x
            corn return); gap = own - its own past exponential average (memory_sessions). Unlike the Kalman level
            filter there is no intercept / slope pair and nothing depends on where a price index starts.
Signal      z = -gap / its typical size (RMS around zero) over the z_window_sessions before the bar.
            + = wheat below fair value -> buy.
Tradable    the same bars as the locked rule (kalman_rule.kalman_signal: not the ~6 sessions after a possible roll),
            so the two rules are compared on the same sample.
Triggers    long when z rises through +entry_z, short when it falls through -entry_z.
Re-entry    add_step None: again every time z leaves the zone and comes back (the first runs).
            add_step a number: after an entry, the next one on that side needs z back at 0 first (price returned
            to fair value), or z at least add_step deeper than the last entry (0 = no adds).
Book        at most max_open positions; signal at a bar's close, fill at the next bar's open.
            Costs cost_ticks x 0.25 c round trip.
Exits       "hold": the close hold_bars later (or the last bar before a possible roll).
            "fair_value": when a bar CLOSES with price back at fair value (z at 0 or through it), sell at the next
            bar's open (fair value moves with corn, so it cannot be a resting limit order); if that never happens,
            the close of the entry's session (session = UTC date = one CBOT session, see strategy.py).
            "fair_value_only": the same without the session stop (held until fair value or a possible roll).
            stop_sessions: with either fair-value exit, also out at the close that many sessions after entry.
"""
import sys
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.walk_forward import run_validation
from strategies.wheat_corn_lag.features import BARS_PER_SESSION, lagged_rms, rolling_stats
from strategies.wheat_corn_lag.kalman_rule import (TICK_CENTS, cap_open, cents_chart, cents_row, finish_trades,
                                                   kalman_signal, make_kalman_evaluator, session_returns,
                                                   tail_triggers, trades_for)
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, AnchoredRuleConfig, load

# three-week windows drawn after the run: fixed before seeing any result, one per year, away from roll windows
CHART_WINDOWS = [("2022-03-14", "2022-04-02"), ("2023-05-15", "2023-06-03"),
                 ("2024-09-16", "2024-10-05"), ("2025-10-06", "2025-10-25")]


def anchored_gap(bars: pd.DataFrame, rule: AnchoredRuleConfig) -> pd.Series:
    """Wheat minus fair value, log units, from information before the bar's own update."""
    beta = rolling_stats(bars, rule.beta_window_sessions * BARS_PER_SESSION)["beta"]     # bars before t only
    own = (bars["ret_w"].fillna(0) - beta.fillna(0) * bars["ret_c"].fillna(0)).cumsum()
    memory_bars = rule.memory_sessions * BARS_PER_SESSION
    return own - own.ewm(alpha=1.0 / memory_bars, adjust=False).mean().shift(1)


def anchored_signal(gap: pd.Series, rule: AnchoredRuleConfig, tradable: pd.Series) -> pd.Series:
    return (-gap / lagged_rms(gap, rule.z_window_sessions * BARS_PER_SESSION)).where(tradable)


def ladder_triggers(z: np.ndarray, entry_z: float, add_step: float, segment: np.ndarray) -> list[tuple[int, int]]:
    """(bar, side). First entry of a divergence when z reaches entry_z; another only when z is add_step deeper than
    the last entry (add_step 0: none). A side starts afresh when z is back at 0 or beyond, and after a possible roll."""
    events = []
    last_long = last_short = None                     # z at the last entry of the current divergence, per side
    for t in range(len(z)):
        if t > 0 and segment[t] != segment[t - 1]:
            last_long = last_short = None
        if np.isnan(z[t]):
            continue
        if z[t] <= 0:
            last_long = None
        if z[t] >= 0:
            last_short = None
        if z[t] >= entry_z and (last_long is None or (add_step > 0 and z[t] >= last_long + add_step)):
            events.append((t, 1))
            last_long = z[t]
        if z[t] <= -entry_z and (last_short is None or (add_step > 0 and z[t] <= last_short - add_step)):
            events.append((t, -1))
            last_short = z[t]
    return events


def fair_value_exits(bars: pd.DataFrame, events: list[tuple[int, int]], z: np.ndarray, session_stop: bool,
                     stop_bars: int = 0) -> pd.DataFrame:
    """One row per trigger, exits = "fair_value" (session_stop) or "fair_value_only". "exit" is the last bar the
    position is held over; the result (move_c, cents before costs) is to the next bar's open after a fair-value
    signal, else to that bar's close. Without the session stop the only other exit is the last bar before a
    possible roll (the price series is cut there)."""
    open_w, close_w = bars["W_open"].values, bars["W"].values
    segment = bars["segment"].values
    session = bars["session"].values if session_stop else segment      # the block of bars a trade may live in
    rows = []
    for t, side in events:
        entry = t + 1
        if entry >= len(bars) or segment[entry] != segment[t]:
            continue                                                  # the fill would be on the other side of a roll
        last, reason = entry, "session" if session_stop else "roll"
        while True:
            if side * z[last] <= 0:                                   # this bar closed at or through fair value
                reason = "fair value"
                break
            if stop_bars and last - entry >= stop_bars:               # the time stop: out at this bar's close
                reason = "time"
                break
            if last + 1 >= len(bars) or session[last + 1] != session[entry]:
                break                                                 # the session's last bar: out at its close
            last += 1
        next_open_in_session = last + 1 < len(bars) and session[last + 1] == session[entry]
        fill = open_w[last + 1] if reason == "fair value" and next_open_in_session else close_w[last]
        rows.append({"time": bars.index[t], "entry": entry, "exit": last, "side": side, "year": bars.index[t].year,
                     "move_c": side * (fill - open_w[entry]), "exit_reason": reason})
    return pd.DataFrame(rows)


def anchored_book(bars: pd.DataFrame, z: pd.Series, rule: AnchoredRuleConfig) -> pd.DataFrame:
    level = np.full(len(z), rule.entry_z)
    if rule.add_step is None:
        events = tail_triggers(z.values, level, -level, bars["segment"].values)
    else:
        events = ladder_triggers(z.values, rule.entry_z, rule.add_step, bars["segment"].values)
    if rule.exits in ("fair_value", "fair_value_only"):
        trades = fair_value_exits(bars, events, z.values, session_stop=rule.exits == "fair_value",
                                  stop_bars=rule.stop_sessions * BARS_PER_SESSION)
    else:
        trades = trades_for(bars, events, rule.hold_bars)
        trades["exit_reason"] = "time"
    if trades.empty:
        return trades
    return finish_trades(bars, cap_open(trades, rule.max_open), rule.cost_ticks)


def make_anchored_evaluator(cfg, bars: pd.DataFrame, tradable: pd.Series):
    """evaluate(params, start, end) for the walk-forward engine, on bars already loaded."""
    def evaluate(params: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        rule = replace(cfg.anchored_rule, **params)
        z = anchored_signal(anchored_gap(bars, rule), rule, tradable)
        trades = anchored_book(bars, z, rule)                     # on ALL bars, then cut
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
    return evaluate


def trade_chart(bars, gap, z, trades, rule, cost_cents, start, end, path) -> None:
    """Candles, fair value and trades on top; z and its entry levels below. Bars side by side (no closed hours)."""
    first = int(np.searchsorted(bars.index, pd.Timestamp(start, tz="UTC")))
    last = int(np.searchsorted(bars.index, pd.Timestamp(end, tz="UTC")))
    x = np.arange(last - first)
    open_, high, low, close = (bars[name].values[first:last] for name in ["W_open", "W_high", "W_low", "W"])
    fair = close * np.exp(-gap.values[first:last])
    zs = z.values[first:last]
    colours = np.where(close >= open_, "#2ca02c", "#d62728")

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(22, 12), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    top.vlines(x, low, high, color=colours, lw=0.7, alpha=0.6)
    top.bar(x, np.maximum(np.abs(close - open_), 0.05), bottom=np.minimum(open_, close), width=0.7, color=colours, alpha=0.6)
    top.plot(x, fair, color="black", lw=1.5, label=f"fair value, {rule.memory_sessions} sessions of memory (not shifted)")
    shown = trades[(trades["entry_bar"] >= first) & (trades["entry_bar"] < last)]
    net_total = 0.0
    for trade in shown.itertuples():
        entry_price = bars["W_open"].values[trade.entry_bar]
        exit_price, net = entry_price + trade.side * trade.move_c, trade.move_c - cost_cents
        net_total += net
        entry_x, exit_x = trade.entry_bar - first, trade.exit_bar - first
        colour = "#1f77b4" if net > 0 else "#7f0000"
        top.plot([entry_x, exit_x], [entry_price, exit_price], color=colour, lw=1.3, ls="--", zorder=4)
        top.scatter(entry_x, entry_price, marker="^" if trade.side > 0 else "v", s=150, color=colour, edgecolor="black", zorder=5)
        top.scatter(exit_x, exit_price, marker="*" if trade.exit_reason == "fair value" else "X", s=130, color=colour,
                    edgecolor="black", zorder=5)
        top.annotate(f"{net:+.1f}", (exit_x, exit_price), textcoords="offset points", xytext=(5, 6), fontsize=9)
        bottom.scatter(entry_x - 1, zs[entry_x - 1] if entry_x >= 1 else np.nan, marker="^" if trade.side > 0 else "v",
                       s=90, color=colour, edgecolor="black", zorder=5)                 # the signal bar
    top.set_ylabel("wheat, cents per bushel")
    top.set_title(f"{start} to {end}: {len(shown)} trades, net {net_total:+.1f} c in total. Triangle = entry (up = long), "
                  f"star = exit at fair value, X = time / session exit, number = net cents. Blue = winner, dark red = loser", loc="left")
    top.legend(loc="upper right")
    bottom.plot(x, zs, color="black", lw=1.1)
    bottom.fill_between(x, rule.entry_z, zs, where=zs >= rule.entry_z, color="#1f77b4", alpha=0.35)
    bottom.fill_between(x, -rule.entry_z, zs, where=zs <= -rule.entry_z, color="#d62728", alpha=0.35)
    for level in [rule.entry_z, -rule.entry_z]:
        bottom.axhline(level, color="grey", ls="--", lw=1)
    bottom.axhline(0, color="grey", lw=0.8)
    bottom.set_ylabel(f"z = -(gap) / typical gap of the previous {rule.z_window_sessions} sessions\n"
                      f"+ = wheat below fair value. Gaps in the line = not tradable (roll)")
    sessions = bars["session"].values[first:last]
    starts = np.flatnonzero(np.r_[True, sessions[1:] != sessions[:-1]])
    bottom.set_xticks(starts, [f"{day:%d %b}" for day in bars.index[first + starts]])
    for axis in (top, bottom):
        for session_start in starts:
            axis.axvline(session_start - 0.5, color="grey", lw=0.4, alpha=0.5)
        axis.grid(axis="y", alpha=0.3)
    top.set_xlim(-1, len(x) + 1)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def engine_cfg(cfg):
    """The engine reads the grid from validation.grid and the selected params from cfg.strategy."""
    return replace(cfg, strategy=cfg.anchored_rule, validation=replace(cfg.validation, grid=cfg.anchored_grid, heatmaps=[]))


def forward_chart(cfg, name: str = "anchored_trades_forward") -> None:
    """The selected rule on the live files, drawn from 2026-11-02. Holdout B feeds the filter; its trades are
    computed but never printed, saved or drawn."""
    from strategies.wheat_corn_lag.forward import LEGS
    live_cfg = replace(cfg, **{field: live_file for field, (_, live_file) in LEGS.items()})
    _, _, bars, features = make_kalman_evaluator(live_cfg, include_holdout=True)
    rule = cfg.anchored_rule
    gap = anchored_gap(bars, rule)
    z = anchored_signal(gap, rule, kalman_signal(features, cfg.kalman_rule).notna())
    trades = anchored_book(bars, z, rule)
    first = int(np.searchsorted(bars.index, pd.Timestamp("2026-11-02", tz="UTC")))
    trades = trades[trades["entry_bar"] >= first]
    path = STRATEGY_DIR / "runs" / "forward" / f"{name}.png"
    trade_chart(bars, gap, z, trades, rule, rule.cost_ticks * TICK_CENTS, "2026-11-02", "2100-01-01", path)
    print(f"{len(trades)} trades from 2026-11-02 (a look, not evidence). Chart: {path}")


if __name__ == "__main__":
    cfg = load()
    if "--forward" in sys.argv:
        forward_chart(cfg)
        sys.exit()
    locked_evaluate, trading_dates, bars, features = make_kalman_evaluator(cfg)        # development data only
    tradable = kalman_signal(features, cfg.kalman_rule).notna()
    evaluate = make_anchored_evaluator(cfg, bars, tradable)
    rule = cfg.anchored_rule
    cost_cents = rule.cost_ticks * TICK_CENTS
    development_start, holdout_start = pd.Timestamp(cfg.split.development_start), pd.Timestamp(cfg.split.holdout_start)

    # self-checks: session P&L adds up to the trades' own returns; every entry had z beyond the entry level
    marked = evaluate({}, development_start, holdout_start)
    selected_trades = marked.attrs["trades"]
    assert abs(marked["ret"].sum() - selected_trades["ret"].sum()) < 1e-9, "session P&L != trade returns"
    gap = anchored_gap(bars, rule)
    z = anchored_signal(gap, rule, tradable)
    z_at_signal = z.values[selected_trades["entry_bar"].values - 1]
    assert (z_at_signal * selected_trades["side"].values >= rule.entry_z).all(), "an entry without a signal"

    result = run_validation(engine_cfg(cfg), evaluate, trading_dates, STRATEGY_DIR)
    run_dir = result["run_dir"]

    # the same result in cents: every fold's pick (made from earlier folds only) next to the selected params
    starts = [pd.Timestamp(first) for first, _ in result["metrics"]["folds"]] + [holdout_start]
    params_of = result["results"].drop_duplicates("combo").set_index("combo")[list(cfg.anchored_grid)]

    def typed(row) -> dict:
        """A table row turns every number into a float: give each param back its own type (2.0 -> 2 sessions)."""
        return {name: type(getattr(rule, name))(value) for name, value in row.items()}

    rows, oos_trades = [], []
    for pick in result["picks"].itertuples():
        params = typed(params_of.loc[pick.combo])
        start, end = starts[pick.fold], starts[pick.fold + 1]
        period = f"{start:%Y-%m}..{end - pd.Timedelta(days=1):%Y-%m}"
        picked = evaluate(params, start, end).attrs["trades"]
        oos_trades.append(picked)
        what = "pick " + ", ".join(f"{name} {value:g}" for name, value in params.items())
        rows.append({"period": period, "what": what, **cents_row(picked, cost_cents)})
        rows.append({"period": period, "what": "selected", **cents_row(evaluate({}, start, end).attrs["trades"], cost_cents)})
    oos_trades = pd.concat(oos_trades)
    rows.append({"period": "all OOS folds", "what": "walk-forward picks", **cents_row(oos_trades, cost_cents)})
    table = pd.DataFrame(rows)

    # the pass rule fixed before the run: walk-forward net cents per week >= the locked rule's over the same folds
    oos_weeks = (holdout_start - starts[1]).days / 7
    locked_trades = locked_evaluate({}, starts[1], holdout_start).attrs["trades"]
    locked_cost = cfg.kalman_rule.cost_ticks * TICK_CENTS
    selected_oos = evaluate({}, starts[1], holdout_start).attrs["trades"]
    compare = pd.DataFrame([
        {"rule": "anchored, walk-forward picks", **cents_row(oos_trades, cost_cents)},
        {"rule": "anchored, selected params", **cents_row(selected_oos, cost_cents)},
        {"rule": "locked Kalman rule", **cents_row(locked_trades, locked_cost)}])
    compare["per_week"] = compare["trades"] / oos_weeks
    compare["worst_trade_c"] = [(t["move_c"] - c).min() for t, c in ((oos_trades, cost_cents), (selected_oos, cost_cents), (locked_trades, locked_cost))]
    compare["hours_held"] = [(t["exit_bar"] - t["entry_bar"] + 1).mean() / 4 for t in (oos_trades, selected_oos, locked_trades)]
    compare["share_at_fair_value"] = [(t["exit_reason"] == "fair value").mean() for t in (oos_trades, selected_oos, locked_trades)]
    compare["net_cents_per_week"] = compare["net_per_trade_c"] * compare["per_week"]

    # every combo over the whole development period (in sample)
    weeks = (holdout_start - development_start).days / 7
    mean_sharpe = result["summary"].set_index("combo")["mean_sharpe"]
    combos = []
    for combo, params in params_of.iterrows():
        traded = evaluate(typed(params), development_start, holdout_start).attrs["trades"]
        combos.append({**params.to_dict(), "per_week": len(traded) / weeks, **cents_row(traded, cost_cents),
                       "cents_per_week": (traded["move_c"] - cost_cents).sum() / weeks,
                       "longs": (traded["side"] > 0).mean(), "hours_held": (traded["exit_bar"] - traded["entry_bar"] + 1).mean() / 4,
                       "share_at_fair_value": (traded["exit_reason"] == "fair value").mean(),
                       "worst_trade_c": (traded["move_c"] - cost_cents).min(), "mean_fold_sharpe": mean_sharpe[combo]})
    combos = pd.DataFrame(combos)

    def by(groups) -> pd.DataFrame:
        return pd.DataFrame([{"group": key, **cents_row(part, cost_cents)} for key, part in selected_oos.groupby(groups)])

    pd.set_option("display.width", 250)
    print(f"\n== in CENTS (cost {cost_cents:.2f} c round trip). net per trade = all trades after costs; average winner /"
          f" loser = the winning / losing trades only")
    print(table.round(2).to_string(index=False))
    print(f"\n== PASS RULE: out-of-sample folds {starts[1]:%Y-%m-%d} to {holdout_start:%Y-%m-%d}, net cents per week vs the locked rule")
    print(compare.round(2).to_string(index=False))
    more = compare["net_cents_per_week"].iloc[0] >= compare["net_cents_per_week"].iloc[2]
    print(f"walk-forward net cents per week >= locked: {'PASSED' if more else 'FAILED'}")
    print("\n== every combo, whole development period (in sample)")
    print(combos.round(2).to_string(index=False))
    for title, groups in [("by year (exit)", selected_oos["exit_time"].dt.year),
                          ("by side", selected_oos["side"].map({1: "long", -1: "short"})),
                          ("by exit", selected_oos["exit_reason"])]:
        print(f"\n-- selected params, out-of-sample folds, {title}")
        print(by(groups).round(2).to_string(index=False))

    table.to_csv(run_dir / "cents_by_fold.csv", index=False)
    compare.to_csv(run_dir / "cents_vs_locked.csv", index=False)
    combos.to_csv(run_dir / "combos_in_cents.csv", index=False)
    cents_chart(table[table["what"] != "selected"].iloc[:-1]).savefig(run_dir / "cents_by_fold.png", dpi=130)

    figure, axis = plt.subplots(figsize=(13, 5))
    for label, traded, cost in [("anchored, selected params", selected_oos, cost_cents), ("locked Kalman rule", locked_trades, locked_cost)]:
        traded = traded.sort_values("exit_time")
        axis.plot(traded["exit_time"], (traded["move_c"] - cost).cumsum().values, lw=1.5,
                  label=f"{label}: {len(traded)} trades, {(traded['move_c'] - cost).mean():+.2f} c net per trade")
    axis.axhline(0, color="grey", lw=0.8)
    axis.set_ylabel("cumulative net cents, 1 contract per trade")
    axis.set_title("Out-of-sample folds, development data: cumulative net cents", loc="left")
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_dir / "cents_equity_vs_locked.png", dpi=130)

    all_selected = evaluate({}, development_start, holdout_start).attrs["trades"]
    for start, end in CHART_WINDOWS:
        trade_chart(bars, gap, z, all_selected, rule, cost_cents, start, end, run_dir / f"trades_{start}.png")
    print(f"\ncents tables and charts (cents_equity_vs_locked.png, trades_<date>.png) added to {run_dir}")
