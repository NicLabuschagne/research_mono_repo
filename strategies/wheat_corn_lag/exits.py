"""
exits.py
--------
Layer 2: the path of the harvested trades, and what exits would do to them. The ENTRIES are the base rule's
(kalman_rule.py config: 20% tails, re-arm every 4.5 h, at most 2 open); only the exit changes. Development data,
IN SAMPLE. Saved to runs/<time>_exits/.

    python -m strategies.wheat_corn_lag.exits

1 Path       every trade followed bar by bar for 2 sessions after entry: average / median move for us, the best
             point (MFE) and worst point (MAE) inside the 1-session hold, and WHEN they happen
2 Time exit  exit after 4 h / 8 h / 12 h / 1 / 1.5 / 2 sessions: is the session close the right time?
3 Stops      no target, time exit 1 session, a stop at a fixed distance:
               cents  5 / 10 / 15 / 20 / 30 c
               ATR    0.5 / 0.75 / 1 / 1.5 / 2 x wheat's typical 1-session move at entry (sigma_w x sqrt 71 x price)
               gap    0.5 / 1 / 1.5 / 2 x the Kalman gap at entry in cents (how far wheat is from fair value)
             next to what the stopped trades would have made by the session close (did the stop help?)
4 Targets    no stop, time exit 1 session, a target: cents 3 / 5 / 8 / 10 / 15 / 20 c, ATR 0.25 / 0.5 / 0.75 /
             1 / 1.5, gap 0.25 / 0.5 / 1 / 1.5; share hit before the close, and what the hit trades would have
             made by the close (did the target cut winners short?)
5 Barriers   target x stop, each unit. Rule fixed before the run: keep the plain 1-session hold unless a pair
             beats its net per trade by >= 0.3 c with a t no lower and no fewer positive years.

Fills (15-minute bars, so the order inside a bar is unknown):
  a level counts as touched when the bar's high / low reaches it; target and stop touched in the same bar -> the
  STOP (conservative). Target = limit order: fills at the level, or at the bar's open if it opens beyond it.
  Stop = stop order: fills at the level, or at the open if the bar gaps through it, then 1 tick (0.25 c) worse.
  Time exit at the close; a possible roll ends the trade at the last bar before it (as everywhere).
Shortcut: entries are held fixed, so an early exit does not free a slot for another trade (max 2 open);
the walk-forward of a chosen exit rebuilds the book. Costs 1.4 ticks round trip on every trade.
Units: cents (1 tick = 0.25 c). net per trade = all trades after costs; average winner / loser = winning / losing
trades only; hit rate = share of trades > 0 after costs.
"""
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import save_run
from core_quant.theme import set_theme
from strategies.wheat_corn_lag.features import build_features
from strategies.wheat_corn_lag.kalman_rule import TICK_CENTS, barrier_exit, book, kalman_signal, trade_paths
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load, load_bars

SESSION = 71
LOOK = 2 * SESSION                             # paths are followed this far
TIME_EXITS = {"4 h": 16, "8 h": 32, "12 h": 48, "1 session": 71, "1.5 sessions": 106, "2 sessions": 142}
STOPS = {"cents": [5, 10, 15, 20, 30], "ATR": [0.5, 0.75, 1.0, 1.5, 2.0], "gap": [0.5, 1.0, 1.5, 2.0]}
TARGETS = {"cents": [3, 5, 8, 10, 15, 20], "ATR": [0.25, 0.5, 0.75, 1.0, 1.5], "gap": [0.25, 0.5, 1.0, 1.5]}
NONE = np.inf


def outcome(result: pd.DataFrame, years: np.ndarray) -> dict:
    net = result["net_c"]
    yearly = net.groupby(years).sum()
    return {"trades": len(net), "net_per_trade_c": net.mean(), "t": net.mean() / (net.std() / np.sqrt(len(net))),
            "hit_rate": (net > 0).mean(), "avg_winner_c": net[net > 0].mean(), "avg_loser_c": net[net <= 0].mean(),
            "share_target": (result["reason"] == "target").mean(), "share_stop": (result["reason"] == "stop").mean(),
            "hours_held": result["bars_held"].mean() / 4, "years_positive": int((yearly > 0).sum())}


# ------------------------------------------------------------------ charts
def path_chart(paths: dict) -> plt.Figure:
    hours = np.arange(LOOK + 1) / 4
    close = paths["close"]
    running_best = np.fmax.accumulate(np.nan_to_num(paths["best"], nan=-np.inf), axis=1)
    running_worst = np.fmin.accumulate(np.nan_to_num(paths["worst"], nan=np.inf), axis=1)
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.fill_between(hours, np.nanpercentile(close, 25, axis=0), np.nanpercentile(close, 75, axis=0), alpha=0.18,
                      label="middle half of trades")
    axis.plot(hours, np.nanmean(close, axis=0), lw=2.2, label="average move for us")
    axis.plot(hours, np.nanmedian(close, axis=0), lw=1.2, ls="--", label="median")
    axis.plot(hours, np.median(running_best, axis=0), lw=1.2, color="#2ca02c", label="median best point so far (MFE)")
    axis.plot(hours, np.median(running_worst, axis=0), lw=1.2, color="#d62728", label="median worst point so far (MAE)")
    axis.axhline(0, color="grey", lw=0.8)
    axis.axvline(SESSION / 4, color="grey", ls=":", lw=0.9)
    axis.set_xlabel("hours after entry (dotted = 1 session, the current exit)")
    axis.set_ylabel("cents, before costs")
    axis.set_title(f"Path of {len(close)} harvested trades, 2 sessions after entry", loc="left")
    axis.legend(fontsize=8, loc="upper left")
    figure.tight_layout()
    return figure


def excursion_chart(mfe: np.ndarray, mae: np.ndarray, final: np.ndarray, atr: np.ndarray) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    colour = np.where(final > 0, "#2ca02c", "#d62728")
    for axis, scale, unit in [(axes[0], 1.0, "cents"), (axes[1], atr, "x ATR")]:
        axis.scatter(mae / scale, mfe / scale, c=colour, s=10, alpha=0.6)
        axis.set_xlabel(f"worst point against us within 1 session, {unit}")
        axis.set_ylabel(f"best point for us within 1 session, {unit}")
        limit = np.nanpercentile(np.concatenate([mae / scale, mfe / scale]), 98)
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.plot([0, limit], [0, limit], color="grey", lw=0.7, ls=":")
    axes[0].set_title("Excursions (green = up at the session close)", loc="left")
    figure.tight_layout()
    return figure


def level_chart(table: pd.DataFrame, kind: str, baseline: float) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for axis, unit in zip(axes, ["cents", "ATR", "gap"]):
        part = table[table["unit"] == unit]
        axis.plot(part["level"], part["net_per_trade_c"], marker="o", label=f"with the {kind}")
        axis.axhline(baseline, color="black", ls="--", lw=1, label="plain 1-session hold")
        axis.set_xlabel(f"{kind} distance, {unit}")
        axis.set_title(f"{kind}s in {unit}", loc="left")
        for row in part.itertuples():
            axis.annotate(f"{row.share:.0%}", (row.level, row.net_per_trade_c), textcoords="offset points",
                          xytext=(0, 6), ha="center", fontsize=7, color="grey")
    axes[0].set_ylabel("net per trade, cents\n(labels: share exiting there)")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    return figure


def barrier_chart(grid: pd.DataFrame, baseline: float) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for axis, unit in zip(axes, ["cents", "ATR", "gap"]):
        part = grid[grid["unit"] == unit].pivot(index="stop", columns="target", values="net_per_trade_c")
        limit = max(abs(part.values - baseline).max(), 0.5)
        image = axis.imshow(part.values - baseline, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto", origin="lower")
        axis.set_xticks(range(len(part.columns)), [str(c) for c in part.columns])
        axis.set_yticks(range(len(part.index)), [str(r) for r in part.index])
        for i in range(part.shape[0]):
            for j in range(part.shape[1]):
                axis.text(j, i, f"{part.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        axis.set_xlabel(f"target, {unit}")
        axis.set_ylabel(f"stop, {unit}")
        axis.set_title(f"net per trade (c), colour vs plain hold {baseline:.2f} c", loc="left", fontsize=9)
        axis.grid(False)
        figure.colorbar(image, ax=axis, fraction=0.04)
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    rule = cfg.kalman_rule
    bars = load_bars(cfg)                                        # development data only
    features = build_features(bars, cfg.features)
    rule = replace(rule, target_gap=0.0, stop_gap=0.0)          # the plain-hold entries; only exits change here
    trades = book(bars, kalman_signal(features, rule), rule)
    cost = rule.cost_ticks * TICK_CENTS
    signal_bar = trades["entry_bar"].values - 1
    price = bars["W_open"].values[trades["entry_bar"].values]
    atr = features["sigma_w"].values[signal_bar] * np.sqrt(SESSION) * price          # known at the signal bar
    gap = np.abs(features[f"kgap_{rule.memory_sessions}s"].values[signal_bar]) * price
    unit_size = {"cents": np.ones(len(trades)), "ATR": atr, "gap": gap}
    years = trades["year"].values
    paths = trade_paths(bars, trades["entry_bar"].values, trades["side"].values, LOOK)
    plain = barrier_exit(paths, np.full(len(trades), NONE), np.full(len(trades), NONE), rule.hold_bars, cost)
    baseline = outcome(plain, years)
    assert np.allclose(plain["net_c"].values, trades["move_c"].values - cost), "plain hold != kalman_rule trades"

    # 1 path
    hold_close, hold_best, hold_worst = (paths[name][:, :rule.hold_bars + 1] for name in ["close", "best", "worst"])
    mfe = np.nanmax(hold_best, axis=1)
    mae = -np.nanmin(hold_worst, axis=1)
    when_best = np.nanargmax(hold_best, axis=1) / 4
    when_worst = np.nanargmin(hold_worst, axis=1) / 4
    final = plain["net_c"].values + cost
    shape = pd.DataFrame({
        "measure": ["best point for us (MFE), c", "worst point against us (MAE), c", "MFE x ATR", "MAE x ATR",
                    "hours to the best point", "hours to the worst point", "ATR at entry, c", "gap at entry, c"],
        "p25": [np.percentile(v, 25) for v in [mfe, mae, mfe / atr, mae / atr, when_best, when_worst, atr, gap]],
        "median": [np.median(v) for v in [mfe, mae, mfe / atr, mae / atr, when_best, when_worst, atr, gap]],
        "p75": [np.percentile(v, 75) for v in [mfe, mae, mfe / atr, mae / atr, when_best, when_worst, atr, gap]]})
    give_back = pd.DataFrame({"group": ["winners at the close", "losers at the close"],
                              "trades": [(final > 0).sum(), (final <= 0).sum()],
                              "median_MFE_c": [np.median(mfe[final > 0]), np.median(mfe[final <= 0])],
                              "median_MAE_c": [np.median(mae[final > 0]), np.median(mae[final <= 0])],
                              "share_MFE_>=5c": [(mfe[final > 0] >= 5).mean(), (mfe[final <= 0] >= 5).mean()],
                              "share_MAE_>=10c": [(mae[final > 0] >= 10).mean(), (mae[final <= 0] >= 10).mean()]})

    # 2 time exits
    none = np.full(len(trades), NONE)
    times = pd.DataFrame([{"exit": name, **outcome(barrier_exit(paths, none, none, bars_held, cost), years)}
                          for name, bars_held in TIME_EXITS.items()])
    times["net_per_session_held_c"] = times["net_per_trade_c"] / (np.array(list(TIME_EXITS.values())) / SESSION)

    # 3 stops and 4 targets on their own
    rows = []
    for unit, levels in STOPS.items():
        for level in levels:
            result = barrier_exit(paths, none, level * unit_size[unit], rule.hold_bars, cost)
            stopped = result["reason"] == "stop"
            held = plain["net_c"][stopped]
            rows.append({"unit": unit, "level": level, "share": stopped.mean(), **outcome(result, years),
                         "stopped_trades_net_c": result["net_c"][stopped].mean(),
                         "same_trades_held_to_close_c": held.mean(),
                         "same_trades_recovered_share": (held > result["net_c"][stopped]).mean()})
    stops = pd.DataFrame(rows)
    rows = []
    for unit, levels in TARGETS.items():
        for level in levels:
            result = barrier_exit(paths, level * unit_size[unit], none, rule.hold_bars, cost)
            hit = result["reason"] == "target"
            rows.append({"unit": unit, "level": level, "share": hit.mean(), **outcome(result, years),
                         "median_hours_to_target": result["bars_held"][hit].median() / 4,
                         "hit_trades_net_c": result["net_c"][hit].mean(),
                         "same_trades_held_to_close_c": plain["net_c"][hit].mean()})
    targets = pd.DataFrame(rows)

    # 5 barriers: target x stop
    rows = []
    for unit in ["cents", "ATR", "gap"]:
        for target in TARGETS[unit]:
            for stop in STOPS[unit]:
                result = barrier_exit(paths, target * unit_size[unit], stop * unit_size[unit], rule.hold_bars, cost)
                rows.append({"unit": unit, "target": target, "stop": stop, **outcome(result, years)})
    grid = pd.DataFrame(rows)
    better = grid[(grid["net_per_trade_c"] >= baseline["net_per_trade_c"] + 0.3) & (grid["t"] >= baseline["t"])
                  & (grid["years_positive"] >= baseline["years_positive"])]
    choice = "plain 1-session hold" if better.empty else better.sort_values("net_per_trade_c").iloc[-1].to_dict()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"{len(trades)} trades; cost {cost:.2f} c round trip; stop fills 1 tick worse. net per trade = all trades"
          f" after costs; average winner / loser = winning / losing trades only")
    print(f"\n== plain 1-session hold: {pd.Series(baseline).round(2).to_dict()}")
    print("\n== 1 path inside the 1-session hold")
    print(shape.round(2).to_string(index=False))
    print(give_back.round(2).to_string(index=False))
    print("\n== 2 time exits (no target, no stop)")
    print(times.round(2).to_string(index=False))
    print("\n== 3 stops only (time exit 1 session); 'same trades held' = what the stopped trades made by the close")
    print(stops.round(2).to_string(index=False))
    print("\n== 4 targets only (time exit 1 session); 'same trades held' = what the hit trades made by the close")
    print(targets.round(2).to_string(index=False))
    print("\n== 5 barriers, best 10 by net per trade")
    print(grid.sort_values("net_per_trade_c", ascending=False).head(10).round(2).to_string(index=False))
    print(f"\nchoice (rule fixed before the run): {choice}")

    figures = {"1_path": path_chart(paths), "1_excursions": excursion_chart(mfe, mae, final, atr),
               "3_stops": level_chart(stops, "stop", baseline["net_per_trade_c"]),
               "4_targets": level_chart(targets, "target", baseline["net_per_trade_c"]),
               "5_barriers": barrier_chart(grid, baseline["net_per_trade_c"])}
    run_dir = save_run(STRATEGY_DIR, "exits", cfg, {"baseline": baseline, "choice": choice},
                       {"path_shape": shape, "give_back": give_back, "time_exits": times, "stops": stops,
                        "targets": targets, "barriers": grid}, figures)
    print(f"\nsaved to {run_dir}")
