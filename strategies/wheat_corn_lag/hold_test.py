"""
hold_test.py
------------
Layer 2, step 1 on the Kalman base: trade wheat OUTRIGHT on the Layer-1 base signal, hold 1 session vs 2 sessions.
Every trigger choice comes from what Layer 1 measured, not from the bean defaults (|z| 1.645 / re-arm 0.6745).
Development data, IN-SAMPLE description; the chosen rule is walk-forward tested next. Saved to runs/<time>_hold_test/.

    python -m strategies.wheat_corn_lag.hold_test

Signal   s = -(Kalman M20 live gap / its size), + = wheat cheap vs corn -> buy, two sizes of the same gap:
           sd    layer1 BASE, kz_20s: 20-session sd (around the window mean; stretched when the gap sits)
           RMS   kzr_20s: 20-session RMS around zero (Layer 1 2026-11-02: same IC, deciles in order)
Sample   only the bars Layer 1 scored like for like (layer1 SPEEDS all present): this leaves out the ~6 sessions
         after a roll window, where K's IC was 0.072 instead of 0.17
Bridge   the Layer-1 deciles in CENTS: every bar entered next open in the signal's direction, held 1 / 2 sessions,
         net of costs, by signal decile (pooled, like the IC test). Shows which deciles are worth trading.
Triggers tail share q (0.10 = the IC test's extreme deciles, 0.20 / 0.30 for more trades): long when s is above the
         (1 - q) quantile of the PAST year's s, short below the q quantile (point in time, shifted one bar).
           per move     fire on entering the tail, re-arm once s is back out of it
           per session  also re-arm at every new session (a tail that lasts 3 sessions = up to 3 trades)
Fill     signal at bar t's close, entry at bar t+1's OPEN; exit at the close `hold` bars later, or at the last bar
         before a possible roll if that comes first; costs 1.4 ticks round trip
Samples  all triggers (the statistical sample) and flat only (one position at a time; trades never overlap, t here)
Random   the same hold from every 71st bar, long and short equally
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import save_run
from core_quant.theme import set_theme
from strategies.wheat_corn_lag.edge_map import COST_TICKS, TICK_CENTS, tail_table
from strategies.wheat_corn_lag.features import build_features
from strategies.wheat_corn_lag.kalman_rule import flat_only, tail_triggers, trades_for
from strategies.wheat_corn_lag.layer1 import BASE, KALMAN, SPEEDS, build_signals
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load, load_bars

HOLDS = {"1 session": 71, "2 sessions": 142}
TAILS = [0.10, 0.20, 0.30]
FOCUS_TAIL = 0.10
QUANTILE_WINDOW = 71 * 252                      # thresholds from the past year of s
QUANTILE_MIN = 71 * 20                          # start once 20 sessions of s exist


def summary(trades: pd.DataFrame, cost_cents: float, weeks: float) -> dict:
    row = tail_table(trades["move_c"], cost_cents)
    row["per_week"] = len(trades) / weeks
    net = trades["move_c"] - cost_cents
    row["t_mean"] = net.mean() / (net.std() / np.sqrt(len(net)))
    return row


def decile_bridge(bars, s: pd.Series, cost_cents: float) -> pd.DataFrame:
    """Layer 1's deciles in cents: every bar of the IC sample, entered next open in the signal's direction."""
    open_w, close_w, segment = bars["W_open"].values, bars["W"].values, bars["segment"].values
    rows = []
    for hold_name, hold in HOLDS.items():
        t = np.flatnonzero(s.notna().values)
        t = t[t + 1 + hold < len(bars)]
        t = t[segment[t + 1 + hold] == segment[t + 1]]
        frame = pd.DataFrame({"s": s.values[t], "wheat_move_c": close_w[t + 1 + hold] - open_w[t + 1]})
        frame["decile"] = pd.qcut(frame["s"], 10, labels=range(1, 11))
        frame["trade_c"] = np.sign(frame["s"]) * frame["wheat_move_c"] - cost_cents
        grouped = frame.groupby("decile", observed=True)
        table = grouped.agg(bars=("s", "count"), s_from=("s", "min"), s_to=("s", "max"),
                            wheat_move_c=("wheat_move_c", "mean"), net_trade_c=("trade_c", "mean"))
        table["hit_rate"] = grouped["trade_c"].apply(lambda x: (x > 0).mean())
        rows.append(table.reset_index().assign(hold=hold_name))
    return pd.concat(rows)


# ------------------------------------------------------------------ charts
def bridge_chart(bridge: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10, 4.4))
    bridge.pivot_table(index="decile", columns=["scale", "hold"], values="wheat_move_c").plot.bar(ax=axis, rot=0)
    axis.axhline(0, color="grey", lw=0.8)
    axis.set_xlabel("signal decile (1 = wheat richest vs corn -> sell, 10 = cheapest -> buy)")
    axis.set_ylabel("wheat's average move after entry, cents")
    axis.set_title("Layer 1 deciles in cents (IC sample, next-open entry)", loc="left")
    figure.tight_layout()
    return figure


def path_chart(bars, events_by_mode: dict, cost_cents: float) -> plt.Figure:
    """Average move after entry over 2 sessions for the focus tail, dotted line = 1 session."""
    open_w, close_w, segment = bars["W_open"].values, bars["W"].values, bars["segment"].values
    figure, axes = plt.subplots(1, len(events_by_mode), figsize=(12, 4.4), sharey=True)
    hours = np.arange(143) / 4
    for axis, (mode, events) in zip(axes, events_by_mode.items()):
        paths = []
        for t, side in events:
            entry = t + 1
            stop = min(entry + 143, len(bars))
            path = side * (close_w[entry:stop] - open_w[entry])
            path = np.where(segment[entry:stop] == segment[entry], path, np.nan)
            paths.append(np.pad(path, (0, 143 - len(path)), constant_values=np.nan))
        paths = np.vstack(paths)
        axis.fill_between(hours, np.nanpercentile(paths, 25, axis=0), np.nanpercentile(paths, 75, axis=0),
                          alpha=0.2, label="middle half of trades")
        axis.plot(hours, np.nanmean(paths, axis=0), lw=2, label="average")
        axis.plot(hours, np.nanmedian(paths, axis=0), lw=1.2, ls="--", label="median")
        axis.axhline(0, color="grey", lw=0.8)
        axis.axhline(cost_cents, color="#d62728", lw=0.8, ls=":", label=f"cost {cost_cents:.2f} c")
        axis.axvline(17.75, color="grey", ls=":", lw=0.8)
        axis.set_title(f"top / bottom {FOCUS_TAIL:.0%}, {mode}: {len(events)} trades", loc="left")
        axis.set_xlabel("hours after entry (dotted = 1 session)")
    axes[0].set_ylabel("move in our favour, cents (before costs)")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    return figure


def year_chart(years: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for axis, mode in zip(axes, ["per move", "per session"]):
        part = years[years["mode"] == mode]
        pivot = part.pivot(index="year", columns="hold", values="mean_c")
        pivot.plot.bar(ax=axis, rot=0)
        counts = part.pivot(index="year", columns="hold", values="trades")
        axis.set_xticks(range(len(pivot)), [f"{year}\nn={counts.loc[year].iloc[0]:.0f}" for year in pivot.index])
        axis.axhline(0, color="grey", lw=0.8)
        axis.set_xlabel("")
        axis.set_ylabel("net cents per trade (all triggers)")
        axis.set_title(f"By year, top / bottom {FOCUS_TAIL:.0%}, {mode}", loc="left")
        axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    bars = load_bars(cfg)                                        # development data only
    features = build_features(bars, cfg.features)
    signals = build_signals(features, cfg.features.level_windows_bars, cfg.features.kalman_memory_sessions)
    ic_sample = signals[[name for _, name in SPEEDS]].notna().all(axis=1)
    scales = {"sd": f"{KALMAN} | M20s", "RMS": BASE}           # same Kalman gap, two ways to size it (RMS = base)
    cost_cents = COST_TICKS * TICK_CENTS
    weeks = (bars.index[-1] - bars.index[0]).days / 7

    close_w, open_w, segment = bars["W"].values, bars["W_open"].values, bars["segment"].values
    rows, year_rows, side_rows, bridges, figures = [], [], [], [], {}
    for hold_name, hold in HOLDS.items():
        starts = np.arange(0, len(bars) - hold - 1, 71)
        same = segment[starts + 1 + hold] == segment[starts + 1]
        moves = (close_w[starts + 1 + hold] - open_w[starts + 1])[same]
        random = pd.DataFrame({"move_c": np.concatenate([moves, -moves])})
        rows.append({"scale": "-", "hold": hold_name, "tail": None, "mode": "random", "sample": "random",
                     **summary(random, cost_cents, weeks)})
    for scale, name in scales.items():
        s = signals[name].where(ic_sample)                       # NaN outside the bars Layer 1 scored
        print(f"{scale}: IC sample {ic_sample.mean():.0%} of bars; share with |s| >= 2: {(s.abs() >= 2).mean():.0%}"
              f" (a normal z: 5%)")
        bridges.append(decile_bridge(bars, s, cost_cents).assign(scale=scale))
        focus_events = {}
        for tail in TAILS:
            rolling = s.rolling(QUANTILE_WINDOW, min_periods=QUANTILE_MIN)
            upper = rolling.quantile(1 - tail).shift(1).values
            lower = rolling.quantile(tail).shift(1).values
            for mode, reset_id in [("per move", bars["segment"].values), ("per session", bars["session"].values)]:
                events = tail_triggers(s.values, upper, lower, reset_id)
                if tail == FOCUS_TAIL:
                    focus_events[mode] = events
                for hold_name, hold in HOLDS.items():
                    trades = trades_for(bars, events, hold)
                    for sample, chosen in [("all triggers", trades), ("flat only", flat_only(trades))]:
                        rows.append({"scale": scale, "hold": hold_name, "tail": tail, "mode": mode,
                                     "sample": sample, **summary(chosen, cost_cents, weeks)})
                    if tail == FOCUS_TAIL:
                        for year, part in trades.groupby("year"):
                            year_rows.append({"scale": scale, "hold": hold_name, "mode": mode, "year": year,
                                              **tail_table(part["move_c"], cost_cents)})
                    if tail == FOCUS_TAIL and mode == "per session" and hold == 71:
                        for (year, side), part in trades.groupby(["year", "side"]):
                            side_rows.append({"scale": scale, "year": year, "side": "long" if side > 0 else "short",
                                              **tail_table(part["move_c"], cost_cents)})
        figures[f"paths_{scale}"] = path_chart(bars, focus_events, cost_cents)
        figures[f"by_year_{scale}"] = year_chart(pd.DataFrame(year_rows).query("scale == @scale"))
    bridge = pd.concat(bridges)
    figures["deciles_in_cents"] = bridge_chart(bridge)
    table = pd.DataFrame(rows).sort_values(["hold", "tail", "mode", "sample", "scale"], na_position="first")
    years = pd.DataFrame(year_rows)
    sides = pd.DataFrame(side_rows)

    pd.set_option("display.width", 260)
    print(f"cost {cost_cents:.2f} c round trip; {weeks:.0f} weeks of development data")
    print("\n== Layer 1 deciles in cents (IC sample, every bar, next-open entry, net trade in the signal's direction)")
    print(bridge.round(2).to_string(index=False))
    shown = ["scale", "hold", "tail", "mode", "sample", "trades", "per_week", "mean_c", "t_mean", "median_c",
             "hit_rate", "avg_win_c", "avg_loss_c", "share_>=+5c", "share_<=-5c", "share_>=+10c", "share_<=-10c"]
    print("\n== wheat outright, tail triggers, NET of costs (cents per trade, wins and losses split)")
    print(table[shown].round(2).to_string(index=False))
    print(f"\n== by year, top / bottom {FOCUS_TAIL:.0%}, all triggers")
    print(years[["scale", "hold", "mode", "year", "trades", "mean_c", "hit_rate", "avg_win_c", "avg_loss_c"]]
          .round(2).to_string(index=False))
    print(f"\n== by year and side, top / bottom {FOCUS_TAIL:.0%}, per session, 1 session hold")
    print(sides[["scale", "year", "side", "trades", "mean_c", "hit_rate", "avg_win_c", "avg_loss_c"]]
          .round(2).to_string(index=False))

    run_dir = save_run(STRATEGY_DIR, "hold_test", cfg, {"tails": TAILS, "holds": HOLDS, "scales": scales},
                       {"deciles_in_cents": bridge, "holds": table, "by_year": years, "by_year_side": sides}, figures)
    print(f"\nsaved to {run_dir}")
