"""
edge_map.py
-----------
Layer 2, step 1: what the Layer 1 edge (signal 1a, model A composite) looks like AFTER an entry, before any rule
is chosen. Development data, IN-SAMPLE description (the rule chosen from it is then walk-forward tested).
Saved to runs/<time>_edge_map/.

    python -m strategies.wheat_corn_lag.edge_map

Signal   s = -(own_ze_106 + own_ze_142 + own_ze_213) / 3   (+ = wheat cheap vs corn -> buy)
Events   first bar where |s| crosses a threshold (1.0 / 1.5 / 2.0); next event only after the longest horizon,
         so events never share a forward window. Direction = sign(s).
Fill     signal at bar t's close, entry at bar t+1's OPEN (as backtest.py); costs 1.4 ticks round trip
Size     D = |own_move_142| at the signal (how far wheat overshot corn, log units), the unit for barriers

Per threshold
  forward move at 1 / 2 / 3 / 4 sessions (71 / 142 / 213 / 284 bars) in cents, with the WIN side and the LOSS
  side shown next to the mean (share of trades beyond +-5 c and +-10 c), against RANDOM entries at the same horizon
  excursions over 4 sessions: best move for us (MFE) and worst move against us (MAE), in units of D
  barrier race: target T x D and stop S x D, time stop 2 or 4 sessions: share target first / stop first / neither,
  and the average result per trade after costs
Paths stop at a segment edge (a possible roll), like every return in this project.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import save_run
from core_quant.theme import set_theme
from strategies.wheat_corn_lag.features import build_features
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load, load_bars

THRESHOLDS = [1.0, 1.5, 2.0]
HORIZONS = [71, 142, 213, 284]
MAX_HOLD = max(HORIZONS)
TICK_CENTS, COST_TICKS = 0.25, 1.4
TAILS_CENTS = [5, 10]
TARGETS, STOPS, HOLDS = [0.25, 0.5, 0.75, 1.0], [1.0, 2.0], [142, 284]


def entry_paths(bars, signal: pd.Series, divergence: pd.Series, threshold: float) -> pd.DataFrame:
    """One row per event: side, D, and the move for us (cents) at every bar from entry, NaN past a segment edge."""
    s = signal.values
    open_w, close_w, segment = bars["W_open"].values, bars["W"].values, bars["segment"].values
    above = np.abs(s) > threshold
    crossing = above & ~np.concatenate([[False], above[:-1]])
    rows, next_allowed = [], 0
    for t in np.flatnonzero(crossing):
        entry = t + 1
        if t < next_allowed or entry + MAX_HOLD >= len(bars) or segment[entry] != segment[t]:
            continue
        next_allowed = t + MAX_HOLD
        side = np.sign(s[t])
        path = side * (close_w[entry:entry + MAX_HOLD + 1] - open_w[entry])          # cents, entry bar's close first
        # segments only increase, so once a path crosses an edge (possible roll) it stays NaN
        path = np.where(segment[entry:entry + MAX_HOLD + 1] == segment[entry], path, np.nan)
        rows.append({"time": bars.index[t], "side": side, "signal": s[t], "price": open_w[entry],
                     "D_cents": divergence.values[t] * open_w[entry], "path": path})
    return pd.DataFrame(rows).set_index("time")


def tail_table(moves_cents: pd.Series, cost_cents: float) -> dict:
    net = moves_cents.dropna() - cost_cents
    row = {"trades": len(net), "mean_c": net.mean(), "median_c": net.median(), "hit_rate": (net > 0).mean(),
           "avg_win_c": net[net > 0].mean(), "avg_loss_c": net[net <= 0].mean()}
    for tail in TAILS_CENTS:
        row[f"share_>=+{tail}c"] = (net >= tail).mean()
        row[f"share_<=-{tail}c"] = (net <= -tail).mean()
    return row


def barrier_race(events: pd.DataFrame, cost_cents: float) -> pd.DataFrame:
    rows = []
    for hold in HOLDS:
        for stop in STOPS:
            for target in TARGETS:
                outcomes, results = [], []
                for _, event in events.iterrows():
                    path = event["path"][:hold + 1]
                    up, down = target * event["D_cents"], stop * event["D_cents"]
                    hit_target = np.flatnonzero(path >= up)
                    hit_stop = np.flatnonzero(path <= -down)
                    first_target = hit_target[0] if len(hit_target) else np.inf
                    first_stop = hit_stop[0] if len(hit_stop) else np.inf
                    if first_target < first_stop:
                        outcomes.append("target"); results.append(path[first_target])
                    elif first_stop < np.inf:
                        outcomes.append("stop"); results.append(path[first_stop])
                    else:
                        last = path[np.isfinite(path)][-1] if np.isfinite(path).any() else 0.0
                        outcomes.append("time"); results.append(last)      # time stop, or the last bar before a roll cut
                outcomes, results = pd.Series(outcomes), pd.Series(results) - cost_cents
                rows.append({"hold_sessions": hold // 71, "stop_xD": stop, "target_xD": target,
                             "target_first": (outcomes == "target").mean(), "stop_first": (outcomes == "stop").mean(),
                             "time_exit": (outcomes == "time").mean(), "mean_c": results.mean(),
                             "avg_win_c": results[results > 0].mean(), "avg_loss_c": results[results <= 0].mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ charts
def path_chart(events_by_threshold: dict) -> plt.Figure:
    figure, axes = plt.subplots(1, len(events_by_threshold), figsize=(5.2 * len(events_by_threshold), 4.4), sharey=True)
    hours = np.arange(MAX_HOLD + 1) / 4
    for axis, (threshold, events) in zip(axes, events_by_threshold.items()):
        paths = np.vstack(events["path"].values)
        axis.fill_between(hours, np.nanpercentile(paths, 25, axis=0), np.nanpercentile(paths, 75, axis=0), alpha=0.2,
                          label="middle half of trades")
        axis.plot(hours, np.nanmean(paths, axis=0), lw=2, label="average")
        axis.plot(hours, np.nanmedian(paths, axis=0), lw=1.2, ls="--", label="median")
        axis.axhline(0, color="grey", lw=0.8)
        for session in range(1, 5):
            axis.axvline(session * 17.75, color="grey", ls=":", lw=0.7)
        axis.set_title(f"|signal| > {threshold}: {len(events)} trades", loc="left")
        axis.set_xlabel("hours after entry (dotted = sessions)")
    axes[0].set_ylabel("move in our favour, cents (before costs)")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    return figure


def excursion_chart(events: pd.DataFrame, threshold: float) -> plt.Figure:
    paths = np.vstack(events["path"].values)
    mfe = np.nanmax(paths, axis=1) / events["D_cents"].values
    mae = -np.nanmin(paths, axis=1) / events["D_cents"].values
    final = paths[:, -1]
    figure, axis = plt.subplots(figsize=(7, 5.5))
    colour = np.where(final > 0, "#2ca02c", "#d62728")
    axis.scatter(mae, mfe, c=colour, s=14, alpha=0.7)
    axis.set_xlabel("worst move against us over 4 sessions, x D")
    axis.set_ylabel("best move for us over 4 sessions, x D")
    axis.set_title(f"Excursions, |signal| > {threshold} (green = up after 4 sessions)", loc="left")
    axis.set_xlim(0, np.nanpercentile(mae, 98))
    axis.set_ylim(0, np.nanpercentile(mfe, 98))
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    bars = load_bars(cfg)                                       # development data only
    features = build_features(bars, cfg.features)
    signal = -features[[f"own_ze_{l}" for l in (106, 142, 213)]].mean(axis=1, skipna=False)
    divergence = features["own_move_142"].abs()
    cost_cents = COST_TICKS * TICK_CENTS
    weeks = (bars.index[-1] - bars.index[0]).days / 7

    # random baseline: the same horizons from every 71st bar, long and short equally (no signal)
    close_w, open_w, segment = bars["W"].values, bars["W_open"].values, bars["segment"].values
    random_moves = {}
    for h in HORIZONS:
        starts = np.arange(0, len(bars) - h - 1, 71)
        ok = segment[starts + 1 + h] == segment[starts + 1]
        moves = (close_w[starts + 1 + h] - open_w[starts + 1])[ok]
        random_moves[h] = pd.Series(np.concatenate([moves, -moves]))

    pd.set_option("display.width", 240)
    events_by_threshold, tables = {}, {}
    for threshold in THRESHOLDS:
        events = entry_paths(bars, signal, divergence, threshold)
        events_by_threshold[threshold] = events
        print(f"\n######## |signal| > {threshold}: {len(events)} trades ({len(events) / weeks:.2f} per week),"
              f" median D {events['D_cents'].median():.1f} c")
        rows = []
        for h in HORIZONS:
            moves = pd.Series([p[h] for p in events["path"]])
            rows.append({"horizon_sessions": h / 71, "who": "signal", **tail_table(moves, cost_cents)})
            rows.append({"horizon_sessions": h / 71, "who": "random", **tail_table(random_moves[h], cost_cents)})
        table = pd.DataFrame(rows)
        tables[f"forward_{threshold}"] = table
        print("forward move after costs, cents (mean is NET of wins and losses):")
        print(table.round(2).to_string(index=False))
        paths = np.vstack(events["path"].values)
        mfe = np.nanmax(paths, axis=1) / events["D_cents"].values
        mae = -np.nanmin(paths, axis=1) / events["D_cents"].values
        print(f"excursions over 4 sessions in units of D: best for us median {np.nanmedian(mfe):.2f} (p25 {np.nanpercentile(mfe, 25):.2f},"
              f" p75 {np.nanpercentile(mfe, 75):.2f}); worst against us median {np.nanmedian(mae):.2f}"
              f" (p75 {np.nanpercentile(mae, 75):.2f}, p90 {np.nanpercentile(mae, 90):.2f})")
        race = barrier_race(events, cost_cents)
        tables[f"barrier_race_{threshold}"] = race
        print("barrier race (target / stop in units of D; mean after costs, cents):")
        print(race.round(2).to_string(index=False))

    figures = {"paths": path_chart(events_by_threshold), "excursions_1.5": excursion_chart(events_by_threshold[1.5], 1.5)}
    run_dir = save_run(STRATEGY_DIR, "edge_map", cfg, {"thresholds": THRESHOLDS}, tables, figures)
    print(f"\nsaved to {run_dir}")
