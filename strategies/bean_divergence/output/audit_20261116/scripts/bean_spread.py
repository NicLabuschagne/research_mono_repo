"""Live fair value 10x slower (drift 0.01). Development data only. Three parts.
1. Is the z a real z? Share of bars inside 1 / 1.645 / 2 / 3 against the normal 68.3 / 90 / 95.4 / 99.7%.
2. The SPREAD (beans hedged with meal and oil at the signal bar's ratios, held fixed) against outright beans, after
   four versions of the alert:
     stretch measured against : the expected 1-hour move of BEANS, or of the SPREAD itself
     turn condition           : gap stopped widening AND fair value flat / turned (as drawn), or gap stopped widening only
   Each: live gap stretched 8% of bars, 5-tick floor, session gap stretched the same way (as often as 0.75 x the
   beans move), entry at the next bar's open on all legs.
   Cost: beans 1 tick; spread = meal and oil crossed, beans passive (labels.hedge_cost_ticks, about 1.1 ticks).
   Pass mark, fixed before the run: move at 60 minutes AFTER cost > 0 with t >= 2 (daily sums) and positive in both halves.
3. Charts: the alert on the five windows."""
import contextlib, io, runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategies.bean_divergence.fair_value import STRATEGY_DIR, adaptive_scale, crush_gap, crush_hedge, session_label
from strategies.bean_divergence.labels import hedge_cost_ticks

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(__file__.replace("bean_spread.py", "bean_limit.py"))
bars, in_dev, session_id, half, dates, weeks = (g[k] for k in ["bars", "in_dev", "session_id", "half", "dates", "weeks"])
open_s, close_s, high_s, low_s, TICK = (g[k] for k in ["open_s", "close_s", "high_s", "low_s", "TICK_CENTS"])
session_gap, fair_move, find_triggers, beans_yard = g["session_gap"], g["fair_move"], g["find_triggers"], g["yardstick"]
DRIFT, STRETCHED_SHARE, MIN_GAP_TICKS, TURN_WAIT_BARS, REARM_SHARE = 0.01, 0.08, 5, 6, 0.4

g["kalman_filter"].DRIFT_TO_NOISE_PER_15_MINUTES = DRIFT
hedge = crush_hedge(bars, True, 5)
g["kalman_filter"].DRIFT_TO_NOISE_PER_15_MINUTES = 0.1
gap = crush_gap(bars, hedge)
assert np.allclose(gap.values, g["live_gaps"][DRIFT].values, equal_nan=True)
to_ticks = bars["ZS_adj_close"] / TICK
spread_ret = bars["ZS_ret"] - hedge["beta_meal"] * bars["ZM_ret"] - hedge["beta_oil"] * bars["ZL_ret"]
spread_yard = adaptive_scale(spread_ret * to_ticks, 48) * np.sqrt(12)             # expected 1-hour move of the spread
session = session_label(bars.index, bars["session_type"])
pd.set_option("display.width", 250)

# ---------------------------------------------------------------- 1: is the z a real z?
window = 20 * 211
years = bars["trading_date"].dt.year.values
rows = []
for name, series in [("live (10x slower)", gap), ("session", session_gap)]:
    scores = {"z, 4-hour size": series / adaptive_scale(series, 48),
              "z, 20-day size": series / np.sqrt((series ** 2).rolling(window, min_periods=window // 2).mean()).shift(1),
              "gap / expected 1-hour beans move": series / beans_yard}
    for score_name, score in scores.items():
        value = score.abs().values
        row = {"gap": name, "score": score_name}
        for level in [1, 1.645, 2, 3]:
            row[f"inside {level}"] = f"{(value[in_dev] < level).mean():.1%}"
        for year in [2021, 2022, 2023, 2024]:
            row[f"inside 1.645, {year}"] = f"{(value[in_dev & (years == year)] < 1.645).mean():.0%}"
        rows.append(row)
print("== 1. Share of 5-minute bars inside each level. A normal z gives 68.3% / 90.0% / 95.4% / 99.7%")
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------- 2: the alert, four versions
gap_v = gap.values
agree_share = float(((session_gap.abs() / beans_yard)[in_dev] >= 0.75).mean())   # how often the session gap 'agrees' now


def alerts(yard: pd.Series, need_fair_turn: bool) -> tuple[list[tuple[int, int]], float, float]:
    """(bar, side) of each alert, plus the live and session levels in units of `yard`. side +1 = beans rich -> sell."""
    ratio, s_ratio = gap / yard, (session_gap / yard).values
    level = float(ratio.abs()[in_dev].quantile(1 - STRETCHED_SHARE))
    agree = float(np.nanquantile(np.abs(s_ratio[in_dev]), 1 - agree_share))
    held = np.sign(ratio) * np.minimum(ratio.abs(), level * 0.99)                 # under the tick floor: cannot fire
    stretches = find_triggers(ratio.where(gap.abs() >= MIN_GAP_TICKS, held).values, session_id, level, REARM_SHARE * level)
    found = []
    for t, side in stretches:
        if side * s_ratio[t] < agree:
            continue
        for bar in range(t, min(t + TURN_WAIT_BARS + 1, len(gap_v))):
            if session_id[bar] != session_id[t] or side * gap_v[bar] < MIN_GAP_TICKS:
                break
            if abs(gap_v[bar]) <= abs(gap_v[bar - 1]) and (not need_fair_turn or side * fair_move[bar] <= 0):
                found.append((bar, side))
                break
    return found, level, agree


log_open = {leg: np.log(bars[f"{leg}_adj_open"].values) for leg in ["ZS", "ZM", "ZL"]}
log_close = {leg: np.log(bars[f"{leg}_adj_close"].values) for leg in ["ZS", "ZM", "ZL"]}
beta_meal, beta_oil = hedge["beta_meal"].values, hedge["beta_oil"].values
hedge_cost = hedge_cost_ticks(bars, hedge, 1.0)


def outcomes(events) -> pd.DataFrame:
    """One row per alert and instrument: the move in the fade direction from the next open, ZS ticks, before costs."""
    rows = []
    for t, rich in events:
        entry = t + 1
        if not in_dev[t] or entry >= len(bars) or session_id[entry] != session_id[t]:
            continue
        end = entry
        while end + 1 < len(bars) and session_id[end + 1] == session_id[entry] and end < entry + 23:
            end += 1
        later = np.arange(entry, end + 1)
        hedged = lambda log, i: log["ZS"][i] - beta_meal[t] * log["ZM"][i] - beta_oil[t] * log["ZL"][i]
        paths = {"beans": (-rich * (close_s[later] - open_s[entry]) / TICK, 1.0),
                 "spread": (-rich * (hedged(log_close, later) - hedged(log_open, entry)) * close_s[t] / TICK, hedge_cost[t])}
        for instrument, (path, cost) in paths.items():
            rows.append({"instrument": instrument, "date": dates[t], "half": half[t], "session": session[t], "cost": cost,
                         "30m": path[min(5, len(path) - 1)], "60m": path[min(11, len(path) - 1)], "120m": path[-1],
                         "best": max(path[:12].max(), 0), "worst": min(path[:12].min(), 0)})
    frame = pd.DataFrame(rows)
    frame["net 60m"] = frame["60m"] - frame["cost"]
    return frame


def mean_t(part: pd.DataFrame, column: str) -> str:
    if len(part) < 30:
        return f"n={len(part)}"
    daily = part.groupby("date")[column].sum()
    return f"{part[column].mean():+.2f} ({daily.mean() / (daily.std() / np.sqrt(len(daily))):+.1f})"


VERSIONS = {"beans yardstick, gap + fair value turn (as drawn)": (beans_yard, True),
            "beans yardstick, gap stopped widening only": (beans_yard, False),
            "spread yardstick, gap + fair value turn": (spread_yard, True),
            "spread yardstick, gap stopped widening only": (spread_yard, False)}
found = {name: alerts(yard, need) for name, (yard, need) in VERSIONS.items()}
print(f"\n== 2. Beans against the spread after each alert. 'ahead' is before costs, 'net 60m' after. mean (t from daily sums). "
      f"Session gap agrees on {agree_share:.0%} of bars")
table, splits = [], []
for name, (events, level, agree) in found.items():
    frame = outcomes(events)
    for instrument, part in frame.groupby("instrument"):
        won, lost = part.loc[part["60m"] > 0, "60m"], part.loc[part["60m"] <= 0, "60m"]
        table.append({"alert": name, "trade": instrument, "a week": round(len(part) / weeks, 1),
                      "levels (live / session)": f"{level:.2f} / {agree:.2f}", "ahead 30m": mean_t(part, "30m"),
                      "ahead 60m": mean_t(part, "60m"), "ahead 120m": mean_t(part, "120m"), "cost": round(part["cost"].mean(), 2),
                      "net 60m": mean_t(part, "net 60m"), "our way at 60m": f"{(part['60m'] > 0).mean():.0%}",
                      "winner / loser 60m": f"{won.mean():+.1f} / {lost.mean():+.1f}",
                      "best / worst 60m": f"{part['best'].median():.1f} / {part['worst'].median():.1f}"})
        split = {"alert": name, "trade": instrument}
        for column in ["half", "session"]:
            for key, piece in part.groupby(column):
                split[key] = f"{mean_t(piece, 'net 60m')} {len(piece) / weeks:.1f}/wk"
        splits.append(split)
print(pd.DataFrame(table).set_index(["alert", "trade"]).T.to_string())
print("\n-- net 60m (after costs) by half and by session: mean (t) and alerts a week")
print(pd.DataFrame(splits).set_index(["alert", "trade"]).to_string())

# ---------------------------------------------------------------- 3: charts
SHOWN = [("beans yardstick, gap + fair value turn (as drawn)", "#1f77b4", 300, "alert, stretch against the beans move"),
         ("spread yardstick, gap stopped widening only", "#9467bd", 130, "alert, stretch against the spread's own move")]
trading_dates = np.array(sorted(bars["trading_date"].unique()))
out_dir = STRATEGY_DIR / "output" / "audit_20261116"
for window_start in ["2021-03-15", "2022-09-12", "2023-06-12", "2024-02-12", "2024-11-21"]:
    days = trading_dates[trading_dates >= pd.Timestamp(window_start)][:3]
    shown = np.flatnonzero(bars["trading_date"].isin(days).values)
    first, last = shown[0], shown[-1] + 1
    x = np.arange(last - first)
    open_, close, high, low = open_s[first:last], close_s[first:last], high_s[first:last], low_s[first:last]
    colours = np.where(close >= open_, "#2ca02c", "#d62728")
    reach = (high.max() - low.min()) * 0.03
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(24, 13), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    top.vlines(x, low, high, color=colours, lw=0.7, alpha=0.6)
    top.bar(x, np.maximum(np.abs(close - open_), 0.05), bottom=np.minimum(open_, close), width=0.7, color=colours, alpha=0.6)
    top.plot(x, close - gap_v[first:last] * TICK, color="black", lw=1.3, label="live fair value, 10x slower")
    top.plot(x, close - session_gap.values[first:last] * TICK, color="darkorange", lw=1.3, label="session fair value")
    bottom.bar(x, gap_v[first:last], width=0.8, color="grey", alpha=0.6, label="live gap (ticks)")
    bottom.plot(x, session_gap.values[first:last], color="darkorange", lw=1.3, label="session gap (ticks)")
    for row, (name, colour, size, label) in enumerate(SHOWN):
        events, level, _ = found[name]
        chosen = [(t - first, side) for t, side in events if first <= t < last]
        for sign, marker in [(1, "v"), (-1, "^")]:
            xs = np.array([i for i, side in chosen if side == sign], dtype=int)
            ys = high[xs] + reach * (1 + 1.6 * row) if sign == 1 else low[xs] - reach * (1 + 1.6 * row)
            top.scatter(xs, ys, marker=marker, s=size, color=colour, edgecolor="black", linewidth=1.5, zorder=5,
                        label=f"{label}: {len(chosen)}" if sign == 1 else None)
            bottom.plot(x, sign * level * VERSIONS[name][0].values[first:last], color=colour, ls="--", lw=1.1,
                        label=f"live level, {label.split('against ')[1]}" if sign == 1 else None)
    ids = session_id[first:last]
    local = bars.index[first:last].tz_convert("Europe/London")
    ticks = np.arange(0, len(x), 24)
    bottom.set_xticks(ticks, [f"{local[i]:%d %b\n%H:%M}" for i in ticks], fontsize=8)
    for axis in (top, bottom):
        for session_start in np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]):
            axis.axvline(session_start - 0.5, color="grey", lw=0.8, alpha=0.7)
        axis.grid(axis="y", alpha=0.3)
    bottom.axhline(0, color="grey", lw=0.8)
    top.set_ylabel("soybeans, cents per bushel")
    bottom.set_ylabel("gap in ticks: + = beans above fair value")
    top.set_title(f"{pd.Timestamp(days[0]):%d %b %Y} to {pd.Timestamp(days[-1]):%d %b %Y}, 5-minute bars, live fair value 10x slower. "
                  f"Triangle down = beans rich (sell), up = beans cheap (buy)", loc="left")
    top.legend(loc="upper left", ncol=4)
    bottom.legend(loc="upper left", ncol=4)
    top.set_xlim(-1, len(x) + 1)
    figure.tight_layout()
    path = out_dir / f"alert_slow_{window_start}.png"
    figure.savefig(path, dpi=100)
    plt.close(figure)
    print(path)
