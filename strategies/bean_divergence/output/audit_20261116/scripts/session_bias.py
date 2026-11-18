"""Does the market move TOWARD the session fair value? (development data only)
At every 5-minute close the bias is the side the session fair value sits on: fair value above beans = up, below = down.
We then measure, from the next open, how far each piece moves in the bias direction (ZS ticks, before costs), inside
the same session (the fair value is re-anchored at the next one):
  beans    : outright beans - this is the directional-filter question
  products : the meal and oil hedge moving TOWARD beans (closing the gap from the other side)
  spread   : beans plus products = how much of the gap closed
Split by how far apart they are (|gap| in ticks), by horizon, by session and by half.
Each number is the average over trading days of that day's mean, so the t-statistic matches it.
Pass mark, fixed before the run: the session fair value works as a directional filter for beans if the beans move
is > 0 with t >= 2 in both halves, for gaps of 5 ticks or more, at one hour or longer."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategies.bean_divergence.fair_value import (STRATEGY_DIR, TICK_CENTS, crush_gap, crush_hedge, half_label, load,
                                                   load_bars, session_label)

pd.set_option("display.width", 250)
cfg = load()
bars = load_bars(cfg)
in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
session_id, dates = bars["session_id"].values, bars["trading_date"].values
session, half = session_label(bars.index, bars["session_type"]), half_label(bars["trading_date"])
hedge = crush_hedge(bars, False, 5)
gap = crush_gap(bars, hedge).values                                                # + = beans above session fair value
beta_meal, beta_oil = hedge["beta_meal"].values, hedge["beta_oil"].values
log_open = {leg: np.log(bars[f"{leg}_adj_open"].values) for leg in ["ZS", "ZM", "ZL"]}
log_close = {leg: np.log(bars[f"{leg}_adj_close"].values) for leg in ["ZS", "ZM", "ZL"]}
price_ticks = bars["ZS_adj_close"].values / TICK_CENTS
n = len(bars)
t = np.arange(n)
session_last = np.searchsorted(session_id, session_id, side="right") - 1          # last bar of each bar's session
bias = -np.sign(gap)                                                               # +1 = fair value above beans
usable = in_dev & ~np.isnan(gap) & (bias != 0) & (t + 1 <= session_last)

HORIZONS = {"30 min": 6, "1 hour": 12, "2 hours": 24, "4 hours": 48, "session end": 10 ** 6}
BUCKETS = {"any gap": (0, 1e9), "under 5 ticks": (0, 5), "5 to 10": (5, 10), "10 to 20": (10, 20), "20 or more": (20, 1e9)}


def moves(bars_ahead: int) -> dict[str, np.ndarray]:
    end = np.minimum(t + bars_ahead, session_last)
    entry = np.minimum(t + 1, n - 1)
    leg = lambda name: log_close[name][end] - log_open[name][entry]
    beans = leg("ZS") * price_ticks
    hedge_move = (beta_meal * leg("ZM") + beta_oil * leg("ZL")) * price_ticks
    return {"beans": bias * beans, "products": -bias * hedge_move, "spread": bias * (beans - hedge_move)}


def day_mean_t(values: np.ndarray, chosen: np.ndarray) -> tuple[float, float, int]:
    by_day = pd.Series(values[chosen]).groupby(dates[chosen]).mean()
    return by_day.mean(), by_day.mean() / (by_day.std() / np.sqrt(len(by_day))), len(by_day)


rows, curves = [], {}
for horizon, bars_ahead in HORIZONS.items():
    piece = moves(bars_ahead)
    for bucket, (low, high) in BUCKETS.items():
        chosen = usable & (np.abs(gap) >= low) & (np.abs(gap) < high)
        row = {"gap": bucket, "held": horizon, "share of bars": f"{chosen.sum() / usable.sum():.0%}"}
        for name in ["beans", "products", "spread"]:
            mean, tstat, _ = day_mean_t(piece[name], chosen)
            row[name] = f"{mean:+.2f} ({tstat:+.1f})"
            curves[(bucket, horizon, name)] = mean
        row["beans our way"] = f"{(piece['beans'][chosen] > 0).mean():.0%}"
        for key in ["2021-22", "2023-Nov24"]:
            mean, tstat, _ = day_mean_t(piece["beans"], chosen & (half == key))
            row[f"beans {key}"] = f"{mean:+.2f} ({tstat:+.1f})"
        for key in ["asia", "europe", "us"]:
            mean, tstat, _ = day_mean_t(piece["beans"], chosen & (session == key))
            row[f"beans {key}"] = f"{mean:+.2f} ({tstat:+.1f})"
        rows.append(row)
table = pd.DataFrame(rows)
print("== Move TOWARD the session fair value from the next open, ticks before costs: mean (t), averaged by day")
for bucket in BUCKETS:
    print(f"\n-- gap {bucket}")
    print(table[table["gap"] == bucket].drop(columns="gap").to_string(index=False))

figure, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
x = np.arange(len(HORIZONS))
for axis, name, title in zip(axes, ["beans", "products", "spread"],
                             ["BEANS move toward the session fair value", "PRODUCTS (meal and oil hedge) move toward beans", "SPREAD: how much of the gap closed"]):
    for bucket, colour in zip(list(BUCKETS)[1:], ["#bdbdbd", "#6baed6", "#3182bd", "#08306b"]):
        axis.plot(x, [curves[(bucket, horizon, name)] for horizon in HORIZONS], marker="o", color=colour, lw=2, label=f"gap {bucket}")
    axis.axhline(0, color="grey", lw=0.8)
    axis.set_xticks(x, list(HORIZONS))
    axis.set_title(title, loc="left")
    axis.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("ticks in the direction of fair value, before costs")
axes[0].legend()
figure.tight_layout()
path = STRATEGY_DIR / "output" / "audit_20261116" / "session_bias.png"
figure.savefig(path, dpi=100)
print(path)
