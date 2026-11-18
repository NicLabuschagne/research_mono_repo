"""The SPREAD on its own terms, step 1 (development data only): does it revert, to what, and over how long?
  spread   : beans minus the hedge (meal and oil at the slow Kalman ratios known before each bar), as one running
             series across sessions, in ZS ticks. Only the spread is traded, so only the spread's own move counts.
  A        : variance ratio. Under 1 = the series reverts over that horizon, over 1 = it trends. Beans alongside.
  B        : memory x horizon. Fair level = the spread's own average with a memory of 30 minutes to 30 days.
             Stretched = |gap| >= 1.645 x its 20-day size (10% of bars). Outcome = the spread's move back toward the
             level over 30 minutes to 3 days, measured from one bar AFTER the signal bar.
  cost     : 1.1 ticks with beans passive, 2.1 with every leg crossed.
  pass     : fixed before the run: a cell is a lead if the move back > 2.1 ticks, t >= 2 (from monthly means, so
             overlapping holds do not inflate it) and positive in both halves."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import strategies.crush_strategy.kalman_filter as kalman_filter
from strategies.bean_divergence.fair_value import STRATEGY_DIR, TICK_CENTS, crush_hedge, half_label, load, load_bars

BARS_PER_DAY = 211
MEMORIES = {"30 min": 6, "2 hours": 24, "8 hours": 96, "1 day": 211, "3 days": 633, "10 days": 2110, "30 days": 6330}
HORIZONS = {"30 min": 6, "1 hour": 12, "2 hours": 24, "4 hours": 48, "1 day": 211, "3 days": 633}
STRETCH_Z = 1.645

pd.set_option("display.width", 250)
cfg = load()
bars = load_bars(cfg)
in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
half = half_label(bars["trading_date"])
month = bars["trading_date"].dt.to_period("M").values
kalman_filter.DRIFT_TO_NOISE_PER_15_MINUTES = 0.01
hedge = crush_hedge(bars, True, 5)
kalman_filter.DRIFT_TO_NOISE_PER_15_MINUTES = 0.1

to_ticks = bars["ZS_adj_close"] / TICK_CENTS
log_close = {leg: np.log(bars[f"{leg}_adj_close"]) for leg in ["ZS", "ZM", "ZL"]}
step = (log_close["ZS"].diff() - hedge["beta_meal"] * log_close["ZM"].diff() - hedge["beta_oil"] * log_close["ZL"].diff()) * to_ticks
spread = step.fillna(0).cumsum()                                                   # running spread, ZS ticks
beans = (log_close["ZS"].diff() * to_ticks).fillna(0).cumsum()

# ---------------------------------------------------------------- A: variance ratio
vr_rows = []
for name, bars_ahead in {"5 min": 1, **HORIZONS, "10 days": 2110}.items():
    row = {"horizon": name}
    for label, series in [("spread", spread), ("beans", beans)]:
        one = series.diff()[in_dev].var()
        row[f"{label}: variance ratio"] = series.diff(bars_ahead)[in_dev].var() / (bars_ahead * one)
        row[f"{label}: typical move (ticks)"] = series.diff(bars_ahead)[in_dev].std()
    vr_rows.append(row)
vr = pd.DataFrame(vr_rows).set_index("horizon")
print("== A. Variance ratio against 5-minute moves (1 = random walk, under 1 = reverts) and the typical move")
print(vr.round(2).to_string())

# ---------------------------------------------------------------- B: memory x horizon
def mean_t(values: pd.Series) -> tuple[float, float]:
    monthly = values.groupby(month[values.index]).mean()                           # values is indexed by bar number
    return values.mean(), monthly.mean() / (monthly.std() / np.sqrt(len(monthly)))


spread_v = spread.values
cells, tables = [], {"move back (ticks)": {}, "t": {}, "share of gap closed": {}, "rank correlation": {}, "halves": {}}
for memory_name, memory_bars in MEMORIES.items():
    gap = spread - spread.ewm(alpha=1.0 / memory_bars, adjust=False).mean().shift(1)
    size = np.sqrt((gap ** 2).rolling(20 * BARS_PER_DAY, min_periods=10 * BARS_PER_DAY).mean()).shift(1)
    z = (gap / size).values
    for horizon_name, horizon_bars in HORIZONS.items():
        ahead = np.full(len(bars), np.nan)
        ahead[:-horizon_bars - 1] = spread_v[1 + horizon_bars:] - spread_v[1:-horizon_bars]   # from one bar after the signal
        usable = in_dev & ~np.isnan(z) & ~np.isnan(ahead)
        stretched = np.flatnonzero(usable & (np.abs(z) >= STRETCH_Z))
        back = pd.Series(-np.sign(z[stretched]) * ahead[stretched], index=stretched)
        mean, t = mean_t(back)
        by_half = back.groupby(half[stretched]).mean()
        closed = mean / np.abs(gap.values[stretched]).mean()
        rank = pd.Series(-gap.values[usable]).corr(pd.Series(ahead[usable]), method="spearman")
        key = (memory_name, horizon_name)
        tables["move back (ticks)"][key], tables["t"][key] = mean, t
        tables["share of gap closed"][key], tables["rank correlation"][key] = closed, rank
        tables["halves"][key] = f"{by_half.get('2021-22', np.nan):+.1f} / {by_half.get('2023-Nov24', np.nan):+.1f}"
        cells.append({"memory": memory_name, "hold": horizon_name, "gap when stretched": np.abs(gap.values[stretched]).mean(),
                      "move back": mean, "t": t, "won": (back > 0).mean(), "winner": back[back > 0].mean(), "loser": back[back <= 0].mean(),
                      "2021-22": by_half.get("2021-22", np.nan), "2023-Nov24": by_half.get("2023-Nov24", np.nan),
                      "lead": mean > 2.1 and t >= 2 and by_half.min() > 0})


def grid(name: str) -> pd.DataFrame:
    return pd.Series(tables[name]).unstack().reindex(index=list(MEMORIES), columns=list(HORIZONS))


print(f"\n== B. Spread stretched (|z| >= {STRETCH_Z}) from its own average. Rows = memory of the average, columns = how long held.")
for name in ["move back (ticks)", "t", "share of gap closed", "rank correlation"]:
    print(f"\n-- {name}" + (" BEFORE costs (1.1 beans passive, 2.1 all crossed)" if name.startswith("move") else ""))
    print(grid(name).round(2 if name != "rank correlation" else 3).to_string())
print("\n-- move back by half: 2021-22 / 2023-Nov24")
print(grid("halves").to_string())
cells = pd.DataFrame(cells)
print("\n-- cells that pass (move back > 2.1, t >= 2, both halves positive); winners and losers in ticks")
print(cells[cells["lead"]].drop(columns="lead").round(2).to_string(index=False))

# ---------------------------------------------------------------- charts
figure, axes = plt.subplots(1, 3, figsize=(26, 8))
axes[0].plot(vr.index, vr["spread: variance ratio"], marker="o", color="#9467bd", label="spread")
axes[0].plot(vr.index, vr["beans: variance ratio"], marker="o", color="#2ca02c", label="beans outright")
axes[0].axhline(1, color="grey", ls="--")
axes[0].set_title("Variance ratio: under 1 = the series reverts over that horizon", loc="left")
axes[0].set_xlabel("horizon")
axes[0].legend()
for axis, name, title in [(axes[1], "move back (ticks)", "Spread move back toward its average after a stretch, ticks before costs"),
                          (axes[2], "t", "t-statistic of that move (monthly means)")]:
    values = grid(name)
    limit = np.abs(values.values).max()
    axis.imshow(values.values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(HORIZONS)), list(HORIZONS))
    axis.set_yticks(range(len(MEMORIES)), list(MEMORIES))
    for i in range(len(MEMORIES)):
        for j in range(len(HORIZONS)):
            axis.text(j, i, f"{values.values[i, j]:+.1f}", ha="center", va="center", fontsize=11)
    axis.set_xlabel("how long held")
    axis.set_ylabel("memory of the average")
    axis.set_title(title, loc="left")
figure.tight_layout()
path = STRATEGY_DIR / "output" / "audit_20261116" / "spread_memory.png"
figure.savefig(path, dpi=100)
print(path)

# the spread itself with three of its averages, the last 15 trading days of development
days = sorted(bars["trading_date"].unique())[-15:]
shown = np.flatnonzero(bars["trading_date"].isin(days).values)
x = np.arange(len(shown))
figure, axis = plt.subplots(figsize=(24, 9))
axis.plot(x, spread_v[shown], color="black", lw=0.9, label="spread (beans minus hedge), ticks")
for (memory_name, colour) in [("2 hours", "#1f77b4"), ("1 day", "darkorange"), ("10 days", "#d62728")]:
    level = spread.ewm(alpha=1.0 / MEMORIES[memory_name], adjust=False).mean().shift(1).values[shown]
    axis.plot(x, level, color=colour, lw=1.4, label=f"its average, memory {memory_name}")
day_starts = np.flatnonzero(np.r_[True, bars["trading_date"].values[shown][1:] != bars["trading_date"].values[shown][:-1]])
axis.set_xticks(day_starts, [f"{pd.Timestamp(d):%d %b}" for d in bars["trading_date"].values[shown][day_starts]])
for start in day_starts:
    axis.axvline(start, color="grey", lw=0.6, alpha=0.6)
axis.set_ylabel("ZS ticks")
axis.set_title("The spread as one running series, 5-minute bars, with three of its own averages", loc="left")
axis.legend(loc="upper left")
axis.grid(axis="y", alpha=0.3)
figure.tight_layout()
path = STRATEGY_DIR / "output" / "audit_20261116" / "spread_series.png"
figure.savefig(path, dpi=100)
print(path)
