"""Two checks on the spread (development data only).
A. Is the spread stationary, and over what lookback? The development sample is cut into back-to-back windows of
   1 / 3 / 10 / 30 / 120 trading days. In each window: an ADF test (6 lags, 5% level) and the half-life of a pull back
   to the window's own mean. Beans outright alongside as the random-walk yardstick (a random walk 'passes' ~5%).
B. Which z? Three ways to size the gap, for the 2-hour and the 8-hour fair value:
     z, 20-day size   : gap / its root-mean-square over the last 20 trading days (used so far)
     z, 4-hour size   : gap / its exponential size with a 4-hour half-life (the beans config)
     1-hour move      : gap / the spread's expected 1-hour move, level set so 10% of bars are outside
   Decision rule, fixed before the run: prefer the score whose share inside 1.645 stays within 88-92% in every year;
   the trade results are shown but a difference under one standard error does not decide it."""
import contextlib, io, runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from strategies.bean_divergence.fair_value import adaptive_scale

with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(__file__.replace("spread_stationary.py", "spread_trades.py"))
bars, spread, in_dev, half, weeks, find_triggers = (g[k] for k in ["bars", "spread", "in_dev", "half", "weeks", "find_triggers"])
beans = (np.log(bars["ZS_adj_close"]).diff() * bars["ZS_adj_close"] / g["TICK"]).fillna(0).cumsum()   # running beans, ticks
session_id, dates, log_open, log_close = (g[k] for k in ["session_id", "dates", "log_open", "log_close"])
beta_meal, beta_oil, price_ticks, out_dir = (g[k] for k in ["beta_meal", "beta_oil", "price_ticks", "out_dir"])
years = bars["trading_date"].dt.year.values
pd.set_option("display.width", 250)

# ---------------------------------------------------------------- A: stationarity by lookback
dev_bars = np.flatnonzero(in_dev)
rows, half_lives = [], {}
for name, days in [("1 day", 1), ("3 days", 3), ("10 days", 10), ("30 days", 30), ("120 days", 120)]:
    length = days * 211
    row = {"lookback": name, "windows": len(dev_bars) // length}
    for label, series in [("spread", spread.values), ("beans", beans.values)]:
        passed, lives = [], []
        for start in range(0, len(dev_bars) - length + 1, length):
            part = series[dev_bars[start:start + length]]
            passed.append(adfuller(part, maxlag=6, autolag=None)[1] < 0.05)
            slope = np.polyfit(part[:-1], np.diff(part), 1)[0]                      # pull per bar toward the window mean
            lives.append(np.log(0.5) / np.log(1 + slope) / 12 if -1 < slope < 0 else np.inf)   # hours
        row[f"{label}: windows stationary"] = f"{np.mean(passed):.0%}"
        row[f"{label}: half-life, hours (median)"] = round(float(np.median(lives)), 1)
        half_lives[(label, name)] = np.mean(passed)
    rows.append(row)
print("== A. Stationary inside a window of that length? (ADF at 5%; a pure random walk passes about 5% of windows)")
print(pd.DataFrame(rows).to_string(index=False))
whole = adfuller(spread.values[dev_bars][::6], maxlag=12, autolag=None)
print(f"whole development sample, 30-minute steps: ADF statistic {whole[0]:.2f}, p-value {whole[1]:.2f}")

# ---------------------------------------------------------------- B: which z
step_ticks = spread.diff()
move_1h = adaptive_scale(step_ticks, 48) * np.sqrt(12)                              # spread's expected 1-hour move


def book(gap: np.ndarray, score: np.ndarray, level: float) -> pd.DataFrame:
    """Same trade as spread_trades.py: in at the next open, out at fair value or after one trading day."""
    rows, busy_until = [], 0
    for t, rich in find_triggers(score, session_id, level, level * 0.41):          # re-arm at 0.6745 / 1.645 of the level
        entry = t + 1
        if not in_dev[t] or entry < busy_until or entry + 212 >= len(bars):
            continue
        held = lambda log, i: log["ZS"][i] - beta_meal[t] * log["ZM"][i] - beta_oil[t] * log["ZL"][i]
        last = entry
        while last < entry + 211 and rich * gap[last] > 0:
            last += 1
        reached = rich * gap[last] <= 0
        exit_level = held(log_open, last + 1) if reached else held(log_close, last)
        busy_until = last + 1
        rows.append({"date": dates[t], "half": half[t], "net": -rich * (exit_level - held(log_open, entry)) * price_ticks[t] - 1.1})
    return pd.DataFrame(rows)


rows = []
for memory_name, memory_bars in [("2 hours", 24), ("8 hours", 96)]:
    gap = spread - spread.ewm(alpha=1.0 / memory_bars, adjust=False).mean().shift(1)
    scores = {"z, 20-day size": gap / np.sqrt((gap ** 2).rolling(20 * 211, min_periods=10 * 211).mean()).shift(1),
              "z, 4-hour size": gap / adaptive_scale(gap, 48),
              "1-hour move": gap / move_1h}
    for score_name, score in scores.items():
        value = score.abs().values
        level = 1.645 if score_name.startswith("z") else float(np.nanquantile(value[in_dev], 0.90))
        trades = book(gap.values, score.values, level)
        daily = trades.groupby("date")["net"].sum()
        by_half = trades.groupby("half")["net"].mean()
        row = {"fair value": memory_name, "score": score_name, "level": round(level, 2),
               "inside 1": f"{(value[in_dev] < 1).mean():.0%}", "inside level": f"{(value[in_dev] < level).mean():.1%}"}
        for year in [2021, 2022, 2023, 2024]:
            row[f"{year}"] = f"{(value[in_dev & (years == year)] < level).mean():.0%}"
        row.update({"trades a week": round(len(trades) / weeks, 1), "stretch at entry": round(float(np.abs(gap.values)[in_dev & (value >= level)].mean()), 1),
                    "net at 1.1": round(trades["net"].mean(), 2), "std error": round(daily.std() / np.sqrt(len(daily)) * len(daily) / len(trades), 2),
                    "t": round(daily.mean() / (daily.std() / np.sqrt(len(daily))), 1), "won": f"{(trades['net'] > 0).mean():.0%}",
                    "winner / loser": f"{trades.loc[trades.net > 0, 'net'].mean():+.1f} / {trades.loc[trades.net <= 0, 'net'].mean():+.1f}",
                    "worst": round(trades["net"].min(), 0), "2021-22 / 2023-24": f"{by_half['2021-22']:+.2f} / {by_half['2023-Nov24']:+.2f}"})
        rows.append(row)
print("\n== B. Three ways to size the gap. Year columns = share of bars inside the level that year (target 90%). Trades net of 1.1 ticks")
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------- chart
figure, axis = plt.subplots(figsize=(12, 6))
names = ["1 day", "3 days", "10 days", "30 days", "120 days"]
x = np.arange(len(names))
axis.bar(x - 0.2, [half_lives[("spread", n)] * 100 for n in names], width=0.4, color="#9467bd", label="spread")
axis.bar(x + 0.2, [half_lives[("beans", n)] * 100 for n in names], width=0.4, color="#2ca02c", label="beans outright")
axis.axhline(5, color="grey", ls="--", label="what a random walk gives (5%)")
axis.set_xticks(x, names)
axis.set_xlabel("length of the window (trading days)")
axis.set_ylabel("% of windows that test as stationary")
axis.set_title("Is the series stationary inside a window of that length?", loc="left")
axis.legend()
figure.tight_layout()
path = out_dir / "spread_stationarity.png"
figure.savefig(path, dpi=100)
print(path)
