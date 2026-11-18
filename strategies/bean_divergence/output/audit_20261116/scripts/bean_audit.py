"""AUDIT (layer 1, statistics only - no trades, no costs): is the bean Kalman fair value built correctly, and how
fast should the fair value be for an OUTRIGHT beans signal? Development data only (the loader stops at the holdout).

Part A  the live Kalman as it is: how much of each error it absorbs per bar, into which coefficient, and whether
        its hedge ratios look like hedge ratios.
Part B  what closes a live gap: beans coming back, the products catching up, or the filter refitting.
Part C  memory test. Three families of fair value, each scored the same way against the forward beans move:
          kalman  : the same level filter at slower speeds (drift-to-noise per 15 min)
          session : hedge frozen at the previous session close (the orange line on the user's chart)
          anchored: fair value moves with meal and oil by the RETURNS hedge ratios (20 trading days of 5-minute
                    returns, lagged) and is pulled toward beans at 1 / memory (the wheat rule's construction)
"""
import numpy as np
import pandas as pd

import strategies.crush_strategy.kalman_filter as kalman_filter
from strategies.bean_divergence.fair_value import (TICK_CENTS, adaptive_scale, crush_gap, crush_hedge, half_label, load,
                                                   load_bars, session_label)

pd.set_option("display.width", 250)
cfg = load()
bars = load_bars(cfg)
in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
bars_per_day = int(bars[in_dev].groupby("trading_date").size().median())
session = session_label(bars.index, bars["session_type"])
half = half_label(bars["trading_date"])
price = bars["ZS_adj_close"]
log_s, log_m, log_o = (np.log(bars[f"{leg}_adj_close"]) for leg in ["ZS", "ZM", "ZL"])
to_ticks = price / TICK_CENTS                                   # log units -> ZS ticks
print(f"{in_dev.sum():,} development bars, {bars_per_day} five-minute bars per trading day, "
      f"{bars.index[in_dev][0]:%Y-%m-%d} to {bars.index[-1]:%Y-%m-%d}")

# ---------------------------------------------------------------- Part A: the live filter as it is
states = kalman_filter.run_kalman_filter(bars, 5)
X = np.column_stack([np.ones(len(bars)), log_m.values, log_o.values])
before = np.sum(X[1:] * states.values[:-1], axis=1)             # fair value from the state of the bar before
after = np.sum(X[1:] * states.values[1:], axis=1)               # fair value after the filter saw this bar
error = log_s.values[1:] - before                               # = the live gap, log units
live_gap = crush_gap(bars, states.shift(1))
assert np.allclose(error * to_ticks.values[1:], live_gap.values[1:], equal_nan=True), "live gap != prediction error"
big = (np.abs(error) > 1e-5) & in_dev[1:]
absorbed = 1 - (log_s.values[1:] - after) / error
parts = X[1:] * (states.values[1:] - states.values[:-1]) / error[:, None]
print("\n== A1. Of each bar's error (the live gap), the share the filter removes by refitting on that same bar")
print(f"median {np.median(absorbed[big]):.3f}  -> the gap halves from refitting alone in "
      f"{np.log(0.5) / np.log(1 - np.median(absorbed[big])):.1f} bars ({5 * np.log(0.5) / np.log(1 - np.median(absorbed[big])):.0f} minutes)")
print("where it goes (medians): intercept {:.3f}, meal ratio {:.3f}, oil ratio {:.3f}".format(*np.median(parts[big], axis=0)))

# returns hedge ratios: beans return on meal and oil returns, 20 trading days of 5-minute returns, known before the bar
window = 20 * bars_per_day
r_s, r_m, r_o = bars["ZS_ret"], bars["ZM_ret"], bars["ZL_ret"]            # inside-session returns (no opening gaps)
cov_mm, cov_oo, cov_mo = r_m.rolling(window).var(), r_o.rolling(window).var(), r_m.rolling(window).cov(r_o)
cov_sm, cov_so = r_s.rolling(window).cov(r_m), r_s.rolling(window).cov(r_o)
determinant = cov_mm * cov_oo - cov_mo ** 2
beta_meal = ((cov_sm * cov_oo - cov_so * cov_mo) / determinant).shift(1)
beta_oil = ((cov_so * cov_mm - cov_sm * cov_mo) / determinant).shift(1)

hedges = pd.DataFrame({"kalman meal": states["beta_meal"].shift(1), "returns meal": beta_meal,
                       "kalman oil": states["beta_oil"].shift(1), "returns oil": beta_oil})[in_dev]
daily_range = hedges.groupby(bars["trading_date"][in_dev]).agg(lambda x: x.max() - x.min())
print("\n== A2. Hedge ratios (beans % move per 1% move in the product)")
print(pd.DataFrame({"mean": hedges.mean(), "lowest": hedges.min(), "highest": hedges.max(),
                    "typical move inside ONE day (median high - low)": daily_range.median()}).round(3).to_string())

print("\n== A3. How much of the beans move each hedge explains (R squared, hedge known before the move)")
rows = []
for bars_ahead in [1, 6, 12, 48]:
    moves = {name: series.shift(-bars_ahead) - series for name, series in [("s", log_s), ("m", log_m), ("o", log_o)]}
    row = {"horizon": f"{bars_ahead * 5} min"}
    for name, (b_meal, b_oil) in {"kalman hedge": (states["beta_meal"].shift(1), states["beta_oil"].shift(1)),
                                  "returns hedge": (beta_meal, beta_oil)}.items():
        left = (moves["s"] - b_meal * moves["m"] - b_oil * moves["o"])[in_dev]
        row[name] = 1 - left.var() / moves["s"][in_dev].var()
    rows.append(row)
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------------------------------------------------------------- Part B: what closes a live gap
print("\n== B. Per tick of live gap now, what happens over the next N minutes (regression slopes, development)")
hedge_now = states.shift(1)
rows = []
for bars_ahead in [1, 3, 6, 12]:
    beans = (log_s.shift(-bars_ahead) - log_s) * to_ticks
    products = (hedge_now["beta_meal"] * (log_m.shift(-bars_ahead) - log_m)
                + hedge_now["beta_oil"] * (log_o.shift(-bars_ahead) - log_o)) * to_ticks
    gap_change = live_gap.shift(-bars_ahead) - live_gap
    refit = beans - products - gap_change                         # what the changed coefficients took out
    frame = pd.DataFrame({"gap": live_gap, "beans": beans, "products": products, "refit": refit})[in_dev].dropna()
    slope = {name: np.polyfit(frame["gap"], frame[name], 1)[0] for name in ["beans", "products", "refit"]}
    rows.append({"minutes": bars_ahead * 5, "beans come back": -slope["beans"], "products catch up": slope["products"],
                 "filter refits": slope["refit"], "gap left": 1 + slope["beans"] - slope["products"] - slope["refit"]})
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------------------------------------------------------------- Part C: memory test
gaps = {}
for drift in [0.1, 0.01, 0.001, 0.0001]:
    kalman_filter.DRIFT_TO_NOISE_PER_15_MINUTES = drift
    gaps[f"kalman live, drift {drift:g}" + (" (CURRENT)" if drift == 0.1 else "")] = crush_gap(bars, crush_hedge(bars, True, 5))
kalman_filter.DRIFT_TO_NOISE_PER_15_MINUTES = 0.1
gaps["session (hedge frozen at prior close)"] = crush_gap(bars, crush_hedge(bars, False, 5))

# anchored: opening gaps are real moves, so use closes (roll-adjusted); the first bar's return includes its gap
own = (log_s.diff() - beta_meal * log_m.diff() - beta_oil * log_o.diff()).fillna(0).cumsum()
for label, memory_bars in [("30 min", 6), ("1 hour", 12), ("2 hours", 24), ("4 hours", 48), ("8 hours", 96),
                           ("1 day", bars_per_day), ("2 days", 2 * bars_per_day), ("5 days", 5 * bars_per_day),
                           ("10 days", 10 * bars_per_day)]:
    gap_log = own - own.ewm(alpha=1.0 / memory_bars, adjust=False).mean().shift(1)
    gaps[f"anchored, memory {label}"] = (gap_log * to_ticks).where(beta_meal.notna())

HORIZONS = [3, 6, 12, 24, 48]
forward = {h: (log_s.shift(-h) - log_s) * to_ticks for h in HORIZONS}       # beans move ahead, ticks
dates = bars["trading_date"]


def fade_after_tail(z: pd.Series, move: pd.Series, mask: np.ndarray) -> tuple[float, float, int]:
    """Mean beans move in the fade direction over the bars with |z| >= 1.645, t from daily sums (bars overlap)."""
    chosen = mask & (z.abs() >= 1.645).values & move.notna().values
    signed = (-np.sign(z) * move)[chosen]
    daily = signed.groupby(dates[chosen]).sum()
    t = daily.mean() / (daily.std() / np.sqrt(len(daily))) if len(daily) > 2 else np.nan
    return signed.mean(), t, int(chosen.sum())


ic_rows, tail_rows, split_rows = [], [], []
for name, gap in gaps.items():
    z = gap / np.sqrt((gap ** 2).rolling(window, min_periods=window // 2).mean()).shift(1)   # 20-day RMS, past bars
    if "CURRENT" in name:
        z_now = gap / adaptive_scale(gap, cfg.research.z_half_life_bars)                     # the 4-hour z in use
    g = gap[in_dev]
    autocorrelation = g.autocorr(1)
    base = {"fair value": name, "typical gap (ticks)": np.sqrt((g ** 2).mean()),
            "gap half-life (min)": 5 * np.log(0.5) / np.log(autocorrelation)}
    ic_rows.append({**base, **{f"IC {h * 5}m": (-z[in_dev]).corr(forward[h][in_dev], method="spearman") for h in HORIZONS}})
    tail = {"fair value": name}
    for h in HORIZONS:
        mean, t, count = fade_after_tail(z, forward[h], in_dev)
        tail[f"{h * 5}m"] = f"{mean:+.2f} ({t:+.1f})"
    tail["share of bars"] = count / in_dev.sum()
    tail_rows.append(tail)
    split = {"fair value": name}
    for label, mask in [("2021-22", half == "2021-22"), ("2023-Nov24", half == "2023-Nov24"),
                        ("asia", session == "asia"), ("europe", session == "europe"), ("us", session == "us")]:
        mean, t, _ = fade_after_tail(z, forward[12], in_dev & mask)
        split[label] = f"{mean:+.2f} ({t:+.1f})"
    split_rows.append(split)
    if "CURRENT" in name:
        tail = {"fair value": "  the same with the 4-hour z in use"}
        for h in HORIZONS:
            mean, t, count = fade_after_tail(z_now, forward[h], in_dev)
            tail[f"{h * 5}m"] = f"{mean:+.2f} ({t:+.1f})"
        tail["share of bars"] = count / in_dev.sum()
        tail_rows.append(tail)

print("\n== C1. Size and speed of each gap; rank correlation of -z with the beans move ahead (z = gap / its 20-day RMS)")
print(pd.DataFrame(ic_rows).round(3).to_string(index=False))
print("\n== C2. Bars with |z| >= 1.645: beans move in the FADE direction, ticks before costs (t from daily sums)")
print(pd.DataFrame(tail_rows).round(3).to_string(index=False))
print("\n== C3. The same at 60 minutes, by half and by session")
print(pd.DataFrame(split_rows).to_string(index=False))
