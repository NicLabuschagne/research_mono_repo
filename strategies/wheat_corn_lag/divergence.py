"""
divergence.py
-------------
The divergence itself, seen continuously, and how fast it closes (half-life). Development data only.
Saved to runs/<time>_divergence/.

    python -m strategies.wheat_corn_lag.divergence

Two ways to see "wheat vs where corn says it should be"
  implied wheat (lookback L)   W(t-L) x exp(beta x corn's log move over the last L bars):
                               where wheat would be now if it had moved exactly beta x corn since L bars ago.
                               Wheat minus this line, in sd, is own_z_L. Drawn continuously, bar by bar.
  residual index R             running total of wheat's own per-bar move (ret_w - beta x ret_c). It never
                               resets; a divergence is R moving away from where it was and coming back.
                               Across a possible-roll cut the return is unknown and counts as 0.

Half-life, three ways (each answers a slightly different question)
  1. OU fit        on R and on the trader's two spreads (cents, roll-free), in blocks of 20 sessions:
                   change(t) = c + lambda x level(t-1); half-life = ln 2 / -ln(1 + lambda) bars.
                   Short blocks bias this toward reversion even for a random walk, so the same fit is run
                   on R rebuilt from SHUFFLED returns (placebo). Only the gap to the placebo is evidence.
  2. Variance ratio  var(own move over k bars) / (k x var(one bar)): 1 = random walk, below 1 = moves
                   partly reverse. Where the curve flattens is the time scale of the reversion.
  3. Impulse response  on the FIRST bar of each |own_z_L| > 1 run (a fresh divergence, what a trade enters
                   on), the share of the L-bar divergence given back k bars later, for each L; overall and when
                   corn moved the same way (the v2 signal context). Averaging every bar in the run instead
                   shows almost no give-back: the reversion happens right after a divergence first appears.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import save_run
from core_quant.theme import set_theme
from strategies.wheat_corn_lag.features import build_features, change_ahead, change_over
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load, load_bars

BLOCK_BARS = 1420                                   # 20 sessions
CHART_LOOKBACK = 16                                 # 4 h: the shorter-term divergence asked for
CHART_SD = 0.75                                     # dots: |corn_z| and |own_z| both above this, same sign
CHART_WEEKS = ["2025-07-28", "2021-05-10", "2023-02-06", "2024-10-14"]   # first = the trader's 2025-08-01 chart
STEPS_AHEAD = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 71, 106, 142, 213, 284]
HOUR_TICKS = [0.25, 1, 4, 17.75, 35.5, 71]                        # 17.75 h = 71 bars = 1 session


def own_returns(bars: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    return bars["ret_w"] - features["beta"] * bars["ret_c"]           # beta is already lagged one bar


def implied_wheat(bars: pd.DataFrame, features: pd.DataFrame, lookback: int) -> pd.Series:
    start_price = bars["W"].groupby(bars["segment"]).shift(lookback)
    corn_move = change_over(bars["log_c"], bars["segment"], lookback)
    return start_price * np.exp(features["beta"] * corn_move)


# ------------------------------------------------------------------ continuous chart
def continuous_chart(bars, features, start: str) -> plt.Figure:
    window = (bars.index >= pd.Timestamp(start, tz="UTC")) & (bars.index < pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=7))
    frame = bars[window]
    implied = implied_wheat(bars, features, CHART_LOOKBACK)[window]
    own = features[f"own_z_{CHART_LOOKBACK}"][window]
    corn = features[f"corn_z_{CHART_LOOKBACK}"][window]
    # plot against bar number, so nights and the weekend do not leave long flat lines
    x = np.arange(len(frame))

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, height_ratios=[1.6, 1])
    top.plot(x, frame["W"], lw=1.4, label="wheat")
    top.plot(x, implied, lw=1.2, ls="--", label=f"corn-implied wheat ({CHART_LOOKBACK} bars back)")
    candidate = (corn.abs() > CHART_SD) & (own.abs() > CHART_SD) & (np.sign(corn) == np.sign(own))
    sell, buy = candidate & (own > 0), candidate & (own < 0)
    top.scatter(x[sell.values], frame["W"][sell], marker="v", s=28, color="#d62728", zorder=5, label="overshoot up (sell)")
    top.scatter(x[buy.values], frame["W"][buy], marker="^", s=28, color="#2ca02c", zorder=5, label="overshoot down (buy)")
    top.set_ylabel("cents/bu")
    top.legend(loc="best", fontsize=8)
    top.set_title(f"Week of {start}: wheat vs corn-implied wheat; markers = |corn_z| and |own_z| > {CHART_SD}, same sign",
                  loc="left")

    bottom.plot(x, own, lw=1.2, label=f"own_z_{CHART_LOOKBACK} (wheat minus implied, sd)")
    bottom.plot(x, corn, lw=1.0, alpha=0.7, label=f"corn_z_{CHART_LOOKBACK}")
    for level in (CHART_SD, -CHART_SD):
        bottom.axhline(level, color="grey", ls=":", lw=0.9)
    bottom.axhline(0, color="grey", lw=0.8)
    bottom.legend(loc="best", fontsize=8)
    # one tick per session
    session_starts = np.flatnonzero((frame["session"] != frame["session"].shift()).values)
    bottom.set_xticks(session_starts, [f"{frame.index[i]:%a %d %b}" for i in session_starts])
    bottom.set_xlabel("bar number (sessions labelled; overnight and weekend gaps removed)")
    figure.tight_layout()
    return figure


# ------------------------------------------------------------------ half-life
def ou_half_life(level: np.ndarray) -> float:
    """Half-life in bars from change(t) = c + lambda * level(t-1); inf when there is no reversion."""
    change, previous = np.diff(level), level[:-1]
    design = np.column_stack([np.ones(len(previous)), previous])
    slope = np.linalg.lstsq(design, change, rcond=None)[0][1]
    return np.log(2) / -np.log(1 + slope) if slope < 0 else np.inf


def block_half_lives(bars, features, seed: int = 11) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    own = own_returns(bars, features)
    spreads = {"1.5W - C": 1.5 * bars["W"] - bars["C"], "W - 1.5C": bars["W"] - 1.5 * bars["C"]}
    # roll-free spread levels: running total of cent changes inside segments
    series = {"residual index R": own.fillna(0).cumsum()}
    for name, spread in spreads.items():
        series[name] = spread.groupby(bars["segment"]).diff().fillna(0).cumsum()

    rows = []
    for start in range(BLOCK_BARS * 2, len(bars) - BLOCK_BARS, BLOCK_BARS):   # skip warm-up blocks
        block = slice(start, start + BLOCK_BARS)
        row = {"block_start": bars.index[start]}
        for name, values in series.items():
            row[name] = ou_half_life(values.values[block])
        shuffled = generator.permutation(own.fillna(0).values[block])
        row["placebo (shuffled R)"] = ou_half_life(np.cumsum(shuffled))
        rows.append(row)
    return pd.DataFrame(rows).set_index("block_start")


def variance_ratio(bars, features) -> pd.DataFrame:
    own = own_returns(bars, features)
    series = {"wheat's own move": own.fillna(0).cumsum(), "wheat": bars["log_w"], "corn": bars["log_c"]}
    one_bar = {"wheat's own move": own, "wheat": bars["ret_w"], "corn": bars["ret_c"]}
    rows = {}
    for name, level in series.items():
        base = one_bar[name].var()
        rows[name] = {k: change_over(level, bars["segment"], k).var() / (k * base) for k in STEPS_AHEAD}
    return pd.DataFrame(rows)


def impulse_response(bars, features, lookbacks: list[int]) -> dict[str, pd.DataFrame]:
    """Share of the L-bar divergence given back k bars later (1 = fully closed, 0 = nothing given back)."""
    residual = own_returns(bars, features).fillna(0).cumsum()
    segment = bars["segment"]
    out = {"fresh |own_z| > 1": {}, "fresh, corn moved the same way (> 1 sd)": {}}
    for lookback in lookbacks:
        own_z, corn_z = features[f"own_z_{lookback}"], features[f"corn_z_{lookback}"]
        built = change_over(residual, segment, lookback)                       # the divergence, in log units
        masks = {"fresh |own_z| > 1": own_z.abs() > 1,
                 "fresh, corn moved the same way (> 1 sd)": (own_z.abs() > 1) & (corn_z.abs() > 1) & (np.sign(own_z) == np.sign(corn_z))}
        for label, mask in masks.items():
            mask = mask & ~mask.shift(1, fill_value=False)                     # first bar of each run only
            given_back = {}
            for k in STEPS_AHEAD:
                ahead = change_ahead(residual, segment, k)
                valid = mask & ahead.notna() & built.notna()
                given_back[k] = (-np.sign(built[valid]) * ahead[valid]).mean() / built[valid].abs().mean()
            out[label][f"L={lookback}"] = given_back
    return {label: pd.DataFrame(table) for label, table in out.items()}


def half_life_charts(ratios: pd.DataFrame, responses: dict) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    hours = np.array(STEPS_AHEAD) / 4
    for name in ratios:
        axes[0].plot(hours, ratios[name].values, marker="o", ms=3, label=name)
    axes[0].axhline(1, color="grey", lw=0.8)
    axes[0].set_title("Variance ratio (1 = random walk, < 1 = reverts)", loc="left")
    axes[0].set_xlabel("hours (log scale; 71 bars = 1 session = 17.75 h)")
    axes[0].legend(fontsize=8)
    for axis, (label, table) in zip(axes[1:], responses.items()):
        for column in table:
            axis.plot(hours, table[column].values, marker="o", ms=3, label=column)
        axis.axhline(0.5, color="grey", ls=":", lw=1)
        axis.axhline(0, color="grey", lw=0.8)
        axis.set_title(f"Share of the divergence given back\n{label}", loc="left")
        axis.set_xlabel("hours ahead (log scale; 17.75 h = 1 session)")
        axis.legend(fontsize=8)
    axes[1].set_ylabel("share given back (0.5 = half-life)")
    for axis in axes:
        # plain-number ticks: the default log-scale labels render as raw "$\mathdefault{10^{0}}$" in this theme
        axis.set_xscale("log")
        axis.set_xticks(HOUR_TICKS, [f"{h:g}" for h in HOUR_TICKS])
        axis.minorticks_off()
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    bars = load_bars(cfg)                                     # development data only
    features = build_features(bars, cfg.features)
    pd.set_option("display.width", 200)

    lives = block_half_lives(bars, features)
    print("== OU half-life per 20-session block, in hours (bars / 4); inf = no reversion in that block")
    summary = pd.DataFrame({
        "median_h": (lives.replace(np.inf, np.nan).median() / 4),
        "p25_h": lives.replace(np.inf, np.nan).quantile(0.25) / 4,
        "p75_h": lives.replace(np.inf, np.nan).quantile(0.75) / 4,
        "share_reverting": np.isfinite(lives).mean(),
    })
    print(summary.round(1).to_string())
    print("\nby year (median hours):")
    print((lives.replace(np.inf, np.nan).groupby(lives.index.year).median() / 4).round(1).to_string())

    ratios = variance_ratio(bars, features)
    print("\n== variance ratio (rows: bars ahead)")
    print(ratios.round(2).to_string())

    responses = impulse_response(bars, features, [8, 16, 32, 71])
    for label, table in responses.items():
        print(f"\n== share of the divergence given back k bars later: {label}")
        print(table.round(2).to_string())

    figures = {f"continuous_{week}": continuous_chart(bars, features, week) for week in CHART_WEEKS}
    figures["half_life"] = half_life_charts(ratios, responses)
    run_dir = save_run(STRATEGY_DIR, "divergence", cfg, {"block_bars": BLOCK_BARS},
                       {"ou_half_life_blocks": lives, "variance_ratio": ratios,
                        **{f"impulse_{i}": t for i, t in enumerate(responses.values())}}, figures)
    print(f"\nsaved to {run_dir}")
