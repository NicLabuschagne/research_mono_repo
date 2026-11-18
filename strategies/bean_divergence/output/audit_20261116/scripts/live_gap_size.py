"""Size of the LIVE gap (beans minus the live Kalman fair value), every 5-minute bar, ZS ticks (development data).
Two speeds: 10x slower (drift 0.01, the one agreed for the alert) and the original (drift 0.1). Descriptive only."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import strategies.crush_strategy.kalman_filter as kalman_filter
from strategies.bean_divergence.fair_value import STRATEGY_DIR, crush_gap, crush_hedge, load, load_bars, session_label
from strategies.bean_divergence.triggers import find_triggers

pd.set_option("display.width", 250)
cfg = load()
bars = load_bars(cfg)
in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
session_id = bars["session_id"].values
session = session_label(bars.index, bars["session_type"])
years = bars["trading_date"].dt.year.values
weeks = bars.loc[in_dev, "trading_date"].nunique() / 5
gaps = {}
for name, drift in [("10x slower (agreed)", 0.01), ("original speed", 0.1)]:
    kalman_filter.DRIFT_TO_NOISE_PER_15_MINUTES = drift
    gaps[name] = crush_gap(bars, crush_hedge(bars, True, 5)).values
kalman_filter.DRIFT_TO_NOISE_PER_15_MINUTES = 0.1

for name, gap in gaps.items():
    size, ok = np.abs(gap), in_dev & ~np.isnan(gap)
    rows = []
    for label, chosen in [("all", ok)] + [(str(y), ok & (years == y)) for y in [2021, 2022, 2023, 2024]] + [(s, ok & (session == s)) for s in ["asia", "europe", "us"]]:
        row = {"": label, "median": np.median(size[chosen]), "75%": np.percentile(size[chosen], 75), "90%": np.percentile(size[chosen], 90),
               "95%": np.percentile(size[chosen], 95), "99%": np.percentile(size[chosen], 99)}
        for level in [5, 8, 10, 15, 20]:
            row[f"bars at {level}+"] = f"{(size[chosen] >= level).mean():.1%}"
        rows.append(row)
    print(f"\n== live gap, {name}: size in ticks, every 5-minute bar")
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    peak = pd.Series(size[ok]).groupby(session_id[ok]).max()
    print("largest gap in a session: median %.1f, 75%% %.1f, 90%% %.1f ticks; sessions reaching 5+ %.0f%%, 10+ %.0f%%, 15+ %.0f%%, 20+ %.0f%%"
          % (peak.median(), peak.quantile(0.75), peak.quantile(0.9), *(100 * (peak >= level).mean() for level in [5, 10, 15, 20])))
    counts = {level: sum(in_dev[t] for t, _ in find_triggers(np.nan_to_num(gap), session_id, level, level / 2)) / weeks for level in [5, 8, 10, 15, 20]}
    print("separate stretches a week (a new one only after the gap is back inside half the level): " + ", ".join(f"{level}+ ticks {count:.1f}" for level, count in counts.items()))

figure, axes = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
for axis, (name, gap) in zip(axes, gaps.items()):
    ok = in_dev & ~np.isnan(gap)
    axis.hist(np.clip(gap[ok], -25, 25), bins=100, color="#3182bd", alpha=0.85)
    for level in [-5, 5]:
        axis.axvline(level, color="black", ls="--")
    axis.set_title(f"Live gap, {name}: {(np.abs(gap[ok]) >= 5).mean():.0%} of bars beyond 5 ticks (dashed)", loc="left")
    axis.set_xlabel("ticks: + = beans above the live fair value (clipped at 25)")
    axis.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("5-minute bars")
figure.tight_layout()
path = STRATEGY_DIR / "output" / "audit_20261116" / "live_gap_size.png"
figure.savefig(path, dpi=100)
print(path)
