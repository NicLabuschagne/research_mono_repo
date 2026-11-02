"""
world_conditioner.py
--------------------
Conditioner test: is wheat's overshoot of corn US-SPECIFIC or GLOBAL, and does that change how it reverts?
Daily (KC and Matif are daily only), development data only. Saved to runs/<time>_world_conditioner/.

    python -m strategies.wheat_corn_lag.world_conditioner

Signal  1a at daily frequency: fade wheat's own move vs corn (composite of L = 2, 3, 5 days; world_wheat.py)
Condition, from CBOT wheat's move vs KC + Matif over the same windows (own_ze_km, composite):
  US-specific  CBOT also overshot world wheat, in the same direction, by > 1 sd: a SRW / CBOT story
  global       CBOT is in line with world wheat (|own vs world| < 0.5 sd): the whole wheat world moved
  mixed        everything else
Hypothesis: US-specific overshoots revert MORE (flow, local), global ones LESS (world repricing, justified).
Useful (fixed before the run, as conditioners.py): |t| >= 2 on signal x bucket (US-specific vs global,
Newey-West, lags = horizon) AND the IC gap has the same sign in >= 4 of 6 years. Everything is known at the
CBOT close of day t.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import save_run
from core_quant.theme import set_theme
from strategies.wheat_corn_lag.conditioners import split_test, yearly_gap
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load
from strategies.wheat_corn_lag.world_wheat import PRIMARY, build_features, composite, daily_panel, targets

US_SPECIFIC_SD, GLOBAL_SD = 1.0, 0.5
HORIZONS = [2, 3]


def buckets(features: pd.DataFrame) -> pd.Series:
    vs_corn = composite(features, "own_ze_c")           # the 1a signal; composite() already returns the FADE sign
    vs_world = composite(features, "own_ze_km")         # same convention, against KC + Matif
    us_specific = (np.sign(vs_world) == np.sign(vs_corn)) & (vs_world.abs() > US_SPECIFIC_SD)
    in_line = vs_world.abs() < GLOBAL_SD
    label = np.where(us_specific, "US-specific", np.where(in_line, "global", "mixed"))
    return pd.Series(label, index=features.index).where(vs_corn.notna() & vs_world.notna())


def chart(tables: dict) -> plt.Figure:
    figure, axes = plt.subplots(1, len(tables), figsize=(6 * len(tables), 4.2), sharey=True)
    for axis, (title, table) in zip(np.atleast_1d(axes), tables.items()):
        order = [b for b in ["US-specific", "mixed", "global"] if b in table.index]
        table = table.loc[order]
        axis.bar(range(len(table)), table["ic"], color=["#d62728", "#7f7f7f", "#2a78d6"][:len(table)])
        for i, (name, row) in enumerate(table.iterrows()):
            note = "" if np.isnan(row["interaction_t_vs_base"]) else f"\nt {row['interaction_t_vs_base']:+.1f} vs global"
            axis.annotate(f"{int(row['bars']):,} days{note}", (i, row["ic"]), xytext=(0, 3), textcoords="offset points",
                          ha="center", fontsize=8)
        axis.set_xticks(range(len(table)), table.index)
        axis.axhline(0, color="grey", lw=0.8)
        axis.set_title(title, loc="left", fontsize=10)
    np.atleast_1d(axes)[0].set_ylabel("rank IC of the corn signal (1a, daily)")
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    holdout = pd.Timestamp(cfg.split.holdout_start, tz="UTC")
    panel = daily_panel(cfg, holdout)
    features = build_features(panel)
    forward = targets(panel)
    signal = composite(features, "own_ze_c")            # + = buy wheat (fade); composite() already negates
    groups = buckets(features)

    pd.set_option("display.width", 200)
    print(f"days with a signal and a bucket: {int(groups.notna().sum())}; buckets: {groups.value_counts().to_dict()}")
    tables, gaps = {}, {}
    for h in HORIZONS:
        title = f"{h}-day forward"
        tables[title] = split_test(signal, forward[f"fwd_{h}"], groups, "global", h)
        gaps[title] = yearly_gap(signal, forward[f"fwd_{h}"], groups, "US-specific", "global", min_count=15)
        overall = tables[title].loc["US-specific", "ic"] - tables[title].loc["global", "ic"]
        gap = gaps[title]
        print(f"\n== {title}")
        print(tables[title].round(3).to_string())
        print(f"   IC(US-specific) - IC(global) by year: {gap.round(3).to_dict()}  -> same sign as the overall gap"
              f" ({overall:+.3f}) in {int((np.sign(gap) == np.sign(overall)).sum())} of {len(gap)} years")
    run_dir = save_run(STRATEGY_DIR, "world_conditioner", cfg, {"us_specific_sd": US_SPECIFIC_SD, "global_sd": GLOBAL_SD},
                       {**{f"split_{h}": tables[f'{h}-day forward'] for h in HORIZONS},
                        **{f"yearly_gap_{h}": gaps[f'{h}-day forward'].to_frame() for h in HORIZONS}},
                       {"world_conditioner": chart(tables)})
    print(f"\nsaved to {run_dir}")
