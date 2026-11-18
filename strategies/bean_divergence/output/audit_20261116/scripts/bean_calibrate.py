"""How often is each gap 'stretched' when the yardstick is beans' own expected 1-hour move? (frequency only, no outcomes)"""
import numpy as np
import pandas as pd
from strategies.bean_divergence.fair_value import adaptive_scale, crush_gap, crush_hedge, load, load_bars, TICK_CENTS

cfg = load()
bars = load_bars(cfg)
in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
sigma = adaptive_scale(bars["ZS_ret"] * bars["ZS_adj_close"] / TICK_CENTS, 48) * np.sqrt(12)
live = crush_gap(bars, crush_hedge(bars, True, 5))
session = crush_gap(bars, crush_hedge(bars, False, 5))
year = bars["trading_date"].dt.year
print("expected 1-hour beans move, ticks (median by year):", sigma[in_dev].groupby(year[in_dev]).median().round(1).to_dict())
for name, gap, levels in [("live", live, [0.3, 0.4, 0.5, 0.6, 0.75]), ("session", session, [0.75, 1.0, 1.25, 1.5, 2.0])]:
    ratio = (gap.abs() / sigma)[in_dev]
    print(name, "share of bars with |gap| >= k x expected 1-hour move:",
          {k: round(float((ratio >= k).mean()), 3) for k in levels},
          "| by year at the middle level:", (ratio >= levels[2]).groupby(year[in_dev]).mean().round(3).to_dict())
