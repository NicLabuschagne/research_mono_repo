"""
triggers.py
-----------
One trigger per move of beans away from the crush fair value (the primary signal for the meta-model).

A trigger is the first bar where |z| >= trigger_z (1.645), in any session. The same side cannot fire again until
|z| is back inside rearm_z (0.67); a new session starts fresh. Side +1 = beans rich -> sell, -1 = beans cheap -> buy.
How far the move went (z, gap) is kept as a FEATURE for the meta-model, instead of separate higher levels that would
label the same move several times. Development period only.

    python -m strategies.bean_divergence.triggers                 # live gap (the base)
    python -m strategies.bean_divergence.triggers prior_close     # prior-close gap
"""
import sys

import numpy as np
import pandas as pd

from strategies.bean_divergence.fair_value import (BeanConfig, adaptive_scale, crush_gap, crush_hedge, half_label,
                                                   load, load_bars, session_label)


def find_triggers(z: np.ndarray, session_id: np.ndarray, level: float, rearm: float) -> list[tuple[int, int]]:
    """(bar, side) for the first bar of each excursion beyond +/- level; side +1 = rich, -1 = cheap."""
    events, armed_rich, armed_cheap = [], True, True
    for t in range(len(z)):
        if t > 0 and session_id[t] != session_id[t - 1]:
            armed_rich = armed_cheap = True                    # a new session starts fresh
        if np.isnan(z[t]):
            continue
        if armed_rich and z[t] >= level:
            events.append((t, 1)); armed_rich = False
        elif not armed_rich and z[t] < rearm:
            armed_rich = True
        if armed_cheap and z[t] <= -level:
            events.append((t, -1)); armed_cheap = False
        elif not armed_cheap and z[t] > -rearm:
            armed_cheap = True
    return events


def build(cfg: BeanConfig, live: bool = True):
    """Bars, hedge, gap (ticks), z and the trigger table for the development period."""
    r = cfg.research
    bars = load_bars(cfg)
    hedge = crush_hedge(bars, live, r.bar_minutes)
    gap = crush_gap(bars, hedge)
    z = gap / adaptive_scale(gap, r.z_half_life_bars)
    session = session_label(bars.index, bars["session_type"])
    half = half_label(bars["trading_date"])
    in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
    rows = []
    for s, side in find_triggers(z.values, bars["session_id"].values, r.trigger_z, r.rearm_z):
        if in_dev[s]:
            rows.append({"bar": s, "time": bars.index[s], "trading_date": bars["trading_date"].iloc[s],
                         "session": session[s], "half": half[s], "side": side,
                         "gap_ticks": gap.iloc[s], "z": z.iloc[s]})
    return bars, hedge, gap, z, pd.DataFrame(rows)


if __name__ == "__main__":
    # self-check on a made-up path: fire, no repeat until re-armed, re-fire, opposite side, session reset
    z = np.array([0.0, 2.0, 2.5, 1.0, 0.5, 1.8, -1.7, -2.0, 1.9, 1.9])
    sid = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    assert find_triggers(z, sid, 1.645, 0.6745) == [(1, 1), (5, 1), (6, -1), (8, 1), (9, 1)]

    cfg = load()
    live = "prior_close" not in sys.argv[1:]
    bars, _, _, _, triggers = build(cfg, live)
    dev = bars[bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)]
    weeks = dev.groupby(half_label(dev["trading_date"]))["trading_date"].nunique() / 5
    per_week = triggers.groupby(["half", "session"]).size().unstack("session").div(weeks, axis=0)
    per_week["all"] = per_week.sum(axis=1)
    print(f"{'live' if live else 'prior-close'} gap, |z| >= {cfg.research.trigger_z}: triggers per week\n",
          per_week.round(1).to_string())
    print("\n|z| at the trigger bar (a meta-model feature):", triggers["z"].abs().describe().round(2).to_dict())
