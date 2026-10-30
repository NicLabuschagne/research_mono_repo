"""
layer1_trigger_harvest.py
-------------------------
Forward returns from the trigger itself, across horizons, with no session cut. Can a frequent trigger harvest a
small, repeatable bounce?

Trigger: prior-close crush gap / its 4-hour adaptive size (layer1_adaptive_extremes) beyond +/- level, once per
excursion (re-armed inside |z| < 0.67), in ANY session. Levels 1.645 / 1.96 / 2.33 / 2.58 - the per-week count picks
the frequency (3-5 a week is the target); the level is chosen by count, not by returns.
Entry: the trigger bar's close (live, the threshold is crossed inside the bar and a limit order sits there).
The next bar's open is shown too: the gap between the two is the part a 5-minute close cannot settle (bid / ask
bounce, fill at the crossing price).
Horizons 5 m - 4 h on continuous bars: breaks and the overnight gap are simply held through.
    outright : beans, + = in the trade's direction (sell rich, buy cheap)
    spread   : the gap with the trigger bar's hedge held fixed, + = back toward fair value
    race     : on 5-minute closes within 30 min, does beans reach +R ticks before -R? A coin flip is 50%; to beat a
               cost of c ticks per round trip the win share must exceed 0.5 + c / (2R).
Baseline: 5 random bars per trigger, same session and half, random direction. Development only; holdout locked.

    python -m strategies.bean_divergence.layer1_trigger_harvest          # prior-close gap
    python -m strategies.bean_divergence.layer1_trigger_harvest live     # live gap (hedge from the bar before)
"""
import sys

import numpy as np
import pandas as pd

from core_quant.config import ROOT
from strategies.bean_divergence.failure_study import session_label
from strategies.bean_divergence.feature_study import STRATEGY_DIR, TICK_CENTS
from strategies.bean_divergence.layer1_adaptive_extremes import REARM, adaptive_scale
from strategies.bean_divergence.layer1_extremes import find_extremes
from strategies.bean_divergence.layer1_signal_days import HALF_LIFE_4H
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.kalman_filter import freeze_at_previous_session_close, run_kalman_filter
from strategies.crush_strategy.strategy import STRATEGY_DIR as CRUSH_DIR, load

LEVELS = [1.645, 1.96, 2.33, 2.58, 3.0, 3.5]
HORIZONS = {"5m": 1, "10m": 2, "15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48}
MAX_BARS = 48
RACE_TICKS, RACE_BARS = [2, 3, 4], 6
RANDOM_PER_TRIGGER = 5
GAP = "live" if sys.argv[1:] == ["live"] else "prior_close"
SUFFIX = "_live" if GAP == "live" else ""


def crush_hedge(bars: pd.DataFrame) -> pd.DataFrame:
    """alpha / beta_meal / beta_oil used for the gap: frozen at the previous session close, or live (the bar before)."""
    states = run_kalman_filter(bars, 5)
    return states.shift(1) if GAP == "live" else freeze_at_previous_session_close(bars, states)


def race(path: np.ndarray, r: float) -> float:
    """1 if the path reaches +r before -r within RACE_BARS closes, 0 if -r first, nan if neither."""
    for k in range(RACE_BARS):
        if path[k] >= r:
            return 1.0
        if path[k] <= -r:
            return 0.0
    return np.nan


if __name__ == "__main__":
    cfg = load(CRUSH_DIR / "config.yaml")
    bars = build_bars(ROOT / cfg.data_path, 5)
    bars = bars[bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start)]        # holdout stays locked
    hedge = crush_hedge(bars)
    a, bm, bo = (hedge[c].values for c in ["alpha", "beta_meal", "beta_oil"])
    log_c = {leg: np.log(bars[f"{leg}_adj_close"]).values for leg in ["ZS", "ZM", "ZL"]}
    log_open_zs = np.log(bars["ZS_adj_open"]).values
    ticks = (bars["ZS_adj_close"] / TICK_CENTS).values
    gap = pd.Series((log_c["ZS"] - (a + bm * log_c["ZM"] + bo * log_c["ZL"])) * ticks, index=bars.index)
    z = (gap / adaptive_scale(gap, HALF_LIFE_4H)).values
    session_id = bars["session_id"].values
    session = session_label(bars.index, bars["session_type"])
    dates = bars["trading_date"]
    half = np.where(dates < pd.Timestamp("2023-01-01"), "2021-22", "2023-Nov24")
    usable = (dates >= pd.Timestamp(cfg.split.development_start)).values & (np.arange(len(bars)) + MAX_BARS < len(bars))
    weeks = pd.Series(dates[usable].values).groupby(half[usable]).nunique() / 5

    def measure(s: int, side: int) -> dict:
        """Forward moves from bar s's close (ticks); the hedge of bar s is held fixed for the spread."""
        later = np.arange(s + 1, s + 1 + MAX_BARS)
        beans = -side * (log_c["ZS"][later] - log_c["ZS"][s]) * ticks[s]
        held = lambda i: log_c["ZS"][i] - (a[s] + bm[s] * log_c["ZM"][i] + bo[s] * log_c["ZL"][i])
        spread = -side * (held(later) - held(s)) * ticks[s]
        from_open = -side * (log_c["ZS"][later] - log_open_zs[s + 1]) * ticks[s]
        row = {}
        for label, h in HORIZONS.items():
            row[f"out_{label}"], row[f"spr_{label}"], row[f"open_{label}"] = beans[h - 1], spread[h - 1], from_open[h - 1]
        for r in RACE_TICKS:
            row[f"race_{r}"] = race(beans, r)
        return row

    rng = np.random.default_rng(42)
    rows = []
    for level in LEVELS:
        constant = pd.DataFrame({"hi": level, "lo": -level, 0.75: REARM, 0.25: -REARM}, index=bars.index)
        for s, side in find_extremes(z, constant, session_id, "hi", "lo"):
            if not usable[s]:
                continue
            rows.append({"level": level, "kind": "signal", "half": half[s], "session": session[s]} | measure(s, side))
            pool = np.flatnonzero((session == session[s]) & (half == half[s]) & usable)
            for r_bar in rng.choice(pool, RANDOM_PER_TRIGGER):
                rows.append({"level": level, "kind": "random", "half": half[s], "session": session[s]}
                            | measure(int(r_bar), int(rng.choice([-1, 1]))))
    t = pd.DataFrame(rows)
    t.to_csv(STRATEGY_DIR / "output" / f"layer1_trigger_harvest{SUFFIX}.csv", index=False)
    print(f"gap: {GAP}")
    sig, rnd = t[t["kind"] == "signal"], t[t["kind"] == "random"]

    # self-check: a random coin-flip race is ~50/50, otherwise the race itself is biased
    coin = rnd["race_3"].dropna().mean()
    assert abs(coin - 0.5) < 0.03, coin

    pd.set_option("display.width", 250)
    counts = sig.groupby(["level", "half", "session"]).size().unstack("session").fillna(0)
    counts = counts.div(weeks, level="half", axis=0)
    counts["all"] = counts.sum(axis=1)
    print("triggers per week, by session:\n", counts.round(1).to_string())

    for level in LEVELS:
        table = []
        for label in HORIZONS:
            row = {"horizon": label}
            for hf in ["2021-22", "2023-Nov24"]:
                s_out = sig[(sig["level"] == level) & (sig["half"] == hf)][f"out_{label}"]
                r_out = rnd[(rnd["level"] == level) & (rnd["half"] == hf)][f"out_{label}"]
                s_spr = sig[(sig["level"] == level) & (sig["half"] == hf)][f"spr_{label}"]
                s_open = sig[(sig["level"] == level) & (sig["half"] == hf)][f"open_{label}"]
                row[f"{hf} beans"] = s_out.mean()
                row[f"{hf} t"] = s_out.mean() / s_out.sem()
                row[f"{hf} hit%"] = (s_out > 0).mean() * 100
                row[f"{hf} rnd hit%"] = (r_out > 0).mean() * 100
                row[f"{hf} spread"] = s_spr.mean()
                row[f"{hf} from open"] = s_open.mean()
            table.append(row)
        print(f"\n|z| >= {level}: forward move from the trigger close (ticks; hit% = share > 0; ties count as misses):\n",
              pd.DataFrame(table).round(1).to_string(index=False))

    race_rows = []
    for level in LEVELS:
        for hf in ["2021-22", "2023-Nov24"]:
            for r in RACE_TICKS:
                s_r = sig[(sig["level"] == level) & (sig["half"] == hf)][f"race_{r}"]
                r_r = rnd[(rnd["level"] == level) & (rnd["half"] == hf)][f"race_{r}"]
                race_rows.append({"level": level, "half": hf, "R": r, "decided%": s_r.notna().mean() * 100,
                                  "win%": s_r.mean() * 100, "random win%": r_r.mean() * 100,
                                  "needed at 1 tick cost": (0.5 + 1 / (2 * r)) * 100,
                                  "needed at 0.4 tick": (0.5 + 0.4 / (2 * r)) * 100,
                                  "ticks per decided race, gross": (2 * s_r.mean() - 1) * r})
    races = pd.DataFrame(race_rows)
    races.to_csv(STRATEGY_DIR / "output" / f"layer1_trigger_harvest_race{SUFFIX}.csv", index=False)
    print("\nrace within 30 min on beans: +R ticks before -R (5-min closes, entry at the trigger close):\n",
          races.round(1).to_string(index=False))
