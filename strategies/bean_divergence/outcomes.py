"""
outcomes.py
-----------
What happens after a trigger, before any filtering: forward moves from the trigger close, across horizons, with no
session cut (breaks and the overnight gap are held through).

    outright : beans, + = in the trade's direction (sell rich, buy cheap)
    spread   : the gap with the trigger bar's hedge held fixed, + = back toward fair value
    from open: beans measured from the next bar's open instead (the part a 5-minute close cannot settle: bid / ask
               bounce, the fill at the crossing price)
    tails    : share of triggers beyond +/-5 and +/-10 ticks - the win side and the loss side, not just the net mean
    race     : on 5-minute closes within 30 min, does beans reach +R ticks before -R? A coin flip is 50%; to beat a
               cost of c ticks per round trip the win share must exceed 0.5 + c / (2R).
    best move: the favourable excursion - the best close in the trade's favour within 30 min / 1 h / 2 h (0 if it
               never went favourable) and the minutes it took to get there. Not net: losses are not in it.
Baseline: 5 random bars per trigger, same session and half, random direction. Development only, no costs.

    python -m strategies.bean_divergence.outcomes                 # live gap (the base)
    python -m strategies.bean_divergence.outcomes prior_close     # prior-close gap
"""
import sys

import numpy as np
import pandas as pd

from strategies.bean_divergence.fair_value import LEGS, STRATEGY_DIR, TICK_CENTS, half_label, load, session_label
from strategies.bean_divergence.triggers import build

HORIZONS = {"5m": 1, "10m": 2, "15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48}
MAX_BARS = 48
RACE_TICKS, RACE_BARS = [2, 3, 4], 6
TAIL_TICKS = [5, 10]
RANDOM_PER_TRIGGER = 5
BEST_WINDOWS = {"30m": 6, "1h": 12, "2h": 24}
BAR_MINUTES = 5


def best_move(path: np.ndarray, bars: int) -> tuple[float, float]:
    """Best close in the trade's favour within the first `bars` closes and the minutes to reach it (nan if never > 0)."""
    k = int(np.argmax(path[:bars]))
    if path[k] <= 0:
        return 0.0, np.nan
    return path[k], (k + 1) * BAR_MINUTES


def race(path: np.ndarray, r: float) -> float:
    """1 if the path reaches +r before -r within RACE_BARS closes, 0 if -r first, nan if neither."""
    for k in range(RACE_BARS):
        if path[k] >= r:
            return 1.0
        if path[k] <= -r:
            return 0.0
    return np.nan


def measure(s: int, side: int, hedge: tuple, log_c: dict, log_open_zs: np.ndarray, ticks: np.ndarray) -> dict:
    """Forward moves from bar s's close (ticks); the hedge of bar s is held fixed for the spread."""
    a, bm, bo = hedge
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
    for label, w in BEST_WINDOWS.items():
        row[f"out_best_{label}"], row[f"out_best_min_{label}"] = best_move(beans, w)
        row[f"spr_best_{label}"], row[f"spr_best_min_{label}"] = best_move(spread, w)
    return row


if __name__ == "__main__":
    cfg = load()
    live = "prior_close" not in sys.argv[1:]
    bars, hedge, _, _, triggers = build(cfg, live)
    hedge_arrays = tuple(hedge[c].values for c in ["alpha", "beta_meal", "beta_oil"])
    log_c = {leg: np.log(bars[f"{leg}_adj_close"]).values for leg in LEGS}
    log_open_zs = np.log(bars["ZS_adj_open"]).values
    ticks = (bars["ZS_adj_close"] / TICK_CENTS).values
    session = session_label(bars.index, bars["session_type"])
    half = half_label(bars["trading_date"])
    usable = ((bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
              & (np.arange(len(bars)) + MAX_BARS < len(bars)))

    rng = np.random.default_rng(42)
    rows = []
    for tr in triggers.itertuples():
        if not usable[tr.bar]:
            continue
        rows.append({"kind": "signal", "half": tr.half, "session": tr.session}
                    | measure(tr.bar, tr.side, hedge_arrays, log_c, log_open_zs, ticks))
        pool = np.flatnonzero((session == tr.session) & (half == tr.half) & usable)
        for r_bar in rng.choice(pool, RANDOM_PER_TRIGGER):
            rows.append({"kind": "random", "half": tr.half, "session": tr.session}
                        | measure(int(r_bar), int(rng.choice([-1, 1])), hedge_arrays, log_c, log_open_zs, ticks))
    t = pd.DataFrame(rows)
    sig, rnd = t[t["kind"] == "signal"], t[t["kind"] == "random"]
    assert abs(rnd["race_3"].dropna().mean() - 0.5) < 0.03          # a coin-flip race is ~50/50, else the race is biased

    halves = ["2021-22", "2023-Nov24"]
    horizon_rows = []
    for label in HORIZONS:
        row = {"horizon": label}
        for hf in halves:
            s_h, r_h = sig[sig["half"] == hf], rnd[rnd["half"] == hf]
            row[f"{hf} beans"] = s_h[f"out_{label}"].mean()
            row[f"{hf} t"] = s_h[f"out_{label}"].mean() / s_h[f"out_{label}"].sem()
            row[f"{hf} hit%"] = (s_h[f"out_{label}"] > 0).mean() * 100
            row[f"{hf} rnd hit%"] = (r_h[f"out_{label}"] > 0).mean() * 100
            row[f"{hf} spread"] = s_h[f"spr_{label}"].mean()
            row[f"{hf} from open"] = s_h[f"open_{label}"].mean()
        horizon_rows.append(row)
    horizons = pd.DataFrame(horizon_rows)

    tail_rows = []
    for kind, g_kind in [("signal", sig), ("random", rnd)]:
        for hf in halves:
            g = g_kind[g_kind["half"] == hf]
            for instrument, prefix in [("beans", "out"), ("spread", "spr")]:
                for label in ["30m", "1h"]:
                    x = g[f"{prefix}_{label}"]
                    row = {"kind": kind, "half": hf, "instrument": instrument, "horizon": label}
                    for k in TAIL_TICKS:
                        row[f">= +{k} %"] = (x >= k).mean() * 100
                        row[f"<= -{k} %"] = (x <= -k).mean() * 100
                    row["net mean"] = x.mean()
                    tail_rows.append(row)
    tails = pd.DataFrame(tail_rows)

    race_rows = []
    for hf in halves:
        for r in RACE_TICKS:
            s_r, r_r = sig[sig["half"] == hf][f"race_{r}"], rnd[rnd["half"] == hf][f"race_{r}"]
            race_rows.append({"half": hf, "R": r, "decided%": s_r.notna().mean() * 100, "win%": s_r.mean() * 100,
                              "random win%": r_r.mean() * 100, "needed at 1 tick cost": (0.5 + 1 / (2 * r)) * 100,
                              "needed at 0.4 tick": (0.5 + 0.4 / (2 * r)) * 100,
                              "ticks per decided race, gross": (2 * s_r.mean() - 1) * r})
    races = pd.DataFrame(race_rows)

    # best move in the trade's favour, Europe and US (pooled halves; the median per half is shown as a stability check)
    assert best_move(np.array([-1.0, 2.0, 5.0, 3.0]), 4) == (5.0, 15) and best_move(np.array([-1.0, -2.0]), 2)[0] == 0.0
    best_rows = []
    for sess in ["europe", "us"]:
        for instrument, prefix in [("beans", "out"), ("spread", "spr")]:
            for label, w in BEST_WINDOWS.items():
                s_g = sig[sig["session"] == sess]
                x, minutes = s_g[f"{prefix}_best_{label}"], s_g[f"{prefix}_best_min_{label}"]
                r_x = rnd[rnd["session"] == sess][f"{prefix}_best_{label}"]
                best_rows.append({"session": sess, "instrument": instrument, "window": label, "n": len(x),
                                  "never favourable %": (x == 0).mean() * 100, "min": x.min(),
                                  "p25": x.quantile(0.25), "median": x.median(), "mean": x.mean(),
                                  "p75": x.quantile(0.75), "p90": x.quantile(0.9), "max": x.max(),
                                  "random median": r_x.median(), "random mean": r_x.mean(),
                                  "median 2021-22": x[s_g["half"] == "2021-22"].median(),
                                  "median 2023-24": x[s_g["half"] == "2023-Nov24"].median(),
                                  "minutes p25": minutes.quantile(0.25), "minutes median": minutes.median(),
                                  "minutes p75": minutes.quantile(0.75),
                                  "best at window end %": (minutes == w * BAR_MINUTES).mean() * 100})
    best = pd.DataFrame(best_rows)

    suffix = "" if live else "_prior_close"
    for name, table in [("horizons", horizons), ("tails", tails), ("race", races), ("best_move", best)]:
        table.to_csv(STRATEGY_DIR / "output" / f"outcomes_{name}{suffix}.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"{'live' if live else 'prior-close'} gap: {len(sig)} triggers")
    print("\nforward move from the trigger close (ticks; hit% = share > 0, ties count as misses):\n",
          horizons.round(1).to_string(index=False))
    print("\nwin side and loss side (share of triggers beyond +/- k ticks) and the net mean:\n",
          tails.round(1).to_string(index=False))
    print("\nrace within 30 min on beans: +R ticks before -R (5-min closes):\n", races.round(1).to_string(index=False))
    print("\nbest move in the trade's favour (ticks, 5-min closes) and minutes to reach it:\n",
          best.round(1).to_string(index=False))
