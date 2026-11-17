"""
regime_test.py
--------------
Layer 1 regime test for the anchored rule (2026-11-17). Pass rule: README, "PLANNED REGIME TEST" (fixed before
the run). Development data only.

    python -m strategies.wheat_corn_lag.regime_test

Question: is the anchored z's edge (IC with the next session's wheat move) different in two kinds of market?
  R1  slow fair value: the 20-session Kalman gap (the locked rule's) points the same way as z, or the other way.
  R2  variance ratio (VR) of the hedged wheat series over the previous 20 sessions:
        VR = variance of 1-session changes / (71 x variance of 15-minute changes)
      A random walk gives 1. Below 1: moves tend to reverse (good for a reversion rule). Above 1: moves tend to
      continue (trending), which is when gaps stay open.
Both measures use bars BEFORE t only (shifted one bar), so they could be known at the signal.
"""
import numpy as np
import pandas as pd

from strategies.wheat_corn_lag.anchored_rule import anchored_book, anchored_gap, anchored_signal
from strategies.wheat_corn_lag.features import BARS_PER_SESSION, build_targets, rolling_stats
from strategies.wheat_corn_lag.kalman_rule import TICK_CENTS, kalman_signal, make_kalman_evaluator
from strategies.wheat_corn_lag.strategy import load

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
MIN_DIFFERENCE, MIN_YEARS, MIN_BAD_SHARE = 0.05, 5, 0.20       # the pass rule, fixed before the run


def hedged_wheat(bars: pd.DataFrame, rule) -> pd.Series:
    """own = running total of (wheat return - beta x corn return): the series the anchored fair value is built on."""
    beta = rolling_stats(bars, rule.beta_window_sessions * BARS_PER_SESSION)["beta"]
    return (bars["ret_w"].fillna(0) - beta.fillna(0) * bars["ret_c"].fillna(0)).cumsum()


def variance_ratio(own: pd.Series, window_bars: int, q: int) -> pd.Series:
    """VR over the window ending at t-1: var of q-bar changes / (q x var of 1-bar changes)."""
    one_bar = own.diff()
    q_bar = own.diff(q)
    var_one = one_bar.rolling(window_bars, min_periods=window_bars // 2).var()
    var_q = q_bar.rolling(window_bars, min_periods=window_bars // 2).var()
    return (var_q / (q * var_one)).shift(1)


def ic(signal: pd.Series, target: pd.Series, mask: pd.Series) -> float:
    return signal[mask].corr(target[mask], method="spearman")


def judge(name: str, signal, target, tradable, groups: dict, years, good: str | None) -> dict:
    """IC per group, overall and per year, and the pass rule. good=None: the group with the higher IC overall."""
    overall = {group: ic(signal, target, tradable & mask) for group, mask in groups.items()}
    if good is None:
        good = max(overall, key=overall.get)
    bad = next(group for group in groups if group != good)
    per_year = {}
    for year in YEARS:
        in_year = tradable & (years == year)
        per_year[year] = {group: ic(signal, target, in_year & mask) for group, mask in groups.items()}
    same_sign = sum(per_year[year][good] - per_year[year][bad] > 0 for year in YEARS)
    bad_share = (tradable & groups[bad]).sum() / tradable.sum()
    checks = [(f"IC({good}) - IC({bad}) >= {MIN_DIFFERENCE}", overall[good] - overall[bad] >= MIN_DIFFERENCE,
               overall[good] - overall[bad]),
              (f"same sign in >= {MIN_YEARS} of 6 years", same_sign >= MIN_YEARS, same_sign),
              (f"bad group >= {MIN_BAD_SHARE:.0%} of bars", bad_share >= MIN_BAD_SHARE, bad_share)]

    table = pd.DataFrame({"all": overall, **{year: per_year[year] for year in YEARS}})
    table["share of bars"] = [(tradable & groups[group]).sum() / tradable.sum() for group in table.index]
    print(f"\n== {name}: IC of z with the next session's wheat move, by group")
    print(table.round(3).to_string())
    for label, ok, value in checks:
        print(f"  {'PASSED' if ok else 'FAILED'}  {label}: {value:.3f}")
    passed = all(ok for _, ok, _ in checks)
    print(f"{name}: {'PASSED' if passed else 'FAILED'}")
    return {"good": good, "bad": bad, "passed": passed}


def trades_by_group(trades, bars, groups: dict, cost: float, name: str) -> None:
    """Descriptive: the validated trades, split by the group at their signal bar."""
    signal_bar = trades["entry_bar"].values - 1
    rows = []
    for group, mask in groups.items():
        part = trades[mask.values[signal_bar]]
        net = part["move_c"] - cost
        rows.append({"group": group, "trades": len(part), "net c per trade": net.mean(), "win rate": (net > 0).mean(),
                     "net c total": net.sum(), "share time stop": (part["exit_reason"] == "time").mean()})
    print(f"-- {name}, validated trades 2020-2025 (descriptive)")
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    cfg = load()
    rule = cfg.anchored_rule
    _, _, bars, features = make_kalman_evaluator(cfg)                       # development data only
    tradable = kalman_signal(features, cfg.kalman_rule).notna()
    z = anchored_signal(anchored_gap(bars, rule), rule, tradable)
    target = build_targets(bars, cfg.features)[f"fwd_{BARS_PER_SESSION}"]
    years = pd.Series(bars.index.year, index=bars.index)
    tradable = tradable & z.notna() & target.notna()

    # R1: z > 0 = wheat cheap vs the fast fair value; -kgap_20s > 0 = wheat cheap vs the slow one.
    # kgap_20s is the live gap (state from before the bar), so it is known at the signal.
    slow = -features[f"kgap_{cfg.kalman_rule.memory_sessions}s"]
    agree = np.sign(slow) == np.sign(z)
    r1_groups = {"agree": agree & slow.notna(), "disagree": ~agree & slow.notna()}

    # R2: variance ratio of the hedged wheat series, 1-session changes over the previous 20 sessions
    vr = variance_ratio(hedged_wheat(bars, rule), 20 * BARS_PER_SESSION, BARS_PER_SESSION)
    r2_groups = {"VR < 1": vr < 1, "VR >= 1": vr >= 1}

    # self-check: the variance ratio of a random walk is about 1, and it uses bars before t only
    rng = np.random.default_rng(0)
    walk = pd.Series(rng.normal(size=60_000).cumsum())
    assert abs(variance_ratio(walk, 1420, 71).median() - 1) < 0.1
    test_vr = variance_ratio(hedged_wheat(bars, rule).where(bars.index < bars.index[50_000], 0), 1420, 71)
    assert np.isclose(test_vr.iloc[50_000], vr.iloc[50_000]), "VR at t uses bar t or later"

    print(f"{tradable.sum():,} tradable bars, {bars.index[0]:%Y-%m} to {bars.index[-1]:%Y-%m}; "
          f"z overall IC {ic(z, target, tradable):.3f}; VR median {vr[tradable].median():.2f}")
    r1 = judge("R1 slow fair value (K20) vs z", z, target, tradable, r1_groups, years, good=None)
    r2 = judge("R2 variance ratio", z, target, tradable, r2_groups, years, good="VR < 1")

    # descriptive: VR in thirds (cut points over the whole period: a look, not a rule)
    thirds = pd.qcut(vr[tradable], 3, labels=["low VR", "mid VR", "high VR"])
    print("\n-- VR in thirds (descriptive): IC by third")
    print({str(label): round(ic(z, target, tradable & (thirds.reindex(bars.index) == label)), 3) for label in thirds.cat.categories})

    trades = anchored_book(bars, z, rule)
    cost = rule.cost_ticks * TICK_CENTS
    print()
    trades_by_group(trades, bars, r1_groups, cost, "R1")
    trades_by_group(trades, bars, r2_groups, cost, "R2")
