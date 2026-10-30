"""
layer1_adaptive_extremes.py
---------------------------
Layer 1, step 7: extremes measured in TODAY's terms, with no 20-day lookback.

The crush filter's own std sqrt(F) barely moves (its noise R is fixed at the warm-up), so dividing by it is
just a fixed number of ticks. Here the scale adapts, using only bars before t:
    live residual        : z = residual / sqrt(F_t), F_t = exponential average of the squared prediction errors
                           (the live residual IS the filter's prediction error) - the standard way an adaptive
                           Kalman filter estimates its own forecast variance. Half-lives 1 h, 4 h, 1 day.
    prior-close residual : z = gap / its own recent typical size (exponential RMS of the gap). Half-lives 4 h, 1 day.
    Neither resets at session breaks, so early in a session the scale still remembers the last session.
Extremes: |z| beyond 1.28 / 1.645 / 2.33 (the 90th / 95th / 99th percentile IF z were normal; the counts per
week show how true that is), once per excursion, re-armed inside |z| < 0.67. Follow-through measured exactly as
in layer1_extremes (shared function), next to the 20-day percentile results. Development only.

    python -m strategies.bean_divergence.layer1_adaptive_extremes
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import ROOT, save_run
from core_quant.theme import LIGHT, PALETTE
from strategies.bean_divergence.failure_study import session_label
from strategies.bean_divergence.feature_study import STRATEGY_DIR, TICK_CENTS
from strategies.bean_divergence.layer1_extremes import PATH_BARS, find_extremes, follow_through, summarise
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.kalman_filter import freeze_at_previous_session_close, run_kalman_filter
from strategies.crush_strategy.strategy import STRATEGY_DIR as CRUSH_DIR, load

BARS_PER_HOUR = 12
HALF_LIVES = {"live": {"1h": 12, "4h": 48, "1d": 211}, "prior close": {"4h": 48, "1d": 211}}   # 211 bars ~ one trading day
LEVELS = {"90th": 1.2816, "95th": 1.6449, "99th": 2.3263}
REARM = 0.6745                     # |z| inside the 25th-75th percentile band of a normal


def adaptive_scale(values: pd.Series, half_life_bars: int) -> pd.Series:
    """Exponential RMS of `values` using bars BEFORE t only (shifted one bar), so bar t is not in its own scale."""
    return np.sqrt((values ** 2).ewm(halflife=half_life_bars, min_periods=half_life_bars).mean()).shift(1)


if __name__ == "__main__":
    cfg = load(CRUSH_DIR / "config.yaml")
    bars = build_bars(ROOT / cfg.data_path, 5)
    bars = bars[bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start)]        # holdout stays locked
    states = run_kalman_filter(bars, 5)
    log_c = {leg: np.log(bars[f"{leg}_adj_close"]).values for leg in ["ZS", "ZM", "ZL"]}
    log_o = {leg: np.log(bars[f"{leg}_adj_open"]).values for leg in ["ZS", "ZM", "ZL"]}
    ticks = (bars["ZS_adj_close"] / TICK_CENTS).values                  # log move x price in cents / 0.25 = ZS ticks
    assert np.isclose(np.log(1201 / 1200) * 1200 / TICK_CENTS, 4.0, rtol=1e-3)     # a 1 cent move at 1200 c = 4 ticks
    session_id = bars["session_id"].values
    session = pd.Series(session_label(bars.index, bars["session_type"]), index=bars.index)
    dates = bars["trading_date"]
    half = np.where(dates < pd.Timestamp("2023-01-01"), "2021-22", "2023-Nov24")

    rows, paths = [], {}
    for hedge_name, hedge in {"prior close": freeze_at_previous_session_close(bars, states), "live": states.shift(1)}.items():
        a, bm, bo = (hedge[c].values for c in ["alpha", "beta_meal", "beta_oil"])
        gap = pd.Series((log_c["ZS"] - (a + bm * log_c["ZM"] + bo * log_c["ZL"])) * ticks, index=bars.index)
        for hl_name, hl in HALF_LIVES[hedge_name].items():
            z = gap / adaptive_scale(gap, hl)
            if hedge_name == "live" and hl_name == "1h":     # self-check: the scale at bar i uses only bars before i
                i = len(gap) // 2
                altered = gap.copy()
                altered.iloc[i:] *= 10                       # change bar i and everything after it
                assert np.isclose(adaptive_scale(gap, hl).iloc[i], adaptive_scale(altered, hl).iloc[i])
                assert not np.isclose(adaptive_scale(gap, hl).iloc[i + 1], adaptive_scale(altered, hl).iloc[i + 1])
            measure = f"adaptive z, half-life {hl_name}"
            for level, threshold in LEVELS.items():
                constant = pd.DataFrame({"hi": threshold, "lo": -threshold, 0.75: REARM, 0.25: -REARM}, index=bars.index)
                for s, side in find_extremes(z.values, constant, session_id, "hi", "lo"):
                    if dates.iloc[s] < pd.Timestamp(cfg.split.development_start) or s + 1 >= len(bars) or session_id[s + 1] != session_id[s]:
                        continue
                    measured, spread_path, beans_path = follow_through(s, side, (a, bm, bo), log_o, log_c, ticks, session_id)
                    rows.append({"hedge": hedge_name, "measure": measure, "level": level, "session": session.iloc[s],
                                 "half": half[s], "trading_date": dates.iloc[s], "side": "rich" if side > 0 else "cheap",
                                 "gap_ticks": gap.iloc[s]} | measured)
                    paths.setdefault((hedge_name, measure, level, session.iloc[s]), []).append((spread_path, beans_path))
        print(f"{hedge_name}: done", flush=True)

    events = pd.DataFrame(rows)
    summary = summarise(events, dates, half, cfg.split.development_start)
    out = STRATEGY_DIR / "output"
    events.to_csv(out / "layer1_adaptive_events.csv", index=False)
    summary.to_csv(out / "layer1_adaptive_summary.csv", index=False)

    # chart: average paths after a 95th-level extreme, Europe and US, one adaptive measure per residual
    chosen = {"prior close": "adaptive z, half-life 1d", "live": "adaptive z, half-life 1h"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    minutes = np.arange(PATH_BARS + 1) * 5
    for row_i, hedge_name in enumerate(["prior close", "live"]):
        for col, sess in enumerate(["europe", "us"]):
            ax = axes[row_i, col]
            for colour, level in zip([PALETTE[2], PALETTE[0], PALETTE[1]], LEVELS):
                pairs = paths.get((hedge_name, chosen[hedge_name], level, sess), [])
                if not pairs:
                    continue
                spread = np.array([p[0] for p in pairs]); beans = np.array([p[1] for p in pairs])
                ax.plot(minutes, np.nanmean(spread, axis=0), color=colour, label=f"spread, |z| beyond {level} level (n {len(pairs)})")
                ax.plot(minutes, np.nanmean(beans, axis=0), color=colour, linestyle="--", label=f"outright beans, {level}")
            ax.axhline(0, color=LIGHT["muted"], linewidth=0.8)
            ax.set_title(f"{sess}, {hedge_name} residual, {chosen[hedge_name]}: average move after an extreme", fontsize=9)
            ax.set_ylabel("ZS ticks (1 tick = 0.25 c/bu)"); ax.legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("minutes after entry (next bar's open); + = back toward fair value")
    fig.tight_layout(); fig.savefig(out / "layer1_adaptive_paths.png", dpi=110)
    print("saved", save_run(STRATEGY_DIR, "layer1_adaptive_extremes", cfg, {"half_lives": HALF_LIVES, "levels": LEVELS},
                            tables={"adaptive_summary": summary}))
