"""
layer1_extremes.py
------------------
Layer 1, step 6: what does the market do when the crush gap reaches an extreme? Descriptive, no costs.

Gap = log beans - crush fair value (prior-close or live hedge), in ZS ticks.
Extreme = the gap beyond the 90th / 95th / 99th percentile (beans rich) or below the 10th / 5th / 1st
(beans cheap) of ITS OWN distribution over the previous 20 trading days of the same session (Asia /
Europe / US). Only past days set the levels. One extreme per excursion: it counts again only after the
gap has come back inside its 25th-75th percentile band.

After each extreme, from the next bar's open, with the hedge of the extreme bar held fixed, inside the session:
    spread    : the gap's move, signed so + = back toward fair value          (what a hedged trade earns)
    outright  : beans' move, signed the same way (+ = beans fell when rich)   (what an outright trade earns)
    at 30 min, 1 h, 2 h, 4 h; plus over the next 2 h how far the gap first went AGAINST (adverse) and how far
    it came back at best (favourable). Counts per day and per week.
Development only; the holdout stays locked.

    python -m strategies.bean_divergence.layer1_extremes
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
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.kalman_filter import freeze_at_previous_session_close, run_kalman_filter
from strategies.crush_strategy.strategy import STRATEGY_DIR as CRUSH_DIR, load

LEVELS = {"90th": (0.90, 0.10), "95th": (0.95, 0.05), "99th": (0.99, 0.01)}
WINDOW_DAYS = 20
HORIZONS = {"30m": 6, "1h": 12, "2h": 24, "4h": 48}
PATH_BARS = 24                      # adverse / favourable measured over the next 2 h


def past_percentiles(gap: pd.Series, session: pd.Series, dates: pd.Series, qs: list[float]) -> pd.DataFrame:
    """For each bar: percentiles of the gap over the previous WINDOW_DAYS trading days of the same session."""
    out = pd.DataFrame(np.nan, index=gap.index, columns=qs)
    for sess in ["asia", "europe", "us"]:
        in_sess = (session == sess).values
        values, day = gap.values[in_sess], dates.values[in_sess]
        positions = np.flatnonzero(in_sess)
        day_list = list(pd.unique(day))
        rows_of_day = {d: np.flatnonzero(day == d) for d in day_list}
        for i in range(WINDOW_DAYS, len(day_list)):
            pool = np.concatenate([values[rows_of_day[d]] for d in day_list[i - WINDOW_DAYS:i]])
            pool = pool[~np.isnan(pool)]
            if len(pool) < 100:
                continue
            today = positions[rows_of_day[day_list[i]]]
            out.iloc[today] = np.quantile(pool, qs)
    return out


def find_extremes(gap: np.ndarray, levels: pd.DataFrame, session_id: np.ndarray, hi_q: float, lo_q: float) -> list[tuple[int, int]]:
    """(bar, side) for the first bar of each excursion beyond the level; side +1 = rich, -1 = cheap."""
    hi, lo = levels[hi_q].values, levels[lo_q].values
    upper_mid, lower_mid = levels[0.75].values, levels[0.25].values
    events, armed_rich, armed_cheap = [], True, True
    for t in range(len(gap)):
        if t > 0 and session_id[t] != session_id[t - 1]:
            armed_rich = armed_cheap = True                    # a new session starts fresh
        if np.isnan(gap[t]) or np.isnan(hi[t]):
            continue
        if armed_rich and gap[t] >= hi[t]:
            events.append((t, 1)); armed_rich = False
        elif not armed_rich and gap[t] < upper_mid[t]:
            armed_rich = True
        if armed_cheap and gap[t] <= lo[t]:
            events.append((t, -1)); armed_cheap = False
        elif not armed_cheap and gap[t] > lower_mid[t]:
            armed_cheap = True
    return events


def follow_through(s: int, side: int, hedge: tuple, log_o: dict, log_c: dict, ticks: np.ndarray,
                   session_id: np.ndarray) -> tuple[dict, list, list]:
    """What happens after an extreme at bar s: enter at the next bar's open with bar s's hedge held fixed,
    measure the spread and outright beans (ticks, + = back toward fair value) at each horizon, the 2 h path,
    and the worst (adverse) and best (favourable) spread move within 2 h. Never past the session close."""
    a, bm, bo = hedge
    e = s + 1
    gap_at = lambda prices, i: prices["ZS"][i] - (a[s] + bm[s] * prices["ZM"][i] + bo[s] * prices["ZL"][i])
    entry_gap, entry_beans = gap_at(log_o, e), log_o["ZS"][e]
    spread_path, beans_path = [], []
    for k in range(PATH_BARS + 1):                          # path over the next 2 h (bar closes)
        i = e + k
        if i >= len(session_id) or session_id[i] != session_id[s]:
            break
        spread_path.append(-side * (gap_at(log_c, i) - entry_gap) * ticks[s])
        beans_path.append(-side * (log_c["ZS"][i] - entry_beans) * ticks[s])
    out = {"adverse_2h": -min(0.0, min(spread_path)) if spread_path else np.nan,
           "favourable_2h": max(0.0, max(spread_path)) if spread_path else np.nan}
    for label, h in HORIZONS.items():
        x = e + h - 1
        inside = x < len(session_id) and session_id[x] == session_id[s]
        out[f"spread_{label}"] = -side * (gap_at(log_c, x) - entry_gap) * ticks[s] if inside else np.nan
        out[f"outright_{label}"] = -side * (log_c["ZS"][x] - entry_beans) * ticks[s] if inside else np.nan
    pad = [np.nan] * (PATH_BARS + 1 - len(spread_path))
    return out, spread_path + pad, beans_path + pad


def summarise(events: pd.DataFrame, dates: pd.Series, half: np.ndarray, development_start) -> pd.DataFrame:
    """Per hedge / level / session / half: count, per-week rate, average follow-through, hit rate, adverse / favourable."""
    keys = [c for c in ["hedge", "measure", "level", "session", "half"] if c in events]
    day_counts = pd.DataFrame({"trading_date": dates, "half": half})
    day_counts = day_counts[dates >= pd.Timestamp(development_start)].drop_duplicates("trading_date").groupby("half").size()
    summary = events.groupby(keys).agg(
        n=("side", "size"), **{f"spread_{h}": (f"spread_{h}", "mean") for h in HORIZONS},
        **{f"outright_{h}": (f"outright_{h}", "mean") for h in HORIZONS},
        spread_1h_se=("spread_1h", "sem"), outright_1h_se=("outright_1h", "sem"),
        spread_1h_hit=("spread_1h", lambda s: (s > 0).mean() * 100), outright_1h_hit=("outright_1h", lambda s: (s > 0).mean() * 100),
        adverse_2h_median=("adverse_2h", "median"), favourable_2h_median=("favourable_2h", "median")).reset_index()
    summary["per_week"] = summary["n"] / summary["half"].map(day_counts) * 5
    return summary


if __name__ == "__main__":
    cfg = load(CRUSH_DIR / "config.yaml")
    bars = build_bars(ROOT / cfg.data_path, 5)
    bars = bars[bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start)]        # holdout stays locked
    states = run_kalman_filter(bars, 5)
    log_c = {leg: np.log(bars[f"{leg}_adj_close"]).values for leg in ["ZS", "ZM", "ZL"]}
    log_o = {leg: np.log(bars[f"{leg}_adj_open"]).values for leg in ["ZS", "ZM", "ZL"]}
    ticks = (bars["ZS_adj_close"] / TICK_CENTS).values
    session_id = bars["session_id"].values
    session = pd.Series(session_label(bars.index, bars["session_type"]), index=bars.index)
    dates = bars["trading_date"]
    half = np.where(dates < pd.Timestamp("2023-01-01"), "2021-22", "2023-Nov24")
    qs = sorted({0.25, 0.75} | {q for pair in LEVELS.values() for q in pair})

    rows, paths = [], {}
    for hedge_name, hedge in {"prior close": freeze_at_previous_session_close(bars, states), "live": states.shift(1)}.items():
        a, bm, bo = (hedge[c].values for c in ["alpha", "beta_meal", "beta_oil"])
        gap = pd.Series((log_c["ZS"] - (a + bm * log_c["ZM"] + bo * log_c["ZL"])) * ticks, index=bars.index)
        levels = past_percentiles(gap, session, dates, qs)
        for level, (hi_q, lo_q) in LEVELS.items():
            for s, side in find_extremes(gap.values, levels, session_id, hi_q, lo_q):
                if dates.iloc[s] < pd.Timestamp(cfg.split.development_start) or s + 1 >= len(bars) or session_id[s + 1] != session_id[s]:
                    continue
                measured, spread_path, beans_path = follow_through(s, side, (a, bm, bo), log_o, log_c, ticks, session_id)
                rows.append({"hedge": hedge_name, "level": level, "session": session.iloc[s], "half": half[s],
                             "trading_date": dates.iloc[s], "side": "rich" if side > 0 else "cheap"} | measured)
                paths.setdefault((hedge_name, level, session.iloc[s]), []).append((spread_path, beans_path))
        print(f"{hedge_name}: done", flush=True)

    events = pd.DataFrame(rows)
    out = STRATEGY_DIR / "output"
    events.to_csv(out / "layer1_extremes_events.csv", index=False)
    summary = summarise(events, dates, half, cfg.split.development_start)
    summary.to_csv(out / "layer1_extremes_summary.csv", index=False)

    # chart: average spread and outright paths after a 95th-percentile extreme, Europe and US, both hedges
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    minutes = np.arange(PATH_BARS + 1) * 5
    for row_i, hedge_name in enumerate(["prior close", "live"]):
        for col, sess in enumerate(["europe", "us"]):
            ax = axes[row_i, col]
            for colour, level in zip([PALETTE[2], PALETTE[0], PALETTE[1]], LEVELS):
                pairs = paths.get((hedge_name, level, sess), [])
                if not pairs:
                    continue
                spread = np.array([p[0] for p in pairs]); beans = np.array([p[1] for p in pairs])
                ax.plot(minutes, np.nanmean(spread, axis=0), color=colour, label=f"spread, beyond {level} pct (n {len(pairs)})")
                ax.plot(minutes, np.nanmean(beans, axis=0), color=colour, linestyle="--", label=f"outright beans, {level}")
            ax.axhline(0, color=LIGHT["muted"], linewidth=0.8)
            ax.set_title(f"{sess}, {hedge_name} residual: average move after an extreme (+ = back toward fair value)", fontsize=9)
            ax.set_ylabel("ZS ticks (1 tick = 0.25 c/bu)"); ax.legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("minutes after entry (next bar's open)")
    fig.tight_layout(); fig.savefig(out / "layer1_extremes_paths.png", dpi=110)
    print("saved", save_run(STRATEGY_DIR, "layer1_extremes", cfg, {"levels": list(LEVELS), "window_days": WINDOW_DAYS},
                            tables={"extremes_summary": summary}))
