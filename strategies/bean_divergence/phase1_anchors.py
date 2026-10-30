"""
phase1_anchors.py
-----------------
Phase 1b, descriptive only: the same fair value, but WITHOUT the session reset.

    session : hedge frozen at the previous session's close (reference, phase 1)
    live    : the filter's estimate after the previous bar, updated every bar (no look-ahead)
    weekly  : hedge frozen at the last bar of the previous week (trading date)
    monthly : hedge frozen at the last bar of the previous month

gap = log beans - log fair value, in ZS ticks. For a gap change from t to t+h, with the
hedge AT t held fixed for the market part:
    gap change = beans move - meal/oil move (hedge at t) - model re-fit (hedge t -> t+h)
Only the first two are the market; the re-fit is the model moving its own fair value.

Horizons in trading bars across session breaks (the gap now carries): 1 h, 4 h, 1 day (~211 bars),
1 week (~1,055 bars). Development only; the holdout stays locked.

    python -m strategies.bean_divergence.phase1_anchors
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import ROOT, load_config
from core_quant.theme import LIGHT, PALETTE
from strategies.bean_divergence.failure_study import session_label
from strategies.bean_divergence.feature_study import STRATEGY_DIR, TICK_CENTS, StudyConfig
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.kalman_filter import freeze_at_previous_session_close, run_kalman_filter

VARIANTS = ["session", "live", "weekly", "monthly"]
HORIZONS = {"1h": 12, "4h": 48, "1d": 211, "1w": 1055}
Z_BUCKETS = [-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf]
STATE = ["alpha", "beta_meal", "beta_oil"]


def hedges(bars: pd.DataFrame, states: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def frozen_at_previous(period: pd.Series) -> pd.DataFrame:
        last = states.groupby(period.values).last().shift(1)            # state at the previous period's last bar
        out = last.reindex(period.values)
        out.index = bars.index
        return out
    date = pd.to_datetime(bars["trading_date"])
    return {"session": freeze_at_previous_session_close(bars, states), "live": states.shift(1),
            "weekly": frozen_at_previous(date.dt.to_period("W-FRI")), "monthly": frozen_at_previous(date.dt.to_period("M"))}


def log_fair(bars: pd.DataFrame, hedge: pd.DataFrame, shift: int = 0) -> pd.Series:
    """Fair value of bar t+shift computed with the hedge of bar t."""
    zm = np.log(bars["ZM_adj_close"]).shift(-shift)
    zl = np.log(bars["ZL_adj_close"]).shift(-shift)
    return hedge["alpha"] + hedge["beta_meal"] * zm + hedge["beta_oil"] * zl


def past_rms_daily(gap: pd.Series, date: pd.Series, days: int = 20) -> pd.Series:
    daily = pd.DataFrame({"sq": gap ** 2, "n": gap.notna().astype(float), "d": date}).groupby("d")[["sq", "n"]].sum()
    rms = np.sqrt(daily["sq"].rolling(days).sum() / daily["n"].rolling(days).sum()).shift(1)
    return pd.Series(rms.reindex(date.values).values, index=gap.index)


def half_life_hours(gap: pd.Series, step: int) -> float:
    pair = pd.concat([gap, gap.shift(step)], axis=1).dropna().iloc[::step]
    phi = np.polyfit(pair.iloc[:, 1], pair.iloc[:, 0], 1)[0]
    return np.log(0.5) / np.log(phi) * step * 5 / 60 if 0 < phi < 1 else np.inf


if __name__ == "__main__":
    cfg = load_config(STRATEGY_DIR / "config.yaml", StudyConfig)
    bars = build_bars(ROOT / cfg.data_path, 5)
    states = run_kalman_filter(bars, 5)                     # the adopted hand-set filter, run over all data in time order
    all_hedges = hedges(bars, states)
    keep = ((bars["trading_date"] >= pd.Timestamp(cfg.split.development_start))
            & (bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start))).values          # holdout stays locked
    ticks = bars["ZS_adj_close"] / TICK_CENTS
    log_beans = np.log(bars["ZS_adj_close"])
    session = session_label(bars.index, bars["session_type"])
    half = np.where(bars["trading_date"] < pd.Timestamp("2023-01-01"), "2021-22", "2023-Nov24")

    size_rows, close_rows, gaps = [], [], {}
    for name in VARIANTS:
        h = all_hedges[name]
        gap = (log_beans - log_fair(bars, h)) * ticks
        gaps[name] = gap
        z = gap / past_rms_daily(gap, bars["trading_date"])
        dev = pd.DataFrame({"gap": gap, "z": z, "session": session, "half": half})[keep]
        for hf, d in dev.groupby("half"):
            g = d["gap"].dropna()
            size_rows.append({"variant": name, "half": hf, "gap_sd_ticks": g.std(), "p05": g.quantile(0.05),
                              "p95": g.quantile(0.95), "half_life_h_30m": half_life_hours(g, 6),
                              "half_life_h_1d": half_life_hours(g, 211)})
        for label, k in HORIZONS.items():
            beans = (log_beans.shift(-k) - log_beans) * ticks
            products = (log_fair(bars, h, k) - log_fair(bars, h)) * ticks                  # hedge held at t
            refit = (log_fair(bars, h.shift(-k), k) - log_fair(bars, h, k)) * ticks         # model moving its fair value
            frame = pd.DataFrame({"session": session, "half": half, "z_bucket": pd.cut(z, Z_BUCKETS), "gap_now": gap,
                                  "d_gap": beans - products - refit, "beans": beans, "products": products, "refit": refit})[keep]
            grp = frame.dropna().groupby(["half", "session", "z_bucket"], observed=True)
            t = grp[["gap_now", "d_gap", "beans", "products", "refit"]].mean()
            t["n_bars"] = grp.size()
            close_rows.append(t.assign(variant=name, horizon=label).reset_index())
        print(f"{name}: done", flush=True)

    # self-check (log units): beans - products - re-fit equals the change in the log gap exactly
    k = HORIZONS["1d"]; h = all_hedges["weekly"]
    log_gap = log_beans - log_fair(bars, h)
    parts = ((log_beans.shift(-k) - log_beans) - (log_fair(bars, h, k) - log_fair(bars, h))
             - (log_fair(bars, h.shift(-k), k) - log_fair(bars, h, k)))
    both = pd.concat([log_gap.shift(-k) - log_gap, parts], axis=1).dropna()
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-12)
    # self-check: every hedge at bar i is unchanged when bars from i onward change (no look-ahead)
    i = int(np.flatnonzero(keep)[len(np.flatnonzero(keep)) // 2])
    altered = bars.copy()
    altered.iloc[i:, altered.columns.get_loc("ZS_adj_close")] *= 1.05
    again = hedges(altered, run_kalman_filter(altered, 5))
    for name in VARIANTS:
        assert np.allclose(all_hedges[name].iloc[: i + 1].values, again[name].iloc[: i + 1].values, equal_nan=True), name

    size, closes = pd.DataFrame(size_rows), pd.concat(close_rows)
    out = STRATEGY_DIR / "output"
    size.to_csv(out / "phase1b_gap_size.csv", index=False)
    closes.to_csv(out / "phase1b_who_closes.csv", index=False)

    # chart: two months, beans vs fair value under each anchor, gaps underneath
    fig, axes = plt.subplots(2, 2, figsize=(18, 8), gridspec_kw={"height_ratios": [2, 1]})
    for col, (a, b) in enumerate([("2022-05-01", "2022-07-01"), ("2024-02-01", "2024-04-01")]):
        m = keep & (bars["trading_date"] >= a).values & (bars["trading_date"] < b).values
        x = np.arange(m.sum())
        axes[0, col].plot(x, bars["ZS_adj_close"][m], color=LIGHT["ink"], linewidth=1.2, label="beans (ZS)")
        for i, name in enumerate(["live", "weekly", "monthly"]):
            fair = np.exp(log_fair(bars, all_hedges[name]))[m]
            axes[0, col].plot(x, fair, color=PALETTE[i], linewidth=1, label=f"fair value: {name}")
            axes[1, col].plot(x, gaps[name][m], color=PALETTE[i], linewidth=1, label=name)
        axes[1, col].axhline(0, color=LIGHT["muted"], linewidth=0.8)
        days = pd.Series(bars["trading_date"][m].values)
        starts = days.index[days != days.shift()][::5]
        axes[1, col].set_xticks(starts, [f"{d:%d %b}" for d in days[starts]])
        axes[0, col].set_xticks(starts, [])
        axes[0, col].set_title(f"{pd.Timestamp(a):%b %Y} - {pd.Timestamp(b) - pd.Timedelta(days=1):%b %Y}")
        axes[0, col].set_ylabel("c/bu"); axes[1, col].set_ylabel("gap (ZS ticks)")
        axes[0, col].legend(fontsize=8); axes[1, col].legend(fontsize=8)
    fig.suptitle("Beans vs fair value from meal and oil, hedge never reset (live) or reset weekly / monthly", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "phase1b_anchors.png", dpi=110)
    pd.set_option("display.width", 250)
    print(size.round(2).to_string(index=False))
