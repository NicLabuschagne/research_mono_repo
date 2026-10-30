"""
phase1_fair_value.py
--------------------
Phase 1, descriptive only: what does the fair-value model say about beans vs their price?
No signal, no lookback, no holding period.

    fair value : beans' price implied by meal and oil, hedge frozen at the previous session's close
    gap        : actual beans - fair value, in ZS ticks (it restarts near 0 every session)

Outputs (development only, by session Asia / Europe / US and by half):
    1. charts of real sessions (beans vs fair value, gap underneath)
    2. gap size: spread, extremes, how often it is unusually large (|z| >= 1.5)
    3. persistence: half-life of the gap from its own autocorrelation (5-min and 30-min sampling)
    4. who closes a large gap: over the next 30 min / 1 h / 2 h, the average change of the gap,
       split into the part beans moved and the part fair value moved, by gap-size bucket

    python -m strategies.bean_divergence.phase1_fair_value
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
from strategies.crush_strategy.features import add_session_zscore
from strategies.crush_strategy.kalman_filter import add_kalman_residual

HORIZONS = {"30m": 6, "1h": 12, "2h": 24}
Z_BUCKETS = [-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf]


def prepare(cfg) -> pd.DataFrame:
    bars = add_session_zscore(add_kalman_residual(build_bars(ROOT / cfg.data_path, 5), 5))
    bars = bars[(bars["trading_date"] >= pd.Timestamp(cfg.split.development_start))
                & (bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start))].copy()    # holdout stays locked
    ticks = bars["ZS_adj_close"] / TICK_CENTS
    log_fair = (bars["frozen_alpha"] + bars["frozen_beta_meal"] * np.log(bars["ZM_adj_close"])
                + bars["frozen_beta_oil"] * np.log(bars["ZL_adj_close"]))
    bars["fair_cents"] = np.exp(log_fair)
    bars["gap_ticks"] = bars["residual"] * ticks
    bars["log_beans"], bars["log_fair"] = np.log(bars["ZS_adj_close"]), log_fair
    bars["session"] = session_label(bars.index, bars["session_type"])
    bars["half"] = np.where(bars["trading_date"] < pd.Timestamp("2023-01-01"), "2021-22", "2023-Nov24")
    return bars


def half_life_minutes(gap: pd.Series, session_id: pd.Series, step_bars: int) -> float:
    """AR(1) of the gap sampled every `step_bars` bars within a session -> half-life in minutes."""
    lagged = gap.shift(step_bars).where(session_id.shift(step_bars) == session_id)
    pair = pd.concat([gap, lagged], axis=1).dropna().iloc[::step_bars]
    phi = np.polyfit(pair.iloc[:, 1], pair.iloc[:, 0], 1)[0]
    return np.log(0.5) / np.log(phi) * 5 * step_bars if 0 < phi < 1 else np.inf


def size_and_persistence(bars: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (session, half), b in bars.groupby(["session", "half"]):
        g = b["gap_ticks"]
        rows.append({"session": session, "half": half, "bars": len(b), "gap_sd_ticks": g.std(),
                     "gap_p05": g.quantile(0.05), "gap_p95": g.quantile(0.95),
                     "pct_time_abs_z_ge_1.5": (b["z"].abs() >= 1.5).mean() * 100,
                     "half_life_min_5m": half_life_minutes(g, b["session_id"], 1),
                     "half_life_min_30m": half_life_minutes(g, b["session_id"], 6)})
    return pd.DataFrame(rows)


def who_closes_the_gap(bars: pd.DataFrame) -> pd.DataFrame:
    """Average future change of the gap, beans and fair value (ticks), by current gap-size bucket."""
    session = bars["session_id"]
    ticks = bars["ZS_adj_close"] / TICK_CENTS
    out = []
    bucket = pd.cut(bars["z"], Z_BUCKETS)
    for name, h in HORIZONS.items():
        same = session.shift(-h) == session
        # same frozen hedge within a session, so gap change = beans change - fair change exactly
        d_beans = ((bars["log_beans"].shift(-h) - bars["log_beans"]) * ticks).where(same)
        d_fair = ((bars["log_fair"].shift(-h) - bars["log_fair"]) * ticks).where(same)
        frame = pd.DataFrame({"session": bars["session"], "half": bars["half"], "z_bucket": bucket,
                              "gap_now": bars["gap_ticks"], "d_gap": d_beans - d_fair, "d_beans": d_beans, "d_fair": d_fair})
        g = frame.dropna().groupby(["session", "half", "z_bucket"], observed=True)
        table = g[["gap_now", "d_gap", "d_beans", "d_fair"]].mean()
        table["n_bars"] = g.size()
        out.append(table.assign(horizon=name).reset_index())
    return pd.concat(out)


def chart_sessions(bars: pd.DataFrame, path) -> None:
    """Four real sessions with a large gap: Europe-heavy evenings and US days, one per half each."""
    picks = []
    for session_type, label in [("evening", "evening"), ("day", "US day")]:
        for half in ["2021-22", "2023-Nov24"]:
            b = bars[(bars["session_type"] == session_type) & (bars["half"] == half)]
            biggest = b.groupby("session_id")["gap_ticks"].apply(lambda g: g.abs().max())
            sid = biggest[biggest.between(biggest.quantile(0.85), biggest.quantile(0.9))].index[0]   # big, not extreme
            picks.append((label, sid))
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), gridspec_kw={"height_ratios": [2, 1]})
    for col, (label, sid) in enumerate(picks):
        s = bars[bars["session_id"] == sid]
        x = np.arange(len(s))
        top, bottom = axes[0, col], axes[1, col]
        top.plot(x, s["ZS_adj_close"], color=PALETTE[0], label="beans (ZS)")
        top.plot(x, s["fair_cents"], color=PALETTE[1], label="fair value from meal + oil")
        bottom.fill_between(x, s["gap_ticks"], 0, color=PALETTE[0], alpha=0.5, linewidth=0)
        bottom.axhline(0, color=LIGHT["muted"], linewidth=0.8)
        local = s.index.tz_convert("America/Chicago")
        top.set_title(f"{label}: {local[0]:%d %b %Y}")
        top.set_ylabel("c/bu"); bottom.set_ylabel("gap (ZS ticks)")
        ticks_at = x[::24]
        bottom.set_xticks(ticks_at, [f"{t:%H:%M}" for t in local[ticks_at]])
        top.set_xticks(ticks_at, [])
        if col == 0:
            top.legend(fontsize=8)
    fig.suptitle("Beans vs fair value from meal and oil (hedge frozen at the previous session close), CT", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == "__main__":
    cfg = load_config(STRATEGY_DIR / "config.yaml", StudyConfig)
    bars = prepare(cfg)
    out = STRATEGY_DIR / "output"
    chart_sessions(bars, out / "phase1_sessions.png")
    size = size_and_persistence(bars)
    closes = who_closes_the_gap(bars)
    size.to_csv(out / "phase1_gap_size.csv", index=False)
    closes.to_csv(out / "phase1_who_closes.csv", index=False)
    pd.set_option("display.width", 250)
    print(size.round(2).to_string(index=False))
