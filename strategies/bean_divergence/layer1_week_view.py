"""
layer1_week_view.py
-------------------
Layer 1, step 3: how does the crush fair-value relationship behave day to day?

    1. bounce-free check of the SPREAD IC: the residual vs its own next move measured from the next
       bar's OPEN (all three legs), Europe and US, prior-close and live residual
    2. two continuous weeks (one busy 2022, one quiet 2024): beans vs fair value, the prior-close gap
       and the live gap, with Asia / Europe / US shaded
    3. the gap's daily behaviour by session: its range within the session (max - min), where it
       ends, and how far it strays (largest absolute gap), in ZS ticks

Development only; the holdout stays locked.

    python -m strategies.bean_divergence.layer1_week_view
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import ROOT
from core_quant.theme import LIGHT, PALETTE
from core_quant.validation import nw_t_stat
from strategies.bean_divergence.failure_study import session_label
from strategies.bean_divergence.feature_study import STRATEGY_DIR, TICK_CENTS
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.kalman_filter import freeze_at_previous_session_close, run_kalman_filter
from strategies.crush_strategy.strategy import STRATEGY_DIR as CRUSH_DIR, load

WEEKS = [("2022-03-07", "2022-03-12"), ("2024-03-04", "2024-03-09")]       # trading dates Mon - Fri
SESSION_COLOUR = {"asia": LIGHT["neutral"], "europe": "#dce9f8", "us": "#fde6da"}


def log_gap(log_legs: dict, hedge: pd.DataFrame, field: str = "close") -> pd.Series:
    """log beans - log fair value with the hedge of each bar (prices: bar close or bar open)."""
    zs, zm, zl = log_legs[field]
    return zs - (hedge["alpha"] + hedge["beta_meal"] * zm + hedge["beta_oil"] * zl)


if __name__ == "__main__":
    cfg = load(CRUSH_DIR / "config.yaml")
    bars = build_bars(ROOT / cfg.data_path, 5)
    bars = bars[bars["trading_date"] < pd.Timestamp(cfg.split.holdout_start)]        # holdout stays locked
    states = run_kalman_filter(bars, 5)
    hedges = {"prior close": freeze_at_previous_session_close(bars, states), "live": states.shift(1)}
    log_legs = {"close": [np.log(bars[f"{leg}_adj_close"]) for leg in ["ZS", "ZM", "ZL"]],
                "open": [np.log(bars[f"{leg}_adj_open"]) for leg in ["ZS", "ZM", "ZL"]]}
    ticks = bars["ZS_adj_close"] / TICK_CENTS
    session_id = bars["session_id"]
    session = pd.Series(session_label(bars.index, bars["session_type"]), index=bars.index)
    half = pd.Series(np.where(bars["trading_date"] < pd.Timestamp("2023-01-01"), "2021-22", "2023-Nov24"), index=bars.index)
    in_dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
    out = STRATEGY_DIR / "output"

    # 1. bounce-free spread IC: gap now vs (gap at close of t+h - gap at OPEN of t+1), hedge of t held fixed
    rows = []
    for name, hedge in hedges.items():
        gap_now = log_gap(log_legs, hedge) * ticks
        for label, h in {"30m": 6, "1h": 12, "2h": 24}.items():
            zs_c, zm_c, zl_c = (s.shift(-h) for s in log_legs["close"])
            zs_o, zm_o, zl_o = (s.shift(-1) for s in log_legs["open"])
            gap_later = zs_c - (hedge["alpha"] + hedge["beta_meal"] * zm_c + hedge["beta_oil"] * zl_c)
            gap_entry = zs_o - (hedge["alpha"] + hedge["beta_meal"] * zm_o + hedge["beta_oil"] * zl_o)
            move = ((gap_later - gap_entry) * ticks).where(session_id.shift(-h) == session_id)
            frame = pd.DataFrame({"gap": gap_now, "move": move, "session": session, "half": half})[in_dev]
            for (sess, hf), f in frame[frame["session"].isin(["europe", "us"])].groupby(["session", "half"]):
                f = f.dropna()
                rows.append({"residual": name, "horizon": label, "session": sess, "half": hf,
                             "ic": f["gap"].corr(f["move"], method="spearman"),
                             "t_nw": nw_t_stat(f, ["gap"], "move", lags=h).loc["gap", "t_nw"]})
    spread_ic = pd.DataFrame(rows)
    spread_ic.to_csv(out / "layer1_spread_ic_bounce_free.csv", index=False)

    # 2. two continuous weeks
    gap_close = log_gap(log_legs, hedges["prior close"]) * ticks
    gap_live = log_gap(log_legs, hedges["live"]) * ticks
    fair_cents = np.exp(log_legs["close"][0] - log_gap(log_legs, hedges["prior close"]))
    fig, axes = plt.subplots(3, 2, figsize=(18, 9), gridspec_kw={"height_ratios": [2, 1, 1]})
    for col, (a, b) in enumerate(WEEKS):
        week = ((bars["trading_date"] >= a) & (bars["trading_date"] < b)).values
        idx = np.flatnonzero(week)
        x = np.arange(len(idx))
        sess = session.values[idx]
        for ax in axes[:, col]:
            start = 0                               # shade each run of bars in one session
            for i in range(1, len(x) + 1):
                if i == len(x) or sess[i] != sess[start] or session_id.values[idx[i]] != session_id.values[idx[start]]:
                    ax.axvspan(start, i - 1, color=SESSION_COLOUR[sess[start]], linewidth=0, zorder=0)
                    start = i
        axes[0, col].plot(x, bars["ZS_adj_close"].values[idx], color=PALETTE[0], linewidth=1, label="beans")
        axes[0, col].plot(x, fair_cents.values[idx], color=PALETTE[1], linewidth=1, label="fair value (prior close)")
        axes[1, col].plot(x, gap_close.values[idx], color=LIGHT["ink"], linewidth=0.9)
        axes[2, col].plot(x, gap_live.values[idx], color=PALETTE[0], linewidth=0.7)
        for ax in axes[1:, col]:
            ax.axhline(0, color=LIGHT["muted"], linewidth=0.8)
        days = pd.Series(bars["trading_date"].values[idx])
        starts = list(days.index[days != days.shift()])
        for ax in axes[:, col]:
            ax.set_xticks(starts, [f"{d:%a %d %b}" for d in days[starts]], fontsize=8)
        axes[0, col].set_title(f"Week of {pd.Timestamp(a):%d %b %Y} (grey = Asia, blue = Europe, orange = US)", fontsize=10)
        axes[0, col].set_ylabel("c/bu"); axes[1, col].set_ylabel("prior-close gap (ticks)"); axes[2, col].set_ylabel("live gap (ticks)")
        axes[0, col].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "layer1_weeks.png", dpi=110)

    # 3. the gap's behaviour inside each session, day by day
    frame = pd.DataFrame({"gap": gap_close, "live_abs": gap_live.abs(), "session": session, "half": half,
                          "sid": session_id})[in_dev]
    frame = frame[frame["session"].isin(["asia", "europe", "us"])]
    per = frame.groupby(["sid", "session", "half"]).agg(range_ticks=("gap", lambda g: g.max() - g.min()),
                                                        end_ticks=("gap", "last"), max_abs_ticks=("gap", lambda g: g.abs().max()),
                                                        live_p90=("live_abs", lambda g: g.quantile(0.9)))
    summary = per.groupby(["session", "half"]).quantile([0.5, 0.9]).unstack()
    summary.to_csv(out / "layer1_gap_daily.csv")
    pd.set_option("display.width", 250)
    print(spread_ic.assign(v=spread_ic["ic"].map("{:+.3f}".format) + " (" + spread_ic["t_nw"].map("{:+.1f}".format) + ")")
          .pivot_table(index=["residual", "session", "half"], columns="horizon", values="v", aggfunc="first")[["30m", "1h", "2h"]].to_string())
    print(summary.round(1).to_string())
