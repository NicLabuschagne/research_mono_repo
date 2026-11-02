"""
labels.py
---------
Triple-barrier labels for the meta-model (Lopez de Prado, AFML ch. 3, meta-labelling): one label per trigger, the
side taken from the trigger. Net of costs, so label 1 = the trade made money after costs.

    entry    : the open of the first minute after the trigger bar (= the next bar's open)
    sigma_1h : the expected 1-hour move at the trigger = exponential RMS of 5-min returns (4-hour half-life, bars
               BEFORE the trigger) x sqrt(12). Beans: ZS returns; spread: the hedged crush return (live hedge).
               A 2026-11-02 scratch test compared recent / same time of day / daily volatility: this one made
               Europe, US and both halves most alike (best move within 1 h ~0.6 sigma_1h in every group).
    barriers : target = k x sigma_1h, stop = stop_mult x target (1x / 1.25x first - triggers often move against you
               first; 0.5x / 0.75x added to test wins bigger than losses), time limit 12 bars = 60 min (the live
               gap's edge is spent by 1 h)
    path     : 1-minute bars from the entry. Beans: a barrier is touched when the minute's high / low reaches it;
               if both are touched in the same minute the stop counts (we cannot know which came first). Spread: on
               1-minute closes (the legs' highs and lows are not simultaneous). Both barriers fill AT the barrier -
               slippage is in the cost - and the time limit at the last close. (Filling stops at the crossing close
               but targets at the barrier made even random entries lose ~1 tick gross.)
    costs    : beans 1 tick a round trip. Spread, per ZS contract, hedge contracts from the trigger bar's betas:
               all legs crossed (1 tick each: ZS $12.50, ZM $10, ZL $6) or beans passive (only meal and oil crossed).
               The exchange-listed board crush needs its own data and a tracking check - not here yet.
    skipped  : barrier under MIN_BARRIER_COSTS x the cost - the cost would eat the edge (AFML's minimum target)
    weight   : average uniqueness - a trade sharing its bars with other open trades counts for less (AFML ch. 4)
Random baseline: 5 random bars per trigger, same session and half, random side. Development only.

Entry rules (2026-11-02; column entry_rule - the meta-model uses "trigger" only):
    trigger  : enter at the bar after the trigger (the original)
    gap_turn : wait for the first bar whose gap is NARROWER than the bar before (the dislocation starts closing)
    z_back   : wait for the first bar with |z| back inside the trigger level (back inside the band)
Confirmation waits at most CONFIRM_WAIT_BARS (30 min) and never across a session break; no confirmation = no trade.
Sigma, hedge and costs are taken at the confirmation bar; the time limit runs from the entry.

    python -m strategies.bean_divergence.labels
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core_quant.config import ROOT
from core_quant.theme import LIGHT, PALETTE
from strategies.bean_divergence.fair_value import (LEGS, STRATEGY_DIR, TICK_CENTS, adaptive_scale, half_label, load,
                                                   session_label)
from strategies.bean_divergence.triggers import build
from strategies.crush_strategy.data_ingest import (add_sessions, add_switch_free_returns, back_adjust,
                                                   flag_contract_switches, load_minute_data)

K = [0.5, 0.75, 1.0]                 # target = k x sigma_1h
STOP_MULT = [0.5, 0.75, 1.0, 1.25]   # stop = stop_mult x target (0.5 / 0.75 added 2026-11-02: wins bigger than losses)
HOLD_BARS = 12                       # 60 min
MIN_BARRIER_COSTS = 4
RANDOM_PER_TRIGGER = 5
TICK_DOLLARS = {"ZS": 12.50, "ZM": 10.0, "ZL": 6.0}
COST_CASES = {"beans": ["beans"], "spread": ["spread_crossed", "spread_beans_passive"]}
ENTRY_RULES = ["trigger", "gap_turn", "z_back"]
CONFIRM_WAIT_BARS = 6


def first_touch(best: np.ndarray, worst: np.ndarray, target: float, stop: float) -> tuple[int, str]:
    """(minute index, reason) of the first minute whose best reaches the target or whose worst reaches the stop
    (both in the same minute -> stop); the last minute if neither."""
    up = np.flatnonzero(best >= target)
    down = np.flatnonzero(worst <= -stop)
    first_up = up[0] if len(up) else len(best)
    first_down = down[0] if len(down) else len(best)
    if first_up == first_down == len(best):
        return len(best) - 1, "time"
    return (first_up, "target") if first_up < first_down else (first_down, "stop")


def average_uniqueness(start: np.ndarray, end: np.ndarray, n_bars: int) -> np.ndarray:
    """Mean over each trade's bars [start, end] of 1 / (trades open on that bar)."""
    change = np.zeros(n_bars + 1)
    for s, e in zip(start, end):
        change[s] += 1
        change[e + 1] -= 1
    open_count = np.cumsum(change)[:n_bars]
    return np.array([np.mean(1 / open_count[s:e + 1]) for s, e in zip(start, end)])


def confirm_bar(s: int, side: int, gap: np.ndarray, z: np.ndarray, session_id: np.ndarray, rule: str,
                level: float, wait: int = CONFIRM_WAIT_BARS) -> int | None:
    """The bar whose close confirms the entry for trigger bar s (s itself for "trigger"); None if no confirmation
    within `wait` bars in the same session."""
    if rule == "trigger":
        return s
    for t in range(s + 1, min(s + wait + 1, len(gap))):
        if session_id[t] != session_id[s]:
            return None
        if rule == "gap_turn" and side * (gap[t] - gap[t - 1]) < 0:
            return t
        if rule == "z_back" and side * z[t] < level:
            return t
    return None


def minute_prices(cfg) -> pd.DataFrame:
    """Back-adjusted 1-minute open / high / low / close, built exactly like the bars."""
    data = load_minute_data(ROOT / cfg.data_path)
    data = back_adjust(flag_contract_switches(add_switch_free_returns(add_sessions(data))))
    for leg in LEGS:
        factor = data[f"{leg}_adj_close"] / data[f"{leg}_close"]          # the same back-adjustment as open / close
        data[f"{leg}_adj_high"], data[f"{leg}_adj_low"] = data[f"{leg}_high"] * factor, data[f"{leg}_low"] * factor
    return data[[f"{leg}_adj_{f}" for leg in LEGS for f in ["open", "high", "low", "close"]]]


if __name__ == "__main__":
    path = np.array([1.0, -2.0, 3.0])
    assert first_touch(path, path, 3, 3) == (2, "target")
    assert first_touch(path, np.array([1.0, -3.0, 3.0]), 3, 3) == (1, "stop")
    assert first_touch(np.array([1.0, -2.0, 2.0]), np.array([1.0, -2.0, 2.0]), 3, 3) == (2, "time")
    assert first_touch(np.array([1.0, 3.0]), np.array([0.0, -3.0]), 3, 3) == (1, "stop")     # same minute: stop
    assert np.allclose(average_uniqueness(np.array([0, 1]), np.array([1, 2]), 3), [0.75, 0.75])
    # rich trigger at bar 1 (side +1): gap widens at 2, narrows at 3; |z| back inside 1.645 at 4; session ends at 5
    g_, z_, sid_ = np.array([0, 5, 6, 5.5, 3, 1.0]), np.array([0, 2.0, 2.2, 1.9, 1.2, 0.5]), np.array([1, 1, 1, 1, 1, 2])
    assert [confirm_bar(1, 1, g_, z_, sid_, rule, 1.645) for rule in ENTRY_RULES] == [1, 3, 4]
    assert confirm_bar(1, 1, g_, np.array([0, 2.0, 2.2, 2.1, 2.0, 0.5]), sid_, "z_back", 1.645) is None  # session ended

    cfg = load()
    r = cfg.research
    bars, hedge, gap, z, triggers = build(cfg)
    session_id = bars["session_id"].values
    n = len(bars)
    a, bm, bo = (hedge[c].values for c in ["alpha", "beta_meal", "beta_oil"])
    ticks = (bars["ZS_adj_close"] / TICK_CENTS).values
    ret = {leg: bars[f"{leg}_ret"] for leg in LEGS}
    sigma = {"beans": adaptive_scale(ret["ZS"] * ticks, r.z_half_life_bars).values * np.sqrt(HOLD_BARS),
             "spread": adaptive_scale((ret["ZS"] - hedge["beta_meal"] * ret["ZM"] - hedge["beta_oil"] * ret["ZL"]) * ticks,
                                      r.z_half_life_bars).values * np.sqrt(HOLD_BARS)}
    # hedge contracts per ZS contract (shortcut: back-adjusted prices for contract values, a few % off before rolls)
    value = {"ZS": bars["ZS_adj_close"].values / 100 * 5000, "ZM": bars["ZM_adj_close"].values * 100,
             "ZL": bars["ZL_adj_close"].values / 100 * 60000}
    hedge_cost = (bm * value["ZS"] / value["ZM"] * TICK_DOLLARS["ZM"]
                  + bo * value["ZS"] / value["ZL"] * TICK_DOLLARS["ZL"]) / TICK_DOLLARS["ZS"] * r.cost_ticks
    cost = {"beans": np.full(n, r.cost_ticks), "spread_crossed": r.cost_ticks + hedge_cost,
            "spread_beans_passive": hedge_cost}

    minutes = minute_prices(cfg)
    m_time = minutes.index
    log_open = {leg: np.log(minutes[f"{leg}_adj_open"].values) for leg in LEGS}
    log_close = {leg: np.log(minutes[f"{leg}_adj_close"].values) for leg in LEGS}
    log_high, log_low = (np.log(minutes[f"ZS_adj_{f}"].values) for f in ["high", "low"])
    bar_pos = m_time.searchsorted(bars.index)                    # each bar is stamped with its last minute
    assert (m_time[bar_pos] == bars.index).all()

    dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
    usable = dev & (np.arange(n) + HOLD_BARS < n) & ~np.isnan(sigma["beans"]) & ~np.isnan(sigma["spread"])
    bar_session, bar_half = session_label(bars.index, bars["session_type"]), half_label(bars["trading_date"])

    events = []                                                  # (kind, entry rule, trigger bar, entry bar, side)
    for tr in triggers.itertuples():
        for rule in ENTRY_RULES:
            e = confirm_bar(tr.bar, tr.side, gap.values, z.values, session_id, rule, r.trigger_z)
            if usable[tr.bar] and e is not None and usable[e]:
                events.append(("signal", rule, tr.bar, e, tr.side))
    rng = np.random.default_rng(42)
    for _, rule, s, _, _ in [ev for ev in events if ev[1] == "trigger"]:
        pool = np.flatnonzero((bar_session == bar_session[s]) & (bar_half == bar_half[s]) & usable)
        for r_bar in rng.choice(pool, RANDOM_PER_TRIGGER):
            events.append(("random", "trigger", int(r_bar), int(r_bar), int(rng.choice([-1, 1]))))

    rows = []
    for event_id, (kind, rule, s, e, side) in enumerate(events):
        p0, p1 = bar_pos[e] + 1, bar_pos[e + HOLD_BARS]          # entry minute .. last minute of the time limit
        assert m_time[p0] > bars.index[e]                         # the entry is after the deciding bar is known
        held = lambda log, i: log["ZS"][i] - (a[e] + bm[e] * log["ZM"][i] + bo[e] * log["ZL"][i])
        window = np.arange(p0, p1 + 1)
        favour = lambda log: -side * (log[window] - log_open["ZS"][p0]) * ticks[e]
        beans_hi_lo = (favour(log_low), favour(log_high)) if side > 0 else (favour(log_high), favour(log_low))
        spread = -side * (held(log_close, window) - held(log_open, p0)) * ticks[e]
        paths = {"beans": (favour(log_close["ZS"]),) + beans_hi_lo,                 # (closes, best, worst)
                 "spread": (spread, spread, spread)}
        base = {"event": event_id, "kind": kind, "entry_rule": rule, "bar": s, "entry_bar": e, "wait_bars": e - s,
                "time": bars.index[e], "session": bar_session[s], "half": bar_half[s], "side": side,
                "crosses_break": session_id[e + HOLD_BARS] != session_id[e]}
        for instrument, (path, best, worst) in paths.items():
            sig = sigma[instrument][e]
            excursion = {"sigma_1h": sig, "mfe_sigma": max(best.max(), 0) / sig, "mae_sigma": max(-worst.min(), 0) / sig}
            for k in K:
                for stop_mult in STOP_MULT:
                    target, stop = k * sig, stop_mult * k * sig
                    i, reason = first_touch(best, worst, target, stop)
                    gross = {"target": target, "stop": -stop, "time": path[i]}[reason]
                    for case in COST_CASES[instrument]:
                        c = cost[case][e]
                        rows.append(base | excursion | {
                            "instrument": case, "k": k, "stop_mult": stop_mult, "target_ticks": target,
                            "stop_ticks": stop, "cost_ticks": c, "skipped": target < MIN_BARRIER_COSTS * c,
                            "exit_reason": reason, "exit_minutes": i + 1, "exit_time": m_time[p0 + i],
                            "gross_ticks": gross, "net_ticks": gross - c, "label": int(gross - c > 0)})
    labels = pd.DataFrame(rows)
    labels["exit_bar"] = bars.index.searchsorted(labels["exit_time"])          # the bar holding the exit minute
    labels["weight"] = np.nan
    groups = labels[(labels.kind == "signal") & ~labels.skipped].groupby(["entry_rule", "instrument", "k", "stop_mult"])
    for _, g in groups:
        labels.loc[g.index, "weight"] = average_uniqueness(g["entry_bar"].values + 1, g["exit_bar"].values, n)
    labels.to_parquet(STRATEGY_DIR / "output" / "labels.parquet")
    all_rules = labels
    labels = labels[labels.entry_rule == "trigger"]               # the summaries and plots below: the original entry

    # summary: exits, win share, net ticks - signal vs random, per half (and per session in the CSV)
    taken = labels[~labels.skipped]
    keys = ["instrument", "k", "stop_mult", "kind"]

    def summarise(group_keys):
        g = taken.groupby(group_keys)
        out = pd.DataFrame({"n": g.size(), "target%": g.exit_reason.apply(lambda x: (x == "target").mean() * 100),
                            "stop%": g.exit_reason.apply(lambda x: (x == "stop").mean() * 100),
                            "time%": g.exit_reason.apply(lambda x: (x == "time").mean() * 100),
                            "win%": g.label.mean() * 100, "net": g.net_ticks.mean(), "gross": g.gross_ticks.mean(),
                            "target ticks": g.target_ticks.median(), "cost": g.cost_ticks.median(),
                            "minutes": g.exit_minutes.median(), "uniqueness": g.weight.mean()})
        skipped = labels.groupby(group_keys).skipped.mean() * 100
        return out.join(skipped.rename("skipped%")).reset_index()

    by_half, by_session = summarise(keys + ["half"]), summarise(keys + ["session"])
    by_half.to_csv(STRATEGY_DIR / "output" / "labels_summary.csv", index=False)
    by_session.to_csv(STRATEGY_DIR / "output" / "labels_summary_session.csv", index=False)
    pd.set_option("display.width", 250)
    sig_events = labels[labels.kind == "signal"].event.nunique()
    print(f"{sig_events} triggers, {labels[labels.kind == 'random'].event.nunique()} random entries; "
          f"{labels[labels.kind == 'signal'].groupby('event').crosses_break.first().mean() * 100:.0f}% of triggers hold "
          "through a session break")
    print("\nnet of costs, per half (ticks; win% = share of trades with net > 0; skipped = barrier < "
          f"{MIN_BARRIER_COSTS} x cost):\n", by_half.round(1).to_string(index=False))
    balance = taken[taken.kind == "signal"].groupby(["instrument", "k", "stop_mult"]).label.mean()
    print("\nlabel balance (share of 1s) - flag if outside 20-80%:", balance.round(2).to_dict())

    # plot 1: best (MFE) and worst (MAE) move within the hour in sigma_1h units, with the barrier choices
    one = labels[(labels.k == K[0]) & (labels.stop_mult == STOP_MULT[0])
                 & labels.instrument.isin(["beans", "spread_crossed"])]
    bins = np.linspace(0, 3, 61)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for row, instrument in enumerate(["beans", "spread_crossed"]):
        for col, sess in enumerate(["europe", "us"]):
            ax = axes[row, col]
            g = one[(one.instrument == instrument) & (one.session == sess)]
            sg, rg = g[g.kind == "signal"], g[g.kind == "random"]
            ax.hist(sg.mfe_sigma.clip(upper=3), bins, density=True, color=PALETTE[5], alpha=0.45,
                    label=f"triggers: best move (median {sg.mfe_sigma.median():.2f})")
            ax.hist(sg.mae_sigma.clip(upper=3), bins, density=True, color=PALETTE[7], alpha=0.35,
                    label=f"triggers: worst move (median {sg.mae_sigma.median():.2f})")
            ax.hist(rg.mfe_sigma.clip(upper=3), bins, density=True, histtype="step", color=LIGHT["ink"],
                    linewidth=1.1, label=f"random: best move (median {rg.mfe_sigma.median():.2f})")
            for j, k in enumerate(K):
                share_up, share_down = (sg.mfe_sigma >= k).mean() * 100, (sg.mae_sigma >= k).mean() * 100
                ax.axvline(k, color=LIGHT["muted"], linestyle="--", linewidth=0.9)
                ax.text(k + 0.03, ax.get_ylim()[1] * (0.93 - 0.11 * j), f"k={k}: +{share_up:.0f}% / -{share_down:.0f}%",
                        fontsize=7, backgroundcolor="white")
            ax.set_title(f"{instrument.replace('_crossed', '')}, {sess}: median sigma_1h "
                         f"{sg.sigma_1h.median():.1f} ticks", fontsize=10)
            ax.legend(fontsize=7, loc="center right")
    fig.supxlabel("move within 60 min, in sigma_1h units (beans: 1-min high / low, spread: 1-min closes; clipped at 3)",
                  fontsize=9)
    fig.suptitle("Where to put the barriers: best and worst move after each trigger, in volatility units "
                 "(+x% / -y% = share of triggers reaching +k / -k sigma)", fontsize=11)
    fig.tight_layout()
    fig.savefig(STRATEGY_DIR / "output" / "labels_excursions.png", dpi=110)

    # plot 2: how trades end and what they make, per barrier choice
    configs = [(k, m) for k in K for m in STOP_MULT]
    names = [f"k {k}\nstop {m}x" for k, m in configs]
    x = np.arange(len(configs))
    instruments = ["beans", "spread_crossed", "spread_beans_passive"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True)
    for col, instrument in enumerate(instruments):
        sg = taken[(taken.instrument == instrument) & (taken.kind == "signal")]
        bottom = np.zeros(len(configs))
        for reason, colour in [("target", PALETTE[5]), ("stop", PALETTE[7]), ("time", LIGHT["muted"])]:
            share = [(sg[(sg.k == k) & (sg.stop_mult == m)].exit_reason == reason).mean() * 100 for k, m in configs]
            axes[0, col].bar(x, share, bottom=bottom, color=colour, label=reason)
            bottom += np.nan_to_num(share)
        win = [sg[(sg.k == k) & (sg.stop_mult == m)].label.mean() * 100 for k, m in configs]
        axes[0, col].plot(x, win, "o", color=LIGHT["ink"], label="win% after costs")
        for xi, (k, m) in zip(x, configs):
            axes[0, col].text(xi, 2, f"n {len(sg[(sg.k == k) & (sg.stop_mult == m)])}", ha="center", fontsize=7,
                              color="white")
        skipped = [labels[(labels.instrument == instrument) & (labels.kind == "signal") & (labels.k == k)
                          & (labels.stop_mult == m)].skipped.mean() * 100 for k, m in configs]
        axes[0, col].set_title(f"{instrument}: how trades end (skipped: {min(skipped):.0f}-{max(skipped):.0f}%)",
                               fontsize=10)
        axes[0, col].set_ylim(0, 100)
        width = 0.27
        for j, (label, kind, hf, colour) in enumerate([("triggers 2021-22", "signal", "2021-22", PALETTE[0]),
                                                       ("triggers 2023-24", "signal", "2023-Nov24", PALETTE[3]),
                                                       ("random", "random", None, LIGHT["muted"])]):
            g = taken[(taken.instrument == instrument) & (taken.kind == kind)]
            if hf:
                g = g[g.half == hf]
            net = [g[(g.k == k) & (g.stop_mult == m)].net_ticks.mean() for k, m in configs]
            axes[1, col].bar(x + (j - 1) * width, net, width, color=colour, label=label)
        axes[1, col].axhline(0, color=LIGHT["ink"], linewidth=0.8)
        axes[1, col].set_title(f"{instrument}: mean net ticks per trade", fontsize=10)
        axes[1, col].set_xticks(x, names, fontsize=8)
    axes[0, 0].set_ylabel("% of trades"); axes[1, 0].set_ylabel("ZS ticks after costs")
    axes[0, 0].legend(fontsize=7, loc="upper right"); axes[1, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Triple barrier, every trigger (no meta-model yet): target = k x sigma_1h, time limit "
                 f"{HOLD_BARS * 5} min, net of costs", fontsize=11)
    fig.tight_layout()
    fig.savefig(STRATEGY_DIR / "output" / "labels_outcomes.png", dpi=110)
    print("saved labels.parquet, labels_summary*.csv, labels_excursions.png, labels_outcomes.png")

    # ---- confirmation entry vs entering at the trigger: Europe and US, every trigger, no model ----
    sig_rows = all_rules[(all_rules.kind == "signal") & ~all_rules.skipped & all_rules.session.isin(["europe", "us"])
                         & all_rules.instrument.isin(["beans", "spread_beans_passive"]) & (all_rules.k == 1.0)]
    weeks = (all_rules.time.max() - all_rules.time.min()).days / 7
    n_triggers = all_rules[(all_rules.kind == "signal") & (all_rules.entry_rule == "trigger")].groupby(
        ["session"]).bar.nunique()
    compare = []
    for (rule, instrument, sess, stop_mult, hf), g in sig_rows.groupby(["entry_rule", "instrument", "session",
                                                                        "stop_mult", "half"]):
        wins, losses = g[g.label == 1].net_ticks, g[g.label == 0].net_ticks
        compare.append({"entry": rule, "instrument": instrument, "session": sess, "stop x target": stop_mult,
                        "half": hf, "trades": len(g), "win%": g.label.mean() * 100, "mean win": wins.mean(),
                        "mean loss": losses.mean(), "net": g.net_ticks.mean(), "target ticks": g.target_ticks.median(),
                        "best move ticks": (g.mfe_sigma * g.sigma_1h).median(),
                        "worst move ticks": (g.mae_sigma * g.sigma_1h).median(), "wait min": g.wait_bars.mean() * 5})
    compare = pd.DataFrame(compare)
    compare.to_csv(STRATEGY_DIR / "output" / "labels_entry_compare.csv", index=False)
    confirmed = all_rules[all_rules.kind == "signal"].groupby(["entry_rule", "session"]).bar.nunique().unstack()
    print("\nshare of triggers that get a confirmed entry within 30 min:\n",
          (confirmed / n_triggers * 100).round(0).to_string())
    print("\nentry rules, k 1.0 (target 1 sigma_1h), net ticks per trade by half (left 2021-22, right 2023-24):")
    view = compare.pivot_table(index=["instrument", "session", "stop x target", "entry"], columns="half",
                               values=["net", "win%"]).round(2)
    print(view.to_string())
    excursion = compare[compare["stop x target"] == 1.0].groupby(["instrument", "session", "entry"])[
        ["best move ticks", "worst move ticks", "mean win", "mean loss", "wait min"]].mean().round(1)
    print("\nbest / worst move within the hour from the entry (medians, ticks), mean win / loss at a symmetric stop:\n",
          excursion.to_string())

    # plot 3: net per trade by entry rule and stop, Europe and US, both halves pooled
    pooled = sig_rows.groupby(["instrument", "session", "stop_mult", "entry_rule"]).net_ticks.mean()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey="row")
    width = 0.27
    for row, instrument in enumerate(["beans", "spread_beans_passive"]):
        for col, sess in enumerate(["europe", "us"]):
            ax = axes[row, col]
            for j, rule in enumerate(ENTRY_RULES):
                net = [pooled.get((instrument, sess, m, rule), np.nan) for m in STOP_MULT]
                ax.bar(np.arange(len(STOP_MULT)) + (j - 1) * width, net, width, color=PALETTE[[0, 3, 5][j]],
                       label=rule)
            ax.axhline(0, color=LIGHT["ink"], linewidth=0.8)
            ax.set_xticks(np.arange(len(STOP_MULT)), [f"stop {m}x" for m in STOP_MULT], fontsize=8)
            ax.set_title(f"{instrument}, {sess}", fontsize=10)
    axes[0, 0].set_ylabel("net ticks per trade"); axes[1, 0].set_ylabel("net ticks per trade")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Entering at the trigger vs waiting for confirmation (target 1 sigma_1h, 60 min, net of costs, "
                 "every trigger, 2021 - Nov 2024)", fontsize=11)
    fig.tight_layout()
    fig.savefig(STRATEGY_DIR / "output" / "labels_entry_compare.png", dpi=110)
    print("saved labels_entry_compare.csv / .png")
