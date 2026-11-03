"""
volume_bars.py
--------------
The same signal on VOLUME bars instead of 5-min time bars (kept 2026-11-03).

    bars    : a new bar every `volume_bar_contracts` ZS contracts (458), restarting at every session open. A bar is
              stamped with the minute that completes it, so it is known at that minute's close. 458 = the mean ZS
              volume of a 5-min bar in Jul - Dec 2020, set BEFORE the development period (the all-development mean,
              379, was a look-ahead and flattered the result). Busy hours get more bars, quiet hours fewer.
    gap / z : live Kalman crush gap with the same drift per BAR as on 5-min bars; z half-life
              `volume_bar_z_half_life` bars (192: a plateau 192 - 768, the most balanced across halves; 48, the
              5-min value, left 2023-24 weak); one trigger per move as triggers.py
    labels  : in CLOCK time, as labels.py - entry at the next minute's open, 60 min, target 1 sigma_1h (taken from
              the 5-min bars, past only), stop 1.25x (main) and 1x; beans 1 tick, spread with beans passive
    model   : meta_model.py's 17 features and walk-forward logistic regression, unchanged
    check   : the whole pipeline is rebuilt from data cut at CHECK_CUT; every bar, hedge, gap, z, trigger and
              feature known before the cut must be identical - anything that changes was using the future

    python -m strategies.bean_divergence.volume_bars
"""
import numpy as np
import pandas as pd

import strategies.bean_divergence.meta_model as mm
from strategies.bean_divergence.fair_value import (LEGS, STRATEGY_DIR, TICK_CENTS, BeanConfig, adaptive_scale,
                                                   crush_gap, half_label, load, session_label)
from strategies.bean_divergence.labels import (average_uniqueness, first_touch, hedge_cost_ticks, minute_data,
                                               sigma_1h)
from strategies.bean_divergence.triggers import build, find_triggers
from strategies.crush_strategy.data_ingest import add_trading_date
from strategies.crush_strategy.kalman_filter import run_kalman_filter

HOLD_MINUTES = 60
STOP_MULT = [1.25, 1.0]
RANDOM_PER_TRIGGER = 2
CHECK_CUT = pd.Timestamp("2023-03-01 12:00", tz="UTC")
PRICE_COLUMNS = [f"{leg}_adj_{f}" for leg in LEGS for f in ["open", "high", "low", "close"]]


def make_volume_bars(minutes: pd.DataFrame, contracts: float, before) -> pd.DataFrame:
    """Session-aligned volume bars: a minute belongs to the bucket of the ZS volume traded BEFORE it, so a bar ends
    with the minute that fills it. bar_number stays in clock time (minutes since the session open // 5)."""
    m = minutes.copy()
    cumulative = m.groupby("session_id")["ZS_volume"].cumsum()
    m["vbar"] = ((cumulative - m["ZS_volume"]) // contracts).astype(int)
    m["bar_end"] = m.index
    rules = {"bar_end": "last", "session_type": "first"}
    for leg in LEGS:
        rules.update({f"{leg}_volume": "sum", f"{leg}_ret": "sum", f"{leg}_adj_close": "last", f"{leg}_adj_open": "first"})
    bars = add_trading_date(m.groupby(["session_id", "vbar"]).agg(rules).reset_index().set_index("bar_end"))
    session_start = pd.DatetimeIndex(m.groupby("session_id")["bar_end"].first().reindex(bars["session_id"]))
    bars["bar_number"] = ((bars.index - session_start) / pd.Timedelta(minutes=5)).astype(int)
    return bars[bars["trading_date"] < pd.Timestamp(before)]


def signal_pipeline(minutes: pd.DataFrame, labels: pd.DataFrame, cfg: BeanConfig, before: pd.Timestamp,
                    bars_before=None):
    """Volume bars (trading dates before `bars_before`, default the holdout start) -> live Kalman gap -> z ->
    triggers from the development start up to `before` -> features."""
    r = cfg.research
    bars = make_volume_bars(minutes, r.volume_bar_contracts, bars_before or cfg.split.holdout_start)
    hedge = run_kalman_filter(bars, 5).shift(1)              # same drift per bar as the 5-min bars
    gap = crush_gap(bars, hedge)
    z = gap / adaptive_scale(gap, r.volume_bar_z_half_life)
    session = session_label(bars.index, bars["session_type"])
    dev = (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values
    rows = []
    for s, side in find_triggers(z.values, bars["session_id"].values, r.trigger_z, r.rearm_z):
        if dev[s] and bars.index[s] < before:
            rows.append({"bar": s, "side": side, "session": session[s]})
    triggers = pd.DataFrame(rows)
    features = mm.build_features(bars, hedge, gap, z, triggers, minutes[PRICE_COLUMNS], labels, cfg) \
        if len(labels) else None
    return bars, hedge, gap, z, triggers, features


def label_triggers(bars, hedge, triggers, minutes, time_bars, time_sigma, cfg) -> pd.DataFrame:
    """Triple-barrier labels in clock time for every trigger, plus random bars (same session and half), in the
    column layout of labels.py so meta_model.label_table can read them."""
    r = cfg.research
    n = len(bars)
    a, bm, bo = (hedge[c].values for c in ["alpha", "beta_meal", "beta_oil"])
    ticks = (bars["ZS_adj_close"] / TICK_CENTS).values
    session, half = session_label(bars.index, bars["session_type"]), half_label(bars["trading_date"])
    # sigma_1h of the last 5-min bar already closed at each volume bar
    last_time_bar = np.searchsorted(time_bars.index.values, bars.index.values, side="right") - 1
    sigma = {k: np.where(last_time_bar >= 0, v[np.maximum(last_time_bar, 0)], np.nan) for k, v in time_sigma.items()}
    cost = {"beans": np.full(n, r.cost_ticks), "spread_beans_passive": hedge_cost_ticks(bars, hedge, r.cost_ticks)}
    m_time = minutes.index
    lo = {leg: np.log(minutes[f"{leg}_adj_open"].values) for leg in LEGS}
    lc = {leg: np.log(minutes[f"{leg}_adj_close"].values) for leg in LEGS}
    l_hi, l_lo = np.log(minutes["ZS_adj_high"].values), np.log(minutes["ZS_adj_low"].values)
    minute_session = minutes["session_id"].values
    pos = m_time.searchsorted(bars.index)
    usable = ~np.isnan(sigma["beans"]) & ~np.isnan(sigma["spread"]) & (pos + HOLD_MINUTES + 1 < len(m_time))

    events = [("signal", t.bar, t.side) for t in triggers.itertuples() if usable[t.bar]]
    rng = np.random.default_rng(42)
    for _, s, _ in list(events):
        pool = np.flatnonzero((session == session[s]) & (half == half[s]) & usable
                              & (bars["trading_date"] >= pd.Timestamp(cfg.split.development_start)).values)
        for random_bar in rng.choice(pool, RANDOM_PER_TRIGGER):
            events.append(("random", int(random_bar), int(rng.choice([-1, 1]))))

    rows = []
    for kind, s, side in events:
        p0 = pos[s] + 1                                       # entry: the open of the next minute
        window = np.arange(p0, p0 + HOLD_MINUTES)
        favour = lambda log: -side * (log[window] - lo["ZS"][p0]) * ticks[s]
        best, worst = (favour(l_lo), favour(l_hi)) if side > 0 else (favour(l_hi), favour(l_lo))
        held = lambda log, i: log["ZS"][i] - (a[s] + bm[s] * log["ZM"][i] + bo[s] * log["ZL"][i])
        spread = -side * (held(lc, window) - held(lo, p0)) * ticks[s]
        paths = {"beans": (favour(lc["ZS"]), best, worst), "spread_beans_passive": (spread, spread, spread)}
        for instrument, (path, b, w) in paths.items():
            target = sigma["spread" if "spread" in instrument else "beans"][s]
            for stop_mult in STOP_MULT:
                i, reason = first_touch(b, w, target, stop_mult * target)
                gross = {"target": target, "stop": -stop_mult * target, "time": path[i]}[reason]
                net = gross - cost[instrument][s]
                rows.append({"kind": kind, "entry_rule": "trigger", "instrument": instrument, "k": 1.0,
                             "stop_mult": stop_mult, "skipped": False, "bar": s, "side": side, "time": bars.index[s],
                             "exit_time": m_time[window[i]], "session": session[s], "half": half[s],
                             "exit_reason": reason, "net_ticks": net, "label": int(net > 0),
                             "crosses_break": minute_session[window[-1]] != bars["session_id"].values[s]})
    labels = pd.DataFrame(rows)
    # exits after the last bar (at the holdout cut) count as ending on it
    labels["exit_bar"] = np.minimum(bars.index.searchsorted(labels["exit_time"]), n - 1)
    labels["weight"] = np.nan
    for _, g in labels[labels.kind == "signal"].groupby(["instrument", "stop_mult"]):
        labels.loc[g.index, "weight"] = average_uniqueness(g["bar"].values + 1, g["exit_bar"].values, n)
    return labels


def summarise(predictions: pd.DataFrame, labels: pd.DataFrame, stop_mult: float, weeks: float) -> list[dict]:
    """Per instrument and session: every trigger and random entries (all development), the model's top two thirds
    and top half by p (out of sample), with the 2022 / 2023-24 split."""
    rows = []
    for (instrument, session), g in predictions.groupby(["instrument", "session"]):
        lab = labels[(labels.instrument == instrument) & (labels.session == session) & (labels.stop_mult == stop_mult)]
        row = {"stop": stop_mult, "instrument": instrument, "session": session,
               "every trigger (2021-24)": lab[lab.kind == "signal"].net_ticks.mean(),
               "random (2021-24)": lab[lab.kind == "random"].net_ticks.mean(),
               "every trigger (2022 on)": g.net_ticks.mean(), "AUC": mm.auc(g.p.values, g.label.values)}
        for share, name in [(2 / 3, "top 2/3"), (1 / 2, "top 1/2")]:
            top = g[g.p >= g.p.quantile(1 - share)]
            row[name] = top.net_ticks.mean()
            row[f"{name} t"] = top.net_ticks.mean() / top.net_ticks.sem()
            row[f"{name} per week"] = len(top) / weeks
            row[f"{name} 2022"] = top[top.time.dt.year == 2022].net_ticks.mean()
            row[f"{name} 2023-24"] = top[top.time.dt.year >= 2023].net_ticks.mean()
        rows.append(row)
    return rows


if __name__ == "__main__":
    # self-check on made-up minutes: buckets of 10 contracts, restarting at a new session
    toy = pd.DataFrame({"session_id": [1, 1, 1, 1, 2, 2], "ZS_volume": [4, 4, 4, 9, 6, 6]})
    toy_bucket = ((toy.groupby("session_id")["ZS_volume"].cumsum() - toy["ZS_volume"]) // 10).tolist()
    assert toy_bucket == [0, 0, 0, 1, 0, 0]        # the minute that fills a bar ends it; the next starts a new one

    cfg = load()
    minutes = minute_data(cfg)
    time_bars, time_hedge, _, _, _ = build(cfg)
    time_sigma = sigma_1h(time_bars, time_hedge, cfg.research.z_half_life_bars)
    far_future = pd.Timestamp("2100-01-01", tz="UTC")
    bars, hedge, gap, z, triggers, _ = signal_pipeline(minutes, pd.DataFrame(), cfg, far_future)
    weeks = bars.loc[bars["trading_date"] >= pd.Timestamp(cfg.split.development_start), "trading_date"].nunique() / 5
    print(f"{len(bars):,} volume bars of {cfg.research.volume_bar_contracts:g} contracts; triggers a week:",
          triggers.session.value_counts().div(weeks).round(1).to_dict())

    labels = label_triggers(bars, hedge, triggers, minutes, time_bars, time_sigma, cfg)
    features = mm.build_features(bars, hedge, gap, z, triggers, minutes[PRICE_COLUMNS], labels, cfg)
    labels.to_parquet(STRATEGY_DIR / "output" / "volume_bars_labels.parquet")

    # leakage check: rebuild from minute data cut at CHECK_CUT (labels cut to exits before it)
    safe = CHECK_CUT - pd.Timedelta(hours=1)
    full = signal_pipeline(minutes, labels, cfg, safe)
    cut = signal_pipeline(minutes[minutes.index <= CHECK_CUT], labels[labels.exit_time <= CHECK_CUT], cfg, safe)
    known = (cut[0].index < safe).sum()
    pd.testing.assert_frame_equal(full[0].iloc[:known], cut[0].iloc[:known])
    pd.testing.assert_frame_equal(full[1].iloc[:known], cut[1].iloc[:known])
    pd.testing.assert_series_equal(full[3].iloc[:known], cut[3].iloc[:known])
    pd.testing.assert_frame_equal(full[4].reset_index(drop=True), cut[4].reset_index(drop=True))
    pd.testing.assert_frame_equal(full[5].set_index("bar").drop(columns="crosses_break"),
                                  cut[5].set_index("bar").drop(columns="crosses_break"), check_exact=False)
    print(f"leakage check passed: {known:,} bars and {len(cut[4]):,} triggers before {CHECK_CUT:%Y-%m-%d} unchanged")

    weeks_oos = (bars.index[-1] - mm.WALK_FORWARD_STARTS[0]).days / 7
    rows = []
    for stop_mult in STOP_MULT:
        predictions, _ = mm.walk_forward(mm.label_table(labels, features, stop_mult))
        predictions.to_parquet(STRATEGY_DIR / "output" / f"volume_bars_predictions_{stop_mult:g}.parquet")
        rows += summarise(predictions, labels, stop_mult, weeks_oos)
    summary = pd.DataFrame(rows)
    summary.to_csv(STRATEGY_DIR / "output" / "volume_bars_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nnet ticks per trade after costs (model columns: out of sample, walk-forward from 2022):\n",
          summary.round(2).to_string(index=False))
