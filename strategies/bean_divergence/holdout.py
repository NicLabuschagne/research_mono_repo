"""
holdout.py
----------
ONE-TIME holdout for the volume-bar Europe spread: 1 Dec 2024 to the end of the minute file (16 Dec 2025).
Pre-registered 2026-11-03, before any holdout result was looked at:

    setup  : volume_bars.py as committed - bars of 458 ZS contracts, z half-life 192 bars, trigger 1.645 / re-arm
             0.6745, clock-time labels (next-minute entry, 60 min, target 1 sigma_1h, stop 1.25x), spread with
             beans passive (cost ~1.1 ticks)
    model  : the 17-feature L2 logistic regression fitted ONCE on all development triggers (C chosen on the last
             quarter of development, as in the walk-forward), then frozen
    take   : Europe spread triggers with p >= the median p of the DEVELOPMENT Europe spread triggers under that
             frozen model ("the model's top half", as a fixed cut-off)
    judge  : net ticks per trade on the taken trades. PASS if > 0; strong if t >= 1.65. Every Europe trigger, the
             other sessions and beans outright are reported, not judged.

The first run writes output/holdout_LOCK; a second run refuses. The Kalman filter, z and features run through the
whole history (causal), so the holdout bars are warmed up by development data exactly as live trading would be.

    python -m strategies.bean_divergence.holdout
"""
import numpy as np
import pandas as pd

import strategies.bean_divergence.meta_model as mm
import strategies.bean_divergence.volume_bars as vbs
from core_quant.config import ROOT
from strategies.bean_divergence.fair_value import STRATEGY_DIR, crush_hedge, load
from strategies.bean_divergence.labels import minute_data, sigma_1h
from strategies.crush_strategy.data_ingest import build_bars

STOP = 1.25
LOCK = STRATEGY_DIR / "output" / "holdout_LOCK"


def take_stats(trades: pd.DataFrame, weeks: float) -> dict:
    """Count, trades a week, net per trade with its t, win share, mean win and mean loss (ticks after costs)."""
    net = trades.net_ticks
    return {"trades": len(trades), "per week": len(trades) / weeks, "net per trade": net.mean(),
            "t": net.mean() / net.sem() if len(trades) > 2 else np.nan, "win%": (net > 0).mean() * 100,
            "mean win": net[net > 0].mean(), "mean loss": net[net <= 0].mean(), "total ticks": net.sum()}


if __name__ == "__main__":
    if LOCK.exists():
        raise SystemExit(f"holdout already run ({LOCK.read_text().strip()}); it is a one-time test")
    cfg = load()
    holdout_start = pd.Timestamp(cfg.split.holdout_start, tz="UTC")
    everything = pd.Timestamp("2100-01-01", tz="UTC")

    minutes = minute_data(cfg)
    time_bars = build_bars(ROOT / cfg.data_path, cfg.research.bar_minutes)          # 5-min bars incl. the holdout
    time_sigma = sigma_1h(time_bars, crush_hedge(time_bars, live=True), cfg.research.z_half_life_bars)
    bars, hedge, gap, z, triggers, _ = vbs.signal_pipeline(minutes, pd.DataFrame(), cfg, everything,
                                                           bars_before="2100-01-01")
    labels = vbs.label_triggers(bars, hedge, triggers, minutes, time_bars, time_sigma, cfg)
    features = mm.build_features(bars, hedge, gap, z, triggers, minutes[vbs.PRICE_COLUMNS], labels, cfg)
    table = mm.label_table(labels, features, STOP)
    development, holdout = table[table.time < holdout_start], table[table.time >= holdout_start]
    weeks = (holdout.time.max() - holdout_start).days / 7
    print(f"holdout {holdout_start:%Y-%m-%d} to {holdout.time.max():%Y-%m-%d} ({weeks:.0f} weeks); "
          f"{holdout.bar.nunique()} triggers")

    rows = []
    for instrument in mm.INSTRUMENTS:
        train = development[development.instrument == instrument]
        model, prep = mm.fit_logistic(train, mm.choose_c(train))
        cutoff = np.median(model.predict_proba(prep(train[train.session == "europe"]))[:, 1])
        test = holdout[holdout.instrument == instrument].copy()
        test["p"] = model.predict_proba(prep(test))[:, 1]
        for session in ["europe", "asia", "us"]:
            s = test[test.session == session]
            rows.append({"instrument": instrument, "session": session, "rule": "every trigger"} | take_stats(s, weeks))
            rows.append({"instrument": instrument, "session": session, "rule": f"p >= {cutoff:.3f}"}
                        | take_stats(s[s.p >= cutoff], weeks))
    result = pd.DataFrame(rows)
    result.to_csv(STRATEGY_DIR / "output" / "holdout_result.csv", index=False)
    LOCK.write_text(f"run {pd.Timestamp.now():%Y-%m-%d %H:%M}")

    pd.set_option("display.width", 220)
    print("\nnet ZS ticks per trade after costs, holdout only:\n", result.round(2).to_string(index=False))
    judged = result[(result.instrument == "spread_beans_passive") & (result.session == "europe")
                    & result.rule.str.startswith("p >=")].iloc[0]
    verdict = "PASS (strong)" if judged["t"] >= 1.65 else "PASS" if judged["net per trade"] > 0 else "FAIL"
    print(f"\nPRE-REGISTERED VERDICT: Europe spread, model's top half: {judged['net per trade']:+.2f} ticks a trade, "
          f"t {judged['t']:.2f}, {judged['trades']} trades -> {verdict}")
