"""
layer1_mispricing_capture.py
----------------------------
How big is the mispricing when a signal fires, and how much of it comes back?

For every A / B / C trigger (layer1_signal_days_events.csv, prior-close crush gap):
    mispricing   : |gap| at the trigger bar, ZS ticks (and cents; 4 ticks = 1 c)
    band         : the level that was crossed: the 20-day 95th / 5th percentile (A) or 1.645 x the 4-hour
                   adaptive size (B, C)
    capture      : from the next bar's open, inside the session (layer1_extremes.follow_through):
                   spread move at 1 h / 4 h and the best spread move within 2 h, in ticks and as a share of the
                   mispricing; outright beans at 1 h / 4 h
Split by mispricing size (terciles within each signal). Development only; no costs.

    python -m strategies.bean_divergence.layer1_mispricing_capture
"""
import numpy as np
import pandas as pd

from core_quant.config import ROOT
from strategies.bean_divergence.failure_study import session_label
from strategies.bean_divergence.feature_study import STRATEGY_DIR, TICK_CENTS
from strategies.bean_divergence.layer1_adaptive_extremes import adaptive_scale
from strategies.bean_divergence.layer1_extremes import follow_through, past_percentiles
from strategies.bean_divergence.layer1_signal_days import HALF_LIFE_4H, Z_95
from strategies.crush_strategy.data_ingest import build_bars
from strategies.crush_strategy.kalman_filter import freeze_at_previous_session_close, run_kalman_filter

RANDOM_PER_SIGNAL = 5

if __name__ == "__main__":
    bars = build_bars(ROOT / "data/raw/latest_df_backup.parquet", 5)
    bars = bars[bars["trading_date"] < pd.Timestamp("2024-12-01")]          # the same bars as the event file
    states = run_kalman_filter(bars, 5)
    hedge = freeze_at_previous_session_close(bars, states)
    a, bm, bo = (hedge[c].values for c in ["alpha", "beta_meal", "beta_oil"])
    log_c = {leg: np.log(bars[f"{leg}_adj_close"]).values for leg in ["ZS", "ZM", "ZL"]}
    log_o = {leg: np.log(bars[f"{leg}_adj_open"]).values for leg in ["ZS", "ZM", "ZL"]}
    ticks = (bars["ZS_adj_close"] / TICK_CENTS).values
    gap = pd.Series((log_c["ZS"] - (a + bm * log_c["ZM"] + bo * log_c["ZL"])) * ticks, index=bars.index)
    session = pd.Series(session_label(bars.index, bars["session_type"]), index=bars.index)
    pct = past_percentiles(gap, session, bars["trading_date"], [0.05, 0.95])
    band_adaptive = Z_95 * adaptive_scale(gap, HALF_LIFE_4H)

    events = pd.read_csv(STRATEGY_DIR / "output" / "layer1_signal_days_events.csv")
    assert np.allclose(events["gap_ticks"].values, gap.values[events["bar"].values])     # same gap as when detected
    rows = []
    for ev in events.itertuples():
        measured, _, _ = follow_through(ev.bar, ev.side, (a, bm, bo), log_o, log_c, ticks, bars["session_id"].values)
        if ev.signal == "A":
            band = pct[0.95].iloc[ev.bar] if ev.side > 0 else -pct[0.05].iloc[ev.bar]
        else:
            band = band_adaptive.iloc[ev.bar]
        rows.append({"signal": ev.signal, "half": "2021-22" if pd.Timestamp(ev.trading_date) < pd.Timestamp("2023-01-01") else "2023-Nov24",
                     "mispricing": abs(ev.gap_ticks), "band": band, "distinct": ev.distinct, "time": ev.time} | measured)
    t = pd.DataFrame(rows)
    t["size"] = t.groupby("signal")["mispricing"].transform(lambda m: pd.qcut(m, 3, labels=["small", "medium", "large"]))

    # baseline: the best move within 2 h is positive even for a coin flip, so measure it at random bars too
    # (same session and half as each signal, random side, next bar in the same session)
    session_id = bars["session_id"].values
    half_of_bar = np.where(bars["trading_date"] < pd.Timestamp("2023-01-01"), "2021-22", "2023-Nov24")
    same_next = np.append(session_id[1:] == session_id[:-1], False)
    in_dev = (bars["trading_date"] >= pd.Timestamp("2021-01-01")).values
    rng = np.random.default_rng(42)
    random_rows = []
    for ev, hf in zip(events.itertuples(), t["half"]):
        pool = np.flatnonzero((session.values == ev.session) & (half_of_bar == hf) & same_next & in_dev)
        for s in rng.choice(pool, RANDOM_PER_SIGNAL):
            measured, _, _ = follow_through(int(s), int(rng.choice([-1, 1])), (a, bm, bo), log_o, log_c, ticks, session_id)
            random_rows.append({"signal": ev.signal, "half": hf, "mispricing": abs(gap.iloc[s])} | measured)
    r = pd.DataFrame(random_rows)

    def summary(g: pd.DataFrame) -> pd.Series:
        return pd.Series({"n": len(g), "mispricing_median": g["mispricing"].median(),
                          "mispricing_cents": g["mispricing"].median() / 4, "mispricing_p90": g["mispricing"].quantile(0.9),
                          "band_median": g["band"].median(),
                          "spread_1h": g["spread_1h"].mean(), "spread_4h": g["spread_4h"].mean(),
                          "best_2h_median": g["favourable_2h"].median(),
                          "pct_back_1h": g["spread_1h"].mean() / g["mispricing"].mean() * 100,
                          "pct_back_best_2h": g["favourable_2h"].median() / g["mispricing"].median() * 100,
                          "outright_1h": g["outright_1h"].mean(), "outright_4h": g["outright_4h"].mean(),
                          "spread_1h_se": g["spread_1h"].sem()})

    pd.set_option("display.width", 250)
    by_signal = t.groupby(["signal", "half"]).apply(summary)
    by_signal["random_best_2h_median"] = r.groupby(["signal", "half"])["favourable_2h"].median()
    by_signal["random_spread_1h"] = r.groupby(["signal", "half"])["spread_1h"].mean()
    by_signal["random_gap_median"] = r.groupby(["signal", "half"])["mispricing"].median()
    by_size = t.groupby(["signal", "size"], observed=True).apply(summary)
    by_signal.to_csv(STRATEGY_DIR / "output" / "layer1_mispricing_by_signal.csv")
    by_size.to_csv(STRATEGY_DIR / "output" / "layer1_mispricing_by_size.csv")
    print("by signal and half (ticks; 4 ticks = 1 c):\n", by_signal.round(1).to_string())
    print("\nby size of the mispricing, both halves (ticks):\n", by_size.round(1).to_string())

    # large Europe triggers, A and B combined: the same dislocation flagged by both counts once (distinct flag)
    weeks = bars[bars["trading_date"] >= pd.Timestamp("2021-01-01")].groupby(half_of_bar[in_dev])["trading_date"].nunique() / 5
    europe = t[t["signal"].isin(["A", "B"]) & t["distinct"]]
    large = []
    for threshold in [0, 15, 20, 25]:
        for hf in ["2021-22", "2023-Nov24"]:
            g = europe[(europe["mispricing"] >= threshold) & (europe["half"] == hf)]
            row = {"gap_at_least": threshold, "half": hf, "n": len(g), "per_week": len(g) / weeks[hf]}
            for kind in ["spread", "outright"]:
                for h in ["30m", "1h", "2h", "4h"]:
                    x = g[f"{kind}_{h}"].dropna()
                    row[f"{kind}_{h}"] = x.mean()
                    row[f"{kind}_{h}_t"] = x.mean() / x.sem()          # signal to noise: mean / its standard error
                    row[f"{kind}_{h}_n"] = len(x)                      # fewer at long horizons: never past the session close
            large.append(row)
    large = pd.DataFrame(large)
    large.to_csv(STRATEGY_DIR / "output" / "layer1_mispricing_large_europe.csv", index=False)
    for kind in ["spread", "outright"]:
        cols = ["gap_at_least", "half", "n", "per_week"] + [f"{kind}_{h}{s}" for h in ["30m", "1h", "2h", "4h"] for s in ["", "_t", "_n"]]
        print(f"\nEurope A + B distinct, {kind} (ticks, t = mean / se, n = triggers with that horizon inside the session):\n",
              large[cols].round(1).to_string(index=False))
    hours = pd.to_datetime(europe["time"], utc=True).dt.tz_convert("America/Chicago").dt.hour
    print("\nEurope trigger hour (CT):", hours.value_counts().sort_index().to_dict())
