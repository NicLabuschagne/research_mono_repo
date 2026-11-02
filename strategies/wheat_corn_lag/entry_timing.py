"""
entry_timing.py
---------------
Layer 2, entry timing, step 1 (diagnosis): which facts known WHEN a trigger fires separate the trades that go
straight to profit from the ones that first run against us? The trades are the base rule's (kalman_rule.py config:
20% tails, re-arm every 4.5 h, at most 2 open, half-gap target, 2 x gap safety stop). Development data, IN SAMPLE,
nothing is chosen here. Saved to runs/<time>_entry_timing/.

    python -m strategies.wheat_corn_lag.entry_timing

At the trigger bar t (all known at its close):
  widen_1h, widen_4h     how far s moved further INTO the tail over the last 1 h / 4 h (> 0 = gap still widening)
  depth                  how far s is beyond the tail threshold
  hours_in_tail          how long s has already been beyond the threshold on this side
  wheat_push_1h_c        wheat's move over the last hour in the direction that widens the gap, cents
                         (> 0 = we are buying while wheat is still falling / selling while it is still rising)
  gap_c, atr_c           the Kalman gap and wheat's typical 1-session move, cents
  leg                    over the last 4 h, did wheat or corn (x beta) move more in the gap-widening direction?
  time of day            entry time, Chicago
Outcomes: net per trade under the rule; worst point against us (MAE) and best point for us (MFE) within 1 session.

A  each feature in fifths (quintiles, pooled cut points). Rule fixed before the run: a feature MATTERS when the top
   and bottom fifth differ by >= 1.5 c net per trade AND the difference has the same sign in >= 4 of 6 years.
   Step 2 builds entry rules only from features that matter.
B  limit entry trade-off (measured on the plain 1-session hold, the simplest yardstick): instead of buying at the
   open after the trigger, rest a limit d cents better for up to 4 hours. Fills only if price trades 1 tick
   THROUGH the level. Filled trades gain d; trades that never come back are missed (and they are the ones that ran
   our way). Per trigger: net cents with a limit vs at market.

Units: cents (1 tick = 0.25 c). net per trade = all trades after costs.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from core_quant.config import save_run
from core_quant.theme import set_theme
from strategies.wheat_corn_lag.backtest import TICK_CENTS
from strategies.wheat_corn_lag.features import build_features
from strategies.wheat_corn_lag.harvest import time_of_day
from strategies.wheat_corn_lag.kalman_rule import book, kalman_signal, trade_paths
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load, load_bars

SESSION = 71
NUMERIC = ["widen_1h", "widen_4h", "depth", "hours_in_tail", "wheat_push_1h_c", "gap_c", "atr_c"]
MATTERS_CENTS, MATTERS_YEARS = 1.5, 4
LIMIT_OFFSETS = [1, 2, 3, 4, 6]
LIMIT_WINDOW = 16                                   # bars a limit entry may rest (4 hours)


def run_length(flags: np.ndarray) -> np.ndarray:
    """How many bars in a row (up to and including t) the flag has been true."""
    run = np.zeros(len(flags), dtype=int)
    for t in range(len(flags)):
        if flags[t]:
            run[t] = run[t - 1] + 1 if t > 0 else 1
    return run


def describe_triggers(bars, features, s: pd.Series, trades: pd.DataFrame, rule) -> pd.DataFrame:
    t = trades["entry_bar"].values - 1
    side = trades["side"].values
    sv = s.values
    rolling = s.rolling(rule.quantile_window_bars, min_periods=rule.quantile_min_bars)
    upper = rolling.quantile(1 - rule.tail).shift(1).values
    lower = rolling.quantile(rule.tail).shift(1).values
    run_up, run_down = run_length(sv >= upper), run_length(sv <= lower)
    index_w = bars["ret_w"].fillna(0).cumsum().values           # roll-free: a move across a possible roll counts 0
    index_c = bars["ret_c"].fillna(0).cumsum().values
    price = bars["W"].values[t]
    beta = features[f"kbeta_{rule.memory_sessions}s"].values[t]
    wheat_part = -side * (index_w[t] - index_w[t - 16])          # > 0: wheat moved the gap-widening way
    corn_part = side * beta * (index_c[t] - index_c[t - 16])     # > 0: corn moved the gap-widening way
    entry_open = bars["W_open"].values[trades["entry_bar"].values]
    return pd.DataFrame({
        "widen_1h": side * (sv[t] - sv[t - 4]),
        "widen_4h": side * (sv[t] - sv[t - 16]),
        "depth": np.where(side > 0, sv[t] - upper[t], lower[t] - sv[t]),
        "hours_in_tail": np.where(side > 0, run_up[t], run_down[t]) / 4,
        "wheat_push_1h_c": -side * (index_w[t] - index_w[t - 4]) * price,
        "gap_c": np.abs(features[f"kgap_{rule.memory_sessions}s"].values[t]) * entry_open,
        "atr_c": features["sigma_w"].values[t] * np.sqrt(SESSION) * entry_open,
        "leg": np.where(wheat_part > corn_part, "wheat-led", "corn-led"),
        "time_of_day": time_of_day(pd.Series(bars.index[t])).values,
    }, index=trades.index)


def fifths(table: pd.DataFrame, feature: str, years: pd.Series) -> tuple[pd.DataFrame, dict]:
    bucket = pd.qcut(table[feature].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])   # rank: ties split evenly
    grouped = table.groupby(bucket, observed=True)
    rows = grouped.agg(trades=("net_c", "size"), from_=(feature, "min"), to=(feature, "max"),
                       net_per_trade_c=("net_c", "mean"), win_rate=("net_c", lambda x: (x > 0).mean()),
                       median_MAE_c=("mae_c", "median"), share_MAE_10c=("mae_c", lambda x: (x >= 10).mean()),
                       median_MFE_c=("mfe_c", "median")).reset_index().rename(columns={feature: "fifth"})
    by_year = table.assign(bucket=bucket).groupby([years, "bucket"], observed=True)["net_c"].mean().unstack()
    yearly_difference = by_year[5] - by_year[1]
    difference = rows["net_per_trade_c"].iloc[-1] - rows["net_per_trade_c"].iloc[0]
    verdict = {"feature": feature, "top_minus_bottom_c": difference,
               "years_same_sign": int((np.sign(yearly_difference) == np.sign(difference)).sum()),
               "spearman_net": spearmanr(table[feature], table["net_c"], nan_policy="omit")[0],
               "spearman_MAE": spearmanr(table[feature], table["mae_c"], nan_policy="omit")[0]}
    verdict["matters"] = abs(difference) >= MATTERS_CENTS and verdict["years_same_sign"] >= MATTERS_YEARS
    return rows, verdict


def limit_tradeoff(paths: dict, cost: float) -> pd.DataFrame:
    """Per original trigger: market entry vs a resting limit d cents better (plain 1-session hold)."""
    close = paths["close"][:, :SESSION + 1]
    valid = ~np.isnan(close)
    last = valid.sum(axis=1) - 1
    market = close[np.arange(len(close)), last] - cost
    worst = np.nan_to_num(paths["worst"][:, :LIMIT_WINDOW + 1], nan=0.0)
    rows = [{"offset_c": 0, "share_filled": 1.0, "net_per_trigger_c": market.mean(), "net_per_filled_trade_c":
             market.mean(), "missed_trades_market_net_c": np.nan, "win_rate_filled": (market > 0).mean()}]
    for offset in LIMIT_OFFSETS:
        filled = (worst <= -(offset + TICK_CENTS)).any(axis=1)
        result = np.where(filled, market + offset, 0.0)
        rows.append({"offset_c": offset, "share_filled": filled.mean(), "net_per_trigger_c": result.mean(),
                     "net_per_filled_trade_c": result[filled].mean(),
                     "missed_trades_market_net_c": market[~filled].mean(),
                     "win_rate_filled": (result[filled] > 0).mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ charts
def fifths_chart(tables: dict, verdicts: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharey=True)
    for axis, (feature, rows) in zip(axes.flat, tables.items()):
        axis.bar(rows["fifth"].astype(int), rows["net_per_trade_c"], color="#2b7bba")
        twin = axis.twinx()
        twin.plot(rows["fifth"].astype(int), rows["median_MAE_c"], marker="v", color="#d62728", lw=1)
        twin.set_ylim(0, 12)
        twin.tick_params(labelsize=7, colors="#d62728")
        mark = "MATTERS" if verdicts.set_index("feature").loc[feature, "matters"] else "no"
        axis.set_title(f"{feature}: {mark}", loc="left", fontsize=9)
        axis.axhline(0, color="grey", lw=0.8)
        axis.set_xlabel("fifth (1 = lowest)", fontsize=8)
    axes[0, 0].set_ylabel("net per trade, c (bars)")
    axes[1, 0].set_ylabel("net per trade, c (bars)")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, 0.5, "bars: net per trade (cents)\nred: median worst point against us (cents, right axis)",
                       fontsize=9)
    figure.tight_layout()
    return figure


def limit_chart(table: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9, 4.4))
    axis.bar(table["offset_c"], table["net_per_trigger_c"], color="#2b7bba", label="net cents per trigger")
    twin = axis.twinx()
    twin.plot(table["offset_c"], table["share_filled"], marker="o", color="#ff7f0e", label="share filled")
    twin.set_ylim(0, 1.05)
    twin.set_ylabel("share of triggers filled")
    axis.axhline(table["net_per_trigger_c"].iloc[0], color="black", ls="--", lw=0.8)
    axis.set_xlabel("limit d cents better than the market entry (0 = market)")
    axis.set_ylabel("net cents per trigger (missed = 0)")
    axis.set_title("Limit entry trade-off (plain 1-session hold)", loc="left")
    figure.legend(fontsize=8, loc="upper right")
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    rule = cfg.kalman_rule
    bars = load_bars(cfg)                                        # development data only
    features = build_features(bars, cfg.features)
    s = kalman_signal(features, rule)
    trades = book(bars, s, rule, features[f"kgap_{rule.memory_sessions}s"])
    cost = rule.cost_ticks * TICK_CENTS

    table = describe_triggers(bars, features, s, trades, rule)
    paths = trade_paths(bars, trades["entry_bar"].values, trades["side"].values, SESSION)
    table["net_c"] = trades["move_c"].values - cost
    table["mae_c"] = -np.nanmin(paths["worst"], axis=1)
    table["mfe_c"] = np.nanmax(paths["best"], axis=1)
    years = pd.Series(trades["year"].values, index=table.index, name="year")

    tables, verdicts = {}, []
    for feature in NUMERIC:
        tables[feature], verdict = fifths(table, feature, years)
        verdicts.append(verdict)
    verdicts = pd.DataFrame(verdicts)
    groups = {}
    for column in ["leg", "time_of_day"]:
        grouped = table.groupby(column)
        groups[column] = grouped.agg(trades=("net_c", "size"), net_per_trade_c=("net_c", "mean"),
                                     win_rate=("net_c", lambda x: (x > 0).mean()), median_MAE_c=("mae_c", "median"),
                                     median_MFE_c=("mfe_c", "median")).reset_index()
    limits = limit_tradeoff(paths, cost)
    correlation = table[NUMERIC].corr(method="spearman")

    pd.set_option("display.width", 220)
    print(f"{len(trades)} trades (the rule's book); net per trade = all trades after costs (rule exits);"
          f" MAE / MFE = worst / best point within 1 session, cents")
    print(f"overall: net per trade {table['net_c'].mean():.2f} c, median MAE {table['mae_c'].median():.2f} c,"
          f" median MFE {table['mfe_c'].median():.2f} c")
    for feature in NUMERIC:
        print(f"\n== {feature}")
        print(tables[feature].round(2).to_string(index=False))
    print(f"\n== verdicts (rule fixed before the run: top - bottom fifth >= {MATTERS_CENTS} c and the same sign in"
          f" >= {MATTERS_YEARS} of 6 years)")
    print(verdicts.round(3).to_string(index=False))
    for column, grouped in groups.items():
        print(f"\n== by {column}")
        print(grouped.round(2).to_string(index=False))
    print("\n== features' rank correlation with each other")
    print(correlation.round(2).to_string())
    print("\n== B limit entry trade-off (plain 1-session hold; per original trigger, a missed trade counts 0)")
    print(limits.round(2).to_string(index=False))

    figures = {"A_fifths": fifths_chart(tables, verdicts), "B_limit": limit_chart(limits)}
    run_dir = save_run(STRATEGY_DIR, "entry_timing", cfg, {"matters": verdicts[verdicts["matters"]]["feature"].tolist()},
                       {"triggers": table, "verdicts": verdicts, "limit_entry": limits,
                        **{f"fifths_{name}": rows for name, rows in tables.items()},
                        **{f"by_{name}": grouped for name, grouped in groups.items()}}, figures)
    print(f"\nsaved to {run_dir}")
