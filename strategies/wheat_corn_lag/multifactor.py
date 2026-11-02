"""
multifactor.py
--------------
Layer 1, extended: does adding soybeans (s) and/or USD (u) to corn (c) improve the fair value of wheat?
Development data only, on the COMMON sample where every factor exists (soybeans start Jul 2020), so corn
alone is re-scored on exactly the same bars. Saved to runs/<time>_multifactor/.

    python -m strategies.wheat_corn_lag.multifactor

Two kinds of answer, kept apart:
1. Explanatory power: how much of wheat's move does the fair value price?
   out-of-sample R2       1 - var(wheat move - predicted) / var(wheat move), predicted = sum(beta x factor
                          move) with the betas known at the START of the move. At 15 min, 1 and 2 sessions.
   residual vol           sd of the unexplained 1-session move, in bp and cents: the width of the fair-value band
   beta stability         mean and sd of each beta over time; soy-corn correlation and its variance inflation
   residual reversion     variance ratio of the unexplained move at 2 sessions (< 1 = it pulls back)
2. Signal quality (the deciding test): IC of the divergence on forward wheat returns, exactly as layer1.py,
   for each factor set's composite signal (L = 106/142/213), plus the decomposition y = a x explained + b x own.
   Pass criteria fixed before the run (as layer1.py): primary horizon 2 sessions, NW t >= 3, same sign in
   >= 5 of 6 years, deciles monotonic >= 0.8.
More explained (higher R2) is NOT automatically better: a factor that moves with the same flow that
overshoots wheat absorbs the overshoot itself, and the IC falls. The IC decides.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from core_quant.config import save_run
from core_quant.theme import set_theme
from strategies.wheat_corn_lag.features import build_features, change_ahead, change_over
from strategies.wheat_corn_lag.layer1 import (HORIZONS, LOOKBACKS, PASS_MONOTONIC, PASS_T, PASS_YEARS, PRIMARY,
                                             deciles, ic_by_year, ic_table)
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load, load_bars

R2_HORIZONS = [1, 71, 142]
SET_LABELS = {"c": "corn", "cs": "corn + soy", "cu": "corn + USD", "csu": "corn + soy + USD"}


def factor_levels(bars):
    return {"c": bars["log_c"], "s": bars["index_s"], "u": bars["index_u"]}


def explanatory_power(bars, features, sets: dict, common: pd.Series) -> pd.DataFrame:
    levels = factor_levels(bars)
    segment = bars["segment"]
    price = bars["W"].median()
    rows = []
    for name, factors in sets.items():
        row = {"set": SET_LABELS[name]}
        for k in R2_HORIZONS:
            move_w = change_over(bars["log_w"], segment, k)
            # betas known at the START of the move (bar t-k), applied to the factor moves over (t-k, t]
            predicted = sum(features[f"beta_{name}_{f}"].shift(k) * change_over(levels[f], segment, k) for f in factors)
            pair = pd.concat([move_w, predicted], axis=1)[common].dropna()
            residual = pair.iloc[:, 0] - pair.iloc[:, 1]
            row[f"R2_oos_{k}"] = 1 - residual.var() / pair.iloc[:, 0].var()
            if k == 71:
                row["resid_sd_1session_bp"] = residual.std() * 1e4
                row["resid_sd_1session_cents"] = residual.std() * price
            if k == 142:
                one_bar = (bars["ret_w"] - sum(features[f"beta_{name}_{f}"] * bars[f"ret_{f}"] for f in factors))[common]
                row["resid_variance_ratio_2s"] = residual.var() / (142 * one_bar.var())
        for f in factors:
            beta = features[f"beta_{name}_{f}"][common]
            row[f"beta_{f}_mean"], row[f"beta_{f}_sd"] = beta.mean(), beta.std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("set")


def decomposition(frame, features, name: str, lookback: int, horizon: int) -> dict:
    forward = frame[f"fwd_{horizon}"]
    data = pd.DataFrame({"y": forward / forward.std(), "explained": features[f"explained_ze_{name}_{lookback}"],
                         "own": features[f"own_ze_{name}_{lookback}"]}).dropna()
    fit = sm.OLS(data["y"], sm.add_constant(data[["explained", "own"]])).fit(cov_type="HAC", cov_kwds={"maxlags": horizon})
    return {"set": SET_LABELS[name], "a_explained": fit.params["explained"], "t_a": fit.tvalues["explained"],
            "b_own": fit.params["own"], "t_b": fit.tvalues["own"]}


# ------------------------------------------------------------------ charts
def r2_chart(power: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / len(power)
    labels = {1: "15 min", 71: "1 session", 142: "2 sessions"}
    for i, (name, row) in enumerate(power.iterrows()):
        values = [row[f"R2_oos_{k}"] for k in R2_HORIZONS]
        axis.bar(np.arange(len(R2_HORIZONS)) + (i - (len(power) - 1) / 2) * width, values, width=width * 0.95, label=name)
    axis.set_xticks(np.arange(len(R2_HORIZONS)), [labels[k] for k in R2_HORIZONS])
    axis.set_ylabel("out-of-sample R2 of wheat's move")
    axis.set_title("How much of wheat's move does each fair value explain? (betas from the past only)", loc="left")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def ic_chart(table: pd.DataFrame, years: pd.DataFrame, frame: pd.DataFrame, names: list[str]) -> plt.Figure:
    figure, (decay, by_year) = plt.subplots(1, 2, figsize=(15, 4.6))
    for name in names:
        block = table[table["signal"] == name]
        decay.plot(block["h"] / 71, block["ic"], marker="o", label=name)
    decay.axhline(0, color="grey", lw=0.8)
    decay.set_xlabel("forward horizon, sessions")
    decay.set_ylabel("rank IC")
    decay.set_title("IC decay by fair value", loc="left")
    decay.legend(fontsize=8)
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        by_year.bar(np.arange(len(years)) + (i - (len(names) - 1) / 2) * width, years[name], width=width * 0.95, label=name)
    by_year.set_xticks(np.arange(len(years)), years.index)
    by_year.axhline(0, color="grey", lw=0.8)
    by_year.set_title(f"IC by year, {PRIMARY // 71}-session forward return", loc="left")
    figure.tight_layout()
    return figure


def decile_chart(frame: pd.DataFrame, names: list[str]) -> plt.Figure:
    figure, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4), sharey=True)
    for axis, name in zip(axes, names):
        means = deciles(frame, name, PRIMARY)
        axis.bar(means.index, means.values, color=["#d62728" if v < 0 else "#2ca02c" for v in means.values])
        axis.axhline(0, color="grey", lw=0.8)
        axis.set_title(name, loc="left", fontsize=9)
        axis.set_xlabel("signal decile (10 = wheat cheapest)")
    axes[0].set_ylabel(f"mean wheat move next {PRIMARY // 71} sessions, bp")
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    bars = load_bars(cfg)                                     # development data only
    features = build_features(bars, cfg.features)
    sets = cfg.features.factor_sets

    # common sample: every set's composite signal exists
    signals = pd.DataFrame({SET_LABELS[name]: -features[[f"own_ze_{name}_{l}" for l in LOOKBACKS]].mean(axis=1, skipna=False)
                            for name in sets})
    common = signals.notna().all(axis=1)
    signals = signals[common]
    targets = pd.DataFrame({f"fwd_{h}": change_ahead(bars["log_w"], bars["segment"], h) for h in HORIZONS})[common]
    frame = signals.join(targets)
    names = list(signals.columns)
    print(f"common sample: {frame.index[0]} -> {frame.index[-1]}, {len(frame):,} bars")

    power = explanatory_power(bars, features, sets, common)
    table = ic_table(frame, names)
    years = ic_by_year(frame, names, PRIMARY)
    primary = table[table["h"] == PRIMARY].set_index("signal")
    verdict = []
    for name in names:
        means = deciles(frame, name, PRIMARY)
        ic, t = primary.loc[name, "ic"], primary.loc[name, "t_nw"]
        same = int((np.sign(years[name]) == np.sign(ic)).sum())
        monotonic = spearmanr(means.index, means.values)[0]
        verdict.append({"signal": name, "ic": ic, "t_nw": t, "years_same_sign": same, "decile_monotonic": monotonic,
                        "PASS": t >= PASS_T and same >= PASS_YEARS and monotonic >= PASS_MONOTONIC})
    verdict = pd.DataFrame(verdict)
    split = pd.DataFrame([decomposition(frame, features, name, 142, PRIMARY) for name in sets])
    correlation = signals.corr()
    soy_on_corn = bars[["ret_c", "ret_s"]][common].dropna()
    soy_corn_corr = soy_on_corn.corr().iloc[0, 1]

    pd.set_option("display.width", 220)
    print("\n== 1. explanatory power (out-of-sample)")
    print(power.round(3).to_string())
    print(f"soy-corn return correlation {soy_corn_corr:.2f}, variance inflation of soy given corn {1 / (1 - soy_corn_corr ** 2):.2f}")
    print("\n== 2. signal quality: rank IC by horizon")
    print(table.pivot(index="signal", columns="h", values="ic").round(3).to_string())
    print("\nNewey-West t")
    print(table.pivot(index="signal", columns="h", values="t_nw").round(2).to_string())
    print(f"\nverdict at {PRIMARY} bars")
    print(verdict.round(3).to_string(index=False))
    print("\nIC by year")
    print(years.round(3).to_string())
    print(f"\ndecomposition y = a x explained + b x own (L142, h{PRIMARY}; negative = reverts)")
    print(split.round(3).to_string(index=False))
    print("\ncorrelation between the signals")
    print(correlation.round(3).to_string())

    figures = {"explained_r2": r2_chart(power), "ic": ic_chart(table, years, frame, names),
               "deciles": decile_chart(frame, names)}
    run_dir = save_run(STRATEGY_DIR, "multifactor", cfg, {"common_start": str(frame.index[0])},
                       {"explanatory_power": power, "ic": table, "ic_by_year": years, "verdict": verdict,
                        "decomposition": split, "signal_correlation": correlation}, figures)
    print(f"\nsaved to {run_dir}")
