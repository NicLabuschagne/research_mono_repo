"""
ic.py
-----
Step 5: does (corn's move, wheat's own move) predict wheat's next move, and at which lookback and horizon?
Development data only (load_bars drops the holdout). Every result is saved to runs/<time>_ic/.

    python -m strategies.wheat_corn_lag.ic

Target  y = fwd_h / (sigma_w * sqrt h): wheat's forward move in its own sd (sigma_w is known at t),
        so a quiet 2020 and a wild 2022 count alike.

1. Grid     y = const + a * corn_z_L + b * own_z_L, for every lookback L and horizon h.
            Newey-West t with lags = h: overlapping targets are not independent, and a 4-session target
            would otherwise be counted ~284 times.
            Story A, the complex reverts:     a < 0
            Story B, wheat's lag informs:     b > 0
            The trader's rule (trade against corn, strongest when wheat lagged) needs a < 0, b > 0.
            Repeated with the Kalman beta (own_kz).
2. By year  a and b fitted per calendar year: in how many of the 6 years does each keep its sign?
3. F0       the literal rule: average forward move in the rule's direction, bp, with Newey-West t.
4. Heatmap  mean forward move (bp) by (corn_z, own_z) bins. The two cells shown (L=8 h=16, L=32 h=71)
            were fixed before looking at any result.
Scale: costs are 1.4 ticks round trip = 0.35 c, about 6 bp of a 600 c wheat price.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from core_quant.config import save_run
from core_quant.theme import diverging_cmap
from strategies.wheat_corn_lag.features import build_features, build_targets
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load, load_bars

HEATMAP_CELLS = [(8, 16), (32, 71)]                       # (lookback, horizon), chosen before any result
BIN_EDGES = [-np.inf, -2, -1, -0.5, 0.5, 1, 2, np.inf]
BIN_LABELS = ["< -2", "-2..-1", "-1..-0.5", "flat", "0.5..1", "1..2", "> 2"]
MIN_BARS_PER_CELL = 200


def newey_west_fit(frame: pd.DataFrame, target: str, regressors: list[str], lags: int):
    data = frame[[target] + regressors].dropna()
    fit = sm.OLS(data[target], sm.add_constant(data[regressors])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return fit, len(data)


def regression_grid(frame: pd.DataFrame, cfg) -> pd.DataFrame:
    rows = []
    for lookback in cfg.features.lookbacks_bars:
        for horizon in cfg.features.horizons_bars:
            row = {"L": lookback, "h": horizon}
            for label, own in [("", f"own_z_{lookback}"), ("_kalman", f"own_kz_{lookback}")]:
                fit, count = newey_west_fit(frame, f"y_{horizon}", [f"corn_z_{lookback}", own], horizon)
                row.update({f"a{label}": fit.params.iloc[1], f"t_a{label}": fit.tvalues.iloc[1],
                            f"b{label}": fit.params.iloc[2], f"t_b{label}": fit.tvalues.iloc[2]})
            row["n"] = count
            # in bp of wheat per 1 sd of the feature, at a typical volatility
            row["bp_scale"] = frame["sigma_w"].median() * np.sqrt(horizon) * 1e4
            rows.append(row)
    grid = pd.DataFrame(rows)
    grid["a_bp"] = grid["a"] * grid["bp_scale"]
    grid["b_bp"] = grid["b"] * grid["bp_scale"]
    return grid


def by_year(frame: pd.DataFrame, cfg) -> pd.DataFrame:
    rows = []
    years = frame.index.year
    for lookback in cfg.features.lookbacks_bars:
        for horizon in cfg.features.horizons_bars:
            columns = [f"y_{horizon}", f"corn_z_{lookback}", f"own_z_{lookback}"]
            signs = {"L": lookback, "h": horizon, "years_a_negative": 0, "years_b_positive": 0, "years": 0}
            for _, block in frame[columns].dropna().groupby(years[frame[columns].notna().all(axis=1)]):
                design = np.column_stack([np.ones(len(block)), block[columns[1]], block[columns[2]]])
                _, a, b = np.linalg.lstsq(design, block[columns[0]].values, rcond=None)[0]
                signs["years"] += 1
                signs["years_a_negative"] += int(a < 0)
                signs["years_b_positive"] += int(b > 0)
            rows.append(signs)
    return pd.DataFrame(rows)


def literal_rule_table(frame: pd.DataFrame, cfg) -> pd.DataFrame:
    rows = []
    for lookback in cfg.features.lookbacks_bars:
        rule = frame[f"f0_{lookback}"]
        for horizon in [4, 16, 71]:
            for label, chosen in [("all signals", rule != 0), ("strongest |2|", rule.abs() == 2)]:
                # forward move in the rule's direction, bp: + = the rule made money before costs
                signed = (np.sign(rule) * frame[f"fwd_{horizon}"] * 1e4).where(chosen & rule.notna())
                block = signed.dropna().to_frame("signed")
                fit = sm.OLS(block["signed"], np.ones(len(block))).fit(cov_type="HAC", cov_kwds={"maxlags": horizon})
                rows.append({"L": lookback, "h": horizon, "signals": label, "bars": len(block),
                             "mean_bp": fit.params.iloc[0], "t_nw": fit.tvalues.iloc[0],
                             "hit_rate": (block["signed"] > 0).mean()})
    return pd.DataFrame(rows)


def heatmap(frame: pd.DataFrame, lookback: int, horizon: int):
    corn_bin = pd.cut(frame[f"corn_z_{lookback}"], BIN_EDGES, labels=BIN_LABELS)
    own_bin = pd.cut(frame[f"own_z_{lookback}"], BIN_EDGES, labels=BIN_LABELS)
    forward_bp = frame[f"fwd_{horizon}"] * 1e4
    grouped = forward_bp.groupby([own_bin, corn_bin], observed=False)
    mean = grouped.mean().unstack().where(grouped.count().unstack() >= MIN_BARS_PER_CELL)
    count = grouped.count().unstack()

    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    limit = np.nanmax(np.abs(mean.values))
    image = axis.imshow(mean.values[::-1], cmap=diverging_cmap(), vmin=-limit, vmax=limit, aspect="auto")
    for i, own_label in enumerate(mean.index[::-1]):
        for j, corn_label in enumerate(mean.columns):
            value = mean.loc[own_label, corn_label]
            if np.isfinite(value):
                axis.text(j, i, f"{value:+.1f}\n{count.loc[own_label, corn_label]:,}", ha="center", va="center", fontsize=7)
    axis.set_xticks(range(len(BIN_LABELS)), BIN_LABELS)
    axis.set_yticks(range(len(BIN_LABELS)), BIN_LABELS[::-1])
    axis.set_xlabel(f"corn move over {lookback} bars (sd)")
    axis.set_ylabel(f"wheat's own move over {lookback} bars (sd)")
    axis.set_title(f"Mean wheat move over the next {horizon} bars (bp), bars per cell below", loc="left")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    return figure, mean


if __name__ == "__main__":
    cfg = load()
    bars = load_bars(cfg)                                     # development data only
    frame = build_features(bars, cfg.features).join(build_targets(bars, cfg.features))
    for horizon in cfg.features.horizons_bars:
        frame[f"y_{horizon}"] = frame[f"fwd_{horizon}"] / (frame["sigma_w"] * np.sqrt(horizon))

    grid = regression_grid(frame, cfg)
    years = by_year(frame, cfg)
    rule = literal_rule_table(frame, cfg)
    figures = {}
    for lookback, horizon in HEATMAP_CELLS:
        figures[f"heatmap_L{lookback}_h{horizon}"], _ = heatmap(frame, lookback, horizon)

    pd.set_option("display.width", 220)
    for column, title in [("t_a", "t-stat of a (corn's move); negative = wheat moves AGAINST corn's move"),
                          ("t_b", "t-stat of b (wheat's own move); positive = wheat's own move CONTINUES"),
                          ("a_bp", "a in bp of wheat per 1 sd of corn move"),
                          ("b_bp", "b in bp of wheat per 1 sd of wheat's own move"),
                          ("t_a_kalman", "t-stat of a, Kalman beta"),
                          ("t_b_kalman", "t-stat of b, Kalman beta")]:
        print(f"\n== {title}   (rows: lookback L bars, columns: horizon h bars)")
        print(grid.pivot(index="L", columns="h", values=column).round(1).to_string())
    print("\n== years (of 6) with a < 0")
    print(years.pivot(index="L", columns="h", values="years_a_negative").to_string())
    print("\n== years (of 6) with b > 0")
    print(years.pivot(index="L", columns="h", values="years_b_positive").to_string())
    print("\n== F0, the literal rule: forward move in the rule's direction (bp, before ~6 bp costs)")
    print(rule.round(2).to_string(index=False))

    run_dir = save_run(STRATEGY_DIR, "ic", cfg, {"cells": len(grid), "heatmaps": list(figures)},
                       {"grid": grid, "by_year": years, "literal_rule": rule}, figures)
    print(f"\nsaved to {run_dir}")
