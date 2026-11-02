"""
meta_model.py
-------------
Layer 2, step 2: the meta-model - which triggers to take, and in which instrument.

Stage 1 (this run): the FEATURE TABLE and a quality screen, before any model is fitted. Every feature is known at
the trigger bar's close (the entry is the next open). Pre-registered list (2026-11-02):

    range_pos       position in the day's running range from 1-min highs / lows, FLIPPED by side so 1 = entering
                    at the extreme (selling the high / buying the low)
    new_extreme     this bar set a new high (rich) / low (cheap) of the day
    range_sigma     the day's range so far in sigma_1h units
    log_abs_z       log |z| (|z| reaches ~50)
    gap_sigma       |gap| / sigma_1h - the dislocation in barrier units
    beans_share_15 / _30   |beans move| / (|beans move| + |fair move|) over 15 / 30 min, trigger hedge held fixed:
                    1 = beans moved into the gap, 0 = meal / oil moved
    oil_share_30    oil's part of the fair-value move vs meal's
    session_*, tod_sin / tod_cos, first_30min   session dummies (Asia = base), clock time CT, first 30 min of a session
    sigma_ratio     sigma_1h vs its median over the previous ~20 days (volatility regime)
    bars_since_calm bars since |z| was last inside the re-arm level (fast spike vs slow drift), capped at 48
    gap_change_15   how much the gap widened over the last 15 min, in sigma_1h units
    vol_zs_20d      trigger-bar ZS volume vs the same time slot over the previous 20 trading days
    vol_zs_today    ...vs the mean bar so far today (note: rises at every session open - read with the session)
    vol_zs_prev     ...vs the mean bar of the same session on the previous trading day
    vol_zm_today / vol_zl_today   meal / oil volume vs their mean bar so far today (which leg had the flow)
    hedge_drift     how far the fair value moved in the last hour from the betas changing, not prices (sigma units)
    hit_rate        win share of the last 20 triggers whose exit was BEFORE this trigger (per instrument)
    side, crosses_break   rich (+1) / cheap (-1); the 60-min hold spans a session break (known from the schedule)
Volume ratios are logged. The label is the triple barrier at the main setting (k 1.0, stop 1.25x), fixed before
any modelling; k 0.75 / stop 1.25x is the one sensitivity check.

Stage 2: the model. One L2 logistic regression per instrument (beans, spread with beans passive), walk-forward in
half-year blocks from 2022: train on every trade that EXITED at least a day before the block (purge + embargo),
predict the block. Inside each training window only: median-fill, clip at the 1st / 99th percentiles, standardise,
pick C on the window's last quarter (purged). Trades weighted by average uniqueness. Expected value per trade =
p x (training mean win) + (1 - p) x (training mean loss); the router takes the instrument with the higher expected
value, or skips. Judged on out-of-sample predictions only: AUC and log loss vs the base rate, calibration,
coefficient stability across blocks, net ticks by threshold vs simple rules.

    python -m strategies.bean_divergence.meta_model          # features + leakage check + screen + model
    python -m strategies.bean_divergence.meta_model fit      # the same from the saved features (skips the rebuild)
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from core_quant.theme import LIGHT, PALETTE
from strategies.bean_divergence.fair_value import LEGS, STRATEGY_DIR, TICK_CENTS, adaptive_scale, load, session_label
from strategies.bean_divergence.labels import HOLD_BARS, minute_prices
from strategies.bean_divergence.triggers import build

LABEL_K, LABEL_STOP = 1.0, 1.25
INSTRUMENTS = ["beans", "spread_beans_passive"]
HIT_RATE_TRADES = 20
VOLUME_DAYS = 20
FEATURES = ["range_pos", "new_extreme", "range_sigma", "log_abs_z", "gap_sigma", "beans_share_15", "beans_share_30",
            "oil_share_30", "session_europe", "session_us", "tod_sin", "tod_cos", "first_30min", "sigma_ratio",
            "bars_since_calm", "gap_change_15", "vol_zs_20d", "vol_zs_today", "vol_zs_prev", "vol_zm_today",
            "vol_zl_today", "hedge_drift", "hit_rate", "side", "crosses_break"]
# dropped 2026-11-02 after the first fit: weak in both instruments (|coefficient mean / std| < 1.6 across the blocks,
# or the sign flipped); still built and screened
DROPPED = ["side", "gap_sigma", "range_sigma", "beans_share_30", "session_europe", "vol_zs_prev", "vol_zl_today",
           "bars_since_calm"]
MODEL_FEATURES = [f for f in FEATURES if f not in DROPPED]
STOP_COMPARE = [0.5, 0.75, 1.0, 1.25]      # stop = this x the target: below 1 = wins bigger than losses
TOP_SHARES = [0.1, 0.2, 0.3]               # compare label settings at the same trade count: the top x% by p
WALK_FORWARD_STARTS = pd.to_datetime(["2022-01-01", "2022-07-01", "2023-01-01", "2023-07-01", "2024-01-01",
                                      "2024-07-01"]).tz_localize("America/Chicago")
EMBARGO = pd.Timedelta(days=1)
C_GRID = [0.003, 0.01, 0.03, 0.1, 1.0]
P_THRESHOLDS = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60]
EV_THRESHOLDS = [0.0, 0.25, 0.5, 1.0, 1.5]


def chicago_date(index: pd.DatetimeIndex) -> pd.Series:
    """CBOT trading date (the evening session belongs to the next day), as in data_ingest.add_trading_date."""
    return pd.Series((index.tz_convert("America/Chicago") + pd.Timedelta(hours=5)).tz_localize(None).normalize(),
                     index=index)


def slot_median(values: pd.Series, date: pd.Series, slot: pd.Series, days: int) -> np.ndarray:
    """Median of `values` at the same slot over the previous `days` trading dates (today excluded)."""
    frame = pd.DataFrame({"date": date.values, "slot": slot.values, "v": values.values})
    by_day = frame.pivot_table(index="date", columns="slot", values="v", aggfunc="mean")
    past = by_day.rolling(days, min_periods=days // 2).median().shift(1)
    return frame.merge(past.stack().rename("m").reset_index(), on=["date", "slot"], how="left")["m"].values


def build_features(bars, hedge, gap, z, triggers, minutes, labels, cfg) -> pd.DataFrame:
    """One row per trigger: the features, known at the trigger bar's close. Inputs may be cut at any time T and the
    rows before T must not change (checked in __main__)."""
    r = cfg.research
    n = len(bars)
    a, bm, bo = (hedge[c].values for c in ["alpha", "beta_meal", "beta_oil"])
    log_c = {leg: np.log(bars[f"{leg}_adj_close"]).values for leg in LEGS}
    ticks = (bars["ZS_adj_close"] / TICK_CENTS).values
    sigma = (adaptive_scale(bars["ZS_ret"] * ticks, r.z_half_life_bars) * np.sqrt(HOLD_BARS)).values
    sigma_median = pd.Series(sigma).rolling(VOLUME_DAYS * 211, min_periods=VOLUME_DAYS * 100).median().shift(1).values
    gap, z = gap.values, z.values
    date = bars["trading_date"]
    session = session_label(bars.index, bars["session_type"])

    # running high / low of the trading day from 1-minute bars, read at each bar's last minute
    m_date = chicago_date(minutes.index)
    run_high = np.log(minutes["ZS_adj_high"]).groupby(m_date.values).cummax().values
    run_low = np.log(minutes["ZS_adj_low"]).groupby(m_date.values).cummin().values
    bar_pos = minutes.index.searchsorted(bars.index)
    high_at, low_at = run_high[bar_pos], run_low[bar_pos]
    same_day_before = np.r_[False, date.values[1:] == date.values[:-1]]
    new_high = ~same_day_before | (high_at > np.r_[np.nan, high_at[:-1]])
    new_low = ~same_day_before | (low_at < np.r_[np.nan, low_at[:-1]])

    # volume ratios
    slot = bars["session_type"] + "_" + bars["bar_number"].astype(str)
    vol = {leg: bars[f"{leg}_volume"].astype(float) for leg in LEGS}
    vol_20d = vol["ZS"].values / slot_median(vol["ZS"], date, slot, VOLUME_DAYS)
    vol_today = {}
    for leg in ["ZS", "ZM", "ZL"]:
        before = vol[leg].groupby(date.values).cumsum() - vol[leg]
        count = vol[leg].groupby(date.values).cumcount()
        vol_today[leg] = vol[leg].values / (before / count.where(count >= 6)).values
    sess_mean = pd.DataFrame({"date": date.values, "session": session, "v": vol["ZS"].values}).pivot_table(
        index="date", columns="session", values="v", aggfunc="mean").shift(1)        # previous trading date
    prev = pd.DataFrame({"date": date.values, "session": session}).merge(
        sess_mean.stack().rename("m").reset_index(), on=["date", "session"], how="left")["m"].values
    vol_prev = vol["ZS"].values / prev

    clock = bars.index.tz_convert("America/Chicago")
    minutes_of_day = clock.hour * 60 + clock.minute
    crosses = labels.drop_duplicates("bar").set_index("bar")["crosses_break"]
    hits = {}
    for instrument in INSTRUMENTS:
        done = labels[(labels.kind == "signal") & (labels.instrument == instrument) & (labels.k == LABEL_K)
                      & (labels.stop_mult == LABEL_STOP) & ~labels.skipped].sort_values(["exit_time", "bar"])
        hits[instrument] = (done["exit_time"].values, np.r_[0, np.cumsum(done["label"].values)])

    rows = []
    for tr in triggers.itertuples():
        s, side = tr.bar, tr.side
        if s < 48 or np.isnan(sigma[s]):
            continue
        moves = {}
        for k in [3, 6]:
            beans = (log_c["ZS"][s] - log_c["ZS"][s - k]) * ticks[s]
            meal = bm[s] * (log_c["ZM"][s] - log_c["ZM"][s - k]) * ticks[s]
            oil = bo[s] * (log_c["ZL"][s] - log_c["ZL"][s - k]) * ticks[s]
            moves[k] = (abs(beans) / (abs(beans) + abs(meal + oil) + 1e-9), abs(oil) / (abs(meal) + abs(oil) + 1e-9))
        calm = np.flatnonzero(np.abs(z[s - 48:s]) < r.rearm_z)
        fair_now = a[s] + bm[s] * log_c["ZM"][s] + bo[s] * log_c["ZL"][s]
        fair_old_betas = a[s - 12] + bm[s - 12] * log_c["ZM"][s] + bo[s - 12] * log_c["ZL"][s]
        pos = (log_c["ZS"][s] - low_at[s]) / max(high_at[s] - low_at[s], 1e-9)
        row = {"bar": s, "range_pos": pos if side > 0 else 1 - pos,
               "new_extreme": float(new_high[s] if side > 0 else new_low[s]),
               "range_sigma": (high_at[s] - low_at[s]) * ticks[s] / sigma[s],
               "log_abs_z": np.log(abs(z[s])), "gap_sigma": abs(gap[s]) / sigma[s],
               "beans_share_15": moves[3][0], "beans_share_30": moves[6][0], "oil_share_30": moves[6][1],
               "session_europe": float(session[s] == "europe"), "session_us": float(session[s] == "us"),
               "tod_sin": np.sin(2 * np.pi * minutes_of_day[s] / 1440),
               "tod_cos": np.cos(2 * np.pi * minutes_of_day[s] / 1440),
               "first_30min": float(bars["bar_number"].values[s] * r.bar_minutes < 30),
               "sigma_ratio": sigma[s] / sigma_median[s],
               "bars_since_calm": 48 - calm[-1] if len(calm) else 48,
               "gap_change_15": side * (gap[s] - gap[s - 3]) / sigma[s],
               "vol_zs_20d": np.log(vol_20d[s]), "vol_zs_today": np.log(vol_today["ZS"][s]),
               "vol_zs_prev": np.log(vol_prev[s]), "vol_zm_today": np.log(vol_today["ZM"][s]),
               "vol_zl_today": np.log(vol_today["ZL"][s]),
               "hedge_drift": abs(fair_now - fair_old_betas) * ticks[s] / sigma[s],
               "side": float(side), "crosses_break": float(crosses.get(s, np.nan))}
        for instrument, (exit_times, cum_wins) in hits.items():
            done = np.searchsorted(exit_times, bars.index[s].to_datetime64(), side="right")   # exits at or before now
            last = min(done, HIT_RATE_TRADES)
            row[f"hit_rate_{instrument}"] = (cum_wins[done] - cum_wins[done - last]) / last if last >= 10 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Area under the ROC curve via ranks: the chance a random win scores above a random loss."""
    ranks = pd.Series(score).rank().values
    wins = label == 1
    n1, n0 = wins.sum(), (~wins).sum()
    return (ranks[wins].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def label_table(labels: pd.DataFrame, features: pd.DataFrame, stop_mult: float = LABEL_STOP) -> pd.DataFrame:
    """Traded triggers at one label setting, one row per trigger and instrument, with their features."""
    main = labels[(labels.kind == "signal") & (labels.k == LABEL_K) & (labels.stop_mult == stop_mult)
                  & ~labels.skipped & labels.instrument.isin(INSTRUMENTS)]
    table = main[["bar", "time", "exit_time", "instrument", "half", "session", "label", "net_ticks", "weight"]].merge(
        features, on="bar")
    table["hit_rate"] = np.where(table.instrument == "beans", table["hit_rate_beans"],
                                 table["hit_rate_spread_beans_passive"])
    return table


def fit_logistic(train: pd.DataFrame, c: float):
    """Median-fill, clip at the training 1st / 99th percentiles, standardise, L2 logistic - all from `train` only."""
    x = train[MODEL_FEATURES]
    median, low, high = x.median(), x.quantile(0.01), x.quantile(0.99)
    clipped = x.fillna(median).clip(low, high, axis=1)
    mean, std = clipped.mean(), clipped.std().replace(0, 1)
    prep = lambda frame: ((frame[MODEL_FEATURES].fillna(median).clip(low, high, axis=1) - mean) / std).values
    model = LogisticRegression(C=c, max_iter=5000).fit(prep(train), train["label"], sample_weight=train["weight"])
    return model, prep


def choose_c(train: pd.DataFrame) -> float:
    """The C with the lowest weighted log loss on the last quarter of the training window (purged)."""
    val_start = train["time"].quantile(0.75)
    inner, val = train[train.exit_time < val_start - EMBARGO], train[train.time >= val_start]
    losses = {}
    for c in C_GRID:
        model, prep = fit_logistic(inner, c)
        losses[c] = log_loss(val["label"], model.predict_proba(prep(val))[:, 1], sample_weight=val["weight"])
    return min(losses, key=losses.get)


def walk_forward(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-sample probability and expected value per trade, block by block; coefficients per block."""
    predictions, coefficients = [], []
    for instrument in INSTRUMENTS:
        data = table[table.instrument == instrument]
        for i, start in enumerate(WALK_FORWARD_STARTS):
            end = WALK_FORWARD_STARTS[i + 1] if i + 1 < len(WALK_FORWARD_STARTS) else data.time.max() + EMBARGO
            train = data[data.exit_time < start - EMBARGO]
            test = data[(data.time >= start) & (data.time < end)].copy()
            assert train.exit_time.max() < test.time.min()                  # no training trade overlaps the block
            c = choose_c(train)
            model, prep = fit_logistic(train, c)
            win, loss = train[train.label == 1].net_ticks.mean(), train[train.label == 0].net_ticks.mean()
            test["p"] = model.predict_proba(prep(test))[:, 1]
            test["base_rate"] = train.label.mean()
            test["ev"] = test.p * win + (1 - test.p) * loss
            test["fold"] = f"{start:%Y-%m}"
            predictions.append(test)
            coefficients.append({"instrument": instrument, "fold": f"{start:%Y-%m}", "C": c, "n_train": len(train)}
                                | dict(zip(MODEL_FEATURES, model.coef_[0])))
    return pd.concat(predictions), pd.DataFrame(coefficients)


def trade_stats(name: str, trades: pd.DataFrame, all_dates: pd.Index, weeks: float) -> dict:
    """Count, trades a week, win share, net ticks, and the t-stat of DAILY net ticks (trades overlap within a day)."""
    daily = trades.groupby(chicago_date(pd.DatetimeIndex(trades.time)).values).net_ticks.sum()
    daily = daily.reindex(all_dates, fill_value=0)
    return {"rule": name, "trades": len(trades), "per week": len(trades) / weeks, "win%": trades.label.mean() * 100,
            "net per trade": trades.net_ticks.mean(), "net per week": trades.net_ticks.sum() / weeks,
            "daily t": daily.mean() / daily.sem()}


if __name__ == "__main__":
    assert np.isclose(auc(np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1])), 1.0)
    assert np.isclose(auc(np.array([1, 2, 3, 4]), np.array([1, 0, 1, 0])), 0.25)

    cfg = load()
    labels = pd.read_parquet(STRATEGY_DIR / "output" / "labels.parquet")
    labels = labels[labels.entry_rule == "trigger"]                    # the original entry (labels.py: entry rules)
    if "fit" in sys.argv[1:]:
        features = pd.read_parquet(STRATEGY_DIR / "output" / "features.parquet")
    else:
        bars, hedge, gap, z, triggers = build(cfg)
        minutes = minute_prices(cfg)
        features = build_features(bars, hedge, gap, z, triggers, minutes, labels, cfg)
        # leakage check: cut every input at time T; features of triggers before T must be identical
        cut = int(len(bars) * 0.6)
        t_cut = bars.index[cut - 1]
        early = build_features(bars.iloc[:cut], hedge.iloc[:cut], gap.iloc[:cut], z.iloc[:cut],
                               triggers[triggers.bar < cut], minutes[minutes.index <= t_cut],
                               labels[labels.exit_time <= t_cut], cfg)
        full = features.set_index("bar").loc[early["bar"]].drop(columns="crosses_break")   # crosses_break: schedule
        pd.testing.assert_frame_equal(early.set_index("bar").drop(columns="crosses_break"), full, check_exact=False)
        print(f"leakage check passed: {len(early)} triggers before {t_cut:%Y-%m-%d} unchanged when the future is cut off")
        features.to_parquet(STRATEGY_DIR / "output" / "features.parquet")

    # quality screen per instrument at the main label setting: coverage, IC vs net ticks, AUC vs the label, per half
    table = label_table(labels, features)
    rows = []
    for instrument in INSTRUMENTS:
        for feature in FEATURES:
            row = {"instrument": instrument, "feature": feature}
            for hf in ["2021-22", "2023-Nov24"]:
                g = table[(table.instrument == instrument) & (table.half == hf)].dropna(subset=[feature])
                q = pd.qcut(g[feature].rank(method="first"), 5, labels=False)
                row[f"IC {hf}"] = g[feature].corr(g["net_ticks"], method="spearman")
                row[f"AUC {hf}"] = auc(g[feature].values, g["label"].values)
                row[f"top-bottom fifth net {hf}"] = g.net_ticks[q == 4].mean() - g.net_ticks[q == 0].mean()
            row["missing%"] = table[table.instrument == instrument][feature].isna().mean() * 100
            row["same sign both halves"] = np.sign(row["IC 2021-22"]) == np.sign(row["IC 2023-Nov24"])
            rows.append(row)
    quality = pd.DataFrame(rows)
    quality.to_csv(STRATEGY_DIR / "output" / "features_quality.csv", index=False)
    pd.set_option("display.width", 250)
    for instrument in INSTRUMENTS:
        print(f"\n{instrument}: feature vs outcome at k {LABEL_K} / stop {LABEL_STOP}x (IC = rank correlation with "
              "net ticks; AUC 0.5 = no information)\n",
              quality[quality.instrument == instrument].drop(columns="instrument").round(3).to_string(index=False))
    corr = table.drop_duplicates("bar")[FEATURES].corr(method="spearman")
    pairs = [(f1, f2, round(corr.loc[f1, f2], 2)) for i, f1 in enumerate(FEATURES) for f2 in FEATURES[i + 1:]
             if abs(corr.loc[f1, f2]) > 0.7]
    print("\nredundant pairs (|rank correlation| > 0.7):", pairs)

    # plot: mean net ticks by feature fifth, per instrument and half
    cols = 5
    fig, axes = plt.subplots(int(np.ceil(len(FEATURES) / cols)), cols, figsize=(18, 17))
    styles = {("beans", "2021-22"): (PALETTE[0], "-"), ("beans", "2023-Nov24"): (PALETTE[0], "--"),
              ("spread_beans_passive", "2021-22"): (PALETTE[3], "-"),
              ("spread_beans_passive", "2023-Nov24"): (PALETTE[3], "--")}
    for ax, feature in zip(axes.flat, FEATURES):
        for (instrument, hf), (colour, line) in styles.items():
            g = table[(table.instrument == instrument) & (table.half == hf)].dropna(subset=[feature])
            if g[feature].nunique() <= 2:                       # flags / dummies: two groups, not fifths
                means = g.groupby(g[feature] > g[feature].min()).net_ticks.mean().values
                x = [1, 5][:len(means)]
            else:
                q = pd.qcut(g[feature].rank(method="first"), 5, labels=False)
                means, x = g.groupby(q).net_ticks.mean().values, np.arange(1, 6)
            ax.plot(x, means, line, color=colour, marker="o", markersize=3,
                    label=f"{instrument.replace('_beans_passive', '')} {hf}")
        ax.axhline(0, color=LIGHT["muted"], linewidth=0.8)
        ax.set_title(feature, fontsize=9)
        ax.set_xticks([1, 3, 5], ["low", "", "high"], fontsize=7)
    for ax in axes.flat[len(FEATURES):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Mean net ticks per trade by feature fifth (lowest -> highest), triggers only, label k {LABEL_K} / "
                 f"stop {LABEL_STOP}x. A feature is useful if the line slopes the same way in both halves", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(STRATEGY_DIR / "output" / "features_by_fifth.png", dpi=100)
    print("saved features.parquet, features_quality.csv, features_by_fifth.png")

    # ---- the model, walk-forward ----
    predictions, coefficients = walk_forward(table)
    predictions.to_parquet(STRATEGY_DIR / "output" / "meta_predictions.parquet")
    coefficients.to_csv(STRATEGY_DIR / "output" / "meta_coefficients.csv", index=False)

    accuracy = []
    blocks = list(predictions.groupby(["instrument", "fold"]))
    blocks += [((instrument, "all"), g) for instrument, g in predictions.groupby("instrument")]
    c_used = coefficients.set_index(["instrument", "fold"])["C"]
    for (instrument, fold), g in blocks:
        accuracy.append({"instrument": instrument, "block": fold, "n": len(g), "win% actual": g.label.mean() * 100,
                         "AUC": auc(g.p.values, g.label.values), "log loss model": log_loss(g.label, g.p),
                         "log loss base rate": log_loss(g.label, g.base_rate), "C": c_used.get((instrument, fold))})
    accuracy = pd.DataFrame(accuracy)
    print("\nout-of-sample accuracy (AUC 0.5 = no information; the model's log loss must be BELOW the base rate's):\n",
          accuracy.round(3).to_string(index=False))

    decile = predictions.groupby("instrument").p.transform(lambda p: pd.qcut(p.rank(method="first"), 10, labels=False))
    calibration = predictions.assign(decile=decile + 1).groupby(["instrument", "decile"]).agg(
        p_mean=("p", "mean"), win_actual=("label", "mean"), net=("net_ticks", "mean"), n=("label", "size"))
    print("\ncalibration by predicted-probability decile (p_mean should match win_actual; net in ticks):\n",
          calibration.round(3).unstack("instrument").to_string())

    stability = []
    for instrument, g in coefficients.groupby("instrument"):
        for feature in MODEL_FEATURES:
            c = g[feature]
            stability.append({"instrument": instrument, "feature": feature, "mean": c.mean(), "min": c.min(),
                              "max": c.max(), "blocks same sign": int(max((c > 0).sum(), (c < 0).sum())),
                              "mean / std": c.mean() / c.std()})
    stability = pd.DataFrame(stability)
    stability.to_csv(STRATEGY_DIR / "output" / "meta_stability.csv", index=False)
    print(f"\ncoefficients across the {len(WALK_FORWARD_STARTS)} blocks (log-odds per 1 standard deviation):\n",
          stability.pivot(index="feature", columns="instrument", values=["mean", "blocks same sign", "mean / std"])
          .loc[MODEL_FEATURES].round(3).to_string())

    weeks = (predictions.time.max() - predictions.time.min()).days / 7
    all_dates = pd.Index(np.unique(chicago_date(pd.DatetimeIndex(predictions.time)).values))
    beans, spread = predictions[predictions.instrument == "beans"], predictions[predictions.instrument != "beans"]
    rules = [trade_stats("every trigger, beans", beans, all_dates, weeks),
             trade_stats("every trigger, spread", spread, all_dates, weeks),
             trade_stats("rule: Europe beans only", beans[beans.session == "europe"], all_dates, weeks),
             trade_stats("rule: Europe spread only", spread[spread.session == "europe"], all_dates, weeks)]
    for threshold in P_THRESHOLDS:
        rules.append(trade_stats(f"beans, p > {threshold}", beans[beans.p > threshold], all_dates, weeks))
        rules.append(trade_stats(f"spread, p > {threshold}", spread[spread.p > threshold], all_dates, weeks))
    wide = predictions.pivot(index="bar", columns="instrument", values="ev")
    best = predictions.set_index(["bar", "instrument"]).loc[list(zip(wide.index, wide.idxmax(axis=1)))].reset_index()
    best["best_ev"] = wide.max(axis=1).values
    for threshold in EV_THRESHOLDS:
        chosen = best[best.best_ev > threshold]
        rules.append(trade_stats(f"router, expected value > {threshold}", chosen, all_dates, weeks)
                     | {"spread share%": (chosen.instrument != "beans").mean() * 100})
    rules = pd.DataFrame(rules)
    rules.to_csv(STRATEGY_DIR / "output" / "meta_rules.csv", index=False)
    print(f"\nout of sample {all_dates[0]:%Y-%m-%d} to {all_dates[-1]:%Y-%m-%d} ({weeks:.0f} weeks), net of costs:\n",
          rules.round(2).to_string(index=False))
    for instrument in INSTRUMENTS:
        g = predictions[(predictions.instrument == instrument) & (predictions.p > 0.54)]
        print(f"\n{instrument}, p > 0.54: net per trade and count by half and session\n",
              g.pivot_table(index="half", columns="session", values="net_ticks", aggfunc=["mean", "size"])
              .round(2).to_string())

    # plot 1: coefficient per block - a stable feature stays on one side of zero
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), sharey=True)
    y = np.arange(len(MODEL_FEATURES))
    for ax, instrument in zip(axes, INSTRUMENTS):
        g = coefficients[coefficients.instrument == instrument].reset_index(drop=True)
        ax.barh(y, g[MODEL_FEATURES].mean(), color=LIGHT["muted"], alpha=0.35, height=0.6, label="mean of blocks")
        for j, row in g.iterrows():
            ax.scatter(row[MODEL_FEATURES].astype(float), y + (j - 2.5) * 0.08, s=12, color=PALETTE[j % len(PALETTE)],
                       label=f"block from {row.fold}")
        ax.axvline(0, color=LIGHT["ink"], linewidth=0.8)
        ax.set_title(f"{instrument}: coefficient in each walk-forward block", fontsize=10)
        ax.set_xlabel("log-odds per 1 standard deviation of the feature")
    axes[0].set_yticks(y, MODEL_FEATURES, fontsize=8)
    axes[0].invert_yaxis()
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Which features to keep: a useful feature stays on one side of zero in every block", fontsize=11)
    fig.tight_layout()
    fig.savefig(STRATEGY_DIR / "output" / "meta_coefficients.png", dpi=100)

    # plot 2: calibration, net per trade and trades a week by threshold
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for instrument, short, colour in zip(INSTRUMENTS, ["beans", "spread"], [PALETTE[0], PALETTE[3]]):
        cal = calibration.loc[instrument]
        axes[0].plot(cal.p_mean, cal.win_actual, "o-", color=colour, label=instrument)
        by_threshold = rules[rules.rule.str.startswith(f"{short}, p >")]
        axes[1].plot(P_THRESHOLDS, by_threshold["net per trade"], "o-", color=colour, label=f"{short}, p > threshold")
        axes[2].plot(P_THRESHOLDS, by_threshold["per week"], "o-", color=colour, label=short)
        every = rules.loc[rules.rule == f"every trigger, {short}", "net per trade"].iloc[0]
        axes[1].axhline(every, color=colour, linestyle=":", linewidth=1, label=f"{short}, every trigger")
    europe = rules.loc[rules.rule == "rule: Europe spread only", "net per trade"].iloc[0]
    axes[1].axhline(europe, color=LIGHT["ink"], linestyle="--", linewidth=1, label="rule: Europe spread only")
    lims = [calibration.p_mean.min(), calibration.p_mean.max()]
    axes[0].plot(lims, lims, color=LIGHT["muted"], linestyle="--", linewidth=0.8, label="perfect calibration")
    axes[0].set_xlabel("predicted win probability (decile mean)"); axes[0].set_ylabel("actual win share")
    axes[0].set_title("Calibration, out of sample", fontsize=10)
    axes[1].axhline(0, color=LIGHT["muted"], linewidth=0.8)
    axes[1].set_xlabel("take trades with p above"); axes[1].set_ylabel("net ticks per trade")
    axes[1].set_title("Net per trade by threshold vs simple rules", fontsize=10)
    axes[2].set_xlabel("take trades with p above"); axes[2].set_ylabel("trades per week")
    axes[2].set_title("Trades a week left", fontsize=10)
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle("Meta-model, walk-forward out of sample 2022 - Nov 2024 (every prediction fitted on earlier data only)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(STRATEGY_DIR / "output" / "meta_results.png", dpi=100)
    print("saved meta_predictions.parquet, meta_coefficients / meta_stability / meta_rules .csv, meta_*.png")

    # ---- which stop: the same model refitted on labels with stop 0.5x / 0.75x / 1x / 1.25x the target (k 1.0) ----
    # Compared at the SAME trade count (top x% of triggers by p), since p means something different per label.
    # (hit_rate stays the one from the main label - any past outcome is a fair feature.)
    compare = []
    for stop_mult in STOP_COMPARE:
        preds, _ = walk_forward(label_table(labels, features, stop_mult))
        for instrument, g in preds.groupby("instrument"):
            win, loss = g[g.label == 1].net_ticks.mean(), g[g.label == 0].net_ticks.mean()
            row = {"stop x target": stop_mult, "instrument": instrument, "AUC": auc(g.p.values, g.label.values),
                   "win% all": g.label.mean() * 100, "breakeven win%": -loss / (win - loss) * 100,
                   "mean win": win, "mean loss": loss, "net all": g.net_ticks.mean()}
            for share in TOP_SHARES:
                top = g[g.p >= g.p.quantile(1 - share)]
                row[f"net top {share:.0%}"] = top.net_ticks.mean()
                row[f"t top {share:.0%}"] = top.net_ticks.mean() / top.net_ticks.sem()
            europe = g[g.session == "europe"]
            row["net Europe only"] = europe.net_ticks.mean()
            compare.append(row)
    compare = pd.DataFrame(compare)
    compare.to_csv(STRATEGY_DIR / "output" / "meta_stop_compare.csv", index=False)
    print(f"\nwhich stop (target {LABEL_K} sigma_1h, out of sample, net ticks per trade; top x% = the triggers with the "
          f"highest p, ~{len(predictions) / 2 / weeks:.0f} triggers a week per instrument in total):\n",
          compare.round(2).to_string(index=False))
