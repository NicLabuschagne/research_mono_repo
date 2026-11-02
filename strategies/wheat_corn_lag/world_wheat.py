"""
world_wheat.py
--------------
Signal 1c: CBOT wheat against WORLD wheat (KC HRW, Euronext/Matif milling wheat), with corn, at DAILY frequency
(the only frequency available for KC and Matif). Development data only (cut at the holdout start before use).
Saved to runs/<time>_world_wheat/.

    python -m strategies.wheat_corn_lag.world_wheat

Data (daily closes, aligned on days ALL markets traded, so no stale prices)
  ZW     LSE daily front month (rolls in it: possible-roll days are cut, see below)
  ZC     roll-free daily returns from Databento contract closes (balance_sheet.roll_free_corn_returns)
  KC     data/raw/kc_hrw_wheat_daily.csv (Investing-style continuous; cents/bu)
  Matif  data/raw/matif_milling_wheat_daily.csv (EUR/t), converted to USD with EURUSD at the Matif close
         (18:30 Paris); Matif closes before CBOT (13:20 CT), so its same-day close is known at the CBOT close
Rolls, by CALENDAR only (each provider's own pattern, found from the jumps in the data):
  ZW     wheat FND-10 .. FND+3 business days (as strategy.py)       KC  7th-18th of Mar, May, Jul, Sep, Dec
  Matif  8th-15th of Mar, May, Sep, Dec
  On an instrument's possible-roll days its daily return counts as 0 in the FEATURES (we do not see its move
  that day). Wheat TARGETS (forward returns) that would cross a ZW possible-roll day are dropped.
  Two unexplained outliers stay in and are flagged: Matif 2021-08-25 (-10.6%), KC 2020-05-08 (+5.4%).

Features (all known at the close of day t)
  betas    multi-factor OLS of wheat's daily return on the set's factor returns, 120 days ending t-1
  own_ze   wheat's move over L days minus sum(beta x factor move over L), / its actual sd of L-day moves (250 days,
           lagged); composite = average over L = 2, 3, 5 days (the 1.5-3.5 session plateau of signal 1a)
  sets     c corn (= signal 1a at daily frequency), k KC, m Matif, ck, cm, km, ckm; control: wheat alone
Signal = -own (fade). Pass criteria fixed before the run (as layer1.py): 2-day horizon, NW t >= 3, same sign in
>= 5 of 6 years, deciles monotonic >= 0.8. Plus: does the world-wheat signal add anything BEYOND corn?
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from core_quant.config import ROOT, save_run
from core_quant.theme import set_theme
from core_quant.validation import nw_t_stat
from strategies.wheat_corn_lag.balance_sheet import CORN_DAILY, WHEAT_DAILY, daily_closes, roll_free_corn_returns
from strategies.wheat_corn_lag.features import lagged_sd, rolling_betas
from strategies.wheat_corn_lag.strategy import STRATEGY_DIR, load, wheat_first_notice_days

BETA_WINDOW, SCALE_WINDOW = 120, 250
LOOKBACKS = [2, 3, 5]
HORIZONS = [1, 2, 3, 4]
PRIMARY = 2
SETS = {"c": ["c"], "k": ["k"], "m": ["m"], "ck": ["c", "k"], "cm": ["c", "m"], "km": ["k", "m"], "ckm": ["c", "k", "m"]}
LABEL = {"c": "corn (1a, daily)", "k": "KC", "m": "Matif", "ck": "corn + KC", "cm": "corn + Matif",
         "km": "KC + Matif", "ckm": "corn + KC + Matif"}
PASS_T, PASS_YEARS, PASS_MONOTONIC = 3.0, 5, 0.8


def investing_csv(name: str) -> pd.Series:
    frame = pd.read_csv(ROOT / "data" / "raw" / name, encoding="utf-8-sig")
    close = pd.to_numeric(frame["Price"].astype(str).str.replace(",", ""), errors="coerce")
    return pd.Series(close.values, index=pd.to_datetime(frame["Date"], format="%m/%d/%Y")).sort_index()


def eurusd_at_matif_close(days: pd.DatetimeIndex) -> pd.Series:
    fx = pd.read_csv(ROOT / "data" / "raw" / "eur_usd_15m.csv")
    fx = pd.Series(fx["close"].values, index=pd.to_datetime(fx["timestamp"], utc=True) + pd.Timedelta(minutes=15))
    fx = fx[~fx.index.duplicated(keep="last")].sort_index()
    close_time = (pd.DatetimeIndex(days) + pd.Timedelta(hours=18, minutes=30)).tz_localize("Europe/Paris").tz_convert("UTC")
    positions = fx.index.searchsorted(close_time, side="right") - 1          # last FX bar ending by the Matif close
    return pd.Series(np.where(positions >= 0, fx.values[np.maximum(positions, 0)], np.nan), index=days)


def possible_roll_days(days: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    zw = np.zeros(len(days), dtype=bool)
    for fnd in wheat_first_notice_days(days[0].year, days[-1].year + 1).tz_localize(None):
        zw |= (days >= fnd - pd.offsets.BDay(10)) & (days <= fnd + pd.offsets.BDay(3))
    kc = days.month.isin([3, 5, 7, 9, 12]) & (days.day >= 7) & (days.day <= 18)
    matif = days.month.isin([3, 5, 9, 12]) & (days.day >= 8) & (days.day <= 15)
    return {"zw": zw, "k": np.asarray(kc), "m": np.asarray(matif)}


def daily_panel(cfg, end: pd.Timestamp) -> pd.DataFrame:
    wheat = daily_closes(WHEAT_DAILY, end)
    corn = daily_closes(CORN_DAILY, end)
    corn_returns = roll_free_corn_returns(cfg, corn)
    kc = investing_csv("kc_hrw_wheat_daily.csv")
    matif = investing_csv("matif_milling_wheat_daily.csv")
    days = wheat.index.intersection(kc.index).intersection(matif.index)
    days = days[(days < end.tz_localize(None)) & (days >= pd.Timestamp("2020-01-30"))]
    panel = pd.DataFrame({"W": wheat[days], "K": kc[days], "M_usd": matif[days] * eurusd_at_matif_close(days)}, index=days)
    corn_index = corn_returns.fillna(0).cumsum()
    panel["index_c"] = corn_index.reindex(days).values
    rolls = possible_roll_days(days)
    panel["zw_roll"] = rolls["zw"]
    panel["ret_w_raw"] = np.log(panel["W"]).diff()
    panel["ret_w"] = panel["ret_w_raw"].where(~panel["zw_roll"], 0.0)          # features: a possible roll counts as 0
    panel["ret_c"] = panel["index_c"].diff()
    panel["ret_k"] = np.log(panel["K"]).diff().where(~rolls["k"], 0.0)
    panel["ret_m"] = np.log(panel["M_usd"]).diff().where(~rolls["m"], 0.0)
    return panel.iloc[1:]


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    features = {}
    cumulative = {f: panel[f"ret_{f}"].cumsum() for f in ["w", "c", "k", "m"]}
    for lookback in LOOKBACKS:
        move = {f: cumulative[f] - cumulative[f].shift(lookback) for f in cumulative}
        own_w = move["w"]
        features[f"wheat_ze_{lookback}"] = own_w / lagged_sd(own_w, SCALE_WINDOW)
    for name, factors in SETS.items():
        betas = rolling_betas(panel, factors, BETA_WINDOW)                        # rows t-120 .. t-1
        for lookback in LOOKBACKS:
            move = {f: cumulative[f] - cumulative[f].shift(lookback) for f in cumulative}
            explained = sum(betas[f] * move[f] for f in factors)
            own = move["w"] - explained
            features[f"own_ze_{name}_{lookback}"] = own / lagged_sd(own, SCALE_WINDOW)
            features[f"explained_ze_{name}_{lookback}"] = explained / lagged_sd(explained, SCALE_WINDOW)
    return pd.DataFrame(features, index=panel.index)


def targets(panel: pd.DataFrame) -> pd.DataFrame:
    """Forward wheat returns from raw closes; dropped when the window touches a ZW possible-roll day."""
    log_w, roll = np.log(panel["W"]).values, panel["zw_roll"].values
    out = {}
    for h in HORIZONS:
        values = np.full(len(panel), np.nan)
        for t in range(len(panel) - h):
            if not roll[t + 1:t + h + 1].any():
                values[t] = log_w[t + h] - log_w[t]
        out[f"fwd_{h}"] = values
    return pd.DataFrame(out, index=panel.index)


def composite(features: pd.DataFrame, prefix: str) -> pd.Series:
    return -features[[f"{prefix}_{l}" for l in LOOKBACKS]].mean(axis=1, skipna=False)


def truncation_check(cfg, full_features: pd.DataFrame) -> float:
    """Rebuild everything with data cut at a date: features before it must not change (look-ahead test)."""
    cut = pd.Timestamp("2023-06-30", tz="UTC")
    short = build_features(daily_panel(cfg, cut))
    common = short.index
    left, right = full_features.loc[common], short
    if (left.isna() != right.isna()).any().any():
        return np.inf
    return float((left - right).abs().max().max())


# ------------------------------------------------------------------ charts
def ic_chart(table: pd.DataFrame, years: pd.DataFrame, names: list[str]) -> plt.Figure:
    figure, (decay, by_year) = plt.subplots(1, 2, figsize=(15, 4.8))
    for name in names:
        block = table[table["signal"] == name]
        decay.plot(block["h"], block["ic"], marker="o", label=name)
    decay.axhline(0, color="grey", lw=0.8)
    decay.set_xlabel("forward horizon, trading days")
    decay.set_ylabel("rank IC")
    decay.set_title("IC by horizon: wheat vs corn, KC, Matif (daily)", loc="left")
    decay.legend(fontsize=7)
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        by_year.bar(np.arange(len(years)) + (i - (len(names) - 1) / 2) * width, years[name], width=width * 0.95, label=name)
    by_year.set_xticks(np.arange(len(years)), years.index)
    by_year.axhline(0, color="grey", lw=0.8)
    by_year.set_title(f"IC by year, {PRIMARY}-day forward", loc="left")
    figure.tight_layout()
    return figure


def betas_chart(panel: pd.DataFrame) -> plt.Figure:
    betas = rolling_betas(panel, ["c", "k", "m"], BETA_WINDOW)
    figure, axis = plt.subplots(figsize=(12, 4))
    for factor, label in [("c", "corn"), ("k", "KC"), ("m", "Matif (USD)")]:
        axis.plot(betas.index, betas[factor], lw=1.2, label=label)
    axis.axhline(0, color="grey", lw=0.8)
    axis.set_title("Rolling 120-day betas of CBOT wheat on corn, KC and Matif together", loc="left")
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


if __name__ == "__main__":
    set_theme()
    cfg = load()
    holdout = pd.Timestamp(cfg.split.holdout_start, tz="UTC")
    panel = daily_panel(cfg, holdout)
    features = build_features(panel)
    frame = pd.DataFrame({"control: wheat alone": composite(features, "wheat_ze"),
                          **{LABEL[s]: composite(features, f"own_ze_{s}") for s in SETS}}).join(targets(panel))
    names = ["control: wheat alone"] + [LABEL[s] for s in SETS]
    common = frame[names].notna().all(axis=1)
    frame = frame[common]

    rows = []
    for h in HORIZONS:
        for name in names:
            block = frame[[name, f"fwd_{h}"]].dropna()
            rows.append({"signal": name, "h": h, "ic": spearmanr(block[name], block[f"fwd_{h}"])[0],
                         "t_nw": float(nw_t_stat(block, [name], f"fwd_{h}", lags=h).loc[name, "t_nw"]), "n": len(block)})
    table = pd.DataFrame(rows)
    years = pd.DataFrame({name: frame[[name, f"fwd_{PRIMARY}"]].dropna().groupby(lambda d: d.year)
                          .apply(lambda b: spearmanr(b.iloc[:, 0], b.iloc[:, 1])[0]) for name in names})
    verdict = []
    for name in names:
        row = table[(table["signal"] == name) & (table["h"] == PRIMARY)].iloc[0]
        block = frame[[name, f"fwd_{PRIMARY}"]].dropna()
        means = block[f"fwd_{PRIMARY}"].groupby(pd.qcut(block[name], 10, labels=False)).mean()
        monotonic = spearmanr(means.index, means.values)[0]
        same = int((np.sign(years[name]) == np.sign(row["ic"])).sum())
        verdict.append({"signal": name, "ic": row["ic"], "t_nw": row["t_nw"], "years_same_sign": f"{same}/{len(years)}",
                        "decile_monotonic": monotonic,
                        "PASS": row["t_nw"] >= PASS_T and same >= PASS_YEARS and monotonic >= PASS_MONOTONIC})
    verdict = pd.DataFrame(verdict)

    # does world wheat add beyond corn? forward return on the corn signal and the KC+Matif signal together
    y = frame[f"fwd_{PRIMARY}"] / frame[f"fwd_{PRIMARY}"].std()
    joint = pd.DataFrame({"y": y, "corn": frame[LABEL["c"]], "world": frame[LABEL["km"]]}).dropna()
    fit = sm.OLS(joint["y"], sm.add_constant(joint[["corn", "world"]] / joint[["corn", "world"]].std())).fit(
        cov_type="HAC", cov_kwds={"maxlags": PRIMARY})
    split = {}
    for name in ["c", "km", "ckm"]:
        data = pd.DataFrame({"y": y, "explained": features[f"explained_ze_{name}_3"], "own": features[f"own_ze_{name}_3"]}).dropna()
        f = sm.OLS(data["y"], sm.add_constant(data[["explained", "own"]])).fit(cov_type="HAC", cov_kwds={"maxlags": PRIMARY})
        split[LABEL[name]] = {"a_explained": f.params["explained"], "t_a": f.tvalues["explained"],
                              "b_own": f.params["own"], "t_b": f.tvalues["own"]}
    betas = rolling_betas(panel, ["c", "k", "m"], BETA_WINDOW)
    r2 = {}
    for name, factors in SETS.items():
        b = rolling_betas(panel, factors, BETA_WINDOW)
        predicted = sum(b[f] * panel[f"ret_{f}"] for f in factors)
        pair = pd.concat([panel["ret_w"], predicted], axis=1).dropna()
        r2[LABEL[name]] = 1 - (pair.iloc[:, 0] - pair.iloc[:, 1]).var() / pair.iloc[:, 0].var()
    leak = truncation_check(cfg, features)

    pd.set_option("display.width", 220)
    print(f"daily panel: {len(panel)} days ({panel.index[0].date()} -> {panel.index[-1].date()}); signals scored on {len(frame)} days")
    print(f"truncation self-check (features with data cut at 2023-06-30 vs full): max diff {leak:.2e}  -> {'PASS' if leak < 1e-9 else 'FAIL'}")
    print("\nout-of-sample R2 of wheat's daily return on the factors (betas from the past 120 days):")
    print({k: round(v, 3) for k, v in r2.items()})
    print(f"average betas when all three are used: {betas.mean().round(2).to_dict()}")
    print("\nrank IC by horizon (trading days)")
    print(table.pivot(index="signal", columns="h", values="ic").round(3).to_string())
    print("\nNewey-West t")
    print(table.pivot(index="signal", columns="h", values="t_nw").round(2).to_string())
    print(f"\nverdict at {PRIMARY} days")
    print(verdict.round(3).to_string(index=False))
    print(f"\nsignal correlation, corn vs KC+Matif: {frame[LABEL['c']].corr(frame[LABEL['km']]):.2f}")
    print(f"joint fit, forward {PRIMARY}-day return on BOTH signals (per sd): corn {fit.params['corn']:+.3f} (t {fit.tvalues['corn']:+.2f}),"
          f" world wheat {fit.params['world']:+.3f} (t {fit.tvalues['world']:+.2f})")
    print("\ndecomposition at L = 3 days, y = a x explained + b x own (negative = reverts)")
    print(pd.DataFrame(split).T.round(3).to_string())

    shown = ["control: wheat alone", LABEL["c"], LABEL["k"], LABEL["m"], LABEL["km"], LABEL["ckm"]]
    figures = {"ic": ic_chart(table, years, shown), "betas": betas_chart(panel)}
    run_dir = save_run(STRATEGY_DIR, "world_wheat", cfg, {"days": len(panel), "truncation_max_diff": leak},
                       {"ic": table, "ic_by_year": years, "verdict": verdict, "decomposition": pd.DataFrame(split).T}, figures)
    print(f"\nsaved to {run_dir}")
