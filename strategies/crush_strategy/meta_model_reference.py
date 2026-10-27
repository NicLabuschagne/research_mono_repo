"""
meta_model_reference.py  (not used by the strategy)
------------------------------------------------------------------
Feature engineering, IC screening and the meta-model for the overnight signal.

Meta-labelling
  The rule (strategy.py) decides when and which way to trade. A model then estimates
  p = P(trade wins) from features known at the signal, and a cost rule decides:
      expected_net = sigma * (p * avg_win_sigma - (1 - p) * avg_loss_sigma) - cost
      take the trade only if expected_net > 0

IC (information coefficient)
  Spearman rank correlation between a feature and the outcome. Rank-based, so outliers
  and scale do not dominate. t-stat = IC * sqrt(n - 2) / sqrt(1 - IC^2).
  Two outcomes: win (net > 0) and net / sigma (scale-free, removes the mechanical
  effect that bigger sigma means bigger cents either way).
  Screening on the full development set is descriptive only: picking features on the
  same data you then test on leaks the outcome into the test.

Model tested (on |z| >= 1.5 signals, for more samples): logistic regression (L2, cost = 1)
on time-of-night sine and the largest abnormal leg move. Strong in development walk-forward (features were picked there),
failed the holdout (AUC 0.45). Kept to read the method, not to trade.
"""
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from core_quant.model_registry import ModelConfig, build_model
from strategies.crush_strategy.strategy import TICK_COST_CENTS, load, run

MODEL = ModelConfig(name="logit", params={"C": 1.0}, scaler="standard")
MODEL_FEATURES = ["hour_sin", "max_abs_leg_abnormal_move"]
TEST_YEARS = [2021, 2022, 2023, 2024]
BARS_PER_HOUR = 12                     # 5-minute bars


# ---------------------------------------------------------------- feature engineering
def session_abnormal_move(bars: pd.DataFrame, leg: str) -> pd.Series:
    """A leg's return since the session open, divided by its usual size at that bar of the session
    (std of the same quantity over the previous 20 days, same session type and bar number)."""
    since_open = bars.groupby("session_id")[f"{leg}_ret"].cumsum()
    usual = since_open.groupby([bars["session_type"], bars["bar_number"]]).transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).std())
    return since_open / usual


def volume_vs_usual(bars: pd.DataFrame, leg: str) -> pd.Series:
    """Volume over the last hour vs its 20-day average at the same bar of the session."""
    last_hour = bars[f"{leg}_volume"].rolling(BARS_PER_HOUR).sum()
    usual = last_hour.groupby([bars["session_type"], bars["bar_number"]]).transform(
        lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    return last_hour / usual


def build_features(bars: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """All candidate features, measured at the signal bar (data up to its close only)."""
    signal = trades["entry_bar"].values - 1
    sign_z = np.sign(trades["entry_z"].values)            # +1: beans rich vs products
    z = bars["z"].values
    log_price = {leg: np.log(bars[f"{leg}_adj_close"].values) for leg in ["ZS", "ZM", "ZL"]}
    london = pd.DatetimeIndex(trades["signal_time"]).tz_convert("Europe/London")
    hour_angle = 2 * np.pi * (london.hour + london.minute / 60) / 24
    signal_date = pd.DatetimeIndex(pd.to_datetime(trades["signal_date"]))
    f = pd.DataFrame(index=trades.index)

    # size and regime
    f["sigma_cents"] = trades["sigma_cents"]
    f["cost_over_sigma"] = TICK_COST_CENTS / trades["sigma_cents"]
    f["abs_entry_z"] = trades["entry_z"].abs()
    # speed: growth of |z| over the last hour (fast spike vs slow drift)
    f["z_growth_1h"] = sign_z * (z[signal] - z[signal - BARS_PER_HOUR])
    # which leg widened the gap over the last 4 hours (+1 beans, -1 products)
    lookback = 4 * BARS_PER_HOUR
    beans_move = sign_z * (log_price["ZS"][signal] - log_price["ZS"][signal - lookback])
    product_move = -sign_z * (trades["beta_meal"].values * (log_price["ZM"][signal] - log_price["ZM"][signal - lookback])
                              + trades["beta_oil"].values * (log_price["ZL"][signal] - log_price["ZL"][signal - lookback]))
    f["beans_vs_products_4h"] = (beans_move - product_move) / (np.abs(beans_move) + np.abs(product_move) + 1e-12)
    # abnormal session moves since the evening open
    abnormal = np.column_stack([session_abnormal_move(bars, leg).values[signal] for leg in ["ZS", "ZM", "ZL"]])
    f["max_abs_leg_abnormal_move"] = np.nanmax(np.abs(abnormal), axis=1)
    # liquidity: the thinnest leg's volume vs normal
    volumes = np.column_stack([volume_vs_usual(bars, leg).values[signal] for leg in ["ZS", "ZM", "ZL"]])
    f["thinnest_leg_volume"] = np.nanmin(volumes, axis=1)
    # crush economics: change in oil's share of crush value over 20 days (previous days only)
    daily = bars.groupby("trading_date")[["ZM_adj_close", "ZL_adj_close"]].last()
    oil_share = 0.11 * daily["ZL_adj_close"] / (0.022 * daily["ZM_adj_close"] + 0.11 * daily["ZL_adj_close"])
    f["oil_share_change_20d"] = signal_date.map((oil_share - oil_share.shift(20)).shift(1)).values
    # time (cyclical encodings) and calendar
    f["hour_sin"], f["hour_cos"] = np.sin(hour_angle), np.cos(hour_angle)
    f["pre_us_hours"] = (london.hour >= 11).astype(int)
    f["season_sin"] = np.sin(2 * np.pi * signal_date.dayofyear / 365.25)
    f["export_sales_night"] = (signal_date.dayofweek == 3).astype(int)   # Thursday 07:30 CT release
    return f


# ---------------------------------------------------------------- IC table
def ic_table(features: pd.DataFrame, trades: pd.DataFrame, cost: float) -> pd.DataFrame:
    net = trades["gross_cents"] - cost
    outcomes = {"win": (net > 0).astype(int), "net_per_sigma": net / trades["sigma_cents"]}
    years = pd.to_datetime(trades["signal_date"]).dt.year
    rows = []
    for name in features.columns:
        x = features[name]
        valid = x.notna()
        row = {"feature": name}
        for label, y in outcomes.items():
            ic = x[valid].rank().corr(y[valid].rank())
            row[f"ic_{label}"] = ic
            row[f"t_{label}"] = ic * np.sqrt(valid.sum() - 2) / np.sqrt(1 - ic ** 2)
        yearly = [x[valid & (years == y)].rank().corr(outcomes["win"][valid & (years == y)].rank())
                  for y in sorted(years.unique())]
        row["same_sign_years"] = f"{sum(np.sign(v) == np.sign(row['ic_win']) for v in yearly)}/{len(yearly)}"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("t_win", key=np.abs, ascending=False).round(3)


# ---------------------------------------------------------------- meta-model
def walk_forward(trades: pd.DataFrame, features: pd.DataFrame, cost: float) -> pd.DataFrame:
    """Train on earlier years, test on the next; compare taking every signal vs the model."""
    data = trades.join(features[MODEL_FEATURES]).assign(
        year=pd.to_datetime(trades["signal_date"]).dt.year,
        win=(trades["gross_cents"] - cost > 0).astype(int),
        gross_sigma=trades["gross_cents"] / trades["sigma_cents"])
    results = []
    for year in TEST_YEARS:
        test = data[data["year"] == year]
        first_signal = test["signal_time"].min().tz_convert(None).normalize()
        train = data[(data["year"] < year) & (pd.to_datetime(data["exit_date"]) < first_signal)]   # purge overlap
        avg_win = train.loc[train["win"] == 1, "gross_sigma"].mean()
        avg_loss = -train.loc[train["win"] == 0, "gross_sigma"].mean()

        model = build_model(MODEL).fit(train[MODEL_FEATURES], train["win"])
        p = model.predict_proba(test[MODEL_FEATURES])[:, 1]
        take = test["sigma_cents"].values * (p * avg_win - (1 - p) * avg_loss) - cost > 0

        net = test["gross_cents"] - cost
        results.append(dict(year=year, train_trades=len(train), auc=roc_auc_score(test["win"], p),
                            take_all_trades=len(test), take_all_net_cents=net.sum(),
                            model_trades=int(take.sum()), model_net_cents=net[take].sum(),
                            coefficients=dict(zip(MODEL_FEATURES, model[-1].coef_[0].round(2)))))
    return pd.DataFrame(results).round(2)


if __name__ == "__main__":
    cfg = load()
    cfg = replace(cfg, strategy=replace(cfg.strategy, threshold=1.5))   # the meta-model was tested on |z| >= 1.5 signals
    config = cfg.strategy
    # start in 2020 so the first walk-forward fold (2021) has a training year
    bars, trades = run(cfg, start=pd.Timestamp("2020-07-01"))
    development = trades[trades["sample"] == "development"].reset_index(drop=True)
    cost = config.cost_ticks * TICK_COST_CENTS
    features = build_features(bars, development)

    pd.set_option("display.width", 200)
    print(f"IC table: {len(development)} development trades\n")
    print(ic_table(features, development, cost).to_string(index=False))
    print("\nMeta-model walk-forward:\n")
    print(walk_forward(development, features, cost).to_string(index=False))
