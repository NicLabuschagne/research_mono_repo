"""
feature_screen.py
-----------------
Step 3a: which features, measured at the signal bar, tell good trades from bad ones?

Trades : the adopted baseline (config.yaml), development only.
Features: meta_model_reference.build_features (regime, signal shape, legs, liquidity, calendar)
          plus three new ones
            hours_left     : time from the signal to the entry session's close (the exit now
                             forces a close there, so late signals have less time to revert)
            beans_stretch_5: the bean leg's own 5-bar Keltner stretch, signed so > 0 means beans
                             are stretched in the same direction as the dislocation
            beans_skew_5   : the bean leg's 5-bar skew, signed the same way
Outcomes: net / sigma (scale-free signal quality) and win (net > 0).

Pre-registered rule (written before looking): a feature passes the screen if, on 2021-22,
|t| >= 2 on net/sigma or on win AND its IC has the same sign in 2021 and in 2022.
Only those are then checked, once, on 2023 - Nov 2024.

Regime is judged separately in cents (the cost is fixed in cents, so net/sigma hides it).

    python -m strategies.crush_strategy.feature_screen
"""
import numpy as np
import pandas as pd

from core_quant.config import save_run
from strategies.crush_strategy.kalman_calibration import FIT_SPLIT
from strategies.crush_strategy.meta_model_reference import build_features
from strategies.crush_strategy.strategy import STRATEGY_DIR, TICK_COST_CENTS, load, run

STRETCH_BARS = 5


def extra_features(bars: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    signal = trades["entry_bar"].values - 1
    sign_z = np.sign(trades["entry_z"].values)
    session_close = pd.Series(bars.index, index=bars.index).groupby(bars["session_id"]).transform("max")
    f = pd.DataFrame(index=trades.index)
    f["hours_left"] = (session_close.values[signal] - bars.index.values[signal]) / np.timedelta64(1, "h")
    close = bars["ZS_adj_close"]
    # bars carry no high/low, so the average absolute close-to-close move stands in for ATR
    atr = close.diff().abs().rolling(STRETCH_BARS).mean()
    stretch = (close - close.rolling(STRETCH_BARS).mean()) / atr.replace(0, np.nan)
    f["beans_stretch_5"] = sign_z * stretch.values[signal]
    f["beans_skew_5"] = sign_z * close.rolling(STRETCH_BARS).skew().values[signal]
    return f


def ic_with_t(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    ok = x.notna() & y.notna()
    n = int(ok.sum())
    if n < 10:
        return np.nan, np.nan, n
    ic = x[ok].rank().corr(y[ok].rank())
    return ic, ic * np.sqrt(n - 2) / np.sqrt(1 - ic ** 2), n


def screen(features: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    net = trades["gross_cents"] - trades["cost_cents"]
    outcomes = {"net_sigma": net / trades["sigma_cents"], "win": (net > 0).astype(float)}
    date = pd.to_datetime(trades["signal_date"])
    windows = {"choose": date < FIT_SPLIT, "y2021": date.dt.year == 2021, "y2022": date.dt.year == 2022,
               "confirm": date >= FIT_SPLIT}
    rows = []
    for name in features.columns:
        row = {"feature": name}
        for w, mask in windows.items():
            for label, y in outcomes.items():
                ic, t, n = ic_with_t(features.loc[mask, name], y[mask])
                row[f"{w}_ic_{label}"], row[f"{w}_t_{label}"] = ic, t
            row[f"{w}_n"] = n
        same_sign = np.sign(row["y2021_ic_net_sigma"]) == np.sign(row["y2022_ic_net_sigma"])
        row["passes_screen"] = bool(same_sign and max(abs(row["choose_t_net_sigma"]), abs(row["choose_t_win"])) >= 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("choose_t_net_sigma", key=np.abs, ascending=False)


def buckets(feature: pd.Series, trades: pd.DataFrame, n_buckets: int = 3) -> pd.DataFrame:
    """Net cents per trade by bucket; bucket edges from the choose window only."""
    net = trades["gross_cents"] - trades["cost_cents"]
    date = pd.to_datetime(trades["signal_date"])
    choose = date < FIT_SPLIT
    edges = np.unique(np.r_[-np.inf, feature[choose].quantile(np.linspace(0, 1, n_buckets + 1)[1:-1]).values, np.inf])
    bucket = pd.cut(feature, edges, labels=[f"B{i + 1}" for i in range(len(edges) - 1)])   # binary features: 2 buckets
    window = np.where(choose, "choose 2021-22", "confirm 2023-24")
    table = net.groupby([window, bucket], observed=False).agg(["size", "mean", "sem"]).round(2)
    table.columns = ["trades", "net_c", "se"]
    table.attrs["edges"] = edges[1:-1].round(3).tolist()
    return table


if __name__ == "__main__":
    cfg = load()
    bars, trades = run(cfg)
    trades = trades[trades["sample"] == "development"].reset_index(drop=True)
    trades["cost_cents"] = cfg.strategy.cost_ticks * TICK_COST_CENTS
    features = build_features(bars, trades).join(extra_features(bars, trades))

    table = screen(features, trades)
    pd.set_option("display.width", 250)
    show = ["feature", "choose_n", "choose_ic_net_sigma", "choose_t_net_sigma", "choose_t_win",
            "y2021_ic_net_sigma", "y2022_ic_net_sigma", "passes_screen"]
    print(f"{len(trades)} development trades; screen on 2021-22 only\n")
    print(table[show].round(3).to_string(index=False))
    survivors = table[table["passes_screen"]]
    print("\nsurvivors, checked once on 2023 - Nov 2024:")
    print(survivors[["feature", "choose_ic_net_sigma", "confirm_n", "confirm_ic_net_sigma", "confirm_t_net_sigma",
                     "confirm_t_win"]].round(3).to_string(index=False))

    regime = {name: buckets(features[name], trades) for name in ["sigma_cents"] + list(survivors["feature"])}
    for name, b in regime.items():
        print(f"\nnet cents by {name} tercile (edges from 2021-22: {b.attrs['edges']})")
        print(b.unstack(0).to_string())
    print("saved", save_run(STRATEGY_DIR, "feature_screen", cfg, {"n_trades": len(trades), "survivors": list(survivors["feature"])},
                            tables={"feature_screen": table, "trade_features": trades.join(features, rsuffix="_feature")}))
