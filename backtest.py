"""Step 4: signal at close of bar t, enter at open of t+1, exit at close of t+1."""
import sys
import numpy as np
import pandas as pd

FEATURES = ("keltner_rev_5", "skew_fade_5")
TICK = 0.0025  # CBOT soybeans (ZS): 1/4 cent per bushel = $12.50 per contract
COSTS_TICKS = (0, 0.5, 1)  # round-trip cost per trade
WINDOWS = {"full": (None, None), "blog Mar22-Feb26": ("2022-03-01", "2026-02-28 23:59")}


def run(df: pd.DataFrame, signal: pd.Series, cost_ticks: float = 0.0) -> pd.Series:
    trade_ret = (np.log(df.close) - np.log(df.open)).shift(-1)  # t+1 bar, aligned to signal bar t
    pos = np.sign(signal)  # long when the feature is positive, short when negative, flat on NaN/0
    cost = cost_ticks * TICK / df.open.shift(-1)
    return (pos * trade_ret - (pos != 0) * cost).dropna()


def stats(r: pd.Series) -> dict:
    monthly = r.groupby(r.index.to_period("M")).sum()  # log returns add up within a month
    eq = r.cumsum()
    return {"bp_per_trade": r.mean() * 1e4, "sharpe_monthly_ann": monthly.mean() / monthly.std() * np.sqrt(12),
            "months_pos_%": (monthly > 0).mean() * 100, "total_logret_%": eq.iloc[-1] * 100,
            "max_dd_%": (eq - eq.cummax()).min() * 100}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "soybean_30m_features.parquet"
    df = pd.read_parquet(path)

    # self-check: a signal equal to the sign of the future bar must win every trade
    oracle = run(df, (df.close - df.open).shift(-1))
    assert (oracle[oracle != 0] > 0).all()

    pd.set_option("display.width", 200)
    print(f"{path}: {len(df)} bars, 1 tick = {np.median(TICK / df.open) * 1e4:.2f} bp (median)\n")
    for f in FEATURES:
        rows = {}
        for w, (a, b) in WINDOWS.items():
            for c in COSTS_TICKS:
                rows[(w, c)] = stats(run(df, df[f], c).loc[a:b])
        print(f"=== {f} ===\n", pd.DataFrame(rows).T.rename_axis(["window", "cost_ticks"]).round(2).to_string())
        g = run(df, df[f])
        print("gross by year (bp/trade):", (g.groupby(g.index.year).mean() * 1e4).round(2).to_dict())
        q = pd.qcut(df[f], 5, labels=["Q1 (most neg)", "Q2", "Q3", "Q4", "Q5 (most pos)"])
        nxt = (np.log(df.close) - np.log(df.open)).shift(-1)
        print("next-bar return by signal quintile (bp):", (nxt.groupby(q).mean() * 1e4).round(2).to_dict(), "\n")
        pd.DataFrame({c: run(df, df[f], c).cumsum() for c in COSTS_TICKS}).to_csv(f"equity_{path.split('_')[1]}_{f}.csv")
