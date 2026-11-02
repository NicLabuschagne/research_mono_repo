"""Step 3: predictive power of each feature vs forward returns (rank IC)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from features import keltner_reversion

HORIZONS = (1, 2, 4, 8, 16, 34)  # bars; 34 bars ~ one trading day


def zrank(s: pd.Series) -> pd.Series:
    r = s.rank()
    return (r - r.mean()) / r.std()


def rank_ic(f: pd.Series, r: pd.Series, h: int):
    """Full-sample rank IC + Newey-West t-stat. Ranks are global: re-ranking inside
    short windows (e.g. per month) mechanically inflates IC for persistent features."""
    x = pd.concat([f, r], axis=1, keys=["f", "r"]).dropna()
    zf, zr = zrank(x.f), zrank(x.r)
    fit = sm.OLS(zr.values, sm.add_constant(zf.values)).fit(cov_type="HAC", cov_kwds={"maxlags": 2 * h + 5})
    return (zf * zr).mean(), fit.tvalues[1], (zf * zr).groupby(x.index.year).mean()


def fwd_returns(close: pd.Series) -> pd.DataFrame:
    # forward return from close t to close t+h; the signal is known at close t
    logc = np.log(close)
    return pd.DataFrame({h: logc.shift(-h) - logc for h in HORIZONS})


if __name__ == "__main__":
    df = pd.read_parquet("soybean_30m_features.parquet")
    feats = list(df.filter(regex="keltner|skew").columns)
    fwd = fwd_returns(df["close"])
    assert np.isclose(fwd[1].iloc[0], np.log(df.close.iloc[1] / df.close.iloc[0]))

    # self-check: on a shuffled-return random walk the slowest feature must show no significant IC
    rng = np.random.default_rng(0)
    p = np.exp(np.log(df.close.iloc[0]) + np.cumsum(rng.permutation(np.log(df.close).diff().fillna(0).values)))
    rw = pd.DataFrame({"close": p, "high": p * df.high / df.close, "low": p * df.low / df.close}, index=df.index)
    for h in (1, 34):
        _, t, _ = rank_ic(keltner_reversion(rw, 120), fwd_returns(rw.close)[h], h)
        assert abs(t) < 3, f"null test failed at h={h}: t={t:.2f}"

    rows, yearly = [], {}
    for f in feats:
        for h in HORIZONS:
            ic, t, yr = rank_ic(df[f], fwd[h], h)
            rows.append({"feature": f, "h": h, "IC": ic, "t_NW": t, "years_pos": (yr > 0).sum()})
            yearly[(f, h)] = yr
    res = pd.DataFrame(rows)
    res.to_csv("soybean_ic.csv", index=False)
    yearly = pd.DataFrame(yearly).T.rename_axis(["feature", "h"])
    yearly.to_csv("soybean_ic_by_year.csv")

    pd.set_option("display.width", 200)
    print("IC x100\n", res.pivot(index="feature", columns="h", values="IC").mul(100).round(2).to_string(), "\n")
    print("Newey-West t\n", res.pivot(index="feature", columns="h", values="t_NW").round(2).to_string(), "\n")
    print("years positive (of 5)\n", res.pivot(index="feature", columns="h", values="years_pos").to_string(), "\n")
    print("IC x100 by year\n", yearly.mul(100).round(2).to_string())
