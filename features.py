import sys
import numpy as np
import pandas as pd

SRC = r"C:\Users\Amin\Downloads\SOYBN_USD_commodities_30m_2022-01-22_2026-10-29.parquet"
WINDOWS = (5, 20, 120)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    # Simple-mean ATR (not Wilder), so the ATR lookback matches the MA lookback exactly.
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def keltner_reversion(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Deviation from MA normalised by ATR. Robust mean reversion."""
    c = df["close"]
    ma = c.rolling(window).mean()
    atr = _atr(df["high"], df["low"], c, window)
    return -(c - ma) / atr.replace(0, np.nan)


def skew_fade(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Negative rolling skewness. Fade distributional asymmetry."""
    return -df["close"].rolling(window).skew()


def load(path: str = SRC) -> pd.DataFrame:
    df = pd.read_parquet(path).set_index("time").sort_index()
    assert df.index.is_unique, "duplicate timestamps"
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    # bins labelled by their start; hours with no trading are dropped, so bars stay session-only
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "trades": "sum"}
    return df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open"])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for w in WINDOWS:
        out[f"keltner_rev_{w}"] = keltner_reversion(df, w)
        out[f"skew_fade_{w}"] = skew_fade(df, w)
    return out


if __name__ == "__main__":
    # usage: features.py [src.parquet] [resample rule, e.g. 1h]
    rule = sys.argv[2] if len(sys.argv) > 2 else None
    bars = load(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else SRC)
    if rule:
        raw, bars = bars, resample(bars, rule)
        assert np.isclose(bars.close.iloc[-1], raw.close.iloc[-1]) and (bars.high >= bars[["open", "close"]].max(axis=1)).all()
    df = add_features(bars)
    # self-check: features at bar t use only bars <= t (truncating the future must not change them)
    cut = add_features(bars.iloc[:300])
    # atol: pandas' online rolling skew differs ~1e-4 depending on series start (checked vs scipy, no drift)
    pd.testing.assert_frame_equal(df.iloc[:300], cut, check_exact=False, atol=1e-3)
    df.to_parquet(f"soybean_{rule or '30m'}_features.parquet")
    print(df.filter(regex="keltner|skew").describe().T.round(3).to_string())
