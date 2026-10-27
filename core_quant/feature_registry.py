"""
feature_registry.py
-------------------
Reusable features and signals.

Convention: every function returns a NEW DataFrame / Series; the input is never changed.
So always assign the result:   df = moving_avg_feat(df, "close", [20])
"""
from typing import Any

import numpy as np
import pandas as pd

def intraday_distribution(df: pd.DataFrame, col: str) -> dict[str, Any]:
    """
    Aggregate a column's mean/std by hour of day and day of week.
    Returns
    -------
    dict with keys: by_hour (DataFrame), by_day (DataFrame)
    """
    clean = df.copy()
    clean.index = pd.to_datetime(clean.index)
    clean["hour"] = clean.index.hour
    clean["day_of_week"] = clean.index.day_name()

    by_hour = clean.groupby("hour").agg(
        ret_mean=(col, "mean"),
        ret_std=(col, "std"),
        count=(col, "count"),
    )
    by_day = clean.groupby("day_of_week").agg(
        ret_mean=(col, "mean"),
        ret_std=(col, "std"),
        count=(col, "count"),
    )

    return {"by_hour": by_hour, "by_day": by_day}

# add volume features
def volume_features(
    df: pd.DataFrame,
    col: str = "volume",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add rolling volume context: the rolling mean, volume relative to it, and a
    rolling z-score.
    Returns a new frame; `df` is not modified.
    """
    windows = windows or [20]
    out = df.copy()
    for window in windows:
        rolling = out[col].rolling(window)
        mean = rolling.mean()
        std = rolling.std()
        out[f"{col}_mean_{window}"] = mean
        out[f"{col}_rel_{window}"] = out[col] / mean
        out[f"{col}_z_{window}"] = (out[col] - mean) / std
    return out

# add time based features
def calculate_time_features(df: pd.DataFrame, session_hour: int = 1) -> pd.DataFrame:
    """Hour, minutes since the session open, and day of week from the index."""
    df = df.copy()
    df["hour"] = df.index.hour
    df["minutes_since_open"] = np.where(df.index.hour >= session_hour,
    (df.index.hour - session_hour) * 60 + df.index.minute,0)
    df["day_of_week"] = df.index.dayofweek
    return df



###
# Feature sets
def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score using differences, so a series that crosses zero is fine."""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)   # flat window: undefined, not inf


def moving_avg_feat(df: pd.DataFrame, col: str, windows: list[int], z_window: int = 252) -> pd.DataFrame:
    """Distance from a moving average, z-scored.

    This measures how STRETCHED price is from its own mean. It is a level,
    not a direction - see ma_momentum_feat for direction.
    """
    df = df.copy()
    for window in windows:
        moving_average = df[col].rolling(window).mean()
        df[f"{col}_ma_{window}"] = moving_average

        distance = df[col] - moving_average
        df[f"{col}_ma_dist_z_{window}"] = zscore(distance, z_window)

    return df


def ma_momentum_feat(df: pd.DataFrame, col: str, cross_pairs: list[tuple], above_windows: list[int],
                     macd_spans: tuple = (12, 26), z_window: int = 252) -> pd.DataFrame:
    """Direction and persistence of the trend, rather than distance from it.

    cross_pairs   - list of (fast, slow) window pairs, e.g. [(10, 50)]
    above_windows - share of the last N days spent above the N-day average
    macd_spans    - (fast, slow) spans for the EMA difference
    """
    df = df.copy()
    for fast, slow in cross_pairs:
        fast_average = df[col].rolling(fast).mean()
        slow_average = df[col].rolling(slow).mean()
        df[f"{col}_macross_{fast}_{slow}"] = zscore(fast_average - slow_average, z_window)

    for window in above_windows:
        is_above = df[col] > df[col].rolling(window).mean()
        df[f"{col}_above_ma_{window}"] = is_above.rolling(window).mean()

    fast_span, slow_span = macd_spans
    macd = df[col].ewm(span=fast_span).mean() - df[col].ewm(span=slow_span).mean()
    df[f"{col}_macd"] = zscore(macd, z_window)

    return df


def seasonality_feat(df: pd.DataFrame) -> pd.DataFrame:
    """Day of year as a sine/cosine pair, so December and January sit next to each other."""
    df = df.copy()
    day_of_year = df.index.dayofyear
    df["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return df


def vol_regime_feat(df: pd.DataFrame, col: str, windows: list[int],
                    regime_window: int = 252) -> pd.DataFrame:
    """Realised volatility over several windows, plus where the last one sits historically.

    vol_regime is a percentile rank, so 0.9 means today is more volatile than
    90 percent of the trailing window.
    """
    df = df.copy()
    returns = df[col].diff()

    for window in windows:
        df[f"vol_{window}"] = returns.rolling(window).std()

    last_window = windows[-1]
    df["vol_regime"] = df[f"vol_{last_window}"].rolling(regime_window).rank(pct=True)

    return df


def volume_feat(df: pd.DataFrame, col: str, ratio_window: int = 20, short_window: int = 5,
                long_window: int = 20) -> pd.DataFrame:
    """Volume against its own recent average, computed within each contract.

    Volume is grouped by expiry because a contract's volume ramps up while it is
    the front month and falls away into delivery - that lifecycle is contract
    specific, so it must not be carried across a roll.

    Keep long_window well under the typical contract length or most rows come
    back empty. The median contract here is about 98 rows.
    """
    df = df.copy()
    grouped = df.groupby("expiry")[col]

    df["volume_ratio"] = df[col] / grouped.transform(lambda s: s.rolling(ratio_window).mean())

    short_average = grouped.transform(lambda s: s.rolling(short_window).mean())
    long_average = grouped.transform(lambda s: s.rolling(long_window).mean())
    df["volume_trend"] = short_average / long_average

    return df


def days_to_harvest_feat(df: pd.DataFrame, harvest_month: int = 4) -> pd.DataFrame:
    """Calendar days until the next harvest window (roughly April for SA maize).

    Dates already past this year's harvest point at next year's.
    """
    df = df.copy()
    year = df.index.year + (df.index.month > harvest_month).astype(int)
    next_harvest = pd.to_datetime(dict(year=year, month=harvest_month, day=1))

    days = (next_harvest.values - df.index.values) / np.timedelta64(1, "D")
    df["days_to_harvest"] = days

    return df


def forward_horizons(df: pd.DataFrame, col: str, horizons: list[int],
                     vol_window: int = 20) -> pd.DataFrame:
    """Forward returns, scaled by expected volatility, one column per horizon.

    `col` must be a LOG PRICE LEVEL (a clog_ column), not a return series -
    the forward return is a difference of two points on that level.

    Dividing by expected vol puts every horizon on the same scale, so a 1-day
    and a 10-day target are comparable.
    """
    df = df.copy()
    returns = df[col].diff()

    for horizon in horizons:
        forward_return = df[col].shift(-horizon) - df[col]
        expected_vol = returns.rolling(vol_window).std() * np.sqrt(horizon)
        df[f"target_{horizon}_d"] = forward_return / expected_vol

    return df

# Hedged beta spread
def rolling_hedge_ratio_naive(df: pd.DataFrame, base_col: str, quote_col: str, window: int = 120,
                              rebalance_days: int = 5) -> pd.Series:
    """Estimate beta by rolling OLS, then freeze it between rebalances.

    """
    base = df[base_col]
    quote = df[quote_col]

    # Rolling OLS slope: cov(base, quote) / var(base).
    covariance = quote.rolling(window).cov(base)
    variance = base.rolling(window).var()
    beta_raw = covariance / variance

    # Freeze: only let beta update every `rebalance_days` bars.
    beta_frozen = beta_raw.copy()
    beta_frozen[:] = np.nan
    beta_frozen.iloc[::rebalance_days] = beta_raw.iloc[::rebalance_days]
    return beta_frozen.ffill()

def hedged_spread(df: pd.DataFrame, base_col: str, quote_col: str, window: int = 120,
                  rebalance_days: int = 5) -> tuple[pd.Series, pd.Series]:
    """Continuous hedged spread built from returns, not level differences.

    Differencing the level across a beta change measures the revaluation from
    re-hedging, not a market move -- the same trap as differencing across a
    roll. So take the hedged return each bar using the beta already in force,
    then sum back up into a continuous series.
    """
    base_ret = df[base_col].diff()
    quote_ret = df[quote_col].diff()

    # Estimate on returns: regressing log levels on log levels is spurious.
    covariance = quote_ret.rolling(window).cov(base_ret)
    variance = base_ret.rolling(window).var()
    beta_raw = covariance / variance

    # Freeze between rebalances.
    beta_frozen = pd.Series(np.nan, index=df.index)
    beta_frozen.iloc[::rebalance_days] = beta_raw.iloc[::rebalance_days]
    beta_frozen = beta_frozen.ffill()

    # Apply the beta that was already in force at the start of the bar, so the
    # hedge ratio is never set with knowledge of the bar it prices.
    hedged_return = quote_ret - beta_frozen.shift(1) * base_ret

    return beta_frozen, hedged_return.cumsum()

def ou_half_life(series: pd.Series, window: int = 60) -> pd.Series:
    """Half-life of mean reversion, from an AR(1) fit on the rolling window.

    Regress the change on the lagged level: delta_t = a + b * level_{t-1}.
    A negative b means reversion; half-life = -ln(2)/ln(1+b). Short half-life
    means reversion trades make sense, long or undefined means they don't.
    """
    def fit(x):
        level = x[:-1]
        delta = np.diff(x)
        if np.var(level) == 0:
            return np.nan
        b = np.cov(level, delta)[0, 1] / np.var(level)
        if b >= 0 or (1 + b) <= 0:
            return np.nan          # not reverting over this window
        return -np.log(2) / np.log(1 + b)

    return series.rolling(window).apply(fit, raw=True)


def variance_ratio(series: pd.Series, window: int = 60, lag: int = 5) -> pd.Series:
    """Variance of k-period returns divided by k times variance of 1-period.

    Above 1 means trending (moves compound), below 1 means mean-reverting
    (moves offset), around 1 means random walk.
    """
    returns_1 = series.diff()
    returns_k = series.diff(lag)
    var_1 = returns_1.rolling(window).var()
    var_k = returns_k.rolling(window).var()
    return var_k / (lag * var_1)


def cross_leg_correlation(df: pd.DataFrame, col_a: str, col_b: str, window: int = 60) -> pd.Series:
    """Rolling correlation between the two legs' returns.

    A direct measure of whether the arb relationship is holding. Falling
    correlation is the decoupling regime where fading the spread is dangerous.
    """
    return df[col_a].diff().rolling(window).corr(df[col_b].diff())


def relative_volatility(df: pd.DataFrame, col_a: str, col_b: str, window: int = 60) -> pd.Series:
    """Which leg is driving. Ratio of the two legs' realised vol."""
    vol_a = df[col_a].diff().rolling(window).std()
    vol_b = df[col_b].diff().rolling(window).std()
    return vol_a / vol_b

# ═══════════════════════════════════════════════════════════════════════
# SIGNALS — 30 formulas as plain pandas from Delphic Alpha
# ═══════════════════════════════════════════════════════════════════════

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()


# --- Standard Atlas (4 families) ---

def fade_breakout(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Short at new highs, long at new lows. Fade exhaustion moves."""
    c = df["close"]
    return_trade = (c - df['open']) / df['open']
    prev_high = c.rolling(window).max().shift(1)
    prev_low = c.rolling(window).min().shift(1)
    signal = pd.Series(0.0, index=df.index)
    signal[c > prev_high] = -1.0
    signal[c < prev_low] = 1.0
    strat_return = signal.shift(1) * return_trade
    position = signal.shift(1)
    return pd.DataFrame({"signal": signal, "strategy_return": strat_return, "bar_return": return_trade, "position": position})


def fade_ma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Short above MA, long below MA. Mean reversion."""
    c = df["close"]
    ma = c.rolling(window).mean()
    signal = pd.Series(0.0, index=df.index)
    signal[c > ma] = -1.0
    signal[c < ma] = 1.0
    return signal


def cs_reversion(prices: dict[str, pd.DataFrame], window: int = 20) -> dict[str, pd.Series]:
    """Cross-sectional reversion: fade relative strength across assets.
    Requires multiple assets. Returns dict of {ticker: signal}.
    """
    ratios = {}
    for ticker, df in prices.items():
        c = df["close"]
        ma = c.rolling(window).mean()
        ratios[ticker] = c / ma
    ratio_df = pd.DataFrame(ratios)
    mean = ratio_df.mean(axis=1)
    std = ratio_df.std(axis=1).replace(0, np.nan)
    zscores = ratio_df.sub(mean, axis=0).div(std, axis=0)
    return {ticker: -zscores[ticker] for ticker in prices}


def mean_reversion(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Negative z-score of close. Short when extended, long when depressed."""
    return -zscore(df["close"], window)


# --- Exotic Atlas (6 families) ---

def skew_fade(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Negative rolling skewness. Fade distributional asymmetry."""
    return -df["close"].rolling(window).skew()


def keltner_reversion(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Deviation from MA normalised by ATR. Robust mean reversion."""
    c = df["close"]
    ma = c.rolling(window).mean()
    atr = _atr(df["high"], df["low"], c, window)
    return -(c - ma) / atr.replace(0, np.nan)


def accel_fade(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Fade the second derivative of price. Captures momentum exhaustion."""
    accel = df["close"].diff().diff()
    return -zscore(accel, window)


def ret_z_reversion(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Z-score of 1-bar returns. Fade extreme recent moves."""
    c = df["close"]
    return -zscore(c / c.shift(1), window)


def skewkurt_fade(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Skewness x kurtosis. Combines direction with tail thickness."""
    c = df["close"]
    return -c.rolling(window).skew() * c.rolling(window).kurt()


def fractal_trend(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Range efficiency x z-score. The only trend-following signal."""
    c = df["close"]
    range_ = c.rolling(window).max() - c.rolling(window).min()
    noise = c.diff().abs().rolling(window).sum()
    efficiency = range_ / noise.replace(0, np.nan)
    return efficiency * zscore(c, window)


# --- Signal registry ---

LOOKBACKS = [5, 20, 120]

_FAMILIES = [
    (fade_breakout,     "Fade-BO",       False),
    (fade_ma,           "Fade-MA",       False),
    (cs_reversion,      "CS-Rev",        True),
    (mean_reversion,    "MR",            False),
    (skew_fade,         "Skew-Fade",     False),
    (keltner_reversion, "Keltner-Rev",   False),
    (accel_fade,        "Accel-Fade",    False),
    (ret_z_reversion,   "Ret-Z-Rev",     False),
    (skewkurt_fade,     "SkewKurt",      False),
    (fractal_trend,     "Fractal-Trend", False),
]

SIGNAL_REGISTRY = []
for _func, _family, _cs in _FAMILIES:
    for _lb in LOOKBACKS:
        SIGNAL_REGISTRY.append({
            "name": f"{_family} {_lb}", "family": _family,
            "lookback": _lb, "function": _func, "cross_sectional": _cs,
        })


def list_signals() -> pd.DataFrame:
    return pd.DataFrame([{"name": s["name"], "family": s["family"], "lookback": s["lookback"]}
                         for s in SIGNAL_REGISTRY])