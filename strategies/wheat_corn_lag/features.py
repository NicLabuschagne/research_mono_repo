"""
features.py
-----------
Step 4: everything the signal may see at the close of bar t, and (separately) the targets.

    python -m strategies.wheat_corn_lag.features      # coverage and sanity summary, no forward returns

Units: lookbacks L and horizons h are counted in 15-minute bars (4 = 1 hour, 71 = about one session).
Every L-bar change is taken INSIDE a segment (strategy.py): a window that would cross a possible roll is NaN.
Windows may cross the overnight close and weekends wherever the series is stitched.

Per-bar statistics: rolling over `stats_window_bars`, then lagged one bar, so bar t never sees itself.
  sigma_w, sigma_c   std of 15-minute log returns
  beta               cov(ret_w, ret_c) / var(ret_c): the rolling OLS hedge ratio        (F1)
  sigma_e            std of wheat's own per-bar move, ret_w - beta * ret_c
  beta_kalman        random-walk hedge ratio, the estimate BEFORE bar t is seen          (F2)

Features per lookback L (all known at the close of bar t)
  corn_z_L    corn's move over L, in corn sd:                 dlog_c / (sigma_c * sqrt L)
  own_z_L     wheat's own move over L, in its own sd:         (dlog_w - beta * dlog_c) / (sigma_e * sqrt L)
  own_kz_L    the same with the Kalman beta
  own_move_L  wheat's own move over L in log units (dlog_w - beta * dlog_c): its size is the divergence D
  own_ze_L    own_move_L / its ACTUAL sd of L-bar moves over the stats window ("empirical" scaling).
  corn_ze_L   dlog_c / its actual sd of L-bar moves. sqrt(L) assumes a random walk and overstates the sd
              because 15-minute moves partly reverse; the empirical scale held a smooth plateau (Step 6)
  f0_L        the trader's literal rule on 1.5W - C and W - 1.5C (cents):
                +2 / -2  both spreads moved opposite to wheat (the trader's strongest)
                +1 / -1  the two spreads diverged
                 0       no signal                              (+ = buy wheat, - = sell wheat)
  F3 (more factors: Matif, KC wheat, DXY) waits for data.

Layer 1 candidate measures (fair value models, judged by IC only in layer1.py)
  own_kze_L   model B: wheat's own move with the KALMAN beta, empirical scaling
  wheat_ze_L  control: wheat's plain move over L, empirical scaling, NO corn at all
  multi-factor fair values (one per factor set in config: c corn, s soybeans, u USD)
  beta_{set}_{f}         OLS of ret_w on the set's factor returns (with intercept), on the stats window ending
                         at t-1; rows with any missing return are left out of the fit
  own_ze_{set}_L         wheat's move minus sum(beta x factor move) over L, empirical scaling
  explained_ze_{set}_L   sum(beta x factor move) over L, empirical scaling (the part the fair value explains)
  kgap_{M}s, kz_{M}s   model K, Kalman LEVEL fair value (the bean_divergence convention, designed for wheat):
              index_w = alpha + beta x index_c on the roll-free indices, alpha and beta a random walk (as
              crush_strategy.kalman_filter: start state and noise R from a warm-up OLS, drift-to-noise ratio d).
              kgap = index_w(t) - fair value from the state BEFORE bar t (the "live" gap: the filter's one-step
              prediction error). Speed: memory of about M sessions, d = 1 / (M x 71)^2 (crush uses d = 0.1 per
              15 min, a memory of a few bars, for its sub-hour edge; wheat's edge is 2-4 sessions).
              kz = kgap / its rolling sd over the stats window (20 sessions), bars BEFORE t only: the SAME scaling
              as model A, so the two are compared like for like (2026-11-02; a 5-40 session exponential RMS gave
              about the same IC). kbeta = the hedge ratio used for that bar's gap (for the spread wheat - beta corn).
              kzr = kgap / its rolling RMS around ZERO over the same window. The gap is a level that can sit on
              one side for weeks; the sd is taken around the window's own mean, so it shrinks while the gap sits
              there and kz stretches (13% of bars beyond |2|). RMS keeps the scale "typical size of the gap".
              Other ways to size the same gap (layer1 scale comparison, 2026-11-02):
              kgapc = gap x wheat price (cents); kzv = gap / wheat's typical 1-session move (sigma_w x sqrt 71);
              kzk = gap / the filter's own prediction sd, sqrt(x P x' + R) (fixed R from the warm-up).
  level_z_W   model C, level fair value: on roll-free log price indices (running totals of the per-bar
              returns; a return across a possible-roll cut counts as 0), fit index_w = a + b x index_c over the
              W bars ending at t-1, fair(t) = a + b x index_c(t). level_z = (index_w - fair) / its past sd.

Targets (FUTURE prices, never a feature): fwd_h = log_w(t+h) - log_w(t), NaN if t+h is in another segment.
"""
import numpy as np
import pandas as pd

from strategies.wheat_corn_lag.strategy import FeatureConfig, load, load_bars

BARS_PER_SESSION = 71


def change_over(series: pd.Series, segment: pd.Series, bars_back: int) -> pd.Series:
    """series(t) - series(t - bars_back); NaN when t - bars_back is in another segment."""
    return series - series.groupby(segment).shift(bars_back)


def change_ahead(series: pd.Series, segment: pd.Series, bars_ahead: int) -> pd.Series:
    """series(t + bars_ahead) - series(t); NaN when t + bars_ahead is in another segment. FUTURE."""
    return series.groupby(segment).shift(-bars_ahead) - series


def lagged_sd(series: pd.Series, window: int) -> pd.Series:
    """sd over the `window` bars ending at t-1."""
    return series.rolling(window, min_periods=window // 2).std().shift(1)


def lagged_rms(series: pd.Series, window: int) -> pd.Series:
    """Root mean square around zero over the `window` bars ending at t-1."""
    return np.sqrt((series ** 2).rolling(window, min_periods=window // 2).mean()).shift(1)


def rolling_stats(bars: pd.DataFrame, window: int) -> pd.DataFrame:
    """sigma_w, sigma_c, beta, sigma_e from the `window` bars ending at t-1."""
    ret_w, ret_c = bars["ret_w"], bars["ret_c"]
    min_periods = window // 2
    var_c = ret_c.rolling(window, min_periods=min_periods).var()
    beta = ret_w.rolling(window, min_periods=min_periods).cov(ret_c) / var_c
    # wheat's own per-bar move, using the beta known before that bar
    own = ret_w - beta.shift(1) * ret_c
    stats = pd.DataFrame({
        "sigma_w": ret_w.rolling(window, min_periods=min_periods).std(),
        "sigma_c": np.sqrt(var_c),
        "beta": beta,
        "sigma_e": own.rolling(window, min_periods=min_periods).std(),
    })
    return stats.shift(1)                      # everything at t comes from bars up to t-1


def kalman_beta(bars: pd.DataFrame, beta_sd_per_session: float, warmup_bars: int) -> pd.Series:
    """Hedge ratio that drifts as a random walk: ret_w(t) = beta(t) * ret_c(t) + noise.

    The value at bar t is the estimate made BEFORE bar t's returns are used (the same lag as rolling_stats).
    Start: a plain least-squares beta and noise level on the first `warmup_bars` valid bars; no output there.
    """
    wheat, corn = bars["ret_w"].values, bars["ret_c"].values
    valid = np.isfinite(wheat) & np.isfinite(corn)
    warmup = np.flatnonzero(valid)[:warmup_bars]
    beta = (wheat[warmup] @ corn[warmup]) / (corn[warmup] @ corn[warmup])
    observation_noise = np.var(wheat[warmup] - beta * corn[warmup])
    process_noise = beta_sd_per_session ** 2 / BARS_PER_SESSION       # beta variance added per bar
    beta_variance = 1.0                                                 # very unsure at the start

    estimates = np.full(len(wheat), np.nan)
    for t in range(len(wheat)):
        beta_variance += process_noise                  # predict: beta may have drifted
        if t > warmup[-1]:
            estimates[t] = beta                         # record BEFORE using bar t
        if valid[t]:                                    # update with bar t
            gain = beta_variance * corn[t] / (corn[t] ** 2 * beta_variance + observation_noise)
            beta += gain * (wheat[t] - beta * corn[t])
            beta_variance -= gain * corn[t] * beta_variance
    return pd.Series(estimates, index=bars.index, name="beta_kalman")


def literal_rule(bars: pd.DataFrame, bars_back: int, ratio: float, flat_sd: float, window: int) -> pd.Series:
    """The trader's rule on the two fixed-ratio spreads, in cents. See the module docstring for the codes."""
    segment = bars["segment"]
    series = {
        "heavy_wheat": ratio * bars["W"] - bars["C"],     # 1.5W - C
        "heavy_corn": bars["W"] - ratio * bars["C"],      # W - 1.5C
        "wheat": bars["W"],
    }
    up, down = {}, {}
    for name, values in series.items():
        move = change_over(values, segment, bars_back)
        # "flat" = within flat_sd of a typical L-bar move, measured on past bars only
        typical = values.groupby(segment).diff().rolling(window, min_periods=window // 2).std().shift(1)
        threshold = flat_sd * typical * np.sqrt(bars_back)
        up[name] = move > threshold
        down[name] = move < -threshold

    diverge_buy = up["heavy_corn"] & ~up["heavy_wheat"]                  # W-1.5C up, 1.5W-C flat or down
    diverge_sell = down["heavy_corn"] & ~down["heavy_wheat"]
    strong_buy = up["heavy_corn"] & up["heavy_wheat"] & ~up["wheat"]       # both up, wheat flat or down
    strong_sell = down["heavy_corn"] & down["heavy_wheat"] & ~down["wheat"]

    signal = (2 * strong_buy + diverge_buy - 2 * strong_sell - diverge_sell).astype(float)
    available = change_over(bars["W"], segment, bars_back).notna()
    return signal.where(available)


def build_features(bars: pd.DataFrame, settings: FeatureConfig) -> pd.DataFrame:
    stats = rolling_stats(bars, settings.stats_window_bars)
    beta_k = kalman_beta(bars, settings.kalman_beta_sd_per_session, settings.kalman_warmup_bars)
    features = stats.assign(beta_kalman=beta_k)

    segment = bars["segment"]
    for lookback in settings.lookbacks_bars:
        move_w = change_over(bars["log_w"], segment, lookback)
        move_c = change_over(bars["log_c"], segment, lookback)
        root = np.sqrt(lookback)
        features[f"corn_z_{lookback}"] = move_c / (stats["sigma_c"] * root)
        features[f"own_z_{lookback}"] = (move_w - stats["beta"] * move_c) / (stats["sigma_e"] * root)
        features[f"own_kz_{lookback}"] = (move_w - beta_k * move_c) / (stats["sigma_e"] * root)
        own_move = move_w - stats["beta"] * move_c
        features[f"own_move_{lookback}"] = own_move
        features[f"own_ze_{lookback}"] = own_move / lagged_sd(own_move, settings.stats_window_bars)
        features[f"corn_ze_{lookback}"] = move_c / lagged_sd(move_c, settings.stats_window_bars)
        own_move_kalman = move_w - beta_k * move_c
        features[f"own_kze_{lookback}"] = own_move_kalman / lagged_sd(own_move_kalman, settings.stats_window_bars)
        features[f"wheat_ze_{lookback}"] = move_w / lagged_sd(move_w, settings.stats_window_bars)
        features[f"f0_{lookback}"] = literal_rule(bars, lookback, settings.f0_ratio, settings.f0_flat_sd,
                                                  settings.stats_window_bars)

    for window in settings.level_windows_bars:
        features[f"level_z_{window}"] = level_fair_value_z(bars, window)
    for memory in settings.kalman_memory_sessions:
        gap, beta, prediction_sd = kalman_level_gap(bars, memory * BARS_PER_SESSION, settings.kalman_warmup_bars)
        features[f"kgap_{memory}s"] = gap
        features[f"kbeta_{memory}s"] = beta
        features[f"kz_{memory}s"] = gap / lagged_sd(gap, settings.stats_window_bars)
        features[f"kzr_{memory}s"] = gap / lagged_rms(gap, settings.stats_window_bars)
        features[f"kgapc_{memory}s"] = gap * bars["W"]                                   # log gap x price = cents
        features[f"kzv_{memory}s"] = gap / (stats["sigma_w"] * np.sqrt(BARS_PER_SESSION))
        features[f"kzk_{memory}s"] = gap / prediction_sd

    # multi-factor fair values
    factor_level = {"c": bars["log_c"], "s": bars["index_s"], "u": bars["index_u"]}
    window = settings.stats_window_bars
    columns = {}                                   # collected, then added at once (one insert per column is slow)
    for name, factors in settings.factor_sets.items():
        betas = rolling_betas(bars, factors, window)
        for factor in factors:
            columns[f"beta_{name}_{factor}"] = betas[factor]
        for lookback in settings.lookbacks_bars:
            explained = sum(betas[factor] * change_over(factor_level[factor], segment, lookback) for factor in factors)
            own_move = change_over(bars["log_w"], segment, lookback) - explained
            columns[f"own_ze_{name}_{lookback}"] = own_move / lagged_sd(own_move, window)
            columns[f"explained_ze_{name}_{lookback}"] = explained / lagged_sd(explained, window)
    return pd.concat([features, pd.DataFrame(columns)], axis=1)


def rolling_betas(bars: pd.DataFrame, factors: list[str], window: int) -> pd.DataFrame:
    """OLS slopes of ret_w on the factor returns (intercept included) over the `window` bars ending at t-1.

    Done with running sums of the cross-products (X'X and X'y), so each bar costs one small solve.
    Rows where wheat or any factor is missing are left out; a fit needs at least half a window of rows.
    """
    wheat = bars["ret_w"].values
    x = np.column_stack([bars[f"ret_{factor}"].values for factor in factors])
    valid = np.isfinite(wheat) & np.isfinite(x).all(axis=1)
    columns = np.column_stack([np.ones(len(wheat)), x, wheat])           # [1, factors..., wheat]
    columns = np.where(valid[:, None], columns, 0.0)
    products = columns[:, :, None] * columns[:, None, :]
    running = np.concatenate([np.zeros((1,) + products.shape[1:]), np.cumsum(products, axis=0)])
    # window sums ENDING AT t-1: rows t-window .. t-1
    count = len(wheat)
    ends = np.arange(count)
    starts = np.maximum(ends - window, 0)
    sums = running[ends] - running[starts]
    size = len(factors) + 1
    xtx, xty = sums[:, :size, :size], sums[:, :size, size]
    enough = sums[:, 0, 0] >= window // 2                                 # [0, 0] is the count of rows used
    slopes = np.full((count, len(factors)), np.nan)
    slopes[enough] = np.linalg.solve(xtx[enough], xty[enough][..., None])[:, 1:, 0]
    return pd.DataFrame(slopes, index=bars.index, columns=factors)


def kalman_level_gap(bars: pd.DataFrame, memory_bars: int, warmup_bars: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Model K: live gap of the wheat index to a Kalman fair value on the corn index, the beta behind it and the
    filter's own sd of that prediction (all from the state BEFORE the bar; see the module docstring)."""
    y = bars["ret_w"].fillna(0).cumsum().values                    # roll-free log index: a cut return counts as 0
    x = np.column_stack([np.ones(len(y)), bars["ret_c"].fillna(0).cumsum().values])
    state = np.linalg.lstsq(x[:warmup_bars], y[:warmup_bars], rcond=None)[0]
    observation_noise = np.var(y[:warmup_bars] - x[:warmup_bars] @ state)
    typical_xx = np.mean(np.sum(x[:warmup_bars] ** 2, axis=1))
    drift_to_noise = 1.0 / memory_bars ** 2
    process_noise = drift_to_noise * observation_noise / typical_xx * np.eye(2)
    state_covariance = np.eye(2)

    gap = np.full(len(y), np.nan)
    beta = np.full(len(y), np.nan)
    prediction_sd = np.full(len(y), np.nan)
    for t in range(len(y)):
        state_covariance = state_covariance + process_noise        # predict: the state may have drifted
        error = y[t] - x[t] @ state                                 # live gap: state from BEFORE bar t
        if t >= warmup_bars:
            gap[t], beta[t] = error, state[1]
            prediction_sd[t] = np.sqrt(x[t] @ state_covariance @ x[t] + observation_noise)
        gain = state_covariance @ x[t] / (x[t] @ state_covariance @ x[t] + observation_noise)
        state = state + gain * error                                # update with bar t, used from bar t+1
        state_covariance = state_covariance - np.outer(gain, x[t]) @ state_covariance
    return pd.Series(gap, index=bars.index), pd.Series(beta, index=bars.index), pd.Series(prediction_sd, index=bars.index)


def level_fair_value_z(bars: pd.DataFrame, window: int) -> pd.Series:
    """Model C: wheat's distance from a level fair value on corn, in its own past sd (see the docstring)."""
    index_w = bars["ret_w"].fillna(0).cumsum()
    index_c = bars["ret_c"].fillna(0).cumsum()
    min_periods = window // 2
    slope = (index_w.rolling(window, min_periods=min_periods).cov(index_c)
             / index_c.rolling(window, min_periods=min_periods).var())
    intercept = index_w.rolling(window, min_periods=min_periods).mean() - slope * index_c.rolling(window, min_periods=min_periods).mean()
    fair = intercept.shift(1) + slope.shift(1) * index_c          # parameters from bars up to t-1, corn at t
    gap = index_w - fair
    return gap / lagged_sd(gap, window)


def build_targets(bars: pd.DataFrame, settings: FeatureConfig) -> pd.DataFrame:
    """Forward wheat log returns. FUTURE information: only ever used as the thing being predicted."""
    return pd.DataFrame({f"fwd_{h}": change_ahead(bars["log_w"], bars["segment"], h)
                         for h in settings.horizons_bars})


if __name__ == "__main__":
    cfg = load()
    bars = load_bars(cfg)
    features = build_features(bars, cfg.features)
    targets = build_targets(bars, cfg.features)
    pd.set_option("display.width", 200)

    print("== coverage (share of bars with a value)")
    print(features.notna().mean().round(3).to_string())
    print("\n== targets: share of bars with a value")
    print(targets.notna().mean().round(3).to_string())

    print("\n== hedge ratio by year: rolling OLS vs Kalman")
    year = features.index.year
    print(features.groupby(year)[["beta", "beta_kalman"]].agg(["mean", "min", "max"]).round(2).to_string())
    print(f"correlation rolling vs Kalman beta: {features['beta'].corr(features['beta_kalman']):.2f}")

    print("\n== scaling check: sd of the z features (near 1 if the units are right)")
    z_columns = [c for c in features if c.startswith(("corn_z", "own_z", "own_kz", "corn_ze", "own_ze"))]
    print(features[z_columns].std().round(2).to_string())

    print("\n== F0: how often the literal rule fires (new signals per week, a signal = first bar of a run)")
    weeks = (bars.index[-1] - bars.index[0]).days / 7
    for lookback in cfg.features.lookbacks_bars:
        f0 = features[f"f0_{lookback}"]
        starts = f0.notna() & (f0 != 0) & (f0 != f0.shift())         # NaN != 0 is True: exclude NaN explicitly
        strong = starts & (f0.abs() == 2)
        print(f"  L={lookback:>3}: {starts.sum() / weeks:5.1f}/week, of which strongest (|2|) {strong.sum() / weeks:4.1f}/week,"
              f" buy share {(f0[starts] > 0).mean():.2f}")
