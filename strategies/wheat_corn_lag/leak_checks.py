"""
leak_checks.py
--------------
Automated look-ahead checks for Steps 1-4. Run after any change:

    python -m strategies.wheat_corn_lag.leak_checks

1. Truncation test (the main one)
   Rebuild bars, segments and features from data cut at T, as if today were T. Every feature at every bar
   up to T must equal the full-history run. A centred window, a missing shift, a full-sample normalisation,
   or a segment rule that knows the future (e.g. a roll window counted from the data's own end) all change
   when the future is removed, and show up here as a mismatch.
2. Independent recomputation
   For random bars, refit beta and sigma_c with numpy on exactly the window of bars t-window .. t-1 and
   rebuild corn_z; must equal the pipeline. Confirms the one-bar lag.
3. Targets
   fwd_h must equal log_w(t+h) - log_w(t) when t+h is in the same segment, and be NaN otherwise.
"""
import numpy as np
import pandas as pd

from strategies.wheat_corn_lag.features import build_features, build_targets
from strategies.wheat_corn_lag.strategy import load, load_bars

# inside a roll window, just after one, and mid-segment
TRUNCATION_DATES = ["2021-02-22 15:00", "2023-03-08 03:00", "2025-06-30 18:00"]
TOLERANCE = 1e-9


def max_difference(full: pd.Series, truncated: pd.Series) -> float:
    """Largest absolute difference on the truncated index; a NaN in one run but not the other counts as inf."""
    full = full.reindex(truncated.index)
    if (full.isna() != truncated.isna()).any():
        return np.inf
    difference = (full - truncated).abs()
    return float(difference.max()) if difference.notna().any() else 0.0


def truncation_test(cfg, full_bars, full_features) -> list[dict]:
    rows = []
    for cutoff in TRUNCATION_DATES:
        end = pd.Timestamp(cutoff, tz="UTC")
        bars = load_bars(cfg, end=end)
        features = build_features(bars, cfg.features)
        rows.append({"check": f"truncate {cutoff}", "item": "segment ids",
                     "max_diff": max_difference(full_bars["segment"].astype(float), bars["segment"].astype(float))})
        rows.append({"check": f"truncate {cutoff}", "item": "returns (ret_w, ret_c)",
                     "max_diff": max(max_difference(full_bars[c], bars[c]) for c in ["ret_w", "ret_c"])})
        worst = max(features.columns, key=lambda c: max_difference(full_features[c], features[c]))
        rows.append({"check": f"truncate {cutoff}", "item": f"all {features.shape[1]} features (worst: {worst})",
                     "max_diff": max_difference(full_features[worst], features[worst])})
    return rows


def numpy_refit(cfg, bars, features, samples: int = 300, seed: int = 7) -> list[dict]:
    window = cfg.features.stats_window_bars
    generator = np.random.default_rng(seed)
    ret_w, ret_c = bars["ret_w"].values, bars["ret_c"].values
    worst_beta = worst_sigma = worst_z = worst_ze = 0.0
    lookback = cfg.features.lookbacks_bars[2]
    corn_z = features[f"corn_z_{lookback}"].values
    corn_ze = features[f"corn_ze_{lookback}"].values
    log_c, segment = bars["log_c"].values, bars["segment"].values
    # corn's L-bar move at every bar, NaN when it would cross a segment edge
    corn_moves = np.full(len(bars), np.nan)
    same = segment[lookback:] == segment[:-lookback]
    corn_moves[lookback:][same] = (log_c[lookback:] - log_c[:-lookback])[same]
    for t in generator.integers(window + lookback + 1, len(bars), samples):
        x, y = ret_c[t - window:t], ret_w[t - window:t]          # ends at t-1: bar t is NOT used
        ok = np.isfinite(x) & np.isfinite(y)
        beta = np.cov(x[ok], y[ok])[0, 1] / np.var(x[ok], ddof=1)
        sigma_c = np.std(x[ok], ddof=1)
        worst_beta = max(worst_beta, abs(beta - features["beta"].values[t]))
        worst_sigma = max(worst_sigma, abs(sigma_c - features["sigma_c"].values[t]))
        if segment[t - lookback] == segment[t]:
            expected = (log_c[t] - log_c[t - lookback]) / (sigma_c * np.sqrt(lookback))
            worst_z = max(worst_z, abs(expected - corn_z[t]))
            # empirical scale: sd of the L-bar moves on bars t-window .. t-1 (bar t NOT used)
            past = corn_moves[t - window:t]
            scale = np.std(past[np.isfinite(past)], ddof=1)
            worst_ze = max(worst_ze, abs(corn_moves[t] / scale - corn_ze[t]))
    return [{"check": "numpy refit", "item": "beta", "max_diff": worst_beta},
            {"check": "numpy refit", "item": "sigma_c", "max_diff": worst_sigma},
            {"check": "numpy refit", "item": f"corn_z_{lookback}", "max_diff": worst_z},
            {"check": "numpy refit", "item": f"corn_ze_{lookback} (empirical scale)", "max_diff": worst_ze}]


def multifactor_refit(cfg, bars, features, samples: int = 200, seed: int = 5) -> list[dict]:
    """Refit each set's betas with numpy least squares on the complete rows t-window .. t-1."""
    window = cfg.features.stats_window_bars
    generator = np.random.default_rng(seed)
    rows = []
    for name, factors in cfg.features.factor_sets.items():
        y_all = bars["ret_w"].values
        x_all = np.column_stack([bars[f"ret_{f}"].values for f in factors])
        worst = 0.0
        for t in generator.integers(window + 1, len(bars), samples):
            y, x = y_all[t - window:t], x_all[t - window:t]            # ends at t-1: bar t is NOT used
            ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
            if ok.sum() < window // 2:
                continue
            design = np.column_stack([np.ones(ok.sum()), x[ok]])
            slopes = np.linalg.lstsq(design, y[ok], rcond=None)[0][1:]
            pipeline = np.array([features[f"beta_{name}_{f}"].values[t] for f in factors])
            worst = max(worst, float(np.max(np.abs(slopes - pipeline))))
        rows.append({"check": "numpy refit", "item": f"multi-factor betas {name}", "max_diff": worst})
    return rows


def target_check(cfg, bars, targets) -> list[dict]:
    rows = []
    log_w, segment = bars["log_w"].values, bars["segment"].values
    for h in cfg.features.horizons_bars:
        expected = np.full(len(bars), np.nan)
        same = segment[h:] == segment[:-h]
        expected[:-h][same] = (log_w[h:] - log_w[:-h])[same]
        rows.append({"check": "targets", "item": f"fwd_{h}",
                     "max_diff": max_difference(pd.Series(expected, bars.index), targets[f"fwd_{h}"])})
    return rows


if __name__ == "__main__":
    cfg = load()
    bars = load_bars(cfg)
    features = build_features(bars, cfg.features)
    targets = build_targets(bars, cfg.features)

    checks = pd.DataFrame(truncation_test(cfg, bars, features) + numpy_refit(cfg, bars, features)
                          + multifactor_refit(cfg, bars, features) + target_check(cfg, bars, targets))
    # numpy and pandas' rolling formulas agree to float precision, not bit for bit
    limit = np.where(checks["check"] == "numpy refit", 1e-8, TOLERANCE)
    checks["result"] = np.where(checks["max_diff"] <= limit, "PASS", "FAIL")
    pd.set_option("display.width", 200)
    print(checks.to_string(index=False))
    print(f"\n{(checks['result'] == 'PASS').sum()} of {len(checks)} checks passed")
